"""Project admitted MCP tools and revalidate bindings before remote execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Protocol

from .._json import FrozenJsonObject, canonical_json
from ..adapters.mcp import (
    MCPBindingState,
    MCPClient,
    MCPClientFactory,
    MCPError,
    MCPProtocolError,
    MCPRemoteToolError,
    MCPServerBinding,
    MCPToolBinding,
    mcp_binding_drift_reason,
)
from ..capabilities import (
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    ToolExecution,
    ToolOutput,
    ToolView,
    validate_tool_schema_value,
)
from ..capability_runtime import CapabilityFailure, SideEffectPlan
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput
from ..security import SecretProvider

MCP_DOMAIN_OWNER_ID = "mcp"
MCP_OUTPUT_KIND = "mcp.read.result"

_MCP_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 32,
        },
        "structured": {"type": "object"},
        "provenance": {
            "type": "object",
            "properties": {
                "binding_id": {"type": "string"},
                "binding_revision": {"type": "integer", "minimum": 1},
                "remote_tool_name": {"type": "string"},
                "input_schema_digest": {"type": "string"},
                "output_schema_digest": {"type": "string"},
                "call_id": {"type": "string"},
                "observed_at": {"type": "string"},
            },
            "required": [
                "binding_id",
                "binding_revision",
                "remote_tool_name",
                "input_schema_digest",
                "output_schema_digest",
                "call_id",
                "observed_at",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["text", "provenance"],
    "additionalProperties": False,
}


class MCPBindingStore(Protocol):
    async def load_mcp_binding(
        self,
        agent_id: str,
        binding_id: str,
    ) -> MCPServerBinding | None: ...

    async def list_mcp_bindings(
        self,
        agent_id: str,
    ) -> tuple[MCPServerBinding, ...]: ...


@dataclass(frozen=True, slots=True)
class MCPActivatedBinding:
    binding: MCPServerBinding
    lock: asyncio.Lock
    executor: MCPToolExecutor

    @property
    def client(self) -> MCPClient | None:
        return self.executor.client


class MCPToolExecutor:
    """Execute exact admitted tools for one binding after last-moment rechecks."""

    def __init__(
        self,
        *,
        binding: MCPServerBinding,
        client_factory: MCPClientFactory,
        secrets: SecretProvider,
        store: MCPBindingStore,
        clock: Callable[[], datetime],
        lock: asyncio.Lock,
    ) -> None:
        self.executor_id = binding.tools[0].executor_id
        self._binding = binding
        self._client_factory = client_factory
        self._secrets = secrets
        self._client: MCPClient | None = None
        self._store = store
        self._clock = clock
        self._lock = lock
        self._tools = {tool.capability_id: tool for tool in binding.tools}

    @property
    def client(self) -> MCPClient | None:
        return self._client

    async def execute(self, request: ToolExecution) -> ToolOutput:
        tool = self._tools.get(request.capability_id)
        if tool is None:
            raise MCPProtocolError(
                "mcp_binding_mismatch",
                "The MCP capability does not belong to its admitted binding.",
            )
        async with self._lock:
            current = await self._store.load_mcp_binding(
                self._binding.agent_id,
                self._binding.binding_id,
            )
            _require_current_binding(current, self._binding, tool)
            if self._client is None:
                self._client = self._client_factory.create(
                    endpoint=self._binding.endpoint,
                    authentication=self._binding.authentication,
                    secrets=self._secrets,
                )
            observed_at = self._clock()
            inspection = await self._client.inspect(observed_at=observed_at)
            drift = mcp_binding_drift_reason(self._binding, inspection)
            if drift is not None:
                raise MCPProtocolError(
                    "mcp_binding_stale",
                    "The MCP server identity or admitted tool schema changed.",
                    {"binding_id": self._binding.binding_id, "reason": drift},
                )
            result = await self._client.call_tool(tool.remote_name, request.arguments)
            if result.is_error:
                raise MCPRemoteToolError(
                    "mcp_remote_tool_error",
                    "The admitted MCP read tool returned an application error.",
                    {
                        "binding_id": self._binding.binding_id,
                        "remote_tool_name": tool.remote_name,
                        "remote_text": result.text,
                    },
                )
            if tool.output_schema is not None:
                if result.structured is None:
                    raise MCPProtocolError(
                        "mcp_result_schema_mismatch",
                        "The MCP tool omitted its admitted structured result.",
                    )
                try:
                    validate_tool_schema_value(tool.output_schema, result.structured)
                except (TypeError, ValueError, RuntimeError):
                    raise MCPProtocolError(
                        "mcp_result_schema_mismatch",
                        "The MCP structured result did not match its admitted schema.",
                    ) from None
            provenance = {
                "binding_id": self._binding.binding_id,
                "binding_revision": self._binding.revision,
                "remote_tool_name": tool.remote_name,
                "input_schema_digest": tool.input_schema_digest,
                "output_schema_digest": tool.output_schema_digest or "none",
                "call_id": request.call_id,
                "observed_at": observed_at.isoformat(),
            }
            data: dict[str, object] = {
                "text": result.text,
                "provenance": provenance,
            }
            if result.structured is not None:
                data["structured"] = result.structured
            sensitivity = max(
                ModelSensitivity.INTERNAL,
                tool.result_sensitivity,
                key=lambda item: item.routing_rank,
            )
            return ToolOutput(
                kind=MCP_OUTPUT_KIND,
                data=data,
                sensitivity=sensitivity,
                sensitivity_provenance={
                    "authority": "mcp_binding_admission",
                    "binding_id": self._binding.binding_id,
                    "binding_revision": self._binding.revision,
                    "capability_id": tool.capability_id,
                },
            )

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.close()


class MCPCapabilityDomain:
    """Own MCP projection, current admission, sensitivity, and safe failures."""

    domain_owner_id = MCP_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        *,
        agent_id: str,
        bindings: tuple[MCPActivatedBinding, ...],
        store: MCPBindingStore,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("MCP declarations belong to another domain")
        self._declarations = declarations
        self._agent_id = agent_id
        self._store = store
        self._binding_by_capability = {
            tool.capability_id: (activated.binding, tool)
            for activated in bindings
            for tool in activated.binding.tools
        }
        self._local_name_by_capability = {
            tool.capability_id: tool.local_name
            for activated in bindings
            for tool in activated.binding.tools
        }

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        if run.agent_id != self._agent_id:
            return ()
        projected: list[str] = []
        bindings: dict[str, MCPServerBinding] = {}
        for binding, _tool in self._binding_by_capability.values():
            bindings[binding.binding_id] = binding
        current = {
            binding.binding_id: await self._store.load_mcp_binding(
                self._agent_id,
                binding.binding_id,
            )
            for binding in bindings.values()
        }
        for capability_id, (binding, _tool) in self._binding_by_capability.items():
            if _binding_revision_is_active(current[binding.binding_id], binding):
                projected.append(self._local_name_by_capability[capability_id])
        return tuple(sorted(projected))

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        del capability
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        del call
        if run.agent_id != self._agent_id:
            raise CapabilityInputError(
                "mcp_binding_unavailable",
                "The MCP binding does not belong to this agent.",
            )
        admitted = self._binding_by_capability.get(capability.id)
        if admitted is None:
            raise CapabilityInputError(
                "mcp_binding_unavailable",
                "The MCP capability is not admitted in this runtime.",
            )
        binding, tool = admitted
        current = await self._store.load_mcp_binding(
            self._agent_id,
            binding.binding_id,
        )
        try:
            _require_current_binding(current, binding, tool)
        except MCPError as error:
            raise CapabilityInputError(error.code, str(error), error.details) from error
        if (
            request_sensitivity.routing_rank
            > binding.maximum_outbound_sensitivity.routing_rank
        ):
            raise CapabilityInputError(
                "mcp_outbound_sensitivity_exceeded",
                "The MCP call exceeds the binding's outbound sensitivity ceiling.",
                {
                    "binding_id": binding.binding_id,
                    "effective_sensitivity": request_sensitivity.value,
                    "maximum_sensitivity": (binding.maximum_outbound_sensitivity.value),
                },
            )
        return arguments

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        del run, call, capability, execution, fingerprint
        raise ValueError("MCP Stage M2 capabilities are never side-effecting")

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> ToolOutput:
        del run, call, arguments
        admitted = self._binding_by_capability.get(capability.id)
        if admitted is None:
            raise MCPProtocolError(
                "mcp_binding_unavailable",
                "The MCP capability is not admitted in this runtime.",
            )
        binding, tool = admitted
        if output.sensitivity is None:
            raise MCPProtocolError(
                "mcp_result_sensitivity_missing",
                "The MCP result omitted its code-owned sensitivity.",
            )
        effective = max(
            output.sensitivity,
            request_sensitivity,
            key=lambda item: item.routing_rank,
        )
        return ToolOutput(
            kind=output.kind,
            data=output.data,
            artifact=output.artifact,
            sensitivity=effective,
            sensitivity_provenance={
                "authority": "mcp_binding_admission_and_run_floor",
                "binding_id": binding.binding_id,
                "binding_revision": binding.revision,
                "capability_id": tool.capability_id,
                "admitted_result_sensitivity": tool.result_sensitivity.value,
                "run_sensitivity_floor": request_sensitivity.value,
            },
        )

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        del call
        if isinstance(error, MCPError):
            return CapabilityFailure(error.code, str(error), error.details)
        return None


async def activate_mcp_domain(
    *,
    agent_id: str,
    store: MCPBindingStore,
    client_factory: MCPClientFactory,
    secrets: SecretProvider,
    clock: Callable[[], datetime],
) -> tuple[
    MCPCapabilityDomain | None,
    tuple[MCPActivatedBinding, ...],
    tuple[Executor, ...],
]:
    """Compose accepted persisted bindings without network activity."""

    activated: list[MCPActivatedBinding] = []
    for binding in await store.list_mcp_bindings(agent_id):
        if binding.state is not MCPBindingState.ACTIVE:
            continue
        lock = asyncio.Lock()
        executor = MCPToolExecutor(
            binding=binding,
            client_factory=client_factory,
            secrets=secrets,
            store=store,
            clock=clock,
            lock=lock,
        )
        activated.append(
            MCPActivatedBinding(
                binding=binding,
                lock=lock,
                executor=executor,
            )
        )
    if not activated:
        return None, (), ()
    capabilities = tuple(
        Capability(
            id=tool.capability_id,
            description=tool.description,
            input_schema=tool.input_schema,
            output_kind=MCP_OUTPUT_KIND,
            output_schema=_MCP_OUTPUT_SCHEMA,
            executor_id=tool.executor_id,
        )
        for item in activated
        for tool in item.binding.tools
    )
    declarations = CapabilityDeclarations(
        domain_owner_id=MCP_DOMAIN_OWNER_ID,
        capabilities=capabilities,
        executor_ids=tuple(
            sorted({capability.executor_id for capability in capabilities})
        ),
        tool_views=tuple(
            ToolView(
                name=tool.local_name,
                capability_id=tool.capability_id,
                description=tool.description,
                discovery=tool.discovery,
                origin_revision_digest="sha256:"
                + sha256(
                    canonical_json(
                        {
                            "binding_id": item.binding.binding_id,
                            "binding_revision": item.binding.revision,
                            "capability_id": tool.capability_id,
                            "executor_id": tool.executor_id,
                            "input_schema_digest": tool.input_schema_digest,
                            "output_schema_digest": tool.output_schema_digest,
                            "discovery": {
                                "summary": tool.discovery.summary,
                                "when_to_use": tool.discovery.when_to_use,
                                "keywords": tool.discovery.keywords,
                                "exposure_class": tool.discovery.exposure_class.value,
                                "eager_priority": tool.discovery.eager_priority,
                            },
                        }
                    ).encode("utf-8")
                ).hexdigest(),
            )
            for item in activated
            for tool in item.binding.tools
        ),
    )
    domain = MCPCapabilityDomain(
        declarations,
        agent_id=agent_id,
        bindings=tuple(activated),
        store=store,
    )
    return domain, tuple(activated), tuple(item.executor for item in activated)


def _require_current_binding(
    current: MCPServerBinding | None,
    accepted: MCPServerBinding,
    tool: MCPToolBinding,
) -> None:
    if not _binding_revision_is_active(current, accepted):
        raise MCPProtocolError(
            "mcp_binding_unavailable",
            "The MCP binding is revoked, stale, or requires agent reopen.",
            {"binding_id": accepted.binding_id},
        )
    assert current is not None
    current_tools = {item.capability_id: item for item in current.tools}
    if current_tools.get(tool.capability_id) != tool:
        raise MCPProtocolError(
            "mcp_binding_stale",
            "The admitted MCP capability mapping changed.",
            {"binding_id": accepted.binding_id},
        )


def _binding_revision_is_active(
    current: MCPServerBinding | None,
    accepted: MCPServerBinding,
) -> bool:
    return (
        current is not None
        and current.state is MCPBindingState.ACTIVE
        and current.binding_id == accepted.binding_id
        and current.agent_id == accepted.agent_id
        and current.revision == accepted.revision
    )


__all__ = [
    "MCPActivatedBinding",
    "MCPCapabilityDomain",
    "MCPBindingStore",
    "MCP_DOMAIN_OWNER_ID",
    "MCP_OUTPUT_KIND",
    "MCPToolExecutor",
    "activate_mcp_domain",
]
