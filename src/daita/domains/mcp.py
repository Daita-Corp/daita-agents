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
    MCPAdmissionError,
    MCPAuthenticationMode,
    MCPBindingState,
    MCPClient,
    MCPClientFactory,
    MCPCompletionSemantics,
    MCPConnectionProvider,
    MCPError,
    MCPPersonalConnectionClient,
    MCPProtocolError,
    MCPProtocolPinnedClient,
    MCPRemoteToolError,
    MCPServerBinding,
    MCPToolBinding,
    check_personal_connection,
    mcp_binding_drift_reason,
    mcp_execution_origin_digest,
    validate_mcp_schema_value,
)
from ..capabilities import (
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    EffectReceiptPolicy,
    ExecutionAdmissionPolicy,
    Executor,
    OperationalEffect,
    RecoveryKind,
    ToolExecution,
    ToolOutput,
    ToolRecovery,
    ToolView,
    recovery_details,
)
from ..capability_runtime import CapabilityFailure, SideEffectPlan
from ..llm.models import ModelSensitivity, ToolCall
from ..llm.provider_definitions import tool_schema_incompatibility
from ..loop.models import RunInput
from ..loop.session import RunSession
from ..security import SecretProvider

MCP_DOMAIN_OWNER_ID = "mcp"
_MCP_BINDING_CHANGED = ToolRecovery(
    RecoveryKind.REINSPECT_BINDING,
    "The connection changed after it was admitted; the person must review "
    "and re-admit it before this tool can run. It was not called.",
)
MCP_OUTPUT_KIND = "mcp.tool.result"

MCP_RECEIPT_POLICY = EffectReceiptPolicy(
    receipt_kind="mcp.tool_call",
    success_evidence_basis=EffectEvidenceBasis.SERVER_REPORTED,
    payload_schema={
        "type": "object",
        "properties": {
            "binding_id": {"type": "string", "maxLength": 256},
            "binding_revision": {"type": "integer", "minimum": 1},
            "remote_tool_name": {"type": "string", "maxLength": 256},
            "origin_digest": {"type": "string", "maxLength": 71},
            "argument_fingerprint": {"type": "string", "maxLength": 71},
            "classification": {
                "type": "string",
                "enum": [
                    "invocation_result",
                    "tool_error",
                    "protocol_or_transport_error",
                    "accepted_async",
                    "local_not_dispatched",
                ],
            },
            "operation_handle": {"type": "string", "maxLength": 256},
        },
        "required": [
            "binding_id",
            "binding_revision",
            "remote_tool_name",
            "origin_digest",
            "argument_fingerprint",
            "classification",
        ],
        "additionalProperties": False,
    },
)


class MCPActionFailure(MCPError):
    """Evidence classified locally at the existing executor's dispatch boundary."""

    def __init__(self, error: MCPError, observation: EffectObservation) -> None:
        super().__init__(error.code, str(error))
        self.observation = observation


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
        connection_provider: MCPConnectionProvider | None,
        store: MCPBindingStore,
        clock: Callable[[], datetime],
        lock: asyncio.Lock,
    ) -> None:
        self.executor_id = binding.tools[0].executor_id
        self._binding = binding
        self._client_factory = client_factory
        self._secrets = secrets
        self._connection_provider = connection_provider
        self._client: MCPClient | None = None
        self._store = store
        self._clock = clock
        self._lock = lock
        self._tools = {tool.capability_id: tool for tool in binding.tools}

    @property
    def client(self) -> MCPClient | None:
        return self._client

    async def _validate_target(
        self, request: ToolExecution, tool: MCPToolBinding
    ) -> MCPClient:
        current = await self._store.load_mcp_binding(
            self._binding.agent_id, self._binding.binding_id
        )
        _require_current_binding(current, self._binding, tool)
        if (
            self._binding.authentication.mode
            is MCPAuthenticationMode.PERSONAL_CONNECTION
            and not request.caller_principal_verified
        ):
            raise MCPProtocolError(
                "needs_authorization",
                "The personal MCP connection needs an authenticated caller.",
            )
        if (
            self._binding.authentication.mode
            is MCPAuthenticationMode.PERSONAL_CONNECTION
            and request.caller_principal_id != self._binding.owner_principal_id
        ):
            raise MCPProtocolError(
                "needs_authorization", "The MCP binding is unavailable to this caller."
            )
        await check_personal_connection(
            self._binding.authentication,
            self._connection_provider,
            request.caller_principal_id,
        )
        if (
            self._binding.authentication.mode
            is MCPAuthenticationMode.PERSONAL_CONNECTION
        ):
            scope = request.execution_scope
            if scope is not None and (
                scope.principal_id != request.caller_principal_id
                or self._binding.binding_id not in scope.allowed_connector_binding_ids
                or scope.contract_bindings.tool_origins.get(tool.capability_id)
                != mcp_execution_origin_digest(self._binding, tool)
            ):
                raise MCPProtocolError(
                    "execution_scope_violation",
                    "The personal MCP connection is outside the frozen machine scope.",
                )
        await validate_mcp_schema_value(
            tool.raw_input_schema or tool.input_schema, request.arguments
        )
        if (
            tool.completion_semantics is not MCPCompletionSemantics.DIRECT_RESULT
            or tool.task_support == "required"
        ):
            raise MCPProtocolError(
                "mcp_completion_unsupported",
                "Asynchronous MCP completion is unsupported; no action was dispatched.",
            )
        if request.request_sensitivity.routing_rank > min(
            self._binding.maximum_outbound_sensitivity.routing_rank,
            tool.maximum_outbound_sensitivity.routing_rank,
        ):
            raise MCPProtocolError(
                "mcp_outbound_sensitivity_exceeded",
                "The request exceeds the admitted outbound sensitivity.",
            )
        if self._client is None:
            client = self._client_factory.create(
                endpoint=self._binding.endpoint,
                authentication=self._binding.authentication,
                secrets=self._secrets,
            )
            try:
                if isinstance(client, MCPProtocolPinnedClient):
                    client.bind_protocol(self._binding.protocol_version)
                if (
                    self._binding.authentication.mode
                    is MCPAuthenticationMode.PERSONAL_CONNECTION
                ):
                    if (
                        not isinstance(client, MCPPersonalConnectionClient)
                        or self._connection_provider is None
                    ):
                        raise MCPProtocolError(
                            "account_unavailable",
                            "The MCP client cannot use a personal connection.",
                        )
                    assert request.caller_principal_id is not None
                    client.bind_personal_connection(
                        self._connection_provider, request.caller_principal_id
                    )
            except BaseException:
                await client.close()
                raise
            self._client = client
        inspection = await self._client.inspect(observed_at=self._clock())
        drift = mcp_binding_drift_reason(self._binding, inspection)
        if drift is not None:
            raise MCPProtocolError(
                "mcp_binding_stale",
                "The MCP server identity, schema or invocation contract changed.",
                recovery_details(
                    _MCP_BINDING_CHANGED,
                    {"binding_id": self._binding.binding_id, "reason": drift},
                ),
            )
        # The bounded remote inspection awaited I/O: recheck local revocation.
        _require_current_binding(
            await self._store.load_mcp_binding(
                self._binding.agent_id, self._binding.binding_id
            ),
            self._binding,
            tool,
        )
        return self._client

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        tool = self._tools[request.capability_id]
        async with self._lock:
            await self._validate_target(request, tool)
            return FrozenJsonObject.from_mapping(
                {
                    "origin_digest": mcp_execution_origin_digest(self._binding, tool),
                    "argument_fingerprint": "sha256:"
                    + sha256(
                        canonical_json(request.arguments).encode("utf-8")
                    ).hexdigest(),
                }
            )

    def _observation(
        self,
        request: ToolExecution,
        tool: MCPToolBinding,
        outcome: EffectOutcome,
        basis: EffectEvidenceBasis,
        classification: str,
        operation_handle: str | None = None,
    ) -> EffectObservation:
        payload: dict[str, object] = {
            "binding_id": self._binding.binding_id,
            "binding_revision": self._binding.revision,
            "remote_tool_name": tool.remote_name,
            "origin_digest": mcp_execution_origin_digest(self._binding, tool),
            "argument_fingerprint": "sha256:"
            + sha256(canonical_json(request.arguments).encode("utf-8")).hexdigest(),
            "classification": classification,
        }
        if operation_handle is not None:
            payload["operation_handle"] = operation_handle
        return EffectObservation(outcome, basis, FrozenJsonObject.from_mapping(payload))

    async def execute(self, request: ToolExecution) -> ToolOutput:
        tool = self._tools.get(request.capability_id)
        if tool is None:
            raise MCPProtocolError(
                "mcp_binding_mismatch",
                "The MCP capability does not belong to its admitted binding.",
            )
        async with self._lock:
            action = tool.operational_effect is not OperationalEffect.NONE
            if action and request.effect_receipt_id is None:
                raise MCPProtocolError(
                    "mcp_receipt_required",
                    "MCP actions require a runtime-owned reservation.",
                )
            try:
                client = await self._validate_target(request, tool)
            except MCPError as error:
                if action:
                    raise MCPActionFailure(
                        error,
                        self._observation(
                            request,
                            tool,
                            EffectOutcome.NOT_APPLIED,
                            EffectEvidenceBasis.LOCAL_NOT_DISPATCHED,
                            "local_not_dispatched",
                        ),
                    ) from error
                raise
            observed_at = self._clock()
            try:
                result = await client.call_tool(tool.remote_name, request.arguments)
            except MCPError as error:
                if action:
                    raise MCPActionFailure(
                        error,
                        self._observation(
                            request,
                            tool,
                            EffectOutcome.UNCERTAIN,
                            EffectEvidenceBasis.UNKNOWN,
                            "protocol_or_transport_error",
                        ),
                    ) from error
                raise
            if result.accepted_async:
                acceptance_error = MCPProtocolError(
                    "mcp_accepted_async",
                    "The server accepted asynchronous work without a completed invocation result; Daita will not poll or repeat it.",
                )
                if action:
                    raise MCPActionFailure(
                        acceptance_error,
                        self._observation(
                            request,
                            tool,
                            EffectOutcome.UNCERTAIN,
                            EffectEvidenceBasis.SERVER_REPORTED,
                            "accepted_async",
                            result.operation_handle,
                        ),
                    )
                raise acceptance_error
            if result.is_error:
                if action:
                    raise MCPActionFailure(
                        MCPRemoteToolError(
                            "mcp_remote_tool_error",
                            "The MCP action returned an error; partial effects may have occurred.",
                        ),
                        self._observation(
                            request,
                            tool,
                            EffectOutcome.UNCERTAIN,
                            EffectEvidenceBasis.SERVER_REPORTED,
                            "tool_error",
                        ),
                    )
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
                    await validate_mcp_schema_value(
                        tool.output_schema, result.structured
                    )
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
                effect_observation=(
                    self._observation(
                        request,
                        tool,
                        EffectOutcome.SUCCEEDED,
                        EffectEvidenceBasis.SERVER_REPORTED,
                        "invocation_result",
                    )
                    if action
                    else None
                ),
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
        connection_provider: MCPConnectionProvider | None,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("MCP declarations belong to another domain")
        self._declarations = declarations
        self._agent_id = agent_id
        self._store = store
        self._connection_provider = connection_provider
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

    async def project(
        self,
        run: RunInput,
        session: RunSession | None = None,
    ) -> tuple[str, ...]:
        if run.agent_id != self._agent_id or (
            session is not None and session.options.files_only
        ):
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
        # Share the provider decision only within this projection; calls recheck.
        personal_access: dict[str, bool] = {}
        for capability_id, (binding, _tool) in self._binding_by_capability.items():
            if (
                binding.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
                and binding.owner_principal_id != run.caller_principal_id
            ):
                continue
            if (
                binding.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
                and not run.caller_principal_verified
            ):
                continue
            if (
                run.execution_scope is not None
                and binding.binding_id
                not in run.execution_scope.allowed_connector_binding_ids
            ):
                continue
            if _binding_revision_is_active(current[binding.binding_id], binding):
                if (
                    binding.authentication.mode
                    is MCPAuthenticationMode.PERSONAL_CONNECTION
                ):
                    if run.execution_scope is not None and (
                        run.execution_scope.principal_id != run.caller_principal_id
                        or binding.binding_id
                        not in run.execution_scope.allowed_connector_binding_ids
                        or run.execution_scope.contract_bindings.tool_origins.get(
                            capability_id
                        )
                        != mcp_execution_origin_digest(binding, _tool)
                    ):
                        continue
                    if binding.binding_id not in personal_access:
                        try:
                            await check_personal_connection(
                                binding.authentication,
                                self._connection_provider,
                                run.caller_principal_id,
                            )
                        except MCPError:
                            personal_access[binding.binding_id] = False
                        else:
                            personal_access[binding.binding_id] = True
                    if not personal_access[binding.binding_id]:
                        continue
                projected.append(self._local_name_by_capability[capability_id])
        return tuple(sorted(projected))

    project_session = project

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
        if (
            binding.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
            and not run.caller_principal_verified
        ):
            raise CapabilityInputError(
                "needs_authorization",
                "The personal MCP connection needs an authenticated caller.",
            )
        if (
            binding.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
            and binding.owner_principal_id != run.caller_principal_id
        ):
            raise CapabilityInputError(
                "needs_authorization", "The MCP binding is unavailable to this caller."
            )
        try:
            await check_personal_connection(
                binding.authentication,
                self._connection_provider,
                run.caller_principal_id,
            )
        except MCPError as error:
            raise CapabilityInputError(error.code, str(error)) from None
        scope = run.execution_scope
        if (
            binding.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
            and scope is not None
            and (
                scope.principal_id != run.caller_principal_id
                or scope.contract_bindings.tool_origins.get(capability.id)
                != mcp_execution_origin_digest(binding, tool)
            )
        ):
            raise CapabilityInputError(
                "execution_scope_violation",
                "The personal MCP connection is outside the frozen machine scope.",
            )
        if (
            scope is not None
            and binding.binding_id not in scope.allowed_connector_binding_ids
        ):
            raise CapabilityInputError(
                "execution_scope_violation",
                "The MCP binding is outside this run's immutable connector ceiling.",
                {"binding_id": binding.binding_id},
            )
        current = await self._store.load_mcp_binding(
            self._agent_id,
            binding.binding_id,
        )
        try:
            _require_current_binding(current, binding, tool)
        except MCPError as error:
            raise CapabilityInputError(error.code, str(error), error.details) from error
        if request_sensitivity.routing_rank > min(
            binding.maximum_outbound_sensitivity.routing_rank,
            tool.maximum_outbound_sensitivity.routing_rank,
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
        # MCP actions in machine runs ask a person at the time, like foreground ones.
        await validate_mcp_schema_value(
            tool.raw_input_schema or tool.input_schema, arguments
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
        del call
        del run
        binding, tool = self._binding_by_capability[capability.id]
        return SideEffectPlan(
            approval_arguments=FrozenJsonObject.from_mapping(
                {
                    "binding_id": binding.binding_id,
                    "binding_revision": binding.revision,
                    "server_name": binding.server_name,
                    "endpoint": binding.endpoint,
                    "remote_tool_name": tool.remote_name,
                    "access_mode": tool.access_mode.value,
                    "operational_effect": tool.operational_effect.value,
                    "arguments": execution.arguments,
                    "completion_evidence": "server_reported_invocation_only",
                }
            ),
            approval_reason="Approve this exact MCP action once? A valid response proves only server-reported invocation, not verified downstream business completion. Ambiguous outcomes block further effects until explicit recovery; no automatic replay.",
            effect_intent=FrozenJsonObject.from_mapping(
                {
                    "binding_id": binding.binding_id,
                    "binding_revision": binding.revision,
                    "remote_tool_name": tool.remote_name,
                    "origin_digest": fingerprint["origin_digest"],
                    "argument_fingerprint": fingerprint["argument_fingerprint"],
                }
            ),
        )

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
        del run
        admitted = self._binding_by_capability.get(capability.id)
        if admitted is None:
            raise MCPProtocolError(
                "mcp_binding_unavailable",
                "The MCP capability is not admitted in this runtime.",
            )
        binding, tool = admitted
        provenance = output.data.get("provenance")
        expected = {
            "binding_id": binding.binding_id,
            "binding_revision": binding.revision,
            "remote_tool_name": tool.remote_name,
            "input_schema_digest": tool.input_schema_digest,
            "output_schema_digest": tool.output_schema_digest or "none",
            "call_id": call.id,
        }
        if not isinstance(provenance, Mapping) or any(
            provenance.get(name) != value for name, value in expected.items()
        ):
            raise MCPProtocolError(
                "mcp_result_provenance_invalid",
                "The MCP result does not match its exact admitted invocation.",
            )
        if (
            output.effect_observation is not None
            and output.effect_observation.payload is not None
        ):
            evidence = output.effect_observation.payload
            if (
                evidence.get("binding_id") != binding.binding_id
                or evidence.get("binding_revision") != binding.revision
                or evidence.get("remote_tool_name") != tool.remote_name
                or evidence.get("origin_digest")
                != mcp_execution_origin_digest(binding, tool)
                or evidence.get("argument_fingerprint")
                != "sha256:"
                + sha256(canonical_json(arguments).encode("utf-8")).hexdigest()
            ):
                raise MCPProtocolError(
                    "mcp_result_evidence_invalid",
                    "The MCP observation does not match the admitted invocation.",
                )
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
            effect_observation=output.effect_observation,
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
        if isinstance(error, MCPActionFailure):
            return CapabilityFailure(
                error.code, str(error), effect_observation=error.observation
            )
        if isinstance(error, MCPError):
            return CapabilityFailure(error.code, str(error), error.details)
        return None


def validate_mcp_model_compatibility(
    bindings: tuple[MCPServerBinding, ...], model_ids: tuple[str, ...]
) -> None:
    """Require each active tool contract to fit every selected model candidate."""
    for binding in bindings:
        if binding.state is not MCPBindingState.ACTIVE:
            continue
        for tool in binding.tools:
            reason = tool_schema_incompatibility(tool.input_schema, model_ids)
            if reason is not None:
                raise MCPAdmissionError(
                    "mcp_schema_unsupported",
                    "The selected model route cannot represent MCP tool "
                    f"{tool.remote_name!r}. Choose another model or update MCP access.",
                    {
                        "binding_id": binding.binding_id,
                        "remote_name": tool.remote_name,
                        "reason": reason,
                    },
                )


async def activate_mcp_domain(
    *,
    agent_id: str,
    store: MCPBindingStore,
    client_factory: MCPClientFactory,
    secrets: SecretProvider,
    connection_provider: MCPConnectionProvider | None,
    clock: Callable[[], datetime],
    bindings: tuple[MCPServerBinding, ...] | None = None,
    existing: tuple[MCPActivatedBinding, ...] = (),
    model_ids: tuple[str, ...] = (),
) -> tuple[
    MCPCapabilityDomain | None,
    tuple[MCPActivatedBinding, ...],
    tuple[Executor, ...],
]:
    """Compose accepted persisted bindings without network activity."""

    activated: list[MCPActivatedBinding] = []
    reusable = {item.binding.binding_id: item for item in existing}
    selected_bindings = (
        await store.list_mcp_bindings(agent_id) if bindings is None else bindings
    )
    validate_mcp_model_compatibility(selected_bindings, model_ids)
    for binding in selected_bindings:
        if binding.state is not MCPBindingState.ACTIVE:
            continue
        previous = reusable.get(binding.binding_id)
        if previous is not None and previous.binding == binding:
            activated.append(previous)
            continue
        lock = asyncio.Lock()
        executor = MCPToolExecutor(
            binding=binding,
            client_factory=client_factory,
            secrets=secrets,
            connection_provider=connection_provider,
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
            external_input_schema=tool.raw_input_schema or tool.input_schema,
            output_kind=MCP_OUTPUT_KIND,
            output_schema=_MCP_OUTPUT_SCHEMA,
            executor_id=tool.executor_id,
            access_mode=tool.access_mode,
            operational_effect=tool.operational_effect,
            effect_receipt_policy=(
                MCP_RECEIPT_POLICY
                if tool.operational_effect is not OperationalEffect.NONE
                else None
            ),
            execution_admission_policy=ExecutionAdmissionPolicy(
                shape=(
                    "read_only_mcp"
                    if tool.operational_effect is OperationalEffect.NONE
                    else "mcp_action"
                ),
                inline_eligible=True,
                graph_v1_eligible=(
                    tool.operational_effect is OperationalEffect.NONE
                    or (
                        tool.completion_semantics
                        is MCPCompletionSemantics.DIRECT_RESULT
                        and tool.task_support != "required"
                    )
                ),
                target_count_argument=None,
                inline_max_targets=1,
                graph_max_targets=1,
            ),
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
                presentation=tool.presentation,
                presentation_sensitivity=max(
                    (candidate.result_sensitivity for candidate in item.binding.tools),
                    key=lambda sensitivity: sensitivity.routing_rank,
                ),
                connector_presentation=FrozenJsonObject.from_mapping(
                    {
                        "kind": "mcp_binding",
                        "id": item.binding.binding_id,
                        "binding_revision": item.binding.revision,
                        "remote_tool_name": tool.remote_name,
                        "label": item.binding.local_label,
                        "summary": item.binding.summary,
                        "when_to_use": item.binding.when_to_use,
                        "keywords": item.binding.keywords,
                        "tool_count": len(item.binding.tools),
                    }
                ),
                origin_revision_digest=mcp_execution_origin_digest(item.binding, tool),
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
        connection_provider=connection_provider,
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
    current_tool = current_tools.get(tool.capability_id)
    if current_tool is None or mcp_execution_origin_digest(
        current, current_tool
    ) != mcp_execution_origin_digest(accepted, tool):
        raise MCPProtocolError(
            "mcp_binding_stale",
            "The admitted MCP capability mapping changed.",
            recovery_details(_MCP_BINDING_CHANGED, {"binding_id": accepted.binding_id}),
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
