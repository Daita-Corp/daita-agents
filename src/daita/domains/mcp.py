"""Project admitted MCP tools and revalidate bindings before remote execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Protocol, cast

from .._json import FrozenJsonObject, canonical_json
from ..adapters.mcp import (
    MCPBindingState,
    MCPClient,
    MCPClientFactory,
    MCPCompletionSemantics,
    MCPError,
    MCPProtocolError,
    MCPRemoteToolError,
    MCPServerBinding,
    MCPToolBinding,
    mcp_binding_drift_reason,
    mcp_execution_origin_digest,
)
from ..capabilities import (
    AutomationEligibility,
    AutomationGrantPolicy,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    EffectReceiptPolicy,
    Executor,
    OperationalEffect,
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
MCP_OUTPUT_KIND = "mcp.tool.result"

MCP_GRANT_POLICY = AutomationGrantPolicy(
    constraints_kind="mcp.tool_call",
    constraints_schema={
        "type": "object",
        "properties": {
            "binding_id": {"type": "string", "maxLength": 256},
            "binding_revision": {"type": "integer", "minimum": 1},
            "remote_tool_name": {"type": "string", "maxLength": 256},
            "fixed_arguments": {"type": "object"},
            "variable_argument_names": {
                "type": "array",
                "maxItems": 128,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
            },
        },
        "required": [
            "binding_id",
            "binding_revision",
            "remote_tool_name",
            "fixed_arguments",
            "variable_argument_names",
        ],
        "additionalProperties": False,
    },
)

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

    async def _validate_target(
        self, request: ToolExecution, tool: MCPToolBinding
    ) -> MCPClient:
        current = await self._store.load_mcp_binding(
            self._binding.agent_id, self._binding.binding_id
        )
        _require_current_binding(current, self._binding, tool)
        validate_tool_schema_value(tool.input_schema, request.arguments)
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
            self._client = self._client_factory.create(
                endpoint=self._binding.endpoint,
                authentication=self._binding.authentication,
                secrets=self._secrets,
            )
        inspection = await self._client.inspect(observed_at=self._clock())
        drift = mcp_binding_drift_reason(self._binding, inspection)
        if drift is not None:
            raise MCPProtocolError(
                "mcp_binding_stale",
                "The MCP server identity, schema or invocation contract changed.",
                {"binding_id": self._binding.binding_id, "reason": drift},
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
        files_only_run_ids: set[str] | None = None,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("MCP declarations belong to another domain")
        self._declarations = declarations
        self._agent_id = agent_id
        self._store = store
        self._files_only_run_ids = (
            files_only_run_ids if files_only_run_ids is not None else set()
        )
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
        if run.agent_id != self._agent_id or run.id in self._files_only_run_ids:
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
            if (
                run.execution_scope is not None
                and run.execution_scope.routine_id is not None
                and binding.binding_id
                not in run.execution_scope.allowed_connector_binding_ids
            ):
                continue
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
        scope = run.execution_scope
        if (
            scope is not None
            and scope.routine_id is not None
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
        if (
            capability.operational_effect is not OperationalEffect.NONE
            and scope is not None
        ):
            grant = next(
                (
                    item
                    for item in scope.capability_grants
                    if item.capability_id == capability.id
                ),
                None,
            )
            if grant is None:
                raise CapabilityInputError(
                    "effect_grant_required",
                    "The MCP action requires its exact retained standing grant.",
                )
            self._validate_constraints(binding, tool, grant.constraints)
            fixed = cast(Mapping[str, object], grant.constraints["fixed_arguments"])
            variable = cast(
                tuple[str, ...], grant.constraints["variable_argument_names"]
            )
            if not set(arguments) <= set(fixed) | set(variable) or any(
                name not in arguments
                or canonical_json(arguments[name]) != canonical_json(value)
                for name, value in fixed.items()
            ):
                raise CapabilityInputError(
                    "mcp_grant_arguments_invalid",
                    "MCP arguments differ from the approved fixed values or variable names.",
                )
        validate_tool_schema_value(tool.input_schema, arguments)
        return arguments

    async def prepare_automation_grant(
        self,
        capability: Capability,
        constraints: FrozenJsonObject,
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> FrozenJsonObject:
        binding, tool = self._binding_by_capability[capability.id]
        if (
            tool.operational_effect is OperationalEffect.NONE
            or tool.automation_eligibility
            is not AutomationEligibility.AUTOMATION_DIRECT
            or tool.completion_semantics is not MCPCompletionSemantics.DIRECT_RESULT
            or tool.task_support == "required"
            or type(max_calls_per_occurrence) is not int
            or not 1 <= max_calls_per_occurrence <= 256
        ):
            raise CapabilityInputError(
                "automation_grant_unsupported",
                "This MCP tool does not support the requested unattended invocation contract.",
            )
        if (
            proposal.agent_id != self._agent_id
            or binding.binding_id not in proposal.allowed_connector_binding_ids
        ):
            raise CapabilityInputError(
                "automation_grant_scope_invalid",
                "The MCP binding is outside the proposed scope.",
            )
        _require_current_binding(
            await self._store.load_mcp_binding(self._agent_id, binding.binding_id),
            binding,
            tool,
        )
        if proposal.sensitivity_ceiling.routing_rank > min(
            binding.maximum_outbound_sensitivity.routing_rank,
            tool.maximum_outbound_sensitivity.routing_rank,
        ):
            raise CapabilityInputError(
                "mcp_outbound_sensitivity_exceeded",
                "The proposed request classification exceeds MCP outbound admission.",
            )
        self._validate_constraints(binding, tool, constraints)
        normalized = constraints.to_dict()
        normalized["variable_argument_names"] = sorted(
            cast(tuple[str, ...], constraints["variable_argument_names"])
        )
        return FrozenJsonObject.from_mapping(normalized)

    @staticmethod
    def _validate_constraints(
        binding: MCPServerBinding, tool: MCPToolBinding, constraints: FrozenJsonObject
    ) -> None:
        validate_tool_schema_value(MCP_GRANT_POLICY.constraints_schema, constraints)
        fixed = cast(Mapping[str, object], constraints["fixed_arguments"])
        variable = cast(tuple[str, ...], constraints["variable_argument_names"])
        properties = cast(
            Mapping[str, Mapping[str, object]], tool.input_schema.get("properties", {})
        )
        names = set(fixed) | set(variable)
        identity_fields = [
            name
            for name, expected in (
                ("binding_id", binding.binding_id),
                ("binding_revision", binding.revision),
                ("remote_tool_name", tool.remote_name),
            )
            if constraints[name] != expected
        ]
        if identity_fields:
            raise CapabilityInputError(
                "mcp_grant_constraints_invalid",
                "The grant must use this exact admitted binding, revision and tool.",
                {
                    "identity_fields": identity_fields,
                    "inspect_tool_name": tool.local_name,
                },
            )
        missing = (
            set(cast(tuple[str, ...], tool.input_schema.get("required", ()))) - names
        )
        unknown = names - set(properties)
        overlapping = set(fixed) & set(variable)
        if (
            len(fixed) > 128
            or overlapping
            or len(variable) != len(set(variable))
            or unknown
            or missing
        ):
            groups = {
                "missing_argument_names": missing,
                "unknown_argument_names": unknown,
                "overlapping_argument_names": overlapping,
            }
            details: dict[str, object] = {
                key: [name[:128] for name in sorted(values)[:32]]
                for key, values in groups.items()
            }
            details.update(
                inspect_tool_name=tool.local_name,
                argument_counts={key: len(values) for key, values in groups.items()},
                names_truncated=any(
                    len(values) > 32 or any(len(name) > 128 for name in values)
                    for values in groups.values()
                ),
                fixed_argument_limit=128,
            )
            raise CapabilityInputError(
                "mcp_grant_constraints_invalid",
                "Correct the listed missing, unknown or overlapping grant arguments. "
                "Use toolbox_inspect for the exact declared names and types; do not invent fields.",
                details,
            )
        # A variable object/array cannot enforce a nested recipient or table limit.
        # This release admits scalar variables; fix composite values in full.
        if any(
            properties[name].get("type")
            not in {"string", "integer", "number", "boolean"}
            for name in variable
        ):
            raise CapabilityInputError(
                "mcp_grant_nested_variable_unsupported",
                "Fix the entire nested argument; variable JSON cannot enforce nested restrictions.",
                {
                    "argument_names": sorted(
                        name
                        for name in variable
                        if properties[name].get("type")
                        not in {"string", "integer", "number", "boolean"}
                    )[:32],
                    "inspect_tool_name": tool.local_name,
                },
            )
        partial_schema = tool.input_schema.to_dict()
        partial_schema["required"] = list(fixed)
        validate_tool_schema_value(partial_schema, fixed)

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        del call
        binding, tool = self._binding_by_capability[capability.id]
        grant = (
            None
            if run.execution_scope is None
            else next(
                (
                    item
                    for item in run.execution_scope.capability_grants
                    if item.capability_id == capability.id
                ),
                None,
            )
        )
        return SideEffectPlan(
            approval_required=run.execution_scope is None,
            capability_grant_digest=None if grant is None else grant.grant_digest,
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


async def activate_mcp_domain(
    *,
    agent_id: str,
    store: MCPBindingStore,
    client_factory: MCPClientFactory,
    secrets: SecretProvider,
    clock: Callable[[], datetime],
    files_only_run_ids: set[str] | None = None,
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
            access_mode=tool.access_mode,
            operational_effect=tool.operational_effect,
            automation_eligibility=tool.automation_eligibility,
            effect_receipt_policy=(
                MCP_RECEIPT_POLICY
                if tool.operational_effect is not OperationalEffect.NONE
                else None
            ),
            automation_grant_policy=(
                MCP_GRANT_POLICY
                if tool.operational_effect is not OperationalEffect.NONE
                and tool.automation_eligibility
                is AutomationEligibility.AUTOMATION_DIRECT
                else None
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
        files_only_run_ids=files_only_run_ids,
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
