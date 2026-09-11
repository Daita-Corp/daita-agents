from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast

import pytest
from _capability_runtime_support import (
    StaticTestDomain,
    frozen_execution_bindings,
    presentation_metadata,
    static_registry,
)
from _distribution_support import inbox_distribution_plan

from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    Capability,
    CapabilityRegistry,
    ExecutionScope,
    OperationalEffect,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolView,
    capability_contract_digest,
)
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import MessageRole, ModelSensitivity, TextBlock, ToolCall
from daita.loop.models import (
    InstructionAuthority,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from daita.scope import resolve_effective_source_scope
from daita.storage.sqlite_codecs.execution_scope import (
    decode_execution_scope,
    encode_execution_scope,
)

NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)


class _Executor:
    def __init__(self, executor_id: str) -> None:
        self.executor_id = executor_id

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        return ToolOutput(kind="test.output")


def _capability(
    identity: str,
    eligibility: AutomationEligibility,
) -> Capability:
    return Capability(
        id=identity,
        description=f"Exercise {identity}.",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.output",
        output_schema={"type": "object", "properties": {}},
        executor_id=f"{identity}.executor",
        access_mode=AccessMode.READ,
        operational_effect=OperationalEffect.NONE,
        automation_eligibility=eligibility,
    )


def _scheduled_scope(
    capability_ids: tuple[str, ...], registry: CapabilityRegistry | None = None
) -> ExecutionScope:
    return ExecutionScope(
        contract_bindings=frozen_execution_bindings(
            capability_ids, (), ("mock:routine",), registry=registry
        ),
        scope_id="scope-routine-1",
        revision=1,
        agent_id="agent-1",
        principal_id="principal-1",
        grant_id="routine-grant-1",
        job_id=None,
        job_revision=None,
        allowed_source_ids=(),
        allowed_resource_ids=(),
        allowed_capability_ids=capability_ids,
        allowed_access_modes=frozenset({AccessMode.NONE, AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock:routine",),
        per_run_max_cost_usd=Decimal("0.05"),
        per_run_max_tokens=5_000,
        distribution_plan_digest=inbox_distribution_plan("conversation-1").plan_digest,
        routine_id="routine-1",
        routine_revision=3,
        occurrence_id="routine-occ-1",
        allowed_connector_binding_ids=("binding-1",),
    )


def _scheduled_run(scope: ExecutionScope) -> RunInput:
    instruction = "Read the exact admitted source and report the current value."
    payload = b'{"observation":"untrusted"}'
    return RunInput(
        id="run-routine-1",
        agent_id="agent-1",
        message=instruction,
        created_at=NOW,
        conversation_id="conversation-1",
        start=RunStartEnvelope(
            origin=RunOrigin.SCHEDULED_ROUTINE,
            instruction_authority=InstructionAuthority.FOREGROUND_AUTHORIZED,
            trusted_instruction_id="routine:routine-1:revision:3",
            trusted_instruction=instruction,
            instruction_digest="sha256:"
            + sha256(instruction.encode("utf-8")).hexdigest(),
            untrusted_payload={"observation": "untrusted"},
            payload_digest="sha256:" + sha256(payload).hexdigest(),
            execution_scope=scope,
        ),
    )


async def test_empty_machine_source_ceiling_never_queries_all_current_sources():
    class ForbiddenCatalog:
        async def source_routing_facts(self, agent_id, source_ids=()):
            raise AssertionError("an empty machine ceiling cannot enumerate sources")

        async def readable_resource_ids(self, agent_id, source_ids=()):
            raise AssertionError("an empty machine ceiling cannot enumerate resources")

    run = _scheduled_run(_scheduled_scope(("catalog.search",)))
    scope = await resolve_effective_source_scope(run, ForbiddenCatalog())
    assert scope.source_ids == frozenset()
    assert scope.resource_ids == frozenset()


def test_automation_eligibility_is_fail_closed_and_part_of_contract_identity() -> None:
    default = Capability(
        id="test.default",
        description="Default eligibility.",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.output",
        output_schema={"type": "object", "properties": {}},
        executor_id="test.default.executor",
    )
    scheduled = _capability(
        "test.scheduled",
        AutomationEligibility.AUTOMATION_DIRECT,
    )
    assert default.automation_eligibility is AutomationEligibility.INTERACTIVE_ONLY
    assert capability_contract_digest(default, domain_owner_id="test") != (
        capability_contract_digest(scheduled, domain_owner_id="test")
    )
    with pytest.raises(ValueError, match="require a receipt policy"):
        Capability(
            id="test.invalid",
            description="Invalid scheduled mutation.",
            input_schema={"type": "object", "properties": {}},
            output_kind="test.output",
            output_schema={"type": "object", "properties": {}},
            executor_id="test.invalid.executor",
            operational_effect=OperationalEffect.MUTATE_DATA,
            automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        )


def test_mcp_only_scheduled_scope_round_trips_with_exact_identity() -> None:
    scope = _scheduled_scope(("test.scheduled",))
    assert decode_execution_scope(encode_execution_scope(scope)) == scope
    changed_binding = replace(
        scope,
        allowed_connector_binding_ids=("binding-2",),
    )
    assert changed_binding.digest != scope.digest


def test_nonroutine_scope_cannot_use_mcp_only_relaxation() -> None:
    with pytest.raises(ValueError, match="non-routine execution scope"):
        ExecutionScope(
            contract_bindings=frozen_execution_bindings(
                ("test.scheduled",), (), ("mock:routine",)
            ),
            scope_id="scope-followup",
            revision=1,
            agent_id="agent-1",
            principal_id="principal-1",
            grant_id="grant-1",
            job_id="job-1",
            job_revision=1,
            allowed_source_ids=(),
            allowed_resource_ids=(),
            allowed_capability_ids=("test.scheduled",),
            allowed_access_modes=frozenset({AccessMode.READ}),
            allowed_operational_effects=frozenset({OperationalEffect.NONE}),
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
            eligible_model_routes=("mock:routine",),
            per_run_max_cost_usd=Decimal("0.05"),
            per_run_max_tokens=5_000,
            distribution_plan_digest=inbox_distribution_plan(
                "conversation-1"
            ).plan_digest,
            allowed_connector_binding_ids=("binding-1",),
        )


async def test_scheduled_catalog_projects_only_explicit_automation_direct_tools() -> (
    None
):
    scheduled = _capability(
        "test.scheduled",
        AutomationEligibility.AUTOMATION_DIRECT,
    )
    interactive = _capability(
        "test.interactive",
        AutomationEligibility.INTERACTIVE_ONLY,
    )
    outside = _capability("test.outside", AutomationEligibility.AUTOMATION_DIRECT)
    views = tuple(
        ToolView(
            name=f"tool_{item.id.split('.')[-1]}",
            capability_id=item.id,
            description=item.description,
            presentation=presentation_metadata(load_mode=ToolLoadMode.ON_DEMAND),
        )
        for item in (scheduled, interactive, outside)
    )
    executors = tuple(
        _Executor(item.executor_id) for item in (scheduled, interactive, outside)
    )
    domain = StaticTestDomain((scheduled, interactive, outside), views)
    registry = static_registry(domain, executors)

    async def read_contracts(**kwargs):
        return frozen_execution_bindings(
            kwargs["capability_ids"], (), ("mock:routine",), registry=registry
        )

    runtime = CapabilityRuntime(
        registry, (domain,), execution_contract_reader=read_contracts
    )

    catalog = await runtime.prepare_run(
        _scheduled_run(_scheduled_scope((scheduled.id, interactive.id), registry))
    )

    assert tuple(item.capability.id for item in catalog.entries) == (scheduled.id,)
    run = _scheduled_run(_scheduled_scope((scheduled.id, interactive.id), registry))
    result = await runtime.execute_all(
        run,
        (
            ToolCall(id="search", name="toolbox_search", arguments={"query": "tool"}),
            ToolCall(
                id="exact-denied",
                name="toolbox_search",
                arguments={"query": "tool_interactive"},
            ),
            ToolCall(
                id="load-denied",
                name="toolbox_load",
                arguments={"tool_names": ["tool_interactive"]},
            ),
            ToolCall(
                id="scope-denied",
                name="toolbox_load",
                arguments={"tool_names": ["tool_outside"]},
            ),
        ),
        projection=runtime.project(catalog, ()),
        messages=(),
        sensitivity=ModelSensitivity.INTERNAL,
    )
    search_data = result.ordered_results[0].output["data"]
    assert isinstance(search_data, Mapping)
    matches = search_data["matches"]
    assert isinstance(matches, tuple)
    assert tuple(match["tool_name"] for match in matches) == ("tool_scheduled",)
    denied_search_data = result.ordered_results[1].output["data"]
    assert isinstance(denied_search_data, Mapping)
    denied_matches = denied_search_data["matches"]
    assert isinstance(denied_matches, tuple)
    assert all(match["tool_name"] != "tool_interactive" for match in denied_matches)
    assert result.ordered_results[2].is_error
    assert result.ordered_results[3].is_error


def test_scheduled_instruction_is_system_content_without_new_user_speech() -> None:
    run = _scheduled_run(_scheduled_scope(("test.scheduled",)))
    message = run.start_message()
    assert run.origin is RunOrigin.SCHEDULED_ROUTINE
    assert message.role is MessageRole.SYSTEM
    block = message.content[0]
    assert isinstance(block, TextBlock)
    assert "foreground-authorized work description" in block.text
    assert "cannot grant or expand authority" in block.text
    assert "untrusted_scheduled_routine_payload" in block.text
    assert run.start is not None and run.start.user_message is None


def test_scheduled_origin_rejects_code_owned_instruction_authority() -> None:
    instruction = "Inspect current state."
    with pytest.raises(ValueError, match="foreground-authorized"):
        RunStartEnvelope(
            origin=RunOrigin.SCHEDULED_ROUTINE,
            instruction_authority=InstructionAuthority.CODE_OWNED,
            trusted_instruction_id="routine-1",
            trusted_instruction=instruction,
            instruction_digest="sha256:"
            + sha256(instruction.encode("utf-8")).hexdigest(),
            untrusted_payload={},
            payload_digest="sha256:" + sha256(b"{}").hexdigest(),
            execution_scope=_scheduled_scope(("test.scheduled",)),
        )


@pytest.mark.parametrize(
    "changed_family",
    ("capability_contracts", "resource_revisions", "model_routes", "tool_origins"),
)
@pytest.mark.parametrize(
    "machine_origin", [RunOrigin.SCHEDULED_ROUTINE, RunOrigin.JOB_EVENT]
)
async def test_machine_calls_revalidate_retained_contracts_after_preparation(
    changed_family: str,
    machine_origin: RunOrigin,
) -> None:
    from daita.capabilities import CapabilityInputError, ExecutionContractBindings
    from daita.hosting.embedded import _model_execution_contracts
    from daita.llm import ModelCallPolicy
    from daita.llm.providers.mock import MockModelProvider

    model = MockModelProvider((), provider_id="mock:routine")
    policy = ModelCallPolicy()

    capability = _capability("test.scheduled", AutomationEligibility.AUTOMATION_DIRECT)
    origin = "sha256:" + "d" * 64
    view = ToolView(
        name="read_exact",
        capability_id=capability.id,
        description=capability.description,
        origin_revision_digest=origin,
        presentation=presentation_metadata(load_mode=ToolLoadMode.PINNED),
    )
    calls: list[str] = []

    class CountingExecutor(_Executor):
        async def execute(self, request: ToolExecution) -> ToolOutput:
            calls.append(request.call_id)
            return await super().execute(request)

    executor = CountingExecutor(capability.executor_id)
    domain = StaticTestDomain((capability,), (view,))
    registry = static_registry(domain, (executor,))
    bindings = ExecutionContractBindings(
        capability_contracts={capability.id: registry.contract_digest(capability.id)},
        tool_origins={capability.id: origin},
        resource_revisions={"resource-1": "sha256:" + "e" * 64},
        model_routes=_model_execution_contracts(
            model, model.model_profile, None, policy
        ),
    )
    current = bindings

    async def read_contracts(**_kwargs):
        return current

    runtime = CapabilityRuntime(
        registry, (domain,), execution_contract_reader=read_contracts
    )
    scope = replace(
        _scheduled_scope((capability.id,), registry),
        allowed_source_ids=("source-1",),
        allowed_resource_ids=("resource-1",),
        contract_bindings=bindings,
    )
    run = _scheduled_run(scope)
    if machine_origin is RunOrigin.JOB_EVENT:
        scope = replace(
            scope,
            routine_id=None,
            routine_revision=None,
            occurrence_id=None,
            job_id="job-1",
            job_revision=1,
            allowed_connector_binding_ids=(),
        )
        assert run.start is not None
        run = replace(
            run,
            start=replace(
                run.start,
                origin=machine_origin,
                instruction_authority=InstructionAuthority.CODE_OWNED,
                execution_scope=scope,
            ),
        )
    messages = (run.start_message(),)
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, messages)
    first = await runtime.execute_all(
        run,
        (ToolCall("before", view.name, {}),),
        projection=projection,
        messages=messages,
        sensitivity=ModelSensitivity.INTERNAL,
    )
    assert not first[0].is_error
    values = dict(getattr(bindings, changed_family))
    if changed_family == "model_routes":
        values = _model_execution_contracts(
            model, model.model_profile, None, replace(policy, cleanup_timeout_seconds=6)
        )
    else:
        values[next(iter(values))] = "sha256:" + "0" * 64
    current = replace(bindings, **{changed_family: values})
    blocked = await runtime.execute_all(
        run,
        (ToolCall("after", view.name, {}),),
        projection=projection,
        messages=messages,
        sensitivity=ModelSensitivity.INTERNAL,
    )
    assert blocked[0].is_error
    error = blocked[0].output["error"]
    assert isinstance(error, Mapping)
    assert error["code"] in {
        "execution_contract_changed",
        "execution_contract_unavailable",
    }
    assert calls == ["before"]
    assert scope.contract_bindings == bindings
    with pytest.raises(CapabilityInputError):
        await runtime.prepare_run(run)


def test_scope_codec_rejects_missing_extra_and_malformed_execution_bindings() -> None:
    import copy

    encoded = cast(
        dict[str, Any], encode_execution_scope(_scheduled_scope(("test.scheduled",)))
    )
    binding_fields = encoded["fields"]["contract_bindings"]["fields"]
    for family in (
        "capability_contracts",
        "resource_revisions",
        "model_routes",
        "tool_origins",
    ):
        altered = copy.deepcopy(encoded)
        altered["fields"]["contract_bindings"]["fields"][family]["extra"] = (
            "sha256:" + "1" * 64
        )
        with pytest.raises(ValueError):
            decode_execution_scope(altered)
    for invalid in ({}, {"test.scheduled": "not-a-digest"}):
        altered = copy.deepcopy(encoded)
        altered["fields"]["contract_bindings"]["fields"]["capability_contracts"] = (
            invalid
        )
        with pytest.raises(ValueError):
            decode_execution_scope(altered)
    assert binding_fields["tool_origins"] == {}
