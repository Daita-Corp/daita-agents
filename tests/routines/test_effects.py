"""Phase D proof: one ordinary loop/supervisor for native and unrelated effects."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita._json import FrozenJsonObject
from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import (
    AccessMode,
    CapabilityInputError,
    CapabilityRegistry,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ExecutionContractBindings,
    OperationalEffect,
    ToolLoadMode,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime, SideEffectPlan
from daita.distribution import (
    DistributionOwner,
    OutcomeState,
    conversation_inbox_destination_id,
)
from daita.distribution.models import EffectRequirement
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.loop import AgentLoop, LoopLimits
from daita.routines.models import (
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    RoutineOccurrenceDisposition,
    RoutineState,
)
from daita.routines.owner import RoutineOwner
from daita.routines.supervisor import RoutineSupervisor
from tests.capabilities._effect_contract_support import _capability
from tests.capabilities._effect_receipt_support import STARTED_AT, _store
from tests.capabilities._effect_runtime_support import _EffectExecutor
from tests.routines._supervisor_support import _ids
from tests.support.capability_runtime import (
    StaticTestDomain,
    presentation_metadata,
)
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.toolbox_model import ToolboxAwareMockModelProvider


class _NoCatalog:
    async def readable_resource_ids(self, *args, **kwargs):
        raise AssertionError(
            "a source-free routine must not query an ambient source catalog"
        )


class _Context:
    async def prepare(self, run, messages, tool_context, *, max_total_tokens=None):
        return None

    def project(
        self,
        snapshot,
        messages,
        *,
        step,
        tool_context,
        previous_request_input_tokens=None,
        remaining_tokens=None,
        request_input_growth_tokens=None,
        remaining_steps=None,
    ):
        return ModelRequest(
            messages=messages,
            tools=tool_context.provider_definitions,
            sensitivity=ModelSensitivity.INTERNAL,
        )


class _StandingDomain(StaticTestDomain):
    async def prepare_automation_grant(
        self, capability, constraints, max_calls_per_occurrence, proposal
    ):
        if constraints["target"] != "admitted-target" or max_calls_per_occurrence > 2:
            raise CapabilityInputError(
                "grant_outside_scope",
                "The exact target or call ceiling is not admitted.",
            )
        return constraints

    async def prepare_call(
        self, run, call, capability, arguments, *, request_sensitivity
    ):
        if run.execution_scope is not None:
            grant = next(
                item
                for item in run.execution_scope.capability_grants
                if item.capability_id == capability.id
            )
            if arguments["target"] != grant.constraints["target"]:
                raise CapabilityInputError(
                    "grant_outside_scope", "The call differs from the standing target."
                )
        return arguments

    async def side_effect_plan(self, run, call, capability, execution, fingerprint):
        scope = run.execution_scope
        assert scope is not None
        grant = next(
            item
            for item in scope.capability_grants
            if item.capability_id == capability.id
        )
        return SideEffectPlan(
            approval_required=False,
            capability_grant_digest=grant.grant_digest,
            effect_intent=FrozenJsonObject.from_mapping(
                {"arguments": execution.arguments}
            ),
        )


class _RoutineExecutor(_EffectExecutor):
    async def execute(self, request):
        output = await super().execute(request)
        if self.mode == "not-applied":
            assert output.effect_observation is not None
            return replace(
                output,
                effect_observation=EffectObservation(
                    EffectOutcome.NOT_APPLIED,
                    EffectEvidenceBasis.ADAPTER_VERIFIED,
                    output.effect_observation.payload,
                ),
            )
        return output


async def _assignment(
    tmp_path: Path,
    *,
    basis=EffectEvidenceBasis.SERVER_REPORTED,
    mode="success",
    minimum=1,
    invoke=True,
    repeated=False,
    defer=False,
    read_only=False,
    through_tool=False,
    approvals=None,
    document_minimum=0,
    after_run=None,
    clock=None,
):
    clock = clock or (lambda: STARTED_AT)
    store = await _store(tmp_path / "state.db")
    store._clock = clock
    identity_factory = _ids()
    capability = _capability()
    assert capability.effect_receipt_policy is not None
    capability = replace(
        capability,
        effect_receipt_policy=replace(
            capability.effect_receipt_policy, success_evidence_basis=basis
        ),
        access_mode=(
            AccessMode.WRITE
            if basis is EffectEvidenceBasis.ADAPTER_VERIFIED
            else AccessMode.NONE
        ),
        operational_effect=(
            OperationalEffect.MUTATE_DATA
            if basis is EffectEvidenceBasis.ADAPTER_VERIFIED
            else OperationalEffect.EXTERNAL_ACTION
        ),
    )
    if read_only:
        capability = replace(
            capability,
            access_mode=AccessMode.NONE,
            operational_effect=OperationalEffect.NONE,
            automation_grant_policy=None,
            effect_receipt_policy=None,
        )
    from collections.abc import Mapping

    properties = capability.input_schema["properties"]
    assert isinstance(properties, Mapping)
    capability = replace(
        capability,
        input_schema={
            **dict(capability.input_schema),
            "properties": {**dict(properties), "value": {"type": "integer"}},
        },
    )
    view = ToolView(
        name="test_action",
        capability_id=capability.id,
        description=capability.description,
        presentation=presentation_metadata(load_mode=ToolLoadMode.ON_DEMAND),
    )
    domain = _StandingDomain((capability,), (view,))
    executor = _RoutineExecutor(store=store, mode=mode, basis=basis)
    from daita.domains.data.export_capabilities import (
        DOCUMENT_CREATE_CAPABILITY_ID,
        DocumentArtifactExecutor,
        artifact_capability_declarations,
    )

    declarations = artifact_capability_declarations(
        include_local_delivery=False, include_local_edit=False
    )
    document = next(
        item
        for item in declarations.capabilities
        if item.id == DOCUMENT_CREATE_CAPABILITY_ID
    )
    document_view = next(
        item for item in declarations.tool_views if item.capability_id == document.id
    )
    document_domain = StaticTestDomain(
        (document,), (document_view,), domain_owner_id="artifact"
    )
    artifacts = await AgentHomeArtifactStore.open(
        agent_id="agent-effect",
        agent_home=tmp_path,
        references=store,
        clock=clock,
    )
    usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
    script = (
        [
            ModelResponse(
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("action-one", view.name, {"target": "admitted-target"}),
                ),
            )
        ]
        if invoke
        else []
    )
    if repeated:
        script.append(
            ModelResponse(
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "action-two",
                        view.name,
                        {"target": "admitted-target", "value": 2},
                    ),
                ),
            )
        )
    script.append(
        ModelResponse(
            usage=usage,
            finish_reason=FinishReason.STOP,
            text="The assignment is finished.",
        )
    )
    if document_minimum:
        script.insert(
            0,
            ModelResponse(
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "partial-report",
                        document_view.name,
                        {
                            "format": "markdown",
                            "content": "Research completed; pending action evidence.",
                        },
                    ),
                ),
            ),
        )
    model = ToolboxAwareMockModelProvider(script, complete_pricing=True)
    registry: CapabilityRegistry | None = None

    async def read_contracts(**kwargs):
        assert (
            kwargs["source_ids"]
            == kwargs["resource_ids"]
            == kwargs["connector_binding_ids"]
            == ()
        )
        current_registry = registry
        assert current_registry is not None
        return ExecutionContractBindings(
            capability_contracts={
                key: current_registry.contract_digest(key)
                for key in kwargs["capability_ids"]
            },
            model_routes={
                key: "sha256:" + "c" * 64 for key in kwargs["model_route_ids"]
            },
        )

    limits = LoopLimits(max_total_tokens=10000, max_estimated_cost_usd=Decimal("1"))
    distribution = DistributionOwner(agent_id="agent-effect", store=store)
    owner = RoutineOwner(
        agent_id="agent-effect",
        store=store,
        catalog=_NoCatalog(),
        distribution=distribution,
        skills=None,
        eligible_model_routes=(model.provider_id,),
        maximum_per_run_tokens=10000,
        maximum_per_run_cost_usd=Decimal("1"),
        clock=clock,
        execution_contract_reader=read_contracts,
    )
    from daita.capabilities import (
        ApprovalDecision,
        CapabilityDeclarations,
    )
    from daita.routines.capabilities import (
        ROUTINE_DOMAIN_OWNER_ID,
        RoutineCapabilityDomain,
        routine_capability_declarations,
    )

    lifecycle = routine_capability_declarations(owner)
    routine_domain = RoutineCapabilityDomain(
        CapabilityDeclarations(
            domain_owner_id=ROUTINE_DOMAIN_OWNER_ID,
            capabilities=lifecycle.capabilities,
            executor_ids=tuple(item.executor_id for item in lifecycle.capabilities),
            tool_views=lifecycle.tool_views,
        ),
        owner,
    )
    registry = CapabilityRegistry(
        declarations=(
            domain.declarations,
            routine_domain.declarations,
            document_domain.declarations,
        ),
        executors=(executor, *lifecycle.executors, DocumentArtifactExecutor()),
    )

    async def approve(request):
        if approvals is not None:
            approvals.append(request)
        return ApprovalDecision.APPROVE

    runtime = CapabilityRuntime(
        registry,
        (domain, routine_domain, document_domain),
        effect_receipts=store,
        artifacts=artifacts,
        execution_contract_reader=read_contracts,
        limits=limits,
        approval_handler=approve,
        clock=clock,
    )
    owner.bind_capability_registry(registry)
    owner.bind_grant_preparer(runtime.prepare_automation_grant)
    contract = replace(
        no_artifact_outcome_contract(),
        effect_requirements=(
            ()
            if read_only
            else (EffectRequirement(capability.id, minimum, frozenset({basis})),)
        ),
    )
    if document_minimum:
        from daita.artifacts.models import ArtifactAuthorship
        from daita.distribution.models import ArtifactRequirement

        contract = replace(
            contract,
            maximum_total_artifact_bytes=4096,
            artifact_requirements=(
                ArtifactRequirement(
                    required=True,
                    minimum_count=document_minimum,
                    maximum_count=2,
                    allowed_media_types=("text/markdown",),
                    allowed_authorships=(ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,),
                    allowed_producer_capability_ids=(document.id,),
                    maximum_artifact_bytes=2048,
                    maximum_total_bytes=4096,
                    maximum_sensitivity=ModelSensitivity.INTERNAL,
                ),
            ),
        )
    proposal = await owner.prepare_create(
        run_id="run-effect",
        conversation_id="conversation-effect",
        call_id="create-effect",
        title="Exact external assignment",
        authorized_instruction="Invoke the exact admitted action and report its evidence.",
        schedule=IntervalSchedule(3600, STARTED_AT),
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        allowed_source_ids=(),
        allowed_resource_ids=(),
        allowed_connector_binding_ids=(),
        allowed_capability_ids=(
            tuple(sorted((capability.id, document.id)))
            if document_minimum
            else (capability.id,)
        ),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        outcome_contract=contract,
        distribution_destination_id=conversation_inbox_destination_id(
            "conversation-effect"
        ),
        eligible_model_routes=(model.provider_id,),
        per_run_max_tokens=10000,
        per_run_max_cost_usd=Decimal("1"),
        cumulative_max_tokens=100000,
        cumulative_max_cost_usd=Decimal("10"),
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        maximum_consecutive_failures=3,
        expires_at=STARTED_AT + timedelta(days=1),
        skill_names=(),
        basis_run_id=None,
        requested_capability_grants=(
            ()
            if read_only
            else (
                RequestedCapabilityGrant(
                    capability.id,
                    FrozenJsonObject.from_mapping({"target": "admitted-target"}),
                    1,
                ),
            )
        ),
        run_immediately=True,
    )
    if through_tool:
        from daita.routines.capabilities import _spec_schema
        from daita.routines.owner import _routine_proposal_payload
        from tests.support.capability_runtime import execute_projected

        payload = _routine_proposal_payload(proposal)
        schema = _spec_schema(update=False)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        arguments = {
            key: value
            for key, value in payload.items()
            if key in properties and value is not None
        }
        arguments.update(
            skill_names=(),
            distribution_destination_id=proposal.distribution_plan.targets[
                0
            ].destination_id,
            requested_capability_grants=tuple(
                {
                    "capability_id": grant.capability_id,
                    "constraints": grant.constraints,
                    "max_calls_per_occurrence": grant.max_calls_per_occurrence,
                }
                for grant in proposal.capability_grants
            ),
        )
        origin = (await store.load("run-effect")).run
        rejected = await execute_projected(
            runtime,
            origin,
            (ToolCall("private-create", "routine_create", arguments),),
            sensitivity=ModelSensitivity.RESTRICTED,
        )
        assert rejected.ordered_results[0].is_error
        assert (
            rejected.ordered_results[0].output["error"]["code"]
            == "routine_instruction_sensitivity_exceeded"
        )
        assert await store.list_scheduled_routines("agent-effect") == ()
        assert not approvals
        outcome = await execute_projected(
            runtime,
            origin,
            (ToolCall("create-effect", "routine_create", arguments),),
        )
        assert not outcome.ordered_results[0].is_error, outcome.ordered_results[
            0
        ].output
        routine = await store.load_scheduled_routine(
            "agent-effect", proposal.routine_id
        )
        assert routine is not None
    else:
        routine = await owner.admit(proposal)
    loop = AgentLoop(
        model=model,
        context_builder=_Context(),
        tools=runtime,
        transcripts=store,
        limits=limits,
        clock=clock,
    )

    async def execute_run(occurrence, run, observation):
        prepared = await loop.prepare(run)
        assert run.execution_scope is not None
        bound = await store.bind_routine_occurrence_run(
            "agent-effect",
            occurrence.occurrence_id,
            claim_token=occurrence.claim_token,
            run_id=run.id,
            execution_scope=run.execution_scope,
            bound_at=clock(),
        )
        assert bound is not None
        result = await loop.run(run, prepared=prepared)
        if after_run is not None:
            await after_run(store, run, result)
        return result

    supervisor = RoutineSupervisor(
        agent_id="agent-effect",
        store=store,
        owner=owner,
        runtime=runtime,
        distribution=distribution,
        artifacts=artifacts,
        execute_run=execute_run,
        clock=clock,
        id_factory=identity_factory,
    )
    assert routine.active_occurrence_id is not None
    occurrence = await store.load_routine_occurrence(
        "agent-effect", routine.active_occurrence_id
    )
    assert occurrence is not None
    if not defer:
        await supervisor._run_claimed(occurrence)
    inspection = await owner.inspect(routine.routine_id)
    assert inspection is not None
    return store, owner, runtime, executor, model, inspection, proposal, supervisor


@pytest.mark.parametrize(
    "basis", (EffectEvidenceBasis.ADAPTER_VERIFIED, EffectEvidenceBasis.SERVER_REPORTED)
)
async def test_routine_uses_one_standing_grant_and_authenticated_successful_receipt(
    tmp_path, basis
):
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, basis=basis)
    )
    try:
        occurrence = inspection.recent_occurrences[0]
        assert occurrence.disposition is RoutineOccurrenceDisposition.COMPLETED
        assert executor.calls == 1
        assert len(occurrence.effect_receipt_ids) == 1
        receipt = await store.load_effect_receipt(
            "agent-effect", occurrence.effect_receipt_ids[0]
        )
        assert receipt is not None and receipt.outcome is EffectOutcome.SUCCEEDED
        assert receipt.evidence_basis is basis
        assert (
            receipt.capability_grant_digest
            == inspection.routine.capability_grants[0].grant_digest
        )
        delivery = (await store.list_deliveries("agent-effect"))[0]
        assert delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        assert delivery.outcome.effect_receipt_ids == occurrence.effect_receipt_ids
        assert inspection.routine.next_due_at == STARTED_AT + timedelta(hours=1)
    finally:
        await store.close()


@pytest.mark.parametrize(
    "minimum,expected",
    (
        (0, RoutineOccurrenceDisposition.COMPLETED),
        (1, RoutineOccurrenceDisposition.TERMINAL_FAILED),
    ),
)
async def test_optional_and_required_actions_cannot_be_satisfied_by_speech(
    tmp_path, minimum, expected
):
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, minimum=minimum, invoke=False)
    )
    try:
        occurrence = inspection.recent_occurrences[0]
        assert occurrence.disposition is expected
        assert occurrence.effect_receipt_ids == () and executor.calls == 0
        if minimum:
            assert occurrence.failure_code == "outcome_effect_requirement_unsatisfied"
    finally:
        await store.close()


@pytest.mark.parametrize(
    "mode,basis,receipt_outcome,failure",
    (
        (
            "disconnect",
            EffectEvidenceBasis.SERVER_REPORTED,
            EffectOutcome.UNCERTAIN,
            "outcome_effect_uncertain",
        ),
        (
            "bad-output",
            EffectEvidenceBasis.SERVER_REPORTED,
            EffectOutcome.UNCERTAIN,
            "outcome_effect_uncertain",
        ),
        (
            "bad-output",
            EffectEvidenceBasis.ADAPTER_VERIFIED,
            EffectOutcome.SUCCEEDED,
            "outcome_effect_result_invalid",
        ),
    ),
)
async def test_uncertain_or_unusable_required_effect_never_reports_success(
    tmp_path, mode, basis, receipt_outcome, failure
):
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, mode=mode, basis=basis, minimum=0)
    )
    try:
        occurrence = inspection.recent_occurrences[0]
        assert occurrence.disposition is RoutineOccurrenceDisposition.TERMINAL_FAILED
        assert occurrence.failure_code == failure
        assert executor.calls == 1
        receipt = await store.load_effect_receipt(
            "agent-effect", occurrence.effect_receipt_ids[0]
        )
        assert receipt is not None and receipt.outcome is receipt_outcome
        if receipt_outcome is EffectOutcome.UNCERTAIN:
            assert inspection.routine.state is RoutineState.PAUSED
    finally:
        await store.close()


async def test_durable_grant_ceiling_counts_a_distinct_second_operation(tmp_path):
    from collections.abc import Mapping

    from daita.llm.models import ToolResultBlock

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, repeated=True)
    )
    try:
        occurrence = inspection.recent_occurrences[0]
        assert executor.calls == 1
        assert len(await store.list_effect_receipts("agent-effect")) == 1
        assert occurrence.terminal_run_id is not None
        transcript = await store.load(occurrence.terminal_run_id)
        second = next(
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "action-two"
        )
        assert second.is_error
        error = second.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "effect_reservation_conflict"
    finally:
        await store.close()


async def test_uncertainty_blocks_resume_manual_claim_and_new_clones_after_reopen(
    tmp_path,
):
    from daita.routines.models import RoutineControlAction
    from daita.storage.sqlite import SQLiteStateStore
    from daita.storage.sqlite_records import EffectUnresolvedError

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, mode="disconnect")
    )
    routine = inspection.routine
    with pytest.raises(EffectUnresolvedError):
        await owner.control(
            routine.routine_id,
            expected_revision=routine.revision,
            action=RoutineControlAction.RESUME,
            authorized_control_call_id="explicit-resume",
        )
    await store.close()
    reopened = await SQLiteStateStore.open(
        tmp_path / "state.db", clock=lambda: STARTED_AT
    )
    try:
        with pytest.raises(EffectUnresolvedError) as blocked:
            await reopened.transition_scheduled_routine(
                "agent-effect",
                routine.routine_id,
                expected_revision=routine.revision,
                state=RoutineState.ACTIVE,
                transitioned_at=STARTED_AT,
            )
        assert (
            blocked.value.receipt_ids
            == inspection.recent_occurrences[0].effect_receipt_ids
        )
        with pytest.raises(EffectUnresolvedError):
            await reopened.admit_scheduled_routine(
                replace(proposal, routine_id="routine-new-clone")
            )
        assert (
            await reopened.load_scheduled_routine("agent-effect", "routine-new-clone")
            is None
        )
        assert len(await reopened.list_effect_receipts("agent-effect")) == 1
        assert executor.calls == 1
    finally:
        await reopened.close()


@pytest.mark.parametrize("expansion", ["schedule", "instruction", "budget", "expiry"])
async def test_unresolved_effect_blocks_expanding_an_unrelated_assignment(
    tmp_path, expansion
):
    from daita.routines.models import text_digest
    from daita.storage.sqlite_records import EffectUnresolvedError

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, mode="disconnect", defer=True)
    )
    try:
        unrelated = await owner.admit(
            replace(proposal, routine_id="routine-unrelated", run_immediately=False)
        )
        await supervisor._run_claimed(inspection.recent_occurrences[0])
        expansions: dict[str, dict[str, object]] = {
            "schedule": {"schedule": IntervalSchedule(60, STARTED_AT)},
            "instruction": {
                "authorized_instruction": "Perform a different action.",
                "instruction_digest": text_digest("Perform a different action."),
            },
            "budget": {"cumulative_max_occurrences": 11},
            "expiry": {"expires_at": proposal.expires_at + timedelta(days=1)},
        }
        changes = expansions[expansion]
        revised = replace(unrelated, revision=unrelated.revision + 1, **changes)
        with pytest.raises(EffectUnresolvedError):
            await store.revise_scheduled_routine(
                revised, expected_revision=unrelated.revision
            )
        assert (
            await store.load_scheduled_routine("agent-effect", unrelated.routine_id)
            == unrelated
        )
        assert executor.calls == 1
    finally:
        await store.close()


async def test_existing_unrelated_routine_can_finish_while_uncertain_routine_stays_blocked(
    tmp_path,
):
    from daita.routines.models import RoutineControlAction
    from daita.storage.sqlite_records import EffectUnresolvedError

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, mode="disconnect", defer=True)
    )
    try:
        independent = await owner.admit(
            replace(
                proposal,
                routine_id="routine-earlier-independent",
                run_immediately=False,
                schedule=IntervalSchedule(3600, STARTED_AT + timedelta(hours=1)),
            )
        )
        await supervisor._run_claimed(inspection.recent_occurrences[0])
        failed = await owner.inspect(inspection.routine.routine_id)
        assert failed is not None and failed.routine.state is RoutineState.PAUSED
        with pytest.raises(EffectUnresolvedError):
            await owner.control(
                failed.routine.routine_id,
                expected_revision=failed.routine.revision,
                action=RoutineControlAction.RUN_NOW,
                authorized_control_call_id="blocked-run-now",
            )
        executor.mode = "success"
        usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
        model.replace_script(
            (
                ModelResponse(
                    usage=usage,
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "independent-action",
                            "test_action",
                            {"target": "admitted-target"},
                        ),
                    ),
                ),
                ModelResponse(
                    usage=usage,
                    finish_reason=FinishReason.STOP,
                    text="The independent action completed.",
                ),
            )
        )
        independent = await owner.control(
            independent.routine_id,
            expected_revision=independent.revision,
            action=RoutineControlAction.RUN_NOW,
            authorized_control_call_id="independent-run-now",
        )
        assert independent.active_occurrence_id is not None
        occurrence = await store.load_routine_occurrence(
            "agent-effect", independent.active_occurrence_id
        )
        assert occurrence is not None
        await supervisor._run_claimed(occurrence)
        completed = await owner.inspect(independent.routine_id)
        assert (
            completed is not None
            and completed.recent_occurrences[0].disposition
            is RoutineOccurrenceDisposition.COMPLETED
        )
        assert executor.calls == 2
        assert (
            len(await store.list_effect_receipts("agent-effect", unresolved_only=True))
            == 1
        )
    finally:
        await store.close()


async def test_known_nonapplication_still_consumes_the_native_invocation_ceiling(
    tmp_path,
):
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(
            tmp_path,
            mode="not-applied",
            basis=EffectEvidenceBasis.ADAPTER_VERIFIED,
            repeated=True,
        )
    )
    try:
        receipts = await store.list_effect_receipts("agent-effect")
        assert executor.calls == len(receipts) == 1
        assert receipts[0].outcome is EffectOutcome.NOT_APPLIED
        assert (
            inspection.recent_occurrences[0].failure_code
            == "outcome_effect_requirement_unsatisfied"
        )
        assert not receipts[0].unresolved
    finally:
        await store.close()


async def test_bound_run_can_dispatch_after_its_initial_claim_lease(
    tmp_path, monkeypatch
):
    now = STARTED_AT
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, defer=True, clock=lambda: now)
    )
    generate = model.generate

    async def slow_model(request):
        nonlocal now
        response = await generate(request)
        now += timedelta(seconds=31)
        return response

    monkeypatch.setattr(model, "generate", slow_model)
    try:
        await supervisor._run_claimed(inspection.recent_occurrences[0])
        updated = await owner.inspect(inspection.routine.routine_id)
        assert updated is not None
        assert (
            updated.recent_occurrences[0].disposition
            is RoutineOccurrenceDisposition.COMPLETED
        )
        receipts = await store.list_effect_receipts("agent-effect")
        assert executor.calls == len(receipts) == 1
        assert receipts[0].outcome is EffectOutcome.SUCCEEDED
    finally:
        await store.close()


async def test_expired_occurrence_claim_prevents_reservation_and_dispatch(tmp_path):
    now = STARTED_AT
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, defer=True, clock=lambda: now)
    )
    try:
        now += timedelta(hours=1)
        await supervisor._run_claimed(inspection.recent_occurrences[0])
        updated = await owner.inspect(inspection.routine.routine_id)
        assert updated is not None
        assert (
            updated.recent_occurrences[0].disposition
            is RoutineOccurrenceDisposition.TERMINAL_FAILED
        )
        assert executor.calls == 0
        assert await store.list_effect_receipts("agent-effect") == ()
    finally:
        await store.close()


async def test_source_free_effect_free_assignment_uses_the_same_occurrence_path(
    tmp_path,
):
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, read_only=True, invoke=False)
    )
    try:
        assert (
            inspection.recent_occurrences[0].disposition
            is RoutineOccurrenceDisposition.COMPLETED
        )
        scope = inspection.recent_occurrences[0].execution_scope
        assert (
            scope is not None
            and scope.allowed_source_ids
            == scope.allowed_resource_ids
            == scope.allowed_connector_binding_ids
            == ()
        )
        assert scope.capability_grants == inspection.routine.capability_grants == ()
        assert inspection.recent_occurrences[0].effect_receipt_ids == ()
        assert executor.calls == 0 and len(model.logical_requests) == 1
    finally:
        await store.close()


async def test_effect_requirement_accepts_supported_alternatives_and_rejects_stronger_proof(
    tmp_path,
):
    from daita.routines.owner import RoutineError

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(tmp_path, defer=True)
    )
    try:
        both = EffectRequirement(
            "test.action",
            1,
            frozenset(
                {
                    EffectEvidenceBasis.ADAPTER_VERIFIED,
                    EffectEvidenceBasis.SERVER_REPORTED,
                }
            ),
        )
        await owner.proposal_authority_snapshot(
            replace(
                proposal,
                outcome_contract=replace(
                    proposal.outcome_contract, effect_requirements=(both,)
                ),
            )
        )
        stronger = replace(
            both,
            accepted_evidence_bases=frozenset({EffectEvidenceBasis.ADAPTER_VERIFIED}),
        )
        with pytest.raises(RoutineError) as unsupported:
            await owner.proposal_authority_snapshot(
                replace(
                    proposal,
                    outcome_contract=replace(
                        proposal.outcome_contract, effect_requirements=(stronger,)
                    ),
                )
            )
        assert unsupported.value.code == "routine_effect_requirement_unsupported"
        excessive = replace(both, minimum_successful_calls=2)
        with pytest.raises(ValueError, match="grants and completion requirements"):
            await owner.proposal_authority_snapshot(
                replace(
                    proposal,
                    outcome_contract=replace(
                        proposal.outcome_contract, effect_requirements=(excessive,)
                    ),
                )
            )
        assert executor.calls == 0
    finally:
        await store.close()


@pytest.mark.parametrize(
    "basis", (EffectEvidenceBasis.ADAPTER_VERIFIED, EffectEvidenceBasis.SERVER_REPORTED)
)
async def test_immediate_assignment_uses_one_creation_approval_and_no_per_call_approval(
    tmp_path, basis
):
    from daita.capabilities import ApprovalRequest

    approvals: list[ApprovalRequest] = []
    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(
            tmp_path,
            through_tool=True,
            approvals=approvals,
            basis=basis,
        )
    )
    try:
        assert len(approvals) == 1
        assert approvals[0].capability_id == "routines.create"
        assert (
            inspection.recent_occurrences[0].disposition
            is RoutineOccurrenceDisposition.COMPLETED
        )
        assert executor.calls == 1
        assert inspection.routine.occurrence_count == 1
        approved = approvals[0].arguments["proposal"]
        assert isinstance(approved, FrozenJsonObject)
        assert approved["run_immediately"] is True
        approved_grants = approved["capability_grants"]
        assert isinstance(approved_grants, tuple)
        assert approved_grants[0] == FrozenJsonObject.from_mapping(
            inspection.routine.capability_grants[0].material()
        )
    finally:
        await store.close()


@pytest.mark.parametrize("document_minimum", (1, 2))
@pytest.mark.parametrize("crash_before_terminal", (False, True))
async def test_failed_assignment_keeps_authenticated_partial_artifacts(
    tmp_path, document_minimum, crash_before_terminal
):
    async def recover_missing_terminal(store, run, result):
        import sqlite3

        # Emulate process loss after the durable tool results, before the loop's
        # terminal write. Use the same recovery entry point as Agent.open.
        with sqlite3.connect(store.path) as connection:
            connection.execute("UPDATE runs SET result = NULL WHERE id = ?", (run.id,))
            connection.execute(
                "DELETE FROM messages WHERE run_id = ? AND position = "
                "(SELECT MAX(position) FROM messages WHERE run_id = ?)",
                (run.id, run.id),
            )
        recovered = await store.recover_unfinished_runs(
            "agent-effect", created_at=STARTED_AT
        )
        recovered_run = next(item for item in recovered if item.run_id == run.id)
        assert recovered_run.artifacts == result.artifacts

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(
            tmp_path,
            mode="disconnect",
            document_minimum=document_minimum,
            after_run=recover_missing_terminal if crash_before_terminal else None,
        )
    )
    try:
        occurrence = inspection.recent_occurrences[0]
        assert occurrence.disposition is RoutineOccurrenceDisposition.TERMINAL_FAILED
        assert occurrence.failure_code == "outcome_effect_uncertain"
        delivery = (await store.list_deliveries("agent-effect"))[0]
        assert len(delivery.outcome.artifact_references) == 1
        assert (
            delivery.outcome.artifact_references[0].producing_run_id
            == occurrence.terminal_run_id
        )
        assert delivery.outcome.effect_receipt_ids == occurrence.effect_receipt_ids
        assert executor.calls == 1
    finally:
        await store.close()


async def test_corrupted_tool_result_cannot_satisfy_an_effect_requirement(tmp_path):
    import sqlite3

    from daita.llm.models import MessageRole, ToolResultBlock
    from daita.storage.sqlite_codecs import decode_message, encode_message

    async def corrupt_result(store, run, result):
        with sqlite3.connect(store.path) as connection:
            for position, data in connection.execute(
                "SELECT position, data FROM messages WHERE run_id = ?", (run.id,)
            ):
                message = decode_message(data)
                if message.role is not MessageRole.TOOL:
                    continue
                block = message.content[0]
                if isinstance(block, ToolResultBlock) and block.call_id == "action-one":
                    forged = replace(
                        block,
                        output={"receipt_id": "a receipt ID alone is not evidence"},
                    )
                    connection.execute(
                        "UPDATE messages SET data = ? WHERE run_id = ? AND position = ?",
                        (
                            encode_message(replace(message, content=(forged,))),
                            run.id,
                            position,
                        ),
                    )

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(
            tmp_path,
            after_run=corrupt_result,
        )
    )
    try:
        assert (
            inspection.recent_occurrences[0].failure_code
            == "outcome_effect_result_invalid"
        )
        receipt = (await store.list_effect_receipts("agent-effect"))[0]
        assert receipt.outcome is EffectOutcome.SUCCEEDED and executor.calls == 1
    finally:
        await store.close()


async def test_human_resolution_keeps_observation_and_requires_explicit_future_work(
    tmp_path,
):
    from daita.routines.models import RoutineControlAction
    from daita.storage.sqlite_records import EffectResolution, EffectResolutionDecision

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(
            tmp_path,
            mode="disconnect",
        )
    )
    try:
        receipt = (await store.list_effect_receipts("agent-effect"))[0]
        resolved = await store.resolve_effect_receipt(
            "agent-effect",
            EffectResolution(
                receipt_id=receipt.receipt_id,
                receipt_digest=receipt.receipt_digest,
                decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
                approving_principal_id="agent:agent-effect",
                control_id="foreground-reviewed-resolution",
                resolved_at=STARTED_AT,
                note="Reviewed external state; allow future authorized work.",
            ),
        )
        assert resolved.outcome is EffectOutcome.UNCERTAIN
        assert (
            resolved.receipt_digest == receipt.receipt_digest
            and not resolved.unresolved
        )
        current = await owner.inspect(proposal.routine_id)
        assert current is not None and current.routine.state is RoutineState.PAUSED
        assert executor.calls == 1 and current.routine.active_occurrence_id is None
        resumed = await owner.control(
            proposal.routine_id,
            expected_revision=current.routine.revision,
            action=RoutineControlAction.RESUME,
            authorized_control_call_id="explicit-resume",
        )
        assert (
            resumed.state is RoutineState.ACTIVE
            and resumed.active_occurrence_id is None
        )
        assert executor.calls == 1
        usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
        model.replace_script(
            (
                ModelResponse(
                    usage=usage,
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "new-authorized-action",
                            "test_action",
                            {"target": "admitted-target"},
                        ),
                    ),
                ),
                ModelResponse(
                    usage=usage,
                    finish_reason=FinishReason.STOP,
                    text="The new invocation completed.",
                ),
            )
        )
        executor.mode = "success"
        claimed = await owner.control(
            proposal.routine_id,
            expected_revision=resumed.revision,
            action=RoutineControlAction.RUN_NOW,
            authorized_control_call_id="new-explicit-run",
        )
        assert claimed.active_occurrence_id is not None
        occurrence = await store.load_routine_occurrence(
            "agent-effect", claimed.active_occurrence_id
        )
        assert occurrence is not None
        await supervisor._run_claimed(occurrence)
        receipts = await store.list_effect_receipts("agent-effect")
        assert executor.calls == len(receipts) == 2
        assert len({item.operation_key for item in receipts}) == 2
        assert (
            await store.load_effect_receipt("agent-effect", receipt.receipt_id)
        ) == resolved
    finally:
        await store.close()


async def test_recovery_disable_survives_delayed_failed_occurrence_finalization(
    tmp_path,
):
    from daita.storage.sqlite_records import EffectResolution, EffectResolutionDecision

    async def resolve_before_finalization(store, run, result):
        receipt = (await store.list_effect_receipts("agent-effect", run_id=run.id))[0]
        await store.resolve_effect_receipt(
            "agent-effect",
            EffectResolution(
                receipt_id=receipt.receipt_id,
                receipt_digest=receipt.receipt_digest,
                decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
                approving_principal_id="agent:agent-effect",
                control_id="foreground-disable-after-review",
                resolved_at=STARTED_AT,
                note="Accept the unknown result and permanently close this assignment.",
            ),
        )

    store, owner, runtime, executor, model, inspection, proposal, supervisor = (
        await _assignment(
            tmp_path,
            mode="disconnect",
            after_run=resolve_before_finalization,
        )
    )
    try:
        assert inspection.routine.state is RoutineState.DISABLED
        assert (
            inspection.routine.active_occurrence_id is None
            and inspection.routine.next_due_at is None
        )
        assert (
            inspection.recent_occurrences[0].disposition
            is RoutineOccurrenceDisposition.TERMINAL_FAILED
        )
        assert executor.calls == 1
        assert (
            await store.list_effect_receipts("agent-effect", unresolved_only=True) == ()
        )
        delivery = (await store.list_deliveries("agent-effect"))[0]
        assert delivery.outcome.conclusion_state is OutcomeState.FAILED
        assert "did not meet" in delivery.outcome.conclusion_preview
        assert "The assignment is finished" not in delivery.outcome.conclusion_preview
    finally:
        await store.close()
