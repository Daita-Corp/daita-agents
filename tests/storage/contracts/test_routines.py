from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from hashlib import sha256

import pytest

from daita.capabilities import (
    AccessMode,
)
from daita.distribution import DeliveryState, DeliverySubjectKind, OutcomeState
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import (
    InstructionAuthority,
    LoopExit,
    LoopExitKind,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from daita.routines.models import (
    RoutineOccurrenceDisposition,
    RoutineState,
)
from daita.storage.protocols import StateStore
from tests.routines._storage_support import NOW, execution_scope, routine_record

pytestmark = [pytest.mark.integration, pytest.mark.contract]


@pytest.mark.parametrize(
    "changed",
    [
        {"principal_id": "different-principal"},
        {"grant_id": "unapproved-grant"},
        {"sensitivity_ceiling": ModelSensitivity.RESTRICTED},
        {"per_run_max_tokens": 6000},
        {"per_run_max_cost_usd": Decimal("0.10")},
        {"allowed_access_modes": frozenset({AccessMode.READ, AccessMode.WRITE})},
        {"distribution_plan_digest": "sha256:" + "e" * 64},
        {"allowed_connector_binding_ids": ("unapproved-binding",)},
    ],
)
async def test_run_binding_rejects_every_changed_authority_field(
    state_store: StateStore, changed
):
    from daita.routines.supervisor import _execution_scope

    store = state_store
    routine = await store.admit_scheduled_routine(routine_record())
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-exact",
    )
    assert claimed is not None
    scope = _execution_scope(routine, claimed)
    bound = await store.bind_routine_occurrence_run(
        routine.agent_id,
        claimed.occurrence_id,
        claim_token="claim-exact",
        run_id="run-exact",
        execution_scope=replace(scope, **changed),
        bound_at=NOW,
    )
    assert bound is None
    current = await store.load_routine_occurrence(
        routine.agent_id, claimed.occurrence_id
    )
    assert current == claimed


async def test_duplicate_ticks_reserve_one_occurrence_and_one_budget(
    state_store: StateStore,
) -> None:
    store = state_store
    routine = await store.admit_scheduled_routine(routine_record())
    first, second = await asyncio.gather(
        store.claim_due_routine_occurrence(
            routine.agent_id,
            routine.routine_id,
            expected_revision=routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-first",
        ),
        store.claim_due_routine_occurrence(
            routine.agent_id,
            routine.routine_id,
            expected_revision=routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-second",
        ),
    )
    assert first is not None and second is not None
    assert first.occurrence_id == second.occurrence_id
    occurrences = await store.list_routine_occurrences("agent-1", "routine-1")
    assert len(occurrences) == 1
    persisted = await store.load_scheduled_routine("agent-1", "routine-1")
    assert persisted is not None
    assert persisted.occurrence_count == 1
    assert persisted.attempt_count == 1
    assert persisted.reserved_tokens == persisted.per_run_max_tokens
    assert persisted.reserved_cost_usd == persisted.per_run_max_cost_usd


async def test_pre_run_failure_below_threshold_advances_with_one_delivery(
    state_store: StateStore,
) -> None:
    store = state_store
    routine = await store.admit_scheduled_routine(routine_record())
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-pre-run-failure",
    )
    assert claimed is not None

    finalized = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id="delivery-pre-run-failure",
        finalized_at=NOW + timedelta(seconds=1),
        failure_code="routine_authority_revoked",
    )

    assert finalized is not None
    occurrence, delivery = finalized
    assert occurrence.disposition is RoutineOccurrenceDisposition.TERMINAL_FAILED
    assert delivery is not None
    assert occurrence.delivery_ids == (delivery.delivery_id,)
    assert delivery.outcome.resulting_run_id is None
    assert delivery.outcome.conclusion_state is OutcomeState.FAILED
    assert delivery.outcome.failure_code == "routine_authority_revoked"
    persisted = await store.load_scheduled_routine(routine.agent_id, routine.routine_id)
    assert persisted is not None
    assert persisted.state is RoutineState.ACTIVE
    assert persisted.consecutive_failures == 1
    assert persisted.active_occurrence_id is None
    assert persisted.next_due_at == NOW + timedelta(hours=1)
    assert persisted.last_delivery_ids == (delivery.delivery_id,)
    assert await store.list_deliveries(routine.agent_id) == (delivery,)


async def test_active_unbound_occurrence_preserves_target_until_delivery_commit(
    state_store: StateStore,
) -> None:
    store = state_store
    origin = RunInput(
        id="run-target-origin",
        agent_id="agent-1",
        conversation_id="conversation-1",
        message="Authorize the scheduled inbox target.",
        created_at=NOW - timedelta(minutes=1),
    )
    await store.start(origin)
    await store.append(origin.id, origin.start_message())
    origin_result = LoopExit(
        run_id=origin.id,
        conversation_id="conversation-1",
        kind=LoopExitKind.COMPLETED,
        reason="assistant_text",
        created_at=NOW - timedelta(seconds=30),
        final_text="Target authorized.",
        steps=1,
    )
    await store.complete(
        origin_result,
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("Target authorized."),),
        ),
    )
    routine = await store.admit_scheduled_routine(routine_record())
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-unbound-target",
    )
    assert claimed is not None and claimed.reserved_run_id is None

    assert await store.clear_conversations(routine.agent_id) == 0
    assert await store.conversation_exists(routine.agent_id, routine.conversation_id)
    finalized = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id="delivery-unbound-target",
        finalized_at=NOW + timedelta(seconds=1),
        failure_code="routine_precheck_unavailable",
    )
    assert finalized is not None and finalized[1] is not None
    assert finalized[1].conversation_id == routine.conversation_id

    assert await store.clear_conversations(routine.agent_id) == 1
    assert not await store.conversation_exists(
        routine.agent_id, routine.conversation_id
    )
    assert len(await store.list_deliveries(routine.agent_id)) == 1


async def test_pre_run_failure_threshold_delivers_one_no_run_escalation(
    state_store: StateStore,
) -> None:
    store = state_store
    routine = await store.admit_scheduled_routine(
        routine_record(maximum_consecutive_failures=1)
    )
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-threshold-failure",
    )
    assert claimed is not None

    first = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id="delivery-threshold-escalation",
        finalized_at=NOW + timedelta(seconds=1),
        failure_code="routine_precheck_unavailable",
    )
    duplicate = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id="delivery-duplicate",
        finalized_at=NOW + timedelta(seconds=2),
        failure_code="routine_precheck_unavailable",
    )

    assert first is not None and duplicate is not None
    occurrence, delivery = first
    assert delivery is not None
    assert occurrence.disposition is RoutineOccurrenceDisposition.TERMINAL_FAILED
    assert occurrence.delivery_ids == (delivery.delivery_id,)
    assert delivery.outcome.resulting_run_id is None
    assert delivery.visibility_state is DeliveryState.AVAILABLE
    assert delivery.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE
    assert delivery.outcome.conclusion_state is OutcomeState.FAILED
    assert delivery.outcome.failure_code == "routine_precheck_unavailable"
    assert duplicate[1] == delivery
    assert len(await store.list_deliveries(routine.agent_id)) == 1

    persisted = await store.load_scheduled_routine(routine.agent_id, routine.routine_id)
    assert persisted is not None
    assert persisted.state is RoutineState.NEEDS_ATTENTION
    assert persisted.consecutive_failures == 1
    assert persisted.active_occurrence_id is None
    assert persisted.next_due_at is None
    assert persisted.last_delivery_ids == (delivery.delivery_id,)


async def test_manual_control_identity_is_stable_and_cannot_overlap(
    state_store: StateStore,
) -> None:
    store = state_store
    routine = await store.admit_scheduled_routine(routine_record())
    first = await store.claim_manual_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        authorized_control_call_id="control-1",
        claimed_at=NOW + timedelta(minutes=1),
        claim_token="manual-claim",
    )
    duplicate = await store.claim_manual_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        authorized_control_call_id="control-1",
        claimed_at=NOW + timedelta(minutes=2),
        claim_token="different-token",
    )
    overlapping = await store.claim_manual_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        authorized_control_call_id="control-2",
        claimed_at=NOW + timedelta(minutes=2),
        claim_token="overlap-token",
    )
    assert first is not None and duplicate is not None
    assert first.occurrence_id == duplicate.occurrence_id
    assert overlapping is None


async def test_stale_claim_recovery_fences_the_old_token(
    state_store: StateStore,
) -> None:
    store = state_store
    routine = await store.admit_scheduled_routine(routine_record())
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="old-token",
    )
    assert claimed is not None
    recovered = await store.recover_stale_routine_occurrences(
        routine.agent_id,
        recovered_at=NOW + timedelta(seconds=31),
        claim_token_factory=lambda identity: f"recovered:{identity}",
    )
    assert len(recovered) == 1
    assert recovered[0].attempt_count == 2
    assert recovered[0].claim_token != "old-token"
    assert (
        await store.bind_routine_occurrence_run(
            routine.agent_id,
            claimed.occurrence_id,
            claim_token="old-token",
            run_id="run-old-token",
            execution_scope=execution_scope(recovered[0]),
            bound_at=NOW + timedelta(seconds=32),
        )
        is None
    )


async def test_terminal_result_sensitivity_escalation_fails_and_blocks_delivery(
    state_store: StateStore,
) -> None:
    store = state_store
    routine = await store.admit_scheduled_routine(routine_record())
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-sensitive",
    )
    assert claimed is not None
    scope = execution_scope(claimed)
    run_id = "run-routine-sensitive"
    bound = await store.bind_routine_occurrence_run(
        routine.agent_id,
        claimed.occurrence_id,
        claim_token="claim-sensitive",
        run_id=run_id,
        execution_scope=scope,
        bound_at=NOW + timedelta(seconds=1),
    )
    assert bound is not None
    run = RunInput(
        id=run_id,
        agent_id=routine.agent_id,
        message=routine.authorized_instruction,
        created_at=NOW + timedelta(seconds=1),
        conversation_id=routine.conversation_id,
        source_scope_ids=("source-1",),
        start=RunStartEnvelope(
            origin=RunOrigin.SCHEDULED_ROUTINE,
            instruction_authority=InstructionAuthority.FOREGROUND_AUTHORIZED,
            trusted_instruction_id="routine:routine-1:revision:1",
            trusted_instruction=routine.authorized_instruction,
            instruction_digest=routine.instruction_digest,
            untrusted_payload={},
            payload_digest="sha256:" + sha256(b"{}").hexdigest(),
            execution_scope=scope,
        ),
    )
    await store.start(run)
    await store.append(run.id, run.start_message())
    call = ToolCall(id="sensitive-read", name="data_sqlite_query", arguments={})
    await store.append(
        run.id,
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
    )
    await store.append(
        run.id,
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=call.id,
                    output={"kind": "data.query", "data": {"value": 42}},
                    sensitivity=ModelSensitivity.CONFIDENTIAL,
                    sensitivity_provenance={"authority": "current_resource"},
                    capability_id="data.query",
                    executor_id="data.query.executor",
                ),
            ),
        ),
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id=routine.conversation_id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=NOW + timedelta(seconds=2),
        final_text="The sensitive current value is 42.",
        steps=2,
    )
    await store.complete(
        result,
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("The sensitive current value is 42."),),
        ),
    )
    terminal = await store.mark_routine_occurrence_run_terminal(
        routine.agent_id,
        claimed.occurrence_id,
        run_id=run.id,
        terminal_at=result.created_at,
    )
    assert terminal is not None
    finalized = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id="delivery-sensitive",
        finalized_at=NOW + timedelta(seconds=3),
    )
    assert finalized is not None
    occurrence, delivery = finalized
    assert delivery is not None
    assert occurrence.disposition is RoutineOccurrenceDisposition.TERMINAL_FAILED
    assert occurrence.failure_code == "outcome_sensitivity_contract_failed"
    assert delivery.outcome.conclusion_state is OutcomeState.FAILED
    assert delivery.outcome.failure_code == "outcome_sensitivity_contract_failed"
    assert delivery.outcome.effective_sensitivity is ModelSensitivity.CONFIDENTIAL
    assert delivery.outcome.conclusion_preview == ""
    assert delivery.visibility_state is DeliveryState.BLOCKED
    assert delivery.blocked_reason_code == "sensitivity_exceeds_destination"


async def test_terminal_failure_at_threshold_uses_its_one_conclusion_as_escalation(
    state_store: StateStore,
) -> None:
    store = state_store
    routine = await store.admit_scheduled_routine(
        routine_record(maximum_consecutive_failures=1)
    )
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-terminal-failure",
    )
    assert claimed is not None
    scope = execution_scope(claimed)
    run_id = "run-routine-threshold-failure"
    bound = await store.bind_routine_occurrence_run(
        routine.agent_id,
        claimed.occurrence_id,
        claim_token="claim-terminal-failure",
        run_id=run_id,
        execution_scope=scope,
        bound_at=NOW + timedelta(seconds=1),
    )
    assert bound is not None
    run = RunInput(
        id=run_id,
        agent_id=routine.agent_id,
        message=routine.authorized_instruction,
        created_at=NOW + timedelta(seconds=1),
        conversation_id=routine.conversation_id,
        source_scope_ids=("source-1",),
        start=RunStartEnvelope(
            origin=RunOrigin.SCHEDULED_ROUTINE,
            instruction_authority=InstructionAuthority.FOREGROUND_AUTHORIZED,
            trusted_instruction_id="routine:routine-1:revision:1",
            trusted_instruction=routine.authorized_instruction,
            instruction_digest=routine.instruction_digest,
            untrusted_payload={},
            payload_digest="sha256:" + sha256(b"{}").hexdigest(),
            execution_scope=scope,
        ),
    )
    await store.start(run)
    await store.append(run.id, run.start_message())
    failed = LoopExit(
        run_id=run.id,
        conversation_id=routine.conversation_id,
        kind=LoopExitKind.FAILED,
        reason="model_limit",
        created_at=NOW + timedelta(seconds=2),
        steps=1,
    )
    await store.finish(failed)
    terminal = await store.mark_routine_occurrence_run_terminal(
        routine.agent_id,
        claimed.occurrence_id,
        run_id=run.id,
        terminal_at=failed.created_at,
    )
    assert terminal is not None

    first = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id="delivery-terminal-escalation",
        finalized_at=NOW + timedelta(seconds=3),
    )
    duplicate = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id="delivery-terminal-duplicate",
        finalized_at=NOW + timedelta(seconds=4),
    )

    assert first is not None and duplicate is not None
    occurrence, delivery = first
    assert delivery is not None
    assert occurrence.failure_code == "routine_run_model_limit"
    assert delivery.outcome.resulting_run_id == run.id
    assert delivery.outcome.conclusion_state is OutcomeState.FAILED
    assert delivery.outcome.failure_code == "routine_run_model_limit"
    assert duplicate[1] == delivery
    assert len(await store.list_deliveries(routine.agent_id)) == 1
    persisted = await store.load_scheduled_routine(routine.agent_id, routine.routine_id)
    assert persisted is not None
    assert persisted.state is RoutineState.NEEDS_ATTENTION
    assert persisted.last_delivery_ids == (delivery.delivery_id,)
