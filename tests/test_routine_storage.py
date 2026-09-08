from __future__ import annotations

from _capability_runtime_support import frozen_execution_bindings

import asyncio
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest
from _distribution_support import (
    inbox_distribution_plan,
    no_artifact_outcome_contract,
)

import daita.storage.sqlite as sqlite_module
from daita.capabilities import AccessMode, ExecutionScope, OperationalEffect
from daita.distribution import DeliveryState, DeliverySubjectKind, OutcomeState
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ModelUsage,
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
from daita.llm.pricing import CostEstimate
from daita.routines.models import (
    IntervalSchedule,
    OnceSchedule,
    MisfirePolicy,
    ReportingMode,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineSlotKind,
    RoutineState,
    ScheduledRoutine,
    text_digest,
)
from daita.routines.schedule import occurrence_id, scheduled_slot_key
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_codecs.routines import (
    decode_routine_occurrence,
    decode_scheduled_routine,
    encode_routine_occurrence,
    encode_scheduled_routine,
)

NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)


def routine_record(
    *,
    routine_id: str = "routine-1",
    agent_id: str = "agent-1",
    conversation_id: str = "conversation-1",
    next_due_at: datetime | None = NOW,
    state: RoutineState = RoutineState.ACTIVE,
    maximum_consecutive_failures: int = 3,
    consecutive_failures: int = 0,
) -> ScheduledRoutine:
    instruction = "Read the exact admitted resource and report its current value."
    return ScheduledRoutine(
        contract_bindings=frozen_execution_bindings(
            ("catalog.inspect", "data.query"), ("resource-1",), ("mock:routine",)
        ),
        routine_id=routine_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        owner_principal_id="principal-1",
        title="Current value report",
        authorized_instruction=instruction,
        instruction_digest=text_digest(instruction),
        schedule=IntervalSchedule(3_600, NOW),
        schedule_interpreter_revision=1,
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        last_acknowledged_precheck_observation=None,
        allowed_source_ids=("source-1",),
        allowed_connector_binding_ids=(),
        allowed_resource_ids=("resource-1",),
        allowed_capability_ids=("catalog.inspect", "data.query"),
        allowed_access_modes=frozenset({AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock:routine",),
        skill_bindings=(),
        outcome_contract=no_artifact_outcome_contract(),
        distribution_plan=inbox_distribution_plan(conversation_id),
        per_run_max_tokens=5_000,
        per_run_max_cost_usd=Decimal("0.05"),
        cumulative_max_tokens=50_000,
        cumulative_max_cost_usd=Decimal("0.50"),
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        reserved_tokens=0,
        reserved_cost_usd=Decimal("0"),
        charged_tokens=0,
        charged_cost_usd=Decimal("0"),
        attempt_count=0,
        occurrence_count=0,
        maximum_consecutive_failures=maximum_consecutive_failures,
        consecutive_failures=consecutive_failures,
        expires_at=NOW + timedelta(days=30),
        next_due_at=next_due_at,
        active_occurrence_id=None,
        last_occurrence_id=None,
        last_delivery_ids=(),
        promotion_evidence=None,
        state=state,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
    )


def occurrence_record(
    *,
    routine: ScheduledRoutine | None = None,
    scheduled_for: datetime = NOW,
) -> RoutineOccurrence:
    routine = routine or routine_record()
    slot_key = scheduled_slot_key(
        routine.routine_id,
        routine.revision,
        scheduled_for,
    )
    return RoutineOccurrence(
        occurrence_id=occurrence_id(routine.routine_id, slot_key),
        agent_id=routine.agent_id,
        routine_id=routine.routine_id,
        routine_revision=routine.revision,
        slot_kind=RoutineSlotKind.SCHEDULED,
        slot_key=slot_key,
        scheduled_for=scheduled_for,
        claimed_at=NOW,
        claim_token="claim-1",
        lease_expires_at=NOW + timedelta(seconds=30),
        precheck_observation=None,
        execution_scope=None,
        execution_scope_digest=None,
        reserved_run_id=None,
        reserved_tokens=5_000,
        reserved_cost_usd=Decimal("0.05"),
        charged_tokens=0,
        charged_cost_usd=Decimal("0"),
        run_bound_at=None,
        run_terminal_at=None,
        conclusion_digest=None,
        terminal_run_id=None,
        delivery_ids=(),
        attempt_count=1,
        failure_code=None,
        retry_at=None,
        disposition=RoutineOccurrenceDisposition.CLAIMED,
        created_at=NOW,
        updated_at=NOW,
    )


def test_routine_codec_v1_round_trip_is_exact() -> None:
    routine = routine_record()
    assert (
        decode_scheduled_routine(
            encode_scheduled_routine(routine),
            agent_id=routine.agent_id,
            routine_id=routine.routine_id,
        )
        == routine
    )


def test_occurrence_codec_v1_round_trip_is_exact() -> None:
    occurrence = occurrence_record()
    assert (
        decode_routine_occurrence(
            encode_routine_occurrence(occurrence),
            agent_id=occurrence.agent_id,
            occurrence_id=occurrence.occurrence_id,
        )
        == occurrence
    )


def execution_scope(occurrence: RoutineOccurrence) -> ExecutionScope:
    return ExecutionScope(
        contract_bindings=frozen_execution_bindings(
            ("catalog.inspect", "data.query"), ("resource-1",), ("mock:routine",)
        ),
        scope_id=f"scope:{occurrence.occurrence_id}",
        revision=1,
        agent_id=occurrence.agent_id,
        principal_id="principal-1",
        grant_id=f"routine:{occurrence.routine_id}:revision:{occurrence.routine_revision}",
        job_id=None,
        job_revision=None,
        allowed_source_ids=("source-1",),
        allowed_resource_ids=("resource-1",),
        allowed_capability_ids=("catalog.inspect", "data.query"),
        allowed_access_modes=frozenset({AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock:routine",),
        per_run_max_cost_usd=Decimal("0.05"),
        per_run_max_tokens=5_000,
        distribution_plan_digest=inbox_distribution_plan("conversation-1").plan_digest,
        routine_id=occurrence.routine_id,
        routine_revision=occurrence.routine_revision,
        occurrence_id=occurrence.occurrence_id,
        allowed_connector_binding_ids=(),
    )


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
async def test_run_binding_rejects_every_changed_authority_field(tmp_path, changed):
    from daita.routines.supervisor import _execution_scope

    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
    finally:
        await store.close()


async def test_store_admits_lists_reopens_and_hides_cross_agent_routines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    admitted = await store.admit_scheduled_routine(routine_record())
    assert admitted.next_due_at == NOW
    assert await store.load_scheduled_routine("agent-1", "routine-1") == admitted
    assert await store.load_scheduled_routine("agent-2", "routine-1") is None
    assert await store.list_scheduled_routines("agent-1") == (admitted,)
    assert await store.list_scheduled_routines("agent-2") == ()
    await store.close()

    reopened = await SQLiteStateStore.open(path)
    try:
        assert await reopened.load_scheduled_routine("agent-1", "routine-1") == admitted
        assert await reopened.next_routine_deadline("agent-1") == NOW
    finally:
        await reopened.close()


async def test_duplicate_ticks_reserve_one_occurrence_and_one_budget(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
    finally:
        await store.close()


async def test_pre_run_failure_below_threshold_advances_with_one_delivery(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
        persisted = await store.load_scheduled_routine(
            routine.agent_id, routine.routine_id
        )
        assert persisted is not None
        assert persisted.state is RoutineState.ACTIVE
        assert persisted.consecutive_failures == 1
        assert persisted.active_occurrence_id is None
        assert persisted.next_due_at == NOW + timedelta(hours=1)
        assert persisted.last_delivery_ids == (delivery.delivery_id,)
        assert await store.list_deliveries(routine.agent_id) == (delivery,)
    finally:
        await store.close()


async def test_active_unbound_occurrence_preserves_target_until_delivery_commit(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
        assert await store.conversation_exists(
            routine.agent_id, routine.conversation_id
        )
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
    finally:
        await store.close()


async def test_pre_run_failure_threshold_delivers_one_no_run_escalation(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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

        persisted = await store.load_scheduled_routine(
            routine.agent_id, routine.routine_id
        )
        assert persisted is not None
        assert persisted.state is RoutineState.NEEDS_ATTENTION
        assert persisted.consecutive_failures == 1
        assert persisted.active_occurrence_id is None
        assert persisted.next_due_at is None
        assert persisted.last_delivery_ids == (delivery.delivery_id,)

    finally:
        await store.close()


async def test_pre_run_escalation_finalization_rolls_back_as_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    routine = await store.admit_scheduled_routine(
        routine_record(maximum_consecutive_failures=1)
    )
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-atomic-escalation",
    )
    assert claimed is not None

    def fail_occurrence_replacement(*_args, **_kwargs):
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(
        sqlite_module,
        "_replace_routine_occurrence_row",
        fail_occurrence_replacement,
    )
    with pytest.raises(RuntimeError, match="injected finalization failure"):
        await store.finalize_routine_occurrence(
            routine.agent_id,
            claimed.occurrence_id,
            delivery_id="delivery-rolled-back",
            finalized_at=NOW + timedelta(seconds=1),
            failure_code="routine_authority_revoked",
        )

    assert await store.list_deliveries(routine.agent_id) == ()
    persisted_occurrence = await store.load_routine_occurrence(
        routine.agent_id, claimed.occurrence_id
    )
    assert persisted_occurrence == claimed
    persisted_routine = await store.load_scheduled_routine(
        routine.agent_id, routine.routine_id
    )
    assert persisted_routine is not None
    assert persisted_routine.active_occurrence_id == claimed.occurrence_id
    await store.close()


async def test_delivery_retention_limit_rolls_back_the_producer_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    monkeypatch.setattr(sqlite_module, "MAX_DELIVERIES_PER_AGENT", 1)
    try:
        first_routine = await store.admit_scheduled_routine(
            routine_record(routine_id="routine-retention-one")
        )
        first = await store.claim_due_routine_occurrence(
            first_routine.agent_id,
            first_routine.routine_id,
            expected_revision=first_routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-retention-one",
        )
        assert first is not None
        assert (
            await store.finalize_routine_occurrence(
                first_routine.agent_id,
                first.occurrence_id,
                delivery_id="delivery-retention-one",
                finalized_at=NOW + timedelta(seconds=1),
                failure_code="routine_test_failure",
            )
            is not None
        )

        second_routine = await store.admit_scheduled_routine(
            routine_record(routine_id="routine-retention-two")
        )
        second = await store.claim_due_routine_occurrence(
            second_routine.agent_id,
            second_routine.routine_id,
            expected_revision=second_routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-retention-two",
        )
        assert second is not None
        with pytest.raises(ValueError, match="delivery_retention_limit_exceeded"):
            await store.finalize_routine_occurrence(
                second_routine.agent_id,
                second.occurrence_id,
                delivery_id="delivery-retention-two",
                finalized_at=NOW + timedelta(seconds=2),
                failure_code="routine_test_failure",
            )

        assert len(await store.list_deliveries("agent-1")) == 1
        persisted_occurrence = await store.load_routine_occurrence(
            "agent-1", second.occurrence_id
        )
        assert persisted_occurrence == second
        persisted_routine = await store.load_scheduled_routine(
            "agent-1", second_routine.routine_id
        )
        assert persisted_routine is not None
        assert persisted_routine.active_occurrence_id == second.occurrence_id
    finally:
        await store.close()


async def test_acknowledged_delivery_is_reclaimed_for_next_atomic_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    monkeypatch.setattr(sqlite_module, "MAX_DELIVERIES_PER_AGENT", 1)
    try:
        first_routine = await store.admit_scheduled_routine(
            routine_record(routine_id="routine-reclaim-one")
        )
        first = await store.claim_due_routine_occurrence(
            first_routine.agent_id,
            first_routine.routine_id,
            expected_revision=first_routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-reclaim-one",
        )
        assert first is not None
        finalized_first = await store.finalize_routine_occurrence(
            first_routine.agent_id,
            first.occurrence_id,
            delivery_id="delivery-reclaim-one",
            finalized_at=NOW + timedelta(seconds=1),
            failure_code="routine_test_failure",
        )
        assert finalized_first is not None and finalized_first[1] is not None
        assert await store.acknowledge_delivery(
            first_routine.agent_id,
            finalized_first[1].delivery_id,
            acknowledged_at=NOW + timedelta(seconds=2),
        )

        second_routine = await store.admit_scheduled_routine(
            routine_record(routine_id="routine-reclaim-two")
        )
        second = await store.claim_due_routine_occurrence(
            second_routine.agent_id,
            second_routine.routine_id,
            expected_revision=second_routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-reclaim-two",
        )
        assert second is not None
        finalized_second = await store.finalize_routine_occurrence(
            second_routine.agent_id,
            second.occurrence_id,
            delivery_id="delivery-reclaim-two",
            finalized_at=NOW + timedelta(seconds=3),
            failure_code="routine_test_failure",
        )
        assert finalized_second is not None and finalized_second[1] is not None
        assert finalized_second[1].delivery_id == "delivery-reclaim-two"
        retained = await store.list_deliveries(
            first_routine.agent_id,
            include_acknowledged=True,
        )
        assert tuple(item.delivery_id for item in retained) == ("delivery-reclaim-two",)
        assert (
            await store.load_delivery(
                first_routine.agent_id,
                "delivery-reclaim-one",
            )
            is None
        )
    finally:
        await store.close()


async def test_cancelled_delivery_finalization_cannot_partially_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    routine = await store.admit_scheduled_routine(routine_record())
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-cancel-finalization",
    )
    assert claimed is not None
    claimed_routine = await store.load_scheduled_routine(
        routine.agent_id, routine.routine_id
    )
    assert claimed_routine is not None
    entered = threading.Event()
    release = threading.Event()
    original_start = sqlite_module._CatalogCommitGate.start

    def blocked_start(gate, connection):
        entered.set()
        release.wait(timeout=5)
        return original_start(gate, connection)

    monkeypatch.setattr(sqlite_module._CatalogCommitGate, "start", blocked_start)
    task = asyncio.create_task(
        store.finalize_routine_occurrence(
            routine.agent_id,
            claimed.occurrence_id,
            delivery_id="delivery-cancelled",
            finalized_at=NOW + timedelta(seconds=1),
            failure_code="routine_test_failure",
        )
    )
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await store.list_deliveries(routine.agent_id) == ()
    assert (
        await store.load_routine_occurrence(routine.agent_id, claimed.occurrence_id)
        == claimed
    )
    assert (
        await store.load_scheduled_routine(routine.agent_id, routine.routine_id)
        == claimed_routine
    )
    await store.close()


async def test_cancelled_routine_admission_cannot_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    entered = threading.Event()
    release = threading.Event()
    original_start = sqlite_module._CatalogCommitGate.start

    def blocked_start(gate, connection):
        entered.set()
        release.wait(timeout=5)
        return original_start(gate, connection)

    monkeypatch.setattr(sqlite_module._CatalogCommitGate, "start", blocked_start)
    task = asyncio.create_task(store.admit_scheduled_routine(routine_record()))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await store.load_scheduled_routine("agent-1", "routine-1") is None
    await store.close()


async def test_claim_fault_rolls_back_occurrence_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    routine = await store.admit_scheduled_routine(routine_record())

    def fail_after_occurrence_insert(*_args, **_kwargs):
        raise RuntimeError("injected routine replacement failure")

    monkeypatch.setattr(
        sqlite_module,
        "_replace_routine_row",
        fail_after_occurrence_insert,
    )
    with pytest.raises(RuntimeError, match="injected"):
        await store.claim_due_routine_occurrence(
            routine.agent_id,
            routine.routine_id,
            expected_revision=routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-fault",
        )
    persisted = await store.load_scheduled_routine("agent-1", "routine-1")
    assert persisted == routine
    assert await store.list_routine_occurrences("agent-1", "routine-1") == ()
    await store.close()


async def test_manual_control_identity_is_stable_and_cannot_overlap(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
    finally:
        await store.close()


async def test_stale_claim_recovery_fences_the_old_token(tmp_path: Path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
    finally:
        await store.close()


@pytest.mark.parametrize("usage_case", ("normal", "once", "overrun", "unknown"))
async def test_terminal_run_finalization_is_idempotent_and_delivers_once(
    tmp_path: Path,
    usage_case: str,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        draft = routine_record()
        if usage_case != "normal":
            draft = replace(
                draft,
                cumulative_max_tokens=5_000,
                cumulative_max_cost_usd=Decimal("0.05"),
            )
        if usage_case == "once":
            draft = replace(draft, schedule=OnceSchedule(exact_at=NOW))
        routine = await store.admit_scheduled_routine(draft)
        claimed = await store.claim_due_routine_occurrence(
            routine.agent_id,
            routine.routine_id,
            expected_revision=routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-terminal",
        )
        assert claimed is not None
        scope = execution_scope(claimed)
        bound = await store.bind_routine_occurrence_run(
            routine.agent_id,
            claimed.occurrence_id,
            claim_token="claim-terminal",
            run_id="run-routine-1",
            execution_scope=scope,
            bound_at=NOW + timedelta(seconds=1),
        )
        assert bound is not None
        instruction = routine.authorized_instruction
        run = RunInput(
            id="run-routine-1",
            agent_id=routine.agent_id,
            message=instruction,
            created_at=NOW + timedelta(seconds=1),
            conversation_id=routine.conversation_id,
            source_scope_ids=("source-1",),
            start=RunStartEnvelope(
                origin=RunOrigin.SCHEDULED_ROUTINE,
                instruction_authority=InstructionAuthority.FOREGROUND_AUTHORIZED,
                trusted_instruction_id="routine:routine-1:revision:1",
                trusted_instruction=instruction,
                instruction_digest=routine.instruction_digest,
                untrusted_payload={},
                payload_digest="sha256:" + sha256(b"{}").hexdigest(),
                execution_scope=scope,
            ),
        )
        await store.start(run)
        await store.append(run.id, run.start_message())
        final_text = "The current value is 42."
        result = LoopExit(
            run_id=run.id,
            conversation_id=routine.conversation_id,
            kind=LoopExitKind.COMPLETED,
            reason="completed",
            created_at=NOW + timedelta(seconds=2),
            final_text=final_text,
            steps=1,
            usage=ModelUsage(
                input_tokens=(
                    8_000
                    if usage_case == "overrun"
                    else (100 if usage_case in {"unknown", "once"} else 0)
                ),
                cost_estimate=(
                    CostEstimate.partial(Decimal("0.01"), code="provider_usage_missing")
                    if usage_case == "unknown"
                    else CostEstimate.complete(
                        Decimal(
                            "0.08"
                            if usage_case == "overrun"
                            else ("0.01" if usage_case == "once" else "0")
                        )
                    )
                ),
            ),
        )
        await store.complete(
            result,
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock(final_text),),
            ),
        )
        assert await store.clear_conversations(routine.agent_id) == 0
        assert await store.result(run.id) == result
        assert await store.conversation_exists(
            routine.agent_id, routine.conversation_id
        )
        terminal = await store.mark_routine_occurrence_run_terminal(
            routine.agent_id,
            claimed.occurrence_id,
            run_id=run.id,
            terminal_at=NOW + timedelta(seconds=2),
        )
        assert terminal is not None
        first = await store.finalize_routine_occurrence(
            routine.agent_id,
            claimed.occurrence_id,
            delivery_id="delivery-routine-1",
            finalized_at=NOW + timedelta(seconds=3),
        )
        duplicate = await store.finalize_routine_occurrence(
            routine.agent_id,
            claimed.occurrence_id,
            delivery_id="delivery-routine-duplicate",
            finalized_at=NOW + timedelta(seconds=4),
        )
        assert first is not None and duplicate is not None
        duplicate_delivery = duplicate[1]
        assert duplicate_delivery is not None
        assert first[0].disposition is (
            RoutineOccurrenceDisposition.COMPLETED
            if usage_case in {"normal", "once"}
            else RoutineOccurrenceDisposition.TERMINAL_FAILED
        )
        assert duplicate_delivery.delivery_id == "delivery-routine-1"
        assert duplicate_delivery.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE
        inbox = await store.list_deliveries(routine.agent_id)
        assert len(inbox) == 1
        if usage_case in {"normal", "once"}:
            assert inbox[0].outcome.conclusion_preview == final_text
        else:
            assert "routine_run_" in inbox[0].outcome.conclusion_preview
        persisted = await store.load_scheduled_routine(
            routine.agent_id,
            routine.routine_id,
        )
        assert persisted is not None
        assert persisted.active_occurrence_id is None
        assert persisted.reserved_tokens == 0
        assert persisted.last_delivery_ids == ("delivery-routine-1",)
        if usage_case == "normal":
            assert persisted.next_due_at == NOW + timedelta(hours=1)
        elif usage_case == "once":
            assert persisted.next_due_at is None
            assert persisted.state is RoutineState.COMPLETED
            assert persisted.charged_tokens == 100
            assert persisted.charged_cost_usd == Decimal("0.01")
            assert persisted.model_budget_exhausted
        else:
            expected_tokens, expected_cost = (
                (8_000, Decimal("0.08"))
                if usage_case == "overrun"
                else (5_000, Decimal("0.05"))
            )
            assert (
                persisted.charged_tokens == first[0].charged_tokens == expected_tokens
            )
            assert (
                persisted.charged_cost_usd == first[0].charged_cost_usd == expected_cost
            )
            assert persisted.next_due_at is None
            assert persisted.state is RoutineState.NEEDS_ATTENTION
            assert duplicate_delivery.outcome.conclusion_state is OutcomeState.FAILED
            await store.close()
            store = await SQLiteStateStore.open(tmp_path / "state.db")
            assert (
                await store.load_scheduled_routine(routine.agent_id, routine.routine_id)
                == persisted
            )
            with pytest.raises(ValueError, match="routine_model_budget_exhausted"):
                await store.transition_scheduled_routine(
                    routine.agent_id,
                    routine.routine_id,
                    expected_revision=persisted.revision,
                    state=RoutineState.ACTIVE,
                    transitioned_at=NOW + timedelta(seconds=5),
                )
    finally:
        await store.close()


async def test_terminal_result_sensitivity_escalation_fails_and_blocks_delivery(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
    finally:
        await store.close()


async def test_terminal_failure_at_threshold_uses_its_one_conclusion_as_escalation(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
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
        persisted = await store.load_scheduled_routine(
            routine.agent_id, routine.routine_id
        )
        assert persisted is not None
        assert persisted.state is RoutineState.NEEDS_ATTENTION
        assert persisted.last_delivery_ids == (delivery.delivery_id,)
    finally:
        await store.close()


async def test_reopen_converges_completed_reserved_run_without_reexecution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    routine = await store.admit_scheduled_routine(routine_record())
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token="claim-before-crash",
    )
    assert claimed is not None
    scope = execution_scope(claimed)
    run_id = "run-before-crash"
    bound = await store.bind_routine_occurrence_run(
        routine.agent_id,
        claimed.occurrence_id,
        claim_token="claim-before-crash",
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
    result = LoopExit(
        run_id=run_id,
        conversation_id=routine.conversation_id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=NOW + timedelta(seconds=2),
        final_text="Recovered report.",
        steps=1,
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
    )
    await store.complete(
        result,
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("Recovered report."),),
        ),
    )
    await store.close()

    reopened = await SQLiteStateStore.open(path)
    try:
        recovered = await reopened.recover_stale_routine_occurrences(
            routine.agent_id,
            recovered_at=NOW + timedelta(seconds=31),
            claim_token_factory=lambda identity: f"unused:{identity}",
        )
        assert len(recovered) == 1
        assert recovered[0].disposition is (
            RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
        )
        assert recovered[0].terminal_run_id == run_id
        finalized = await reopened.finalize_routine_occurrence(
            routine.agent_id,
            claimed.occurrence_id,
            delivery_id="delivery-after-reopen",
            finalized_at=NOW + timedelta(seconds=32),
        )
        assert finalized is not None
        assert finalized[0].disposition is RoutineOccurrenceDisposition.COMPLETED
        assert len(await reopened.list_deliveries(routine.agent_id)) == 1
    finally:
        await reopened.close()


async def test_immediate_first_creation_claims_once_with_normal_budget_and_next_anchor(
    tmp_path,
):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    proposal = replace(routine_record(), run_immediately=True)
    created = await store.admit_scheduled_routine(proposal)
    assert created.active_occurrence_id is not None
    assert created.occurrence_count == created.attempt_count == 1
    assert created.reserved_tokens == created.per_run_max_tokens
    assert created.reserved_cost_usd == created.per_run_max_cost_usd
    assert created.next_due_at == NOW + timedelta(hours=1)
    occurrences = await store.list_routine_occurrences(
        created.agent_id, created.routine_id
    )
    assert len(occurrences) == 1
    assert occurrences[0].slot_kind is RoutineSlotKind.MANUAL
    assert occurrences[0].scheduled_for == NOW
    with pytest.raises(ValueError, match="routine_identity_already_exists"):
        await store.admit_scheduled_routine(proposal)
    assert (
        await store.list_routine_occurrences(created.agent_id, created.routine_id)
        == occurrences
    )
    await store.close()


async def test_immediate_first_claim_failure_rolls_back_the_whole_creation(
    tmp_path, monkeypatch
):
    import daita.storage.sqlite as sqlite_module

    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    proposal = replace(routine_record(), run_immediately=True)

    def fail_claim(*_args):
        raise OSError("cannot persist immediate occurrence")

    monkeypatch.setattr(sqlite_module, "_insert_routine_occurrence", fail_claim)
    with pytest.raises(OSError, match="immediate occurrence"):
        await store.admit_scheduled_routine(proposal)
    assert (
        await store.load_scheduled_routine(proposal.agent_id, proposal.routine_id)
        is None
    )
    assert (
        await store.list_routine_occurrences(proposal.agent_id, proposal.routine_id)
        == ()
    )
    await store.close()


async def test_cancelled_immediate_admission_settles_one_atomic_assignment(
    tmp_path, monkeypatch
):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    proposal = replace(routine_record(), run_immediately=True)
    entered, release = threading.Event(), threading.Event()
    original = sqlite_module._insert_routine_occurrence

    def hold_claim(*args):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the atomic admission")
        return original(*args)

    monkeypatch.setattr(sqlite_module, "_insert_routine_occurrence", hold_claim)
    admission = asyncio.create_task(store.admit_scheduled_routine(proposal))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        admission.cancel()
        release.set()
        try:
            await admission
        except asyncio.CancelledError:
            pass
        stored = await store.load_scheduled_routine(
            proposal.agent_id, proposal.routine_id
        )
        assert stored is not None and stored.active_occurrence_id is not None
        occurrences = await store.list_routine_occurrences(
            proposal.agent_id, proposal.routine_id
        )
        assert len(occurrences) == 1 and stored.occurrence_count == 1
        assert stored.reserved_tokens == occurrences[0].reserved_tokens
        assert stored.reserved_cost_usd == occurrences[0].reserved_cost_usd
        with pytest.raises(ValueError, match="routine_identity_already_exists"):
            await store.admit_scheduled_routine(proposal)
        assert await store.list_effect_receipts(proposal.agent_id) == ()
    finally:
        release.set()
        await store.close()
