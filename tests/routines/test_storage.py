from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

import daita.storage.sql as sqlite_module
import daita.storage.sql_deliveries as delivery_store_module
import daita.storage.sql_routines as routine_store_module
from daita.distribution import DeliverySubjectKind, OutcomeState
from daita.jobs import JobState
from daita.jobs.owner import JobOwner
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import STALL_USAGE_UNREPORTED, CostEstimate
from daita.loop.models import (
    InstructionAuthority,
    LoopExit,
    LoopExitKind,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from daita.routines.models import (
    OnceSchedule,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineSlotKind,
    RoutineState,
)
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_codecs.routines import (
    decode_routine_occurrence,
    decode_scheduled_routine,
    encode_routine_occurrence,
    encode_scheduled_routine,
)
from tests.routines._storage_support import (
    NOW,
    execution_scope,
    occurrence_record,
    routine_record,
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


async def test_pre_run_escalation_finalization_rolls_back_as_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    routine = await store.admit_scheduled_routine(
        # A one-time routine that fails escalates to Needs you.
        routine_record(schedule=OnceSchedule(NOW))
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
        routine_store_module,
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
    monkeypatch.setattr(delivery_store_module, "MAX_DELIVERIES_PER_AGENT", 1)
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
    monkeypatch.setattr(delivery_store_module, "MAX_DELIVERIES_PER_AGENT", 1)
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


async def _finish_job_with_delivery(
    store: SQLiteStateStore, ordinal: int, at: datetime
) -> str:
    owner = JobOwner(
        agent_id="agent-1",
        store=store,
        clock=lambda: at,
        id_factory=lambda prefix: f"{prefix}-{ordinal}",
        per_run_max_tokens=1_000,
    )
    job = await owner.start("Count the rows.", conversation_id="conversation-1")
    run_id = f"run-{ordinal}"
    assert await owner.begin_run(job.job_id, run_id) is not None
    done = await owner.end_run(
        job.job_id, run_id, tokens=0, state=JobState.DONE, final_text="Counted."
    )
    assert done is not None and done.state is JobState.DONE
    return job.job_id


async def _finalize_failed_routine(
    store: SQLiteStateStore, routine_id: str
) -> RoutineOccurrence:
    routine = await store.admit_scheduled_routine(routine_record(routine_id=routine_id))
    claimed = await store.claim_due_routine_occurrence(
        routine.agent_id,
        routine.routine_id,
        expected_revision=routine.revision,
        expected_due_at=NOW,
        claimed_at=NOW,
        claim_token=f"{routine_id}:claim",
    )
    assert claimed is not None
    finalized = await store.finalize_routine_occurrence(
        routine.agent_id,
        claimed.occurrence_id,
        delivery_id=f"delivery-{routine_id}",
        finalized_at=NOW + timedelta(seconds=1),
        failure_code="routine_test_failure",
    )
    assert finalized is not None and finalized[1] is not None
    return claimed


async def test_job_result_deliveries_yield_their_slots_to_routine_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Job results are never acknowledged; the job record stays authoritative.
    for module in (sqlite_module, delivery_store_module):
        monkeypatch.setattr(module, "MAX_DELIVERIES_PER_AGENT", 2)
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    try:
        for ordinal in range(2):
            await _finish_job_with_delivery(
                store, ordinal, NOW + timedelta(seconds=ordinal + 2)
            )
        await _finalize_failed_routine(store, "routine-after-jobs")
        await _finish_job_with_delivery(store, 2, NOW + timedelta(seconds=4))
    finally:
        await store.close()

    reopened = await SQLiteStateStore.open(path)
    try:
        assert {
            (item.subject_kind, item.delivery_id)
            for item in await reopened.list_deliveries("agent-1")
        } == {
            (DeliverySubjectKind.ROUTINE_OCCURRENCE, "delivery-routine-after-jobs"),
            (DeliverySubjectKind.JOB, "delivery-2"),
        }
        assert await reopened.load_job("agent-1", "job-0") is not None
    finally:
        await reopened.close()


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
        routine_store_module,
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


@pytest.mark.parametrize(
    "usage_case",
    (
        "normal",
        "once",
        "overrun",
        "unknown",
        "subscription",
        "subscription_no_tokens",
        "subscription_replayed_stall",
        "subscription_other_unknown",
    ),
)
async def test_terminal_run_finalization_is_idempotent_and_delivers_once(
    tmp_path: Path,
    usage_case: str,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        draft = routine_record()
        # A subscription run has no dollar limit; its run's token limit still bounds it.
        unpriced = usage_case.startswith("subscription")
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
        if unpriced:
            scope = replace(scope, per_run_max_cost_usd=None)
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
                    else (
                        100
                        if usage_case
                        in {
                            "unknown",
                            "once",
                            "subscription",
                            "subscription_replayed_stall",
                            "subscription_other_unknown",
                        }
                        else 0
                    )
                ),
                cost_estimate=(
                    CostEstimate.partial(Decimal("0.01"), code="provider_usage_missing")
                    if usage_case == "unknown"
                    # A replayed stall adds an attempt that reported no usage.
                    else (
                        CostEstimate.unavailable(STALL_USAGE_UNREPORTED)
                        if usage_case == "subscription_replayed_stall"
                        # Other unknown consumption on an unpriced route still fails.
                        else (
                            CostEstimate.unavailable("multiple_incomplete_reasons")
                            if usage_case == "subscription_other_unknown"
                            else (
                                CostEstimate.unavailable("subscription_billing")
                                if unpriced
                                else CostEstimate.complete(
                                    Decimal(
                                        "0.08"
                                        if usage_case == "overrun"
                                        else ("0.01" if usage_case == "once" else "0")
                                    )
                                )
                            )
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
        completed = usage_case in {
            "normal",
            "once",
            "subscription",
            "subscription_replayed_stall",
        }
        assert first[0].disposition is (
            RoutineOccurrenceDisposition.COMPLETED
            if completed
            else RoutineOccurrenceDisposition.TERMINAL_FAILED
        )
        assert duplicate_delivery.delivery_id == "delivery-routine-1"
        assert duplicate_delivery.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE
        deliveries = await store.list_deliveries(routine.agent_id)
        assert len(deliveries) == 1
        if completed:
            assert deliveries[0].outcome.conclusion_preview == final_text
        else:
            assert "routine_run_" in deliveries[0].outcome.conclusion_preview
        persisted = await store.load_scheduled_routine(
            routine.agent_id,
            routine.routine_id,
        )
        assert persisted is not None
        assert persisted.active_occurrence_id is None
        assert persisted.last_delivery_ids == ("delivery-routine-1",)
        if usage_case == "normal":
            assert persisted.next_due_at == NOW + timedelta(hours=1)
        elif usage_case == "subscription_replayed_stall":
            # Unknown consumption is charged the run's whole token limit.
            assert persisted.next_due_at == NOW + timedelta(hours=1)
            assert persisted.charged_tokens == first[0].charged_tokens
            assert persisted.charged_tokens == scope.per_run_max_tokens > 100
            assert persisted.charged_cost_usd == Decimal("0")
        elif usage_case == "subscription":
            assert persisted.next_due_at == NOW + timedelta(hours=1)
            assert persisted.charged_tokens == 100
            assert persisted.charged_cost_usd == Decimal("0")
        elif usage_case == "once":
            assert persisted.next_due_at is None
            assert persisted.state is RoutineState.COMPLETED
            assert persisted.charged_tokens == 100
            assert persisted.charged_cost_usd == Decimal("0.01")
        else:
            expected_tokens, expected_cost = (
                (8_000, Decimal("0.08"))
                if usage_case == "overrun"
                else (5_000, Decimal("0") if unpriced else Decimal("0.05"))
            )
            assert (
                persisted.charged_tokens == first[0].charged_tokens == expected_tokens
            )
            assert (
                persisted.charged_cost_usd == first[0].charged_cost_usd == expected_cost
            )
            # One failure keeps a recurring routine going at its next slot.
            assert persisted.state is RoutineState.ACTIVE
            assert persisted.consecutive_failures == 1
            assert persisted.next_due_at == NOW + timedelta(hours=1)
            assert duplicate_delivery.outcome.conclusion_state is OutcomeState.FAILED
            await store.close()
            store = await SQLiteStateStore.open(tmp_path / "state.db")
            assert (
                await store.load_scheduled_routine(routine.agent_id, routine.routine_id)
                == persisted
            )
    finally:
        await store.close()


async def test_routine_charges_equal_the_sum_of_its_run_usage(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    runs = (
        ModelUsage(
            input_tokens=300,
            output_tokens=20,
            cost_estimate=CostEstimate.complete(Decimal("0.004")),
        ),
        ModelUsage(
            input_tokens=450,
            output_tokens=30,
            cost_estimate=CostEstimate.complete(Decimal("0.006")),
        ),
    )
    try:
        routine = await store.admit_scheduled_routine(routine_record())
        for index, usage in enumerate(runs):
            due_at = routine.next_due_at
            assert due_at is not None
            claimed = await store.claim_due_routine_occurrence(
                routine.agent_id,
                routine.routine_id,
                expected_revision=routine.revision,
                expected_due_at=due_at,
                claimed_at=due_at,
                claim_token=f"claim-{index}",
            )
            assert claimed is not None
            run_id = f"run-routine-sum-{index}"
            bound = await store.bind_routine_occurrence_run(
                routine.agent_id,
                claimed.occurrence_id,
                claim_token=f"claim-{index}",
                run_id=run_id,
                execution_scope=execution_scope(claimed),
                bound_at=due_at + timedelta(seconds=1),
            )
            assert bound is not None
            run = RunInput(
                id=run_id,
                agent_id=routine.agent_id,
                message=routine.authorized_instruction,
                created_at=due_at + timedelta(seconds=1),
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
                    execution_scope=execution_scope(claimed),
                ),
            )
            await store.start(run)
            await store.append(run.id, run.start_message())
            final_text = f"Run {index} completed."
            await store.complete(
                LoopExit(
                    run_id=run.id,
                    conversation_id=routine.conversation_id,
                    kind=LoopExitKind.COMPLETED,
                    reason="completed",
                    created_at=due_at + timedelta(seconds=2),
                    final_text=final_text,
                    steps=1,
                    usage=usage,
                ),
                CanonicalMessage(
                    role=MessageRole.ASSISTANT,
                    content=(TextBlock(final_text),),
                ),
            )
            assert await store.mark_routine_occurrence_run_terminal(
                routine.agent_id,
                claimed.occurrence_id,
                run_id=run.id,
                terminal_at=due_at + timedelta(seconds=2),
            )
            for attempt in range(2):
                # A repeated finalization returns the first outcome and charges nothing.
                finalized = await store.finalize_routine_occurrence(
                    routine.agent_id,
                    claimed.occurrence_id,
                    delivery_id=f"delivery-routine-sum-{index}-{attempt}",
                    finalized_at=due_at + timedelta(seconds=3 + attempt),
                )
                assert finalized is not None
                assert finalized[0].charged_tokens == usage.total_tokens
                assert finalized[0].charged_cost_usd == usage.cost_estimate.amount_usd
            current = await store.load_scheduled_routine(
                routine.agent_id, routine.routine_id
            )
            assert current is not None
            routine = current
        assert routine.charged_tokens == sum(item.total_tokens for item in runs)
        assert routine.charged_cost_usd == sum(
            (item.cost_estimate.amount_usd or Decimal(0) for item in runs),
            Decimal(0),
        )
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


async def test_immediate_first_creation_claims_once_and_keeps_the_next_anchor(
    tmp_path,
):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    proposal = routine_record()
    created = await store.admit_scheduled_routine(proposal, run_immediately=True)
    assert created.active_occurrence_id is not None
    assert created.occurrence_count == created.attempt_count == 1
    assert created.next_due_at == NOW + timedelta(hours=1)
    occurrences = await store.list_routine_occurrences(
        created.agent_id, created.routine_id
    )
    assert len(occurrences) == 1
    assert occurrences[0].slot_kind is RoutineSlotKind.MANUAL
    assert occurrences[0].scheduled_for == NOW
    with pytest.raises(ValueError, match="routine_identity_already_exists"):
        await store.admit_scheduled_routine(proposal, run_immediately=True)
    assert (
        await store.list_routine_occurrences(created.agent_id, created.routine_id)
        == occurrences
    )
    await store.close()


async def test_immediate_first_claim_failure_rolls_back_the_whole_creation(
    tmp_path, monkeypatch
):

    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    proposal = routine_record()

    def fail_claim(*_args):
        raise OSError("cannot persist immediate occurrence")

    monkeypatch.setattr(routine_store_module, "_insert_routine_occurrence", fail_claim)
    with pytest.raises(OSError, match="immediate occurrence"):
        await store.admit_scheduled_routine(proposal, run_immediately=True)
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
    proposal = routine_record()
    entered, release = threading.Event(), threading.Event()
    original = routine_store_module._insert_routine_occurrence

    def hold_claim(*args):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the atomic admission")
        return original(*args)

    monkeypatch.setattr(routine_store_module, "_insert_routine_occurrence", hold_claim)
    admission = asyncio.create_task(
        store.admit_scheduled_routine(proposal, run_immediately=True)
    )
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
        with pytest.raises(ValueError, match="routine_identity_already_exists"):
            await store.admit_scheduled_routine(proposal, run_immediately=True)
        assert await store.list_effect_receipts(proposal.agent_id) == ()
    finally:
        release.set()
        await store.close()


async def _seed_occurrences(
    store: SQLiteStateStore, *occurrences: RoutineOccurrence
) -> None:
    def write(connection) -> None:
        for occurrence in occurrences:
            routine_store_module._insert_routine_occurrence(connection, occurrence)

    await sqlite_module.run_sql_transaction(store._database, write)


def _occurrence_in(
    disposition: RoutineOccurrenceDisposition,
    *,
    hour: int,
    agent_id: str = "agent-1",
) -> RoutineOccurrence:
    return replace(
        occurrence_record(
            routine=routine_record(agent_id=agent_id),
            scheduled_for=NOW + timedelta(hours=hour),
        ),
        disposition=disposition,
    )


async def test_active_occurrence_reads_return_only_queued_or_running_work(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        await store.admit_scheduled_routine(routine_record())
        await store.admit_scheduled_routine(routine_record(agent_id="agent-2"))
        seeded = tuple(
            _occurrence_in(disposition, hour=hour)
            for hour, disposition in enumerate(RoutineOccurrenceDisposition)
        )
        other_agent = _occurrence_in(
            RoutineOccurrenceDisposition.RUNNING, hour=0, agent_id="agent-2"
        )
        await _seed_occurrences(store, *seeded, other_agent)

        active = await store.list_active_routine_occurrences("agent-1")
        assert {item.disposition for item in active} == {
            RoutineOccurrenceDisposition.CLAIMED,
            RoutineOccurrenceDisposition.RUNNING,
            RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
            RoutineOccurrenceDisposition.RETRYABLE,
        }
        assert all(item.agent_id == "agent-1" for item in active)
        ids = [item.occurrence_id for item in active]
        assert ids == sorted(ids, reverse=True)
        assert await store.list_active_routine_occurrences("agent-1", limit=2) == (
            active[:2]
        )
        assert await store.list_active_routine_occurrences("agent-3") == ()
        for limit in (0, 101, True):
            with pytest.raises(ValueError, match="active occurrence limit"):
                await store.list_active_routine_occurrences("agent-1", limit=limit)
    finally:
        await store.close()


async def test_occurrence_lookup_returns_only_known_owned_identities(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        await store.admit_scheduled_routine(routine_record())
        await store.admit_scheduled_routine(routine_record(agent_id="agent-2"))
        first = _occurrence_in(RoutineOccurrenceDisposition.COMPLETED, hour=1)
        second = _occurrence_in(RoutineOccurrenceDisposition.TERMINAL_FAILED, hour=2)
        foreign = _occurrence_in(
            RoutineOccurrenceDisposition.COMPLETED, hour=3, agent_id="agent-2"
        )
        await _seed_occurrences(store, first, second, foreign)

        loaded = await store.load_routine_occurrences(
            "agent-1",
            (second.occurrence_id, "occurrence-unknown", foreign.occurrence_id)
            + (first.occurrence_id,),
        )
        assert loaded == tuple(
            sorted((first, second), key=lambda item: item.occurrence_id)
        )
        assert await store.load_routine_occurrences("agent-1", ()) == ()
        assert (
            await store.load_routine_occurrences(
                "agent-1", tuple(f"occurrence-{index}" for index in range(50))
            )
            == ()
        )
        for invalid in (
            tuple(f"occurrence-{index}" for index in range(51)),
            ("",),
        ):
            with pytest.raises(ValueError, match="occurrence identities"):
                await store.load_routine_occurrences("agent-1", invalid)
    finally:
        await store.close()


@pytest.mark.parametrize("carried", (True, False))
async def test_older_receipt_with_a_grant_verifies_against_its_runs_scope(
    tmp_path: Path, carried: bool
) -> None:
    """A receipt that used a standing grant, from before grants were removed.

    The routine record no longer has grants; the run's own execution scope
    still carries the one the receipt used, and finalization checks it there.
    """

    from daita._json import FrozenJsonObject, canonical_json
    from daita.capabilities import CapabilityGrant, EffectEvidenceBasis, EffectOutcome
    from daita.llm.models import ToolCall, ToolResultBlock
    from daita.storage.sqlite_codecs import encode_receipt
    from daita.storage.sqlite_records import EffectReceipt, effect_receipt_id

    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        routine = await store.admit_scheduled_routine(routine_record())
        claimed = await store.claim_due_routine_occurrence(
            routine.agent_id,
            routine.routine_id,
            expected_revision=routine.revision,
            expected_due_at=NOW,
            claimed_at=NOW,
            claim_token="claim-granted",
        )
        assert claimed is not None
        unscoped = execution_scope(claimed)
        grant = CapabilityGrant(
            grant_id="grant-released",
            domain_owner_id="data",
            capability_id="data.query",
            capability_contract_digest=(
                unscoped.contract_bindings.capability_contracts["data.query"]
            ),
            constraints_kind="released",
            constraints=FrozenJsonObject.from_mapping({}),
            max_calls_per_occurrence=1,
        )
        scope = replace(unscoped, capability_grants=(grant,))
        run_id = "run-routine-granted"
        assert await store.bind_routine_occurrence_run(
            routine.agent_id,
            claimed.occurrence_id,
            claim_token="claim-granted",
            run_id=run_id,
            execution_scope=scope,
            bound_at=NOW + timedelta(seconds=1),
        )
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
        operation_key = "sha256:" + "5" * 64
        receipt = EffectReceipt(
            receipt_id=effect_receipt_id(
                agent_id=routine.agent_id,
                run_id=run_id,
                call_id="granted-call",
                operation_key=operation_key,
            ),
            receipt_kind="effect",
            agent_id=routine.agent_id,
            run_id=run_id,
            call_id="granted-call",
            capability_id="data.query",
            domain_owner_id="data",
            capability_contract_digest=grant.capability_contract_digest,
            operation_key=operation_key,
            argument_fingerprint="sha256:" + "6" * 64,
            sensitivity=ModelSensitivity.INTERNAL,
            started_at=NOW + timedelta(seconds=1),
            routine_id=routine.routine_id,
            routine_revision=routine.revision,
            occurrence_id=claimed.occurrence_id,
            # Older data: the receipt names the grant that authorized it.
            capability_grant_digest=(
                grant.grant_digest if carried else "sha256:" + "9" * 64
            ),
            outcome=EffectOutcome.SUCCEEDED,
            evidence_basis=EffectEvidenceBasis.SERVER_REPORTED,
            payload=FrozenJsonObject.from_mapping({}),
            finished_at=NOW + timedelta(seconds=1),
        )

        def insert(connection) -> None:
            connection.execute(
                "INSERT INTO effect_receipts(agent_id, id, run_id, call_id, "
                "operation_key, routine_id, occurrence_id, grant_digest, "
                "unresolved, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    receipt.agent_id,
                    receipt.receipt_id,
                    receipt.run_id,
                    receipt.call_id,
                    receipt.operation_key,
                    receipt.routine_id,
                    receipt.occurrence_id,
                    receipt.capability_grant_digest,
                    encode_receipt(receipt),
                ),
            )

        await sqlite_module.run_sql_transaction(store._database, insert)
        output = {
            "effect_receipt": {
                "receipt_id": receipt.receipt_id,
                "receipt_digest": receipt.receipt_digest,
                "outcome": receipt.outcome.value,
                "evidence_basis": receipt.evidence_basis.value,
            }
        }
        call = ToolCall("granted-call", "data_query", {})
        await store.append(
            run.id, CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(call,))
        )
        await store.append(
            run.id,
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id=call.id,
                        output=output,
                        sensitivity=ModelSensitivity.INTERNAL,
                        sensitivity_provenance={"authority": "test"},
                        capability_id="data.query",
                        executor_id="data.query.executor",
                        output_sha256="sha256:"
                        + sha256(
                            canonical_json(
                                FrozenJsonObject.from_mapping(output)
                            ).encode("utf-8")
                        ).hexdigest(),
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
            final_text="The value was written.",
            steps=2,
            usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
        )
        await store.complete(
            result,
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock("The value was written."),),
            ),
        )
        assert await store.mark_routine_occurrence_run_terminal(
            routine.agent_id,
            claimed.occurrence_id,
            run_id=run.id,
            terminal_at=result.created_at,
        )
        finalized = await store.finalize_routine_occurrence(
            routine.agent_id,
            claimed.occurrence_id,
            delivery_id="delivery-granted",
            finalized_at=NOW + timedelta(seconds=3),
        )
        assert finalized is not None and finalized[1] is not None
        occurrence, delivery = finalized
        assert delivery is not None
        assert occurrence.effect_receipt_ids == (receipt.receipt_id,)
        if carried:
            assert occurrence.disposition is RoutineOccurrenceDisposition.COMPLETED
            assert delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        else:
            assert occurrence.failure_code == "outcome_effect_binding_invalid"
            assert delivery.outcome.conclusion_state is OutcomeState.FAILED
    finally:
        await store.close()
