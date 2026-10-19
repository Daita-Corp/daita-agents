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
from daita.distribution.models import (
    CONVERSATION_INBOX_DESTINATION_REVISION,
    ConversationInboxTarget,
    conversation_inbox_destination_id,
    target_fingerprint,
)
from daita.distribution.owner import construct_graph_job_delivery
from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    TaskAttempt,
    TaskControl,
    canonical_digest,
)
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
from tests.support.graph import GRAPH_NOW, graph_admission, task_result


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


async def _start_graph_task(
    store: SQLiteStateStore, job_id: str, task_id: str, budget: int
) -> TaskAttempt:
    attempt_id = f"{job_id}:{task_id}"
    attempt = await store.claim_graph_task(
        "agent-1",
        job_id,
        task_id,
        attempt_id=attempt_id,
        claim_token=f"{attempt_id}:claim",
        run_id=f"{attempt_id}:run",
        executor_id="executor-1",
        claimed_at=GRAPH_NOW,
        lease_seconds=30,
        absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
        budget_reservations=(BudgetAmount("work_units", budget),),
    )
    assert attempt is not None
    started = await store.start_graph_attempt(
        "agent-1",
        job_id,
        task_id,
        attempt_id,
        claim_token=f"{attempt_id}:claim",
        fencing_epoch=attempt.fencing_epoch,
        started_at=GRAPH_NOW + timedelta(seconds=1),
    )
    assert started is not None and started.claim_token is not None
    return started


async def _finish_graph_job_with_delivery(
    store: SQLiteStateStore, job_id: str, completed_at: datetime
) -> None:
    await store.admit_graph(graph_admission(job_id=job_id))
    for task_id, budget in (("worker", 2), ("finalizer", 1)):
        attempt = await _start_graph_task(store, job_id, task_id, budget)
        assert attempt.claim_token is not None
        result = task_result(
            job_id=job_id,
            task_id=task_id,
            attempt_id=attempt.attempt_id,
            run_id=attempt.run_id,
            result_id=f"{attempt.attempt_id}:result",
            completed_at=completed_at,
        )
        if task_id == "worker":
            await store.complete_graph_attempt(
                result,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                usage=None,
            )
            continue
        destination_id = conversation_inbox_destination_id("conversation-1")
        target = ConversationInboxTarget(
            conversation_id="conversation-1",
            destination_id=destination_id,
            destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
            sensitivity_ceiling=ModelSensitivity.RESTRICTED,
            target_fingerprint=target_fingerprint(
                conversation_id="conversation-1",
                destination_id=destination_id,
                destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
                sensitivity_ceiling=ModelSensitivity.RESTRICTED,
            ),
        )
        await store.finalize_graph_attempt(
            result,
            construct_graph_job_delivery(
                delivery_id=f"delivery-{job_id}",
                agent_id="agent-1",
                conversation_id="conversation-1",
                job_id=job_id,
                target=target,
                result_id=result.result_id,
                result_digest=result.result_digest,
                result_summary=result.summary,
                artifact_references=(),
                effective_sensitivity=ModelSensitivity.RESTRICTED,
                provenance_digest=result.result_digest,
                observed_at=completed_at,
            ),
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            usage=None,
        )


async def _block_graph_job_on_input(store: SQLiteStateStore, job_id: str) -> None:
    await store.admit_graph(graph_admission(job_id=job_id))
    attempt = await _start_graph_task(store, job_id, "worker", 2)
    assert attempt.claim_token is not None
    payload = {"question": "Select the bounded source."}
    await store.open_graph_control(
        TaskControl(
            agent_id="agent-1",
            job_id=job_id,
            task_id="worker",
            control_id=f"{job_id}:control",
            kind=ControlKind.NEEDS_INPUT,
            state=ControlState.OPEN,
            requesting_attempt_id=attempt.attempt_id,
            payload=payload,
            created_at=GRAPH_NOW + timedelta(seconds=2),
            payload_digest=canonical_digest(payload),
        ),
        claim_token=attempt.claim_token,
        fencing_epoch=attempt.fencing_epoch,
    )


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
            await _finish_graph_job_with_delivery(
                store, f"job-{ordinal}", GRAPH_NOW + timedelta(seconds=ordinal + 2)
            )
        await _finalize_failed_routine(store, "routine-after-jobs")
        await _finish_graph_job_with_delivery(
            store, "job-2", GRAPH_NOW + timedelta(seconds=4)
        )
        assert tuple(
            item.job_id for item in await store.list_graph_deliveries("agent-1")
        ) == ("job-2",)
    finally:
        await store.close()

    reopened = await SQLiteStateStore.open(path)
    try:
        assert tuple(
            item.delivery_id for item in await reopened.list_deliveries("agent-1")
        ) == ("delivery-routine-after-jobs",)
        assert await reopened.inspect_graph("agent-1", "job-0") is not None
    finally:
        await reopened.close()


async def test_attention_on_a_job_still_waiting_for_the_user_is_never_reclaimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(delivery_store_module, "MAX_DELIVERIES_PER_AGENT", 2)
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        # "job-1" is a prefix of "job-10"; only the exact job's state may protect.
        await _block_graph_job_on_input(store, "job-1")
        await _block_graph_job_on_input(store, "job-10")
        assert await store.resolve_graph_control(
            "agent-1",
            "job-10",
            "worker",
            "job-10:control",
            state=ControlState.RESOLVED,
            resolved_at=GRAPH_NOW + timedelta(seconds=3),
            resolved_by_kind="user",
            resolved_by_id="user-1",
            resolution={"source_id": "source-1"},
            make_ready=True,
        )
        attention = await store.list_deliveries("agent-1")
        assert {item.subject_id.split("/")[0] for item in attention} == {
            "job-1",
            "job-10",
        }

        await _finalize_failed_routine(store, "routine-after-attention")
        assert {
            item.delivery_id for item in await store.list_deliveries("agent-1")
        } == {
            "delivery-routine-after-attention",
            *(
                item.delivery_id
                for item in attention
                if item.subject_id.startswith("job-1/")
            ),
        }
        with pytest.raises(ValueError, match="delivery_retention_limit_exceeded"):
            await _finalize_failed_routine(store, "routine-without-room")
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
        "occurrence_exhausted",
    ),
)
async def test_terminal_run_finalization_is_idempotent_and_delivers_once(
    tmp_path: Path,
    usage_case: str,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        draft = routine_record()
        # Subscription routines carry no dollar ceilings; tokens still bound them.
        unpriced = usage_case.startswith("subscription")
        if unpriced:
            draft = replace(
                draft, per_run_max_cost_usd=None, cumulative_max_cost_usd=None
            )
        if usage_case == "occurrence_exhausted":
            draft = replace(draft, cumulative_max_occurrences=1)
        elif usage_case not in {
            "normal",
            "subscription",
            "subscription_replayed_stall",
        }:
            draft = replace(
                draft,
                cumulative_max_tokens=5_000,
                cumulative_max_cost_usd=None if unpriced else Decimal("0.05"),
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
        if unpriced:
            assert claimed.reserved_cost_usd == Decimal("0")
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
            "occurrence_exhausted",
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
        inbox = await store.list_deliveries(routine.agent_id)
        assert len(inbox) == 1
        if completed:
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
        elif usage_case == "subscription_replayed_stall":
            # Unknown consumption is charged the whole per-run reservation.
            assert persisted.next_due_at == NOW + timedelta(hours=1)
            assert persisted.charged_tokens == first[0].charged_tokens
            assert persisted.charged_tokens == claimed.reserved_tokens > 100
            assert persisted.charged_cost_usd == Decimal("0")
        elif usage_case == "subscription":
            assert persisted.next_due_at == NOW + timedelta(hours=1)
            assert persisted.charged_tokens == 100
            assert persisted.charged_cost_usd == Decimal("0")
            assert persisted.reserved_cost_usd == Decimal("0")
        elif usage_case == "occurrence_exhausted":
            # A used-up allowance cannot stay ACTIVE with an unclaimable past due.
            assert persisted.state is RoutineState.NEEDS_ATTENTION
            assert persisted.next_due_at is None
            assert await store.next_routine_deadline(routine.agent_id) is None
            with pytest.raises(ValueError, match="routine_occurrence_budget_exhausted"):
                await store.transition_scheduled_routine(
                    routine.agent_id,
                    routine.routine_id,
                    expected_revision=persisted.revision,
                    state=RoutineState.ACTIVE,
                    transitioned_at=NOW + timedelta(hours=2),
                )
            assert (
                await store.load_scheduled_routine(routine.agent_id, routine.routine_id)
                == persisted
            )
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
                else (5_000, Decimal("0") if unpriced else Decimal("0.05"))
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
        assert routine.reserved_tokens == 0
        assert routine.reserved_cost_usd == Decimal("0")
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

    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    proposal = replace(routine_record(), run_immediately=True)

    def fail_claim(*_args):
        raise OSError("cannot persist immediate occurrence")

    monkeypatch.setattr(routine_store_module, "_insert_routine_occurrence", fail_claim)
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
    original = routine_store_module._insert_routine_occurrence

    def hold_claim(*args):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the atomic admission")
        return original(*args)

    monkeypatch.setattr(routine_store_module, "_insert_routine_occurrence", hold_claim)
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
            RoutineOccurrenceDisposition.PRECHECKING,
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
