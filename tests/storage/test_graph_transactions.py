from __future__ import annotations

import asyncio
import random
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    EdgeKind,
    GraphMutationRequest,
    MutationDecision,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskDependency,
    TaskState,
    canonical_digest,
    topology_digest,
)
from daita.jobs.graph.validation import GraphValidationError
from daita.llm.models import ModelSensitivity
from daita.storage import sqlite_graph
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_graph import GraphStoreConflictError
from tests.support.graph import GRAPH_NOW, graph_admission, task_result

pytestmark = pytest.mark.integration


async def _store(tmp_path: Path) -> SQLiteStateStore:
    return await SQLiteStateStore.open(tmp_path / "state.db")


async def _claim_and_start(
    store: SQLiteStateStore,
    *,
    task_id: str = "worker",
    attempt_id: str = "attempt-1",
    run_id: str = "run-1",
    claimed_at=GRAPH_NOW,
    budget: int = 2,
):
    attempt = await store.claim_graph_task(
        "agent-1",
        "job-1",
        task_id,
        attempt_id=attempt_id,
        claim_token=f"{attempt_id}:claim",
        run_id=run_id,
        executor_id="executor-1",
        claimed_at=claimed_at,
        lease_seconds=30,
        absolute_deadline_at=claimed_at + timedelta(minutes=2),
        budget_reservations=(BudgetAmount("work_units", budget),),
    )
    assert attempt is not None
    started = await store.start_graph_attempt(
        "agent-1",
        "job-1",
        task_id,
        attempt_id,
        claim_token=f"{attempt_id}:claim",
        fencing_epoch=attempt.fencing_epoch,
        started_at=claimed_at + timedelta(seconds=1),
    )
    assert started is not None
    return started


@pytest.mark.asyncio
async def test_concurrent_claim_has_exactly_one_winner(tmp_path: Path):
    store = await _store(tmp_path)
    await store.admit_graph(graph_admission())

    async def claim(index: int):
        return await store.claim_graph_task(
            "agent-1",
            "job-1",
            "worker",
            attempt_id=f"attempt-{index}",
            claim_token=f"claim-{index}",
            run_id=f"run-{index}",
            executor_id="executor-1",
            claimed_at=GRAPH_NOW,
            lease_seconds=30,
            absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
            budget_reservations=(BudgetAmount("work_units", 1),),
        )

    winners = await asyncio.gather(*(claim(index) for index in range(12)))
    assert sum(item is not None for item in winners) == 1
    inspection = await store.inspect_graph("agent-1", "job-1")
    assert inspection is not None
    assert len(inspection.attempts) == 1
    assert inspection.graph.active_attempt_count == 1


@pytest.mark.asyncio
async def test_stale_fence_rejects_checkpoint_and_completion(tmp_path: Path):
    store = await _store(tmp_path)
    await store.admit_graph(graph_admission())
    attempt = await _claim_and_start(store)
    fenced = await store.fence_graph_attempt(
        "agent-1",
        "job-1",
        "worker",
        "attempt-1",
        fencing_epoch=attempt.fencing_epoch,
        fenced_at=GRAPH_NOW + timedelta(seconds=40),
        requeue=True,
        reason_code="lease_expired",
    )
    assert fenced is not None

    checkpoint = TaskCheckpoint(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        fencing_epoch=attempt.fencing_epoch,
        ordinal=1,
        milestone="late",
        payload={"late": True},
        created_at=GRAPH_NOW + timedelta(seconds=41),
        payload_digest=canonical_digest({"late": True}),
    )
    with pytest.raises(GraphValidationError, match="stale"):
        await store.checkpoint_graph_attempt(checkpoint, claim_token="attempt-1:claim")
    with pytest.raises(GraphValidationError, match="stale"):
        await store.complete_graph_attempt(
            task_result(completed_at=GRAPH_NOW + timedelta(seconds=41)),
            claim_token="attempt-1:claim",
            fencing_epoch=attempt.fencing_epoch,
            usage=None,
        )

    replacement = await store.claim_graph_task(
        "agent-1",
        "job-1",
        "worker",
        attempt_id="attempt-2",
        claim_token="claim-2",
        run_id="run-2",
        executor_id="executor-1",
        claimed_at=GRAPH_NOW + timedelta(seconds=42),
        lease_seconds=30,
        absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
        budget_reservations=(BudgetAmount("work_units", 1),),
    )
    assert replacement is not None
    assert replacement.fencing_epoch > attempt.fencing_epoch


@pytest.mark.asyncio
async def test_response_loss_completion_is_idempotent_and_conserves_budget(
    tmp_path: Path,
):
    store = await _store(tmp_path)
    await store.admit_graph(graph_admission())
    attempt = await _claim_and_start(store)
    result = task_result()

    first = await store.complete_graph_attempt(
        result,
        claim_token="attempt-1:claim",
        fencing_epoch=attempt.fencing_epoch,
        usage=None,
    )
    second = await store.complete_graph_attempt(
        result,
        claim_token="attempt-1:claim",
        fencing_epoch=attempt.fencing_epoch,
        usage=None,
    )
    assert first == second == result
    ledgers = await store.list_graph_budget_ledgers("agent-1", "job-1")
    assert [(item.settled, item.reserved) for item in ledgers] == [(2, 0)]
    reservations = await store.list_graph_attempt_reservations(
        "agent-1", "job-1", "worker", "attempt-1"
    )
    assert [(item.reserved, item.settled) for item in reservations] == [(2, 2)]

    conflicting = task_result(result_id="different-result")
    with pytest.raises(GraphStoreConflictError, match="different content"):
        await store.complete_graph_attempt(
            conflicting,
            claim_token="attempt-1:claim",
            fencing_epoch=attempt.fencing_epoch,
            usage=None,
        )


@pytest.mark.asyncio
async def test_mutation_response_loss_is_idempotent_and_conflicts_are_rejected(
    tmp_path: Path,
):
    store = await _store(tmp_path)
    admission = graph_admission()
    await store.admit_graph(admission)
    added_specification = replace(
        admission.tasks[0].specification,
        budgets=(BudgetAmount("work_units", 0),),
    )
    added = replace(
        admission.tasks[0],
        task_id="worker-2",
        state=TaskState.PENDING,
        priority=5,
        specification=added_specification,
        task_spec_digest=added_specification.digest,
    )
    request = GraphMutationRequest(
        agent_id="agent-1",
        job_id="job-1",
        mutation_id="mutation-1",
        actor_kind="owner",
        actor_key="owner-1",
        idempotency_key="request-1",
        expected_revision=0,
        created_at=GRAPH_NOW + timedelta(seconds=1),
        tasks=(added,),
    )

    first = await store.apply_graph_mutation(request)
    second = await store.apply_graph_mutation(request)
    assert first == second
    assert first.decision is MutationDecision.COMMITTED
    inspection = await store.inspect_graph("agent-1", "job-1")
    assert inspection is not None
    assert inspection.graph.revision == 1
    assert inspection.graph.mutation_count == 1

    changed = replace(
        request,
        mutation_id="mutation-2",
        tasks=(replace(added, priority=6),),
    )
    with pytest.raises(GraphStoreConflictError, match="idempotency"):
        await store.apply_graph_mutation(changed)


@pytest.mark.asyncio
async def test_graph_inspection_uses_one_read_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _store(tmp_path)
    admission = graph_admission()
    await store.admit_graph(admission)
    added_specification = replace(
        admission.tasks[0].specification,
        budgets=(BudgetAmount("work_units", 0),),
    )
    added = replace(
        admission.tasks[0],
        task_id="worker-2",
        state=TaskState.PENDING,
        priority=5,
        specification=added_specification,
        task_spec_digest=added_specification.digest,
    )
    request = GraphMutationRequest(
        agent_id="agent-1",
        job_id="job-1",
        mutation_id="mutation-1",
        actor_kind="owner",
        actor_key="owner-1",
        idempotency_key="request-1",
        expected_revision=0,
        created_at=GRAPH_NOW + timedelta(seconds=1),
        tasks=(added,),
    )
    graph_loaded = threading.Event()
    continue_inspection = threading.Event()
    original_load_graph = sqlite_graph._load_graph

    def pause_after_graph_projection(connection, agent_id, job_id):
        graph = original_load_graph(connection, agent_id, job_id)
        if not graph_loaded.is_set():
            graph_loaded.set()
            if not continue_inspection.wait(timeout=2):
                raise AssertionError("inspection did not resume")
        return graph

    monkeypatch.setattr(sqlite_graph, "_load_graph", pause_after_graph_projection)
    inspection_task = asyncio.create_task(store.inspect_graph("agent-1", "job-1"))
    assert await asyncio.to_thread(graph_loaded.wait, 2)
    try:
        mutation = await store.apply_graph_mutation(request)
        assert mutation.decision is MutationDecision.COMMITTED
    finally:
        continue_inspection.set()

    inspection = await asyncio.wait_for(inspection_task, timeout=2)
    assert inspection is not None
    assert inspection.graph.revision == 0
    assert inspection.graph.task_count == len(inspection.tasks) == 2

    current = await store.inspect_graph("agent-1", "job-1")
    assert current is not None
    assert current.graph.revision == 1
    assert current.graph.task_count == len(current.tasks) == 3


@pytest.mark.parametrize("seed", range(24))
@pytest.mark.asyncio
async def test_seeded_mutations_preserve_counts_and_acyclic_topology(
    tmp_path: Path, seed: int
):
    randomizer = random.Random(seed)
    job_id = f"job-{seed}"
    store = await SQLiteStateStore.open(tmp_path / f"state-{seed}.db")
    admission = graph_admission(job_id=job_id)
    await store.admit_graph(admission)
    task_count = randomizer.randint(1, 8)
    base_specification = replace(
        admission.tasks[0].specification,
        budgets=(BudgetAmount("work_units", 0),),
    )
    tasks = tuple(
        replace(
            admission.tasks[0],
            task_id=f"generated-{index}",
            state=TaskState.PENDING,
            priority=randomizer.randint(-10, 10),
            specification=base_specification,
            task_spec_digest=base_specification.digest,
        )
        for index in range(task_count)
    )
    dependencies: list[TaskDependency] = []
    for downstream in range(task_count):
        candidates = ["worker", *(f"generated-{index}" for index in range(downstream))]
        for upstream in candidates:
            if randomizer.random() < 0.3:
                dependencies.append(
                    TaskDependency(
                        agent_id="agent-1",
                        job_id=job_id,
                        upstream_task_id=upstream,
                        downstream_task_id=f"generated-{downstream}",
                        edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                        created_at=GRAPH_NOW + timedelta(seconds=1),
                        creator_key=f"seed-{seed}",
                        mutation_id=f"mutation-{seed}",
                    )
                )
    request = GraphMutationRequest(
        agent_id="agent-1",
        job_id=job_id,
        mutation_id=f"mutation-{seed}",
        actor_kind="owner",
        actor_key="owner-1",
        idempotency_key=f"seed-{seed}",
        expected_revision=0,
        created_at=GRAPH_NOW + timedelta(seconds=1),
        tasks=tasks,
        dependencies=tuple(dependencies),
    )

    mutation = await store.apply_graph_mutation(request)
    assert mutation.decision is MutationDecision.COMMITTED
    inspection = await store.inspect_graph("agent-1", job_id)
    assert inspection is not None
    assert inspection.graph.task_count == 2 + task_count
    assert inspection.graph.edge_count == 1 + len(dependencies)
    assert inspection.graph.topology_digest == topology_digest(
        inspection.tasks, inspection.dependencies
    )


@pytest.mark.asyncio
async def test_finalizer_seal_is_the_only_terminal_success_path(tmp_path: Path):
    store = await _store(tmp_path)
    await store.admit_graph(graph_admission())
    worker_attempt = await _claim_and_start(store)
    await store.complete_graph_attempt(
        task_result(),
        claim_token="attempt-1:claim",
        fencing_epoch=worker_attempt.fencing_epoch,
        usage=(BudgetAmount("work_units", 1),),
    )
    inspection = await store.inspect_graph("agent-1", "job-1")
    assert inspection is not None
    assert inspection.job.state.value == "active"
    assert (
        next(item for item in inspection.tasks if item.task_id == "finalizer").state
        is TaskState.READY
    )

    final_attempt = await _claim_and_start(
        store,
        task_id="finalizer",
        attempt_id="final-attempt",
        run_id="final-run",
        claimed_at=GRAPH_NOW + timedelta(seconds=3),
        budget=1,
    )
    final_result = task_result(
        task_id="finalizer",
        attempt_id="final-attempt",
        run_id="final-run",
        result_id="final-result",
        completed_at=GRAPH_NOW + timedelta(seconds=5),
    )
    await store.complete_graph_attempt(
        final_result,
        claim_token="final-attempt:claim",
        fencing_epoch=final_attempt.fencing_epoch,
        usage=(BudgetAmount("work_units", 1),),
    )
    completed = await store.inspect_graph("agent-1", "job-1")
    assert completed is not None
    assert completed.job.state.value == "succeeded"
    assert completed.job.terminal_result_id == "final-result"

    with pytest.raises(GraphValidationError, match="terminal"):
        await store.apply_graph_mutation(
            GraphMutationRequest(
                agent_id="agent-1",
                job_id="job-1",
                mutation_id="too-late",
                actor_kind="owner",
                actor_key="owner-1",
                idempotency_key="too-late",
                expected_revision=0,
                created_at=GRAPH_NOW + timedelta(seconds=6),
            )
        )


@pytest.mark.asyncio
async def test_inspection_and_event_pages_are_bounded(tmp_path: Path):
    store = await _store(tmp_path)
    await store.admit_graph(graph_admission())
    ready = await store.list_ready_graph_tasks("agent-1", now=GRAPH_NOW, limit=1)
    assert [item.task_id for item in ready] == ["worker"]

    page = await store.list_graph_events("agent-1", "job-1", limit=1)
    assert len(page.events) == 1
    assert page.events[0].kind == "graph_admitted"


@pytest.mark.asyncio
async def test_checkpoint_comment_and_control_round_trip_transactionally(
    tmp_path: Path,
):
    store = await _store(tmp_path)
    await store.admit_graph(graph_admission())
    attempt = await _claim_and_start(store)
    heartbeat = await store.heartbeat_graph_attempt(
        "agent-1",
        "job-1",
        "worker",
        "attempt-1",
        claim_token="attempt-1:claim",
        fencing_epoch=attempt.fencing_epoch,
        heartbeat_at=GRAPH_NOW + timedelta(seconds=11),
    )
    assert heartbeat is not None
    checkpoint = TaskCheckpoint(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        fencing_epoch=attempt.fencing_epoch,
        ordinal=1,
        milestone="source-validated",
        payload={"revision": "sha256:" + "1" * 64},
        created_at=GRAPH_NOW + timedelta(seconds=12),
        payload_digest=canonical_digest({"revision": "sha256:" + "1" * 64}),
    )
    assert (
        await store.checkpoint_graph_attempt(checkpoint, claim_token="attempt-1:claim")
        == checkpoint
    )
    comment = TaskComment(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        comment_id="comment-1",
        author_kind="supervisor",
        author_id="supervisor-1",
        sensitivity=ModelSensitivity.RESTRICTED,
        body="The source revision was validated.",
        created_at=GRAPH_NOW + timedelta(seconds=13),
        body_digest=canonical_digest({"body": "The source revision was validated."}),
    )
    assert await store.add_graph_comment(comment) == comment
    payload = {"question": "Select the bounded source."}
    control = TaskControl(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        control_id="control-1",
        kind=ControlKind.NEEDS_INPUT,
        state=ControlState.OPEN,
        requesting_attempt_id="attempt-1",
        payload=payload,
        created_at=GRAPH_NOW + timedelta(seconds=14),
        payload_digest=canonical_digest(payload),
    )
    assert (
        await store.open_graph_control(
            control,
            claim_token="attempt-1:claim",
            fencing_epoch=attempt.fencing_epoch,
        )
        == control
    )
    resolved = await store.resolve_graph_control(
        "agent-1",
        "job-1",
        "worker",
        "control-1",
        state=ControlState.RESOLVED,
        resolved_at=GRAPH_NOW + timedelta(seconds=15),
        resolved_by_kind="user",
        resolved_by_id="user-1",
        resolution={"source_id": "source-1"},
        make_ready=True,
    )
    assert resolved is not None and resolved.state is ControlState.RESOLVED
    inspection = await store.inspect_graph("agent-1", "job-1")
    assert inspection is not None
    assert inspection.job.state.value == "active"
    assert (
        inspection.tasks[1 if inspection.tasks[0].task_id == "finalizer" else 0].state
        is TaskState.READY
    )
    assert len(inspection.controls) == 1
