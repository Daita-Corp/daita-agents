from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from daita.jobs.graph.models import (
    BudgetAmount,
    GraphMutationRequest,
    MutationDecision,
    TaskState,
)
from daita.storage import sql_graph as sqlite_graph
from daita.storage.sqlite import SQLiteStateStore
from tests.support.graph import GRAPH_NOW, graph_admission

pytestmark = pytest.mark.integration


async def test_graph_inspection_uses_one_read_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
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
