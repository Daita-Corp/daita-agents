"""Stable Phase 7 timeline, board, and diagnostics projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from daita.capabilities import ToolExecution
from daita.cli import _graph_inspection_mapping
from daita.distribution.models import DeliverySubjectKind
from daita.jobs.capabilities import JOB_INSPECT_CAPABILITY_ID, JobInspectExecutor
from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    GraphEvent,
    GraphState,
    TaskControl,
    TaskState,
    canonical_digest,
)
from daita.jobs.owner import JobOwner
from daita.jobs.projections import graph_board, graph_diagnostics
from daita.storage.sqlite import SQLiteStateStore
from daita.tui.screens.jobs import (
    render_graph_inspection,
    render_job_board,
    render_job_timeline,
)
from tests.support.graph import GRAPH_NOW, graph_admission
from tests.support.static_graph_integration import DeterministicIds


async def _store_with_open_control(tmp_path: Path):
    store = await SQLiteStateStore.open(tmp_path / "operational.sqlite")
    admission = graph_admission()
    await store.admit_graph(admission)
    attempt = await store.claim_graph_task(
        "agent-1",
        "job-1",
        "worker",
        attempt_id="attempt-operational",
        claim_token="claim-operational",
        run_id="run-operational",
        executor_id="executor-1",
        claimed_at=GRAPH_NOW,
        lease_seconds=30,
        absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
        budget_reservations=(BudgetAmount("work_units", 1),),
    )
    assert attempt is not None
    started = await store.start_graph_attempt(
        "agent-1",
        "job-1",
        "worker",
        attempt.attempt_id,
        claim_token=attempt.claim_token,
        fencing_epoch=attempt.fencing_epoch,
        started_at=GRAPH_NOW + timedelta(seconds=1),
    )
    assert started is not None
    payload = {
        "message": "Choose one bounded value.",
        "response_schema": {
            "type": "object",
            "properties": {"choice": {"type": "string"}},
            "required": ["choice"],
            "additionalProperties": False,
        },
        "expires_at": admission.job.deadline_at.isoformat(),
    }
    control = TaskControl(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        control_id="control-operational",
        kind=ControlKind.NEEDS_INPUT,
        state=ControlState.OPEN,
        requesting_attempt_id=attempt.attempt_id,
        payload=payload,
        created_at=GRAPH_NOW + timedelta(seconds=2),
        payload_digest=canonical_digest(payload),
    )
    await store.open_graph_control(
        control,
        claim_token=attempt.claim_token,
        fencing_epoch=attempt.fencing_epoch,
    )
    owner = JobOwner(
        agent_id="agent-1",
        store=store,
        clock=lambda: GRAPH_NOW + timedelta(seconds=3),
        id_factory=DeterministicIds("operational"),
    )
    return store, owner, control


async def test_timeline_cursor_is_stable_bounded_and_task_filterable(
    tmp_path: Path,
) -> None:
    store, owner, control = await _store_with_open_control(tmp_path)
    try:
        first = await owner.graph_timeline("job-1", limit=2)
        assert first is not None
        assert len(first.events) == 2
        assert first.next_cursor == first.events[-1].event_id
        assert [event.event_id for event in first.events] == sorted(
            event.event_id for event in first.events
        )

        resolved = await owner.answer_graph_task_input(
            "job-1",
            "worker",
            control.control_id,
            principal_id="agent-1",
            answer={"choice": "bounded"},
            idempotency_key="timeline-answer",
        )
        assert resolved is not None

        stable_first = await owner.graph_timeline("job-1", limit=2)
        assert stable_first is not None
        assert stable_first.events == first.events
        second = await owner.graph_timeline(
            "job-1", after_event_id=first.next_cursor or 0, limit=100
        )
        assert second is not None
        assert {event.event_id for event in first.events}.isdisjoint(
            event.event_id for event in second.events
        )
        all_events = (*first.events, *second.events)
        assert [event.event_id for event in all_events] == sorted(
            event.event_id for event in all_events
        )
        task_page = await owner.graph_timeline("job-1", task_id="worker", limit=100)
        assert task_page is not None and task_page.events
        assert {event.task_id for event in task_page.events} == {"worker"}
        assert task_page.graph_state is GraphState.ACTIVE
        assert "Current state: active" in render_job_timeline(task_page)

        with pytest.raises(ValueError, match="page limit"):
            await owner.graph_timeline("job-1", limit=0)
        with pytest.raises(ValueError, match="page limit"):
            await owner.graph_timeline("job-1", limit=101)
        with pytest.raises(ValueError, match="cursor"):
            await owner.graph_timeline("job-1", after_event_id=-1)
    finally:
        await store.close()


async def test_board_uses_same_bounded_records_as_owner_and_store(
    tmp_path: Path,
) -> None:
    store, owner, _control = await _store_with_open_control(tmp_path)
    try:
        inspection = await owner.inspect_graph("job-1")
        board = await owner.graph_board("job-1")
        assert inspection is not None and board is not None
        assert board == graph_board(inspection)
        assert {task_id for column in board.columns for task_id in column.task_ids} == {
            task.task_id for task in inspection.tasks
        }
        assert len(board.dependencies) == len(inspection.dependencies)
        edge = board.dependencies[0]
        stored_edge = inspection.dependencies[0]
        assert (edge.upstream_task_id, edge.downstream_task_id, edge.edge_kind) == (
            stored_edge.upstream_task_id,
            stored_edge.downstream_task_id,
            stored_edge.edge_kind.value,
        )
        rendered = render_job_board(board)
        assert board.job_id in rendered
        assert all(task.task_id in rendered for task in inspection.tasks)
        assert "blocked: worker" in rendered
    finally:
        await store.close()


async def test_store_model_cli_and_tui_project_the_same_graph_state(
    tmp_path: Path,
) -> None:
    store, owner, _control = await _store_with_open_control(tmp_path)
    try:
        inspection = await owner.inspect_graph("job-1")
        assert inspection is not None
        expected_tasks = {task.task_id: task.state.value for task in inspection.tasks}
        expected_edges = {
            (
                edge.upstream_task_id,
                edge.downstream_task_id,
                edge.edge_kind.value,
            )
            for edge in inspection.dependencies
        }

        model_output = await JobInspectExecutor(owner).execute(
            ToolExecution(
                run_id="run-surface-equivalence",
                call_id="call-surface-equivalence",
                capability_id=JOB_INSPECT_CAPABILITY_ID,
                arguments={"job_id": "job-1"},
            )
        )
        raw_model_tasks = model_output.data["tasks"]
        raw_model_edges = model_output.data["dependencies"]
        assert isinstance(raw_model_tasks, tuple)
        assert isinstance(raw_model_edges, tuple)
        model_tasks = {
            str(item["task_id"]): str(item["state"])
            for item in raw_model_tasks
            if isinstance(item, Mapping)
        }
        model_edges = {
            (
                str(item["upstream_task_id"]),
                str(item["downstream_task_id"]),
                str(item["edge_kind"]),
            )
            for item in raw_model_edges
            if isinstance(item, Mapping)
        }

        cli = _graph_inspection_mapping(inspection)
        raw_cli_tasks = cli["tasks"]
        raw_cli_edges = cli["dependencies"]
        assert isinstance(raw_cli_tasks, list)
        assert isinstance(raw_cli_edges, list)
        cli_tasks = {
            str(item["task_id"]): str(item["state"])
            for item in raw_cli_tasks
            if isinstance(item, Mapping)
        }
        cli_edges = {
            (
                str(item["upstream_task_id"]),
                str(item["downstream_task_id"]),
                str(item["edge_kind"]),
            )
            for item in raw_cli_edges
            if isinstance(item, Mapping)
        }

        assert model_output.data["state"] == inspection.job.state.value
        assert cli["state"] == inspection.job.state.value
        assert model_output.data["graph_revision"] == inspection.graph.revision
        assert cli["topology_revision"] == inspection.graph.revision
        assert model_tasks == cli_tasks == expected_tasks
        assert model_edges == cli_edges == expected_edges
        tui = render_graph_inspection(inspection)
        assert f"State: {inspection.job.state.value}" in tui
        assert all(task_id in tui for task_id in expected_tasks)
        assert all(
            f"{upstream} -> {downstream}" in tui
            for upstream, downstream, _ in expected_edges
        )
    finally:
        await store.close()


async def test_attention_delivery_and_expired_control_replay_are_restart_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operational.sqlite"
    store, owner, control = await _store_with_open_control(tmp_path)
    try:
        attention = await store.list_deliveries("agent-1", limit=10)
        assert len(attention) == 1
        assert attention[0].subject_kind is DeliverySubjectKind.GRAPH_ATTENTION
        assert control.control_id in attention[0].subject_id
        assert await owner.graph_board("job-1") is not None
        assert await owner.graph_timeline("job-1") is not None
        assert await store.list_deliveries("agent-1", limit=10) == attention
    finally:
        await store.close()

    expired_at = graph_admission().job.deadline_at + timedelta(seconds=1)
    reopened = await SQLiteStateStore.open(path)
    expired_owner = JobOwner(
        agent_id="agent-1",
        store=reopened,
        clock=lambda: expired_at,
        id_factory=DeterministicIds("expired-control"),
    )
    try:
        expired = await expired_owner.answer_graph_task_input(
            "job-1",
            "worker",
            control.control_id,
            principal_id="agent-1",
            answer={"choice": "too-late"},
            idempotency_key="expired-answer",
        )
        assert expired is not None and expired.state is ControlState.EXPIRED
        assert expired.resolution is not None
        assert expired.resolution["requested_by_id"] == "agent-1"
    finally:
        await reopened.close()

    replay_store = await SQLiteStateStore.open(path)
    replay_owner = JobOwner(
        agent_id="agent-1",
        store=replay_store,
        clock=lambda: expired_at + timedelta(seconds=1),
        id_factory=DeterministicIds("expired-control-replay"),
    )
    try:
        replay = await replay_owner.answer_graph_task_input(
            "job-1",
            "worker",
            control.control_id,
            principal_id="agent-1",
            answer={"choice": "too-late"},
            idempotency_key="expired-answer",
        )
        assert replay == expired
        assert await replay_store.list_deliveries("agent-1", limit=10) == attention
    finally:
        await replay_store.close()


async def test_diagnostics_use_authoritative_rows_not_event_replay(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "deadlock.sqlite")
    try:
        await store.admit_graph(graph_admission())
        inspection = await store.inspect_graph("agent-1", "job-1")
        assert inspection is not None
        tasks = tuple(
            replace(task, state=TaskState.PENDING, not_before=None)
            for task in inspection.tasks
        )
        misleading = GraphEvent(
            event_id=999,
            agent_id="agent-1",
            job_id="job-1",
            kind="graph_succeeded",
            created_at=GRAPH_NOW,
            payload={"state": "succeeded"},
        )
        ledgers = tuple(
            replace(ledger, settled=ledger.ceiling, reserved=0)
            for ledger in inspection.budget_ledgers
        )
        projected = replace(
            inspection,
            job=replace(inspection.job, state=GraphState.BLOCKED),
            tasks=tasks,
            controls=(),
            budget_ledgers=ledgers,
            events=(misleading,),
        )

        diagnostics = graph_diagnostics(projected)
        assert diagnostics.graph_state is GraphState.BLOCKED
        assert diagnostics.latest_event_id == 999
        assert diagnostics.deadlocked is True
        assert diagnostics.deadlock_reason == "dependency_wait_without_runnable_parent"
        assert diagnostics.exhausted_budgets
        assert any(item["kind"] == "deadlock" for item in diagnostics.blockers)

        ready = next(task for task in tasks if task.task_id == "worker")
        budget_deadlock = graph_diagnostics(
            replace(
                projected,
                tasks=tuple(
                    (
                        replace(task, state=TaskState.READY)
                        if task.task_id == ready.task_id
                        else task
                    )
                    for task in tasks
                ),
            )
        )
        assert budget_deadlock.deadlocked is True
        assert (
            budget_deadlock.deadlock_reason == "budget_exhaustion_without_runnable_task"
        )
        assert any(
            item["scope"] == "root_lane" for item in budget_deadlock.exhausted_budgets
        )
    finally:
        await store.close()
