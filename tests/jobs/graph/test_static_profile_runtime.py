from __future__ import annotations

import asyncio
import contextvars
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from daita.cli import _graph_inspection_mapping
from daita.hosting.execution_governor import PermitKind
from daita.jobs.graph.models import (
    AttemptState,
    GraphState,
    TaskRole,
    TaskState,
)
from daita.tui.screens.jobs import render_graph_inspection
from tests.support.static_graph_integration import (
    AGENT_ID,
    ObservedReadBackend,
    StaticGraphIntegration,
    StaticProfileCatalog,
)

pytestmark = pytest.mark.integration


async def test_static_profile_graph_has_bounded_work_and_one_finalizer(
    tmp_path: Path,
) -> None:
    resources = (("source-a", "resource-a"),)
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    try:
        admission = await integration.build(("resource-a",))
        assert len(admission.tasks) == 2
        work = tuple(item for item in admission.tasks if item.role is TaskRole.INTERNAL)
        finalizers = tuple(
            item for item in admission.tasks if item.role is TaskRole.FINALIZER
        )
        assert len(work) == len(finalizers) == 1
        assert work[0].state is TaskState.READY
        assert finalizers[0].state is TaskState.PENDING
        assert admission.job.finalizer_task_id == finalizers[0].task_id
        assert len(admission.dependencies) == 1
        assert admission.dependencies[0].upstream_task_id == work[0].task_id
        assert admission.dependencies[0].downstream_task_id == finalizers[0].task_id
        assert admission.job.specification.budgets[0].ceiling == 6
        assert admission.job.specification.budgets[0].control_reserved == 3
    finally:
        await integration.close()


async def test_multi_resource_static_profile_graph_completes_once(
    tmp_path: Path,
) -> None:
    resources = (
        ("source-a", "resource-a"),
        ("source-b", "resource-b"),
        ("source-b", "resource-c"),
    )
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    try:
        admission = await integration.build(tuple(item[1] for item in resources))
        assert len(admission.tasks) == 4
        assert len(admission.dependencies) == 3
        assert sum(item.role is TaskRole.FINALIZER for item in admission.tasks) == 1
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        assert terminal.job.state is GraphState.SUCCEEDED
        assert all(item.state is TaskState.SUCCEEDED for item in terminal.tasks)
        assert len(terminal.results) == 4
        assert len(await integration.artifacts.list_refs()) == 4
        deliveries = await integration.store.list_draft_graph_deliveries(
            AGENT_ID,
            job_id=admission.job.job_id,
        )
        assert len(deliveries) == 1
        assert deliveries[0].outcome.conclusion_id == terminal.job.terminal_result_id
        ledgers = await integration.store.list_graph_budget_ledgers(
            AGENT_ID, admission.job.job_id
        )
        assert tuple(
            (item.ceiling, item.settled, item.reserved) for item in ledgers
        ) == (
            (12, 4, 0),
        )
    finally:
        await integration.close()


async def test_integration_start_data_profile_emits_only_graph_work(
    tmp_path: Path,
) -> None:
    resources = (("source-a", "resource-a"),)
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    try:
        job_id = await integration.start_profile(("resource-a",))
        terminal = await integration.wait_terminal(job_id)
        assert terminal.job.state is GraphState.SUCCEEDED
        assert integration.supervisor._driver is None
        assert integration.supervisor._workers == {}
        assert len(terminal.tasks) == 2
        assert all(
            item.specification.created_by == "job_owner" for item in terminal.tasks
        )
        assert len(terminal.delivery_ids) == 1
    finally:
        await integration.close()


async def test_finalizer_barrier_rejects_early_claim(tmp_path: Path) -> None:
    resources = (
        ("source-a", "resource-a"),
        ("source-b", "resource-b"),
    )
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    try:
        admission = await integration.build(("resource-a", "resource-b"))
        await integration.owner.admit_static_graph(admission)
        claimed = await integration.store.claim_graph_task(
            AGENT_ID,
            admission.job.job_id,
            admission.job.finalizer_task_id,
            attempt_id="attempt-early-finalizer",
            claim_token="claim-early-finalizer",
            run_id="run-" + "e" * 32,
            executor_id="jobs.data_profile.finalize",
            claimed_at=integration.clock(),
            lease_seconds=30,
            absolute_deadline_at=integration.clock() + timedelta(seconds=60),
            budget_reservations=next(
                item for item in admission.tasks if item.role is TaskRole.FINALIZER
            ).specification.budgets,
        )
        assert claimed is None
        inspection = await integration.owner.inspect_graph(admission.job.job_id)
        assert inspection is not None
        finalizer = next(
            item for item in inspection.tasks if item.role is TaskRole.FINALIZER
        )
        assert finalizer.state is TaskState.PENDING
    finally:
        await integration.close()


async def test_expired_graph_fails_without_launching_work(tmp_path: Path) -> None:
    resources = (("source-a", "resource-a"),)
    current = [datetime(2026, 9, 17, 12, tzinfo=UTC)]
    integration = await StaticGraphIntegration.open(
        tmp_path,
        resources,
        clock=lambda: current[0],
    )
    try:
        admission = await integration.build(("resource-a",))
        await integration.owner.admit_static_graph(admission)
        current[0] += timedelta(seconds=301)
        await integration.supervisor.start_graph_integration()
        terminal = await integration.wait_terminal(admission.job.job_id)
        assert terminal.job.state is GraphState.FAILED
        assert terminal.job.failure_code == "deadline_exceeded"
        assert all(item.state is TaskState.SKIPPED for item in terminal.tasks)
        assert terminal.attempts == ()
        assert terminal.results == ()
        assert terminal.delivery_ids == ()
        assert integration.backend.calls == Counter()
    finally:
        await integration.close()


async def test_restart_never_reruns_successful_work(tmp_path: Path) -> None:
    resources = (
        ("source-a", "resource-a"),
        ("source-b", "resource-b"),
    )
    catalog = StaticProfileCatalog(resources)
    backend = ObservedReadBackend(catalog)
    backend.block_after = 1
    first = await StaticGraphIntegration.open(
        tmp_path,
        resources,
        namespace="before-restart",
        backend=backend,
    )
    admission = await first.build(("resource-a", "resource-b"))
    await first.admit_and_start(admission)
    await asyncio.wait_for(backend.blocked.wait(), timeout=3)
    successful_before = Counter(backend.completed)
    assert sum(successful_before.values()) == 1
    await first.close()

    backend.release.set()
    second = await StaticGraphIntegration.open(
        tmp_path,
        resources,
        initialize=False,
        namespace="after-restart",
        backend=backend,
    )
    try:
        await second.supervisor.start_graph_integration()
        assert second.supervisor._graph_driver is not None
        await asyncio.sleep(1.2)
        assert second.supervisor._graph_driver is not None
        assert (
            not second.supervisor._graph_driver.done()
        ), second.supervisor._graph_driver.exception()
        terminal = await second.wait_terminal(admission.job.job_id, timeout=8)
        assert terminal.job.state is GraphState.SUCCEEDED
        for resource_id, completed in successful_before.items():
            assert completed == 1
            assert backend.completed[resource_id] == 1
        assert all(item.state is TaskState.SUCCEEDED for item in terminal.tasks)
        assert any(item.state is AttemptState.FENCED for item in terminal.attempts)
    finally:
        await second.close()


async def test_independent_reads_obey_coordinator_capacities(tmp_path: Path) -> None:
    resources = (
        ("source-a", "resource-a1"),
        ("source-a", "resource-a2"),
        ("source-b", "resource-b1"),
        ("source-b", "resource-b2"),
    )
    catalog = StaticProfileCatalog(resources)
    backend = ObservedReadBackend(catalog, delay=0.2)
    integration = await StaticGraphIntegration.open(
        tmp_path,
        resources,
        graph_parallelism=3,
        execution_capacity=3,
        source_capacities={"source:source-a": 1, "source:source-b": 2},
        backend=backend,
    )
    try:
        admission = await integration.build(tuple(item[1] for item in resources))
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        assert terminal.job.state is GraphState.SUCCEEDED
        assert backend.maximum_by_source["source-a"] == 1
        assert backend.maximum_by_source["source-b"] <= 2
        assert backend.maximum_global >= 2
        for _ in range(100):
            if integration.coordinator.diagnostics().active_permits == 0:
                break
            await asyncio.sleep(0.001)
        assert integration.coordinator.diagnostics().active_permits == 0
    finally:
        await integration.close()


async def test_graph_store_calls_do_not_span_capability_or_source_io(
    tmp_path: Path,
) -> None:
    resources = (("source-a", "resource-a"),)
    catalog = StaticProfileCatalog(resources)
    backend = ObservedReadBackend(catalog)
    integration = await StaticGraphIntegration.open(
        tmp_path,
        resources,
        backend=backend,
    )
    store_call = contextvars.ContextVar("graph_store_call", default=False)
    original = integration.supervisor._graph_store_call

    async def observed(operation):
        token = store_call.set(True)
        try:
            return await original(operation)
        finally:
            store_call.reset(token)

    integration.supervisor._graph_store_call = observed  # type: ignore[method-assign]
    original_execute = integration.runtime.execute_internal

    async def observed_execute(request):
        assert store_call.get() is False
        value = await original_execute(request)
        assert store_call.get() is False
        return value

    integration.runtime.execute_internal = observed_execute  # type: ignore[method-assign]
    backend.assert_store_idle = lambda: (
        (_ for _ in ()).throw(AssertionError("source I/O overlapped graph store work"))
        if store_call.get()
        else None
    )
    original_commit = integration.artifacts.commit

    async def observed_commit(*args, **kwargs):
        assert store_call.get() is False
        return await original_commit(*args, **kwargs)

    integration.artifacts.commit = observed_commit  # type: ignore[method-assign]
    try:
        admission = await integration.build(("resource-a",))
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        assert terminal.job.state is GraphState.SUCCEEDED
        for _ in range(100):
            if integration.coordinator.diagnostics().active_permits == 0:
                break
            await asyncio.sleep(0.001)
        assert integration.coordinator.diagnostics().active_permits == 0
        assert not any(
            key[0] is PermitKind.SQLITE_PRESSURE
            for key in integration.coordinator._permit_active
        )
    finally:
        await integration.close()


async def test_claim_artifact_result_finalization_and_delivery_are_response_loss_safe(
    tmp_path: Path,
) -> None:
    resources = (("source-a", "resource-a"),)
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    original_claim = integration.store.claim_graph_task
    original_commit = integration.artifacts.commit
    original_complete = integration.store.complete_graph_attempt
    original_finalize = integration.store.finalize_graph_attempt
    losses = {
        "claim": False,
        "artifact": False,
        "result": False,
        "finalize": False,
    }

    async def lose_claim(*args, **kwargs):
        value = await original_claim(*args, **kwargs)
        if not losses["claim"] and value is not None:
            losses["claim"] = True
            raise ConnectionError("claim response lost")
        return value

    async def lose_artifact(*args, **kwargs):
        value = await original_commit(*args, **kwargs)
        if not losses["artifact"]:
            losses["artifact"] = True
            raise ConnectionError("artifact response lost")
        return value

    async def lose_result(*args, **kwargs):
        value = await original_complete(*args, **kwargs)
        if not losses["result"]:
            losses["result"] = True
            raise ConnectionError("result response lost")
        return value

    async def lose_finalization(*args, **kwargs):
        value = await original_finalize(*args, **kwargs)
        if not losses["finalize"]:
            losses["finalize"] = True
            raise ConnectionError("finalization response lost")
        return value

    integration.store.claim_graph_task = lose_claim  # type: ignore[method-assign]
    integration.artifacts.commit = lose_artifact  # type: ignore[method-assign]
    integration.store.complete_graph_attempt = lose_result  # type: ignore[method-assign]
    integration.store.finalize_graph_attempt = lose_finalization  # type: ignore[method-assign]
    try:
        admission = await integration.build(("resource-a",))
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        assert terminal.job.state is GraphState.SUCCEEDED
        assert all(losses.values())
        assert len(terminal.attempts) == 2
        assert all(item.state is AttemptState.SUCCEEDED for item in terminal.attempts)
        assert len(await integration.artifacts.list_refs()) == 2
        assert (
            len(
                await integration.store.list_draft_graph_deliveries(
                    AGENT_ID, job_id=admission.job.job_id
                )
            )
            == 1
        )
        ledgers = await integration.store.list_graph_budget_ledgers(
            AGENT_ID, admission.job.job_id
        )
        assert tuple((item.settled, item.reserved) for item in ledgers) == ((2, 0),)
    finally:
        await integration.close()


async def test_retryable_internal_failure_is_charged_and_requeued(
    tmp_path: Path,
) -> None:
    resources = (("source-a", "resource-a"),)
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    original = integration.runtime.execute_internal
    failed = False

    async def fail_once(request):
        nonlocal failed
        if not failed:
            failed = True
            raise ConnectionError("injected capability failure")
        return await original(request)

    integration.runtime.execute_internal = fail_once  # type: ignore[method-assign]
    try:
        admission = await integration.build(("resource-a",))
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id, timeout=8)
        assert terminal.job.state is GraphState.SUCCEEDED
        worker_attempts = tuple(
            item
            for item in terminal.attempts
            if item.task_id != admission.job.finalizer_task_id
        )
        assert tuple(item.state for item in worker_attempts) == (
            AttemptState.FAILED,
            AttemptState.SUCCEEDED,
        )
        assert all(item.checkpoint_ids for item in terminal.attempts)
        ledgers = await integration.store.list_graph_budget_ledgers(
            AGENT_ID, admission.job.job_id
        )
        assert tuple((item.settled, item.reserved) for item in ledgers) == ((3, 0),)
    finally:
        await integration.close()


async def test_contract_failure_is_charged_and_fails_without_retry(
    tmp_path: Path,
) -> None:
    resources = (("source-a", "resource-a"),)
    integration = await StaticGraphIntegration.open(tmp_path, resources)

    async def reject_contract(request):
        del request
        raise ValueError("injected immutable contract failure")

    integration.runtime.execute_internal = reject_contract  # type: ignore[method-assign]
    try:
        admission = await integration.build(("resource-a",))
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        assert terminal.job.state is GraphState.FAILED
        assert terminal.job.failure_code == "graph_internal_execution_failed"
        assert len(terminal.attempts) == 1
        assert terminal.attempts[0].state is AttemptState.FAILED
        assert terminal.attempts[0].measured_usage[0].amount == 1
        assert await integration.artifacts.list_refs() == ()
        assert terminal.delivery_ids == ()
        ledgers = await integration.store.list_graph_budget_ledgers(
            AGENT_ID, admission.job.job_id
        )
        assert tuple((item.settled, item.reserved) for item in ledgers) == ((1, 0),)
    finally:
        await integration.close()


async def test_stale_finalizer_cannot_settle_or_publish(tmp_path: Path) -> None:
    resources = (("source-a", "resource-a"),)
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    original = integration.store.finalize_graph_attempt
    entered = asyncio.Event()
    release = asyncio.Event()
    intercepted = False

    async def gate_finalization(*args, **kwargs):
        nonlocal intercepted
        if not intercepted:
            intercepted = True
            entered.set()
            await release.wait()
        return await original(*args, **kwargs)

    integration.store.finalize_graph_attempt = gate_finalization  # type: ignore[method-assign]
    try:
        admission = await integration.build(("resource-a",))
        await integration.admit_and_start(admission)
        await asyncio.wait_for(entered.wait(), timeout=3)
        inspection = await integration.owner.inspect_graph(admission.job.job_id)
        assert inspection is not None
        attempt = next(
            item
            for item in inspection.attempts
            if item.task_id == admission.job.finalizer_task_id
            and item.state is AttemptState.RUNNING
        )
        fenced = await integration.store.fence_graph_attempt(
            AGENT_ID,
            admission.job.job_id,
            admission.job.finalizer_task_id,
            attempt.attempt_id,
            fencing_epoch=attempt.fencing_epoch,
            fenced_at=integration.clock(),
            requeue=True,
            reason_code="injected_stale_finalizer",
        )
        assert fenced is not None and fenced.state is AttemptState.FENCED
        release.set()
        await asyncio.sleep(0.05)
        assert (
            await integration.store.list_draft_graph_deliveries(
                AGENT_ID, job_id=admission.job.job_id
            )
            == ()
        )
        current = await integration.owner.inspect_graph(admission.job.job_id)
        assert current is not None
        assert current.job.state is not GraphState.SUCCEEDED
        assert current.job.terminal_result_id is None
        assert (
            next(
                item
                for item in current.tasks
                if item.task_id == admission.job.finalizer_task_id
            ).state
            is TaskState.READY
        )
    finally:
        release.set()
        await integration.close()


async def test_heartbeat_and_checkpoint_are_durable(tmp_path: Path) -> None:
    resources = (("source-a", "resource-a"),)
    catalog = StaticProfileCatalog(resources)
    backend = ObservedReadBackend(catalog, delay=0.2)
    integration = await StaticGraphIntegration.open(
        tmp_path,
        resources,
        backend=backend,
    )
    integration.supervisor._graph_heartbeat_seconds = 0.01
    original_heartbeat = integration.store.heartbeat_graph_attempt
    heartbeat_ticks = 0

    async def accelerated_heartbeat(*args, **kwargs):
        nonlocal heartbeat_ticks
        heartbeat_ticks += 1
        kwargs["heartbeat_at"] += timedelta(seconds=10 * heartbeat_ticks)
        return await original_heartbeat(*args, **kwargs)

    integration.store.heartbeat_graph_attempt = accelerated_heartbeat  # type: ignore[method-assign]
    try:
        admission = await integration.build(("resource-a",))
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        worker = next(
            item
            for item in terminal.attempts
            if item.task_id != admission.job.finalizer_task_id
        )
        assert worker.checkpoint_ids == (f"{worker.attempt_id}:started",)
        assert worker.started_at is not None
        assert worker.heartbeat_at is not None
        assert worker.heartbeat_at > worker.started_at
        assert heartbeat_ticks >= 1
    finally:
        await integration.close()


async def test_cli_and_tui_graph_inspection_projections_are_bounded_read_only(
    tmp_path: Path,
) -> None:
    resources = (("source-a", "resource-a"),)
    integration = await StaticGraphIntegration.open(tmp_path, resources)
    try:
        admission = await integration.build(("resource-a",))
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        projected = _graph_inspection_mapping(terminal)
        assert projected["state"] == "succeeded"
        assert projected["task_count"] == 2
        assert len(cast(list[object], projected["tasks"])) == 2
        assert projected["delivery_ids"]
        assert len(cast(list[object], projected["events"])) <= 100
        rendered = render_graph_inspection(terminal)
        assert "2 tasks" in rendered
        assert "Accepted results: 2" in rendered
        assert "Deliveries: 1" in rendered
        assert "retry" not in rendered.casefold()
        assert "cancel" not in rendered.casefold()
    finally:
        await integration.close()
