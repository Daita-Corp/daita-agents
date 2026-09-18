from __future__ import annotations

import asyncio
import statistics

import pytest

from daita.hosting.execution_governor import (
    AdmissionClosedError,
    AdmissionDrainTimeout,
    CoordinatorState,
    PermitKind,
    RunAdmissionCoordinator,
    WorkloadClass,
)
from daita.loop.session import RunCancellationToken


def _deadline(seconds: float = 1.0) -> float:
    return asyncio.get_running_loop().time() + seconds


async def test_serial_admission_is_fifo_for_one_conversation():
    coordinator = RunAdmissionCoordinator()
    first = await coordinator.admit_execution(
        WorkloadClass.FOREGROUND,
        "run-one",
        conversation_id="conversation",
        absolute_deadline=_deadline(),
    )
    order: list[str] = []
    second_acquired = asyncio.Event()
    release_second = asyncio.Event()

    async def waiter(label: str) -> None:
        lease = await coordinator.admit_execution(
            WorkloadClass.FOREGROUND,
            label,
            conversation_id="conversation",
            absolute_deadline=_deadline(),
        )
        async with lease:
            order.append(label)
            if label == "run-two":
                second_acquired.set()
                await release_second.wait()

    second = asyncio.create_task(waiter("run-two"))
    await asyncio.sleep(0)
    third = asyncio.create_task(waiter("run-three"))
    await asyncio.sleep(0)
    assert coordinator.diagnostics().waiting_admissions == 2

    await first.release()
    await asyncio.wait_for(second_acquired.wait(), timeout=1)
    assert order == ["run-two"]
    release_second.set()
    await asyncio.gather(second, third)
    assert order == ["run-two", "run-three"]
    await coordinator.close(deadline=_deadline())


async def test_queued_admission_cancellation_does_not_leak_or_reorder():
    coordinator = RunAdmissionCoordinator()
    active = await coordinator.admit_execution(
        WorkloadClass.SYSTEM,
        "active",
        absolute_deadline=_deadline(),
    )
    queued = asyncio.create_task(
        coordinator.admit_execution(
            WorkloadClass.SYSTEM,
            "cancelled",
            absolute_deadline=_deadline(),
        )
    )
    await asyncio.sleep(0)
    assert coordinator.diagnostics().waiting_admissions == 1

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert coordinator.diagnostics().waiting_admissions == 0
    await active.release()

    replacement = await coordinator.admit_execution(
        WorkloadClass.SYSTEM,
        "replacement",
        absolute_deadline=_deadline(),
    )
    await replacement.release()
    with pytest.raises(RuntimeError, match="exactly once"):
        await replacement.release()
    await coordinator.close(deadline=_deadline())


async def test_drain_cancels_active_owner_and_rejects_new_admission():
    coordinator = RunAdmissionCoordinator(cancellation_grace_seconds=0.1)
    acquired = asyncio.Event()
    cancelled = asyncio.Event()

    async def owner() -> None:
        lease = await coordinator.admit_execution(
            WorkloadClass.ROUTINE,
            "routine",
            absolute_deadline=_deadline(),
        )
        async with lease:
            acquired.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert lease.cancellation.cancelled
                cancelled.set()
                raise

    task = asyncio.create_task(owner())
    await acquired.wait()
    await coordinator.close(deadline=_deadline(0.01))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert coordinator.state is CoordinatorState.CLOSED
    with pytest.raises(AdmissionClosedError):
        await coordinator.admit_execution(WorkloadClass.SYSTEM, "late")


async def test_drain_timeout_keeps_coordinator_draining_until_owner_settles():
    coordinator = RunAdmissionCoordinator(cancellation_grace_seconds=0.01)
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_owner() -> None:
        lease = await coordinator.admit_execution(
            WorkloadClass.SYSTEM,
            "resistant",
            absolute_deadline=_deadline(),
        )
        try:
            acquired.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
        finally:
            await lease.release()

    task = asyncio.create_task(cancellation_resistant_owner())
    await acquired.wait()
    with pytest.raises(AdmissionDrainTimeout):
        await coordinator.drain(deadline=_deadline(0.01))
    assert coordinator.state is CoordinatorState.DRAINING
    assert coordinator.diagnostics().active_leases == 1

    release.set()
    await task
    await coordinator.close(deadline=_deadline())
    assert coordinator.state is CoordinatorState.CLOSED


async def test_conservative_permit_interfaces_are_keyed_capacity_one():
    coordinator = RunAdmissionCoordinator()
    permit_methods = (
        (PermitKind.PROVIDER, coordinator.provider_permit, "provider:key"),
        (
            PermitKind.SOURCE_RESOURCE,
            coordinator.source_resource_permit,
            "source:one",
        ),
        (PermitKind.MCP_BINDING, coordinator.mcp_permit, "mcp:agent:binding"),
        (
            PermitKind.SQLITE_PRESSURE,
            coordinator.sqlite_pressure_permit,
            "sqlite:state",
        ),
        (PermitKind.EFFECT, coordinator.effect_permit, "effect:v1:global"),
    )
    for kind, acquire, key in permit_methods:
        first = await acquire(key, deadline=_deadline())
        waiting = asyncio.create_task(acquire(key, deadline=_deadline()))
        await asyncio.sleep(0)
        assert not waiting.done()
        await first.release()
        second = await waiting
        assert second.kind is kind
        assert second.key == key
        await second.release()
    await coordinator.close(deadline=_deadline())


async def test_waiting_permit_observes_session_cancellation():
    coordinator = RunAdmissionCoordinator()
    active = await coordinator.source_resource_permit(
        "source:one",
        deadline=_deadline(),
    )
    cancellation = RunCancellationToken()
    waiting = asyncio.create_task(
        coordinator.source_resource_permit(
            "source:one",
            deadline=_deadline(),
            cancellation=cancellation,
        )
    )
    await asyncio.sleep(0)
    cancellation.cancel("test_cancelled")

    with pytest.raises(asyncio.CancelledError, match="test_cancelled"):
        await asyncio.wait_for(waiting, timeout=0.2)
    await active.release()
    await coordinator.close(deadline=_deadline())


async def test_different_conversations_overlap_but_same_conversation_is_ordered():
    coordinator = RunAdmissionCoordinator(
        execution_capacity=5,
        foreground_execution_reserve=1,
        provider_capacity=2,
        foreground_provider_reserve=1,
    )
    first = await coordinator.admit_execution(
        WorkloadClass.FOREGROUND,
        "first-a",
        conversation_id="conversation-a",
    )
    acquired_b = asyncio.Event()
    acquired_second_a = asyncio.Event()
    release_b = asyncio.Event()

    async def run_b() -> None:
        lease = await coordinator.admit_execution(
            WorkloadClass.FOREGROUND,
            "first-b",
            conversation_id="conversation-b",
        )
        async with lease:
            acquired_b.set()
            await release_b.wait()

    async def run_second_a() -> None:
        lease = await coordinator.admit_execution(
            WorkloadClass.FOREGROUND,
            "second-a",
            conversation_id="conversation-a",
        )
        async with lease:
            acquired_second_a.set()

    second_a_task = asyncio.create_task(run_second_a())
    await asyncio.sleep(0)
    b_task = asyncio.create_task(run_b())
    await asyncio.wait_for(acquired_b.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not acquired_second_a.is_set()
    await first.release()
    await asyncio.wait_for(acquired_second_a.wait(), timeout=1)
    release_b.set()
    await asyncio.gather(b_task, second_a_task)
    await coordinator.close(deadline=_deadline())


async def test_background_admission_uses_exact_weighted_deficit_cycle():
    coordinator = RunAdmissionCoordinator(execution_capacity=1)
    holder = await coordinator.admit_execution(
        WorkloadClass.FOREGROUND,
        "holder",
        conversation_id="holder-conversation",
    )
    order: list[WorkloadClass] = []

    async def wait_for_slot(workload: WorkloadClass, index: int) -> None:
        lease = await coordinator.admit_execution(workload, f"{workload.value}-{index}")
        async with lease:
            order.append(workload)

    tasks = tuple(
        asyncio.create_task(wait_for_slot(workload, index))
        for workload, count in (
            (WorkloadClass.SYSTEM, 8),
            (WorkloadClass.ROUTINE, 3),
            (WorkloadClass.GRAPH, 2),
        )
        for index in range(count)
    )
    while coordinator.diagnostics().waiting_admissions != len(tasks):
        await asyncio.sleep(0)
    await holder.release()
    await asyncio.gather(*tasks)
    assert order == [
        *([WorkloadClass.SYSTEM] * 8),
        *([WorkloadClass.ROUTINE] * 3),
        *([WorkloadClass.GRAPH] * 2),
    ]
    await coordinator.close(deadline=_deadline())


async def test_background_admission_adds_bounded_thirty_second_aging_quanta():
    now = [0.0]
    coordinator = RunAdmissionCoordinator(
        execution_capacity=1,
        clock=lambda: now[0],
    )
    holder = await coordinator.admit_execution(
        WorkloadClass.FOREGROUND,
        "holder",
        conversation_id="holder-conversation",
    )
    order: list[WorkloadClass] = []

    async def wait_for_slot(workload: WorkloadClass, index: int) -> None:
        lease = await coordinator.admit_execution(workload, f"{workload.value}-{index}")
        async with lease:
            order.append(workload)

    graph_tasks = tuple(
        asyncio.create_task(wait_for_slot(WorkloadClass.GRAPH, index))
        for index in range(6)
    )
    while coordinator.diagnostics().waiting_admissions != len(graph_tasks):
        await asyncio.sleep(0)
    now[0] = 120.0
    later_tasks = tuple(
        asyncio.create_task(wait_for_slot(workload, index))
        for workload, count in (
            (WorkloadClass.SYSTEM, 16),
            (WorkloadClass.ROUTINE, 6),
        )
        for index in range(count)
    )
    all_tasks = (*graph_tasks, *later_tasks)
    while coordinator.diagnostics().waiting_admissions != len(all_tasks):
        await asyncio.sleep(0)
    await holder.release()
    await asyncio.gather(*all_tasks)
    assert order[:17] == [
        *([WorkloadClass.SYSTEM] * 8),
        *([WorkloadClass.ROUTINE] * 3),
        *([WorkloadClass.GRAPH] * 6),
    ]
    await coordinator.close(deadline=_deadline())


async def test_provider_capacity_two_keeps_one_strict_foreground_reservation():
    coordinator = RunAdmissionCoordinator(
        execution_capacity=5,
        foreground_execution_reserve=1,
        provider_capacity=2,
        foreground_provider_reserve=1,
    )
    first_background_ready = asyncio.Event()
    release_first_background = asyncio.Event()
    second_background_ready = asyncio.Event()
    foreground_ready = asyncio.Event()
    release_foreground = asyncio.Event()

    async def background(
        identity: str, ready: asyncio.Event, release: asyncio.Event
    ) -> None:
        lease = await coordinator.admit_execution(WorkloadClass.GRAPH, identity)
        async with lease:
            permit = await coordinator.provider_permit("provider:account")
            async with permit:
                ready.set()
                await release.wait()

    first = asyncio.create_task(
        background("graph-one", first_background_ready, release_first_background)
    )
    await first_background_ready.wait()
    release_second_background = asyncio.Event()
    second = asyncio.create_task(
        background("graph-two", second_background_ready, release_second_background)
    )
    await asyncio.sleep(0)
    assert not second_background_ready.is_set()

    async def foreground() -> None:
        lease = await coordinator.admit_execution(
            WorkloadClass.FOREGROUND,
            "foreground",
            conversation_id="foreground-conversation",
        )
        async with lease:
            permit = await coordinator.provider_permit("provider:account")
            async with permit:
                foreground_ready.set()
                await release_foreground.wait()

    foreground_task = asyncio.create_task(foreground())
    await asyncio.wait_for(foreground_ready.wait(), timeout=1)
    assert not second_background_ready.is_set()
    release_foreground.set()
    release_first_background.set()
    await asyncio.wait_for(second_background_ready.wait(), timeout=1)
    release_second_background.set()
    await asyncio.gather(first, second, foreground_task)
    await coordinator.close(deadline=_deadline())


async def test_source_mcp_and_effect_conflict_keys_have_frozen_capacities():
    coordinator = RunAdmissionCoordinator(
        execution_capacity=5,
        source_resource_capacities={"source:postgres": 2, "source:sqlite": 1},
        sqlite_pressure_capacity=4,
    )
    postgres = (
        await coordinator.source_resource_permit("source:postgres"),
        await coordinator.source_resource_permit("source:postgres"),
    )
    third_postgres = asyncio.create_task(
        coordinator.source_resource_permit("source:postgres", deadline=_deadline())
    )
    await asyncio.sleep(0)
    assert not third_postgres.done()
    await postgres[0].release()
    acquired = await third_postgres
    await acquired.release()
    await postgres[1].release()

    for acquire, key in (
        (coordinator.source_resource_permit, "source:sqlite"),
        (coordinator.mcp_permit, "mcp:agent:binding"),
        (coordinator.effect_permit, "effect:v1:global"),
    ):
        first = await acquire(key)
        waiting = asyncio.create_task(acquire(key, deadline=_deadline()))
        await asyncio.sleep(0)
        assert not waiting.done()
        await first.release()
        second = await waiting
        await second.release()
    await coordinator.close(deadline=_deadline())


async def test_saturated_graph_work_preserves_foreground_admission_p95():
    coordinator = RunAdmissionCoordinator(
        execution_capacity=5,
        foreground_execution_reserve=1,
        provider_capacity=2,
        foreground_provider_reserve=1,
    )
    backgrounds = tuple(
        [
            await coordinator.admit_execution(WorkloadClass.GRAPH, f"graph-{index}")
            for index in range(4)
        ]
    )
    waits: list[float] = []
    loop = asyncio.get_running_loop()
    for index in range(40):
        started = loop.time()
        foreground = await coordinator.admit_execution(
            WorkloadClass.FOREGROUND,
            f"foreground-{index}",
            conversation_id=f"conversation-{index}",
            absolute_deadline=_deadline(2.0),
        )
        waits.append(loop.time() - started)
        await foreground.release()
    p95 = statistics.quantiles(waits, n=20, method="inclusive")[18]
    assert p95 <= 2.0
    for lease in backgrounds:
        await lease.release()
    await coordinator.close(deadline=_deadline())
