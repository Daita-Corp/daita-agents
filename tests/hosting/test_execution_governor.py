from __future__ import annotations

import asyncio

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
