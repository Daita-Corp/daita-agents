"""Coordinate admitted host work and conservative keyed resource permits."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Self
from uuid import uuid4

from ..loop.session import RunCancellationToken


class AdmissionError(RuntimeError):
    pass


class AdmissionClosedError(AdmissionError):
    pass


class AdmissionDrainTimeout(AdmissionError):
    pass


class CoordinatorState(str, Enum):
    OPEN = "open"
    DRAINING = "draining"
    CLOSED = "closed"


class AdmissionLeaseState(str, Enum):
    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"


class WorkloadClass(str, Enum):
    FOREGROUND = "foreground"
    SYSTEM = "system"
    ROUTINE = "routine"


class PermitKind(str, Enum):
    PROVIDER = "provider"
    SOURCE_RESOURCE = "source_resource"
    MCP_BINDING = "mcp_binding"
    SQLITE_PRESSURE = "sqlite_pressure"
    EFFECT = "effect"


@dataclass(frozen=True, slots=True)
class CoordinatorDiagnostics:
    state: CoordinatorState
    active_leases: int
    waiting_admissions: int
    active_permits: int
    acquisition_sequence: int


@dataclass(slots=True)
class _AdmissionWaiter:
    sequence: int
    workload_class: WorkloadClass
    identity: str
    conversation_id: str | None


class AdmissionLease:
    """Opaque once-releasable host-lifecycle capability."""

    __slots__ = (
        "_coordinator",
        "_owner_task",
        "_state",
        "absolute_deadline",
        "acquired_at",
        "acquisition_sequence",
        "cancellation",
        "conversation_id",
        "identity",
        "lease_id",
        "workload_class",
    )

    def __init__(
        self,
        coordinator: RunAdmissionCoordinator,
        *,
        lease_id: str,
        workload_class: WorkloadClass,
        identity: str,
        conversation_id: str | None,
        acquisition_sequence: int,
        acquired_at: float,
        absolute_deadline: float,
        owner_task: asyncio.Task[object] | None,
    ) -> None:
        self._coordinator = coordinator
        self._owner_task = owner_task
        self._state = AdmissionLeaseState.ACTIVE
        self.lease_id = lease_id
        self.workload_class = workload_class
        self.identity = identity
        self.conversation_id = conversation_id
        self.acquisition_sequence = acquisition_sequence
        self.acquired_at = acquired_at
        self.absolute_deadline = absolute_deadline
        self.cancellation = RunCancellationToken()

    @property
    def state(self) -> AdmissionLeaseState:
        return self._state

    @property
    def owner_task(self) -> asyncio.Task[object] | None:
        return self._owner_task

    async def release(self) -> None:
        if self._state is not AdmissionLeaseState.ACTIVE:
            raise RuntimeError("admission lease may be released exactly once")
        self._state = AdmissionLeaseState.RELEASING
        worker = asyncio.create_task(self._coordinator._release_execution(self))
        cancelled = False
        try:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancelled = True
            worker.result()
        finally:
            self._state = AdmissionLeaseState.RELEASED
        if cancelled:
            raise asyncio.CancelledError

    async def __aenter__(self) -> Self:
        if self._state is not AdmissionLeaseState.ACTIVE:
            raise RuntimeError("admission lease is not active")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        await self.release()


class PermitLease:
    """One short-lived keyed permit, independent of an execution lease."""

    __slots__ = ("_coordinator", "_released", "key", "kind")

    def __init__(
        self,
        coordinator: RunAdmissionCoordinator,
        kind: PermitKind,
        key: str,
    ) -> None:
        self._coordinator = coordinator
        self._released = False
        self.kind = kind
        self.key = key

    async def release(self) -> None:
        if self._released:
            raise RuntimeError("permit lease may be released exactly once")
        self._released = True
        worker = asyncio.create_task(
            self._coordinator._release_permit(self.kind, self.key)
        )
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        worker.result()
        if cancelled:
            raise asyncio.CancelledError

    async def __aenter__(self) -> Self:
        if self._released:
            raise RuntimeError("permit lease is released")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        await self.release()


class EffectCoordinator:
    """The one Phase-1 global effect-lane interface."""

    __slots__ = ("_coordinator",)

    KEY = "effect:v1:global"

    def __init__(self, coordinator: RunAdmissionCoordinator) -> None:
        self._coordinator = coordinator

    async def acquire(
        self,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> PermitLease:
        return await self._coordinator.effect_permit(
            self.KEY,
            deadline=deadline,
            cancellation=cancellation,
        )


class RunAdmissionCoordinator:
    """Own serial host admission, lifecycle drain, and keyed permit seams."""

    def __init__(
        self,
        *,
        execution_capacity: int = 1,
        clock: Callable[[], float] | None = None,
        id_factory: Callable[[str], str] | None = None,
        cancellation_grace_seconds: float = 5.0,
    ) -> None:
        if execution_capacity != 1:
            raise ValueError("Phase 1 execution capacity must remain one")
        if (
            not isinstance(cancellation_grace_seconds, (int, float))
            or isinstance(cancellation_grace_seconds, bool)
            or not 0 < float(cancellation_grace_seconds) <= 30
        ):
            raise ValueError("cancellation grace must be positive and at most 30")
        self._clock = clock or (lambda: asyncio.get_running_loop().time())
        self._id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid4().hex}")
        self._cancellation_grace_seconds = float(cancellation_grace_seconds)
        self._condition = asyncio.Condition()
        self._state = CoordinatorState.OPEN
        self._sequence = 0
        self._waiters: deque[_AdmissionWaiter] = deque()
        self._active: dict[str, AdmissionLease] = {}
        self._active_conversations: set[str] = set()
        self._permit_conditions: dict[tuple[PermitKind, str], asyncio.Condition] = {}
        self._permit_active: set[tuple[PermitKind, str]] = set()
        self.effect_coordinator = EffectCoordinator(self)

    @property
    def state(self) -> CoordinatorState:
        return self._state

    def diagnostics(self) -> CoordinatorDiagnostics:
        return CoordinatorDiagnostics(
            state=self._state,
            active_leases=len(self._active),
            waiting_admissions=len(self._waiters),
            active_permits=len(self._permit_active),
            acquisition_sequence=self._sequence,
        )

    async def admit_execution(
        self,
        workload_class: WorkloadClass,
        identity: str,
        *,
        conversation_id: str | None = None,
        absolute_deadline: float | None = None,
    ) -> AdmissionLease:
        if not isinstance(workload_class, WorkloadClass):
            raise TypeError("workload_class must be WorkloadClass")
        if not isinstance(identity, str) or not identity:
            raise ValueError("admitted identity must be non-empty text")
        if conversation_id is not None and (
            not isinstance(conversation_id, str) or not conversation_id
        ):
            raise ValueError("conversation_id must be non-empty text or None")
        deadline = _deadline_or_infinity(absolute_deadline)
        if deadline <= asyncio.get_running_loop().time():
            raise TimeoutError("execution admission deadline expired")
        waiter: _AdmissionWaiter | None = None
        async with self._condition:
            if self._state is not CoordinatorState.OPEN:
                raise AdmissionClosedError("host admission is not open")
            self._sequence += 1
            waiter = _AdmissionWaiter(
                self._sequence,
                workload_class,
                identity,
                conversation_id,
            )
            self._waiters.append(waiter)
            try:
                while not self._waiter_ready(waiter):
                    if self._state is not CoordinatorState.OPEN:
                        raise AdmissionClosedError("host admission started draining")
                    await _wait_condition(self._condition, deadline)
                self._waiters.popleft()
                acquired_at = self._clock()
                lease = AdmissionLease(
                    self,
                    lease_id=self._id_factory("admission"),
                    workload_class=workload_class,
                    identity=identity,
                    conversation_id=conversation_id,
                    acquisition_sequence=waiter.sequence,
                    acquired_at=acquired_at,
                    absolute_deadline=deadline,
                    owner_task=asyncio.current_task(),
                )
                self._active[lease.lease_id] = lease
                if conversation_id is not None:
                    self._active_conversations.add(conversation_id)
                return lease
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    def _waiter_ready(self, waiter: _AdmissionWaiter) -> bool:
        return (
            self._state is CoordinatorState.OPEN
            and bool(self._waiters)
            and self._waiters[0] is waiter
            and not self._active
            and (
                waiter.conversation_id is None
                or waiter.conversation_id not in self._active_conversations
            )
        )

    async def _release_execution(self, lease: AdmissionLease) -> None:
        async with self._condition:
            current = self._active.get(lease.lease_id)
            if current is not lease:
                raise RuntimeError("admission lease is not active in this coordinator")
            del self._active[lease.lease_id]
            if lease.conversation_id is not None:
                self._active_conversations.discard(lease.conversation_id)
            self._condition.notify_all()

    async def begin_draining(self) -> None:
        async with self._condition:
            if self._state is CoordinatorState.CLOSED:
                return
            self._state = CoordinatorState.DRAINING
            self._condition.notify_all()
        for condition in tuple(self._permit_conditions.values()):
            async with condition:
                condition.notify_all()

    async def drain(self, *, deadline: float) -> None:
        deadline = _required_deadline(deadline)
        await self.begin_draining()
        try:
            await self._wait_for_zero(deadline)
            return
        except TimeoutError:
            pass
        active = tuple(self._active.values())
        for lease in active:
            lease.cancellation.cancel("host_drain_timeout")
            task = lease.owner_task
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ):
                task.cancel("host_drain_timeout")
        cancellation_deadline = (
            asyncio.get_running_loop().time() + self._cancellation_grace_seconds
        )
        try:
            await self._wait_for_zero(cancellation_deadline)
        except TimeoutError as error:
            raise AdmissionDrainTimeout(
                "admitted work did not settle after cancellation"
            ) from error

    async def _wait_for_zero(self, deadline: float) -> None:
        async with self._condition:
            while self._active or self._permit_active:
                await _wait_condition(self._condition, deadline)

    async def close(self, *, deadline: float) -> None:
        if self._state is CoordinatorState.CLOSED:
            return
        await self.drain(deadline=deadline)
        async with self._condition:
            self._state = CoordinatorState.CLOSED
            self._condition.notify_all()

    async def provider_permit(
        self,
        key: str,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> PermitLease:
        return await self._acquire_permit(
            PermitKind.PROVIDER, key, deadline=deadline, cancellation=cancellation
        )

    async def source_resource_permit(
        self,
        key: str,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> PermitLease:
        return await self._acquire_permit(
            PermitKind.SOURCE_RESOURCE,
            key,
            deadline=deadline,
            cancellation=cancellation,
        )

    async def mcp_permit(
        self,
        key: str,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> PermitLease:
        return await self._acquire_permit(
            PermitKind.MCP_BINDING,
            key,
            deadline=deadline,
            cancellation=cancellation,
        )

    async def sqlite_pressure_permit(
        self,
        key: str = "sqlite:state",
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> PermitLease:
        return await self._acquire_permit(
            PermitKind.SQLITE_PRESSURE,
            key,
            deadline=deadline,
            cancellation=cancellation,
        )

    async def effect_permit(
        self,
        key: str,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> PermitLease:
        return await self._acquire_permit(
            PermitKind.EFFECT, key, deadline=deadline, cancellation=cancellation
        )

    async def _acquire_permit(
        self,
        kind: PermitKind,
        key: str,
        *,
        deadline: float | None,
        cancellation: RunCancellationToken | None,
    ) -> PermitLease:
        if (
            not isinstance(key, str)
            or not key
            or len(key.encode("utf-8")) > 2_048
            or any(character in key for character in "\r\n\x00")
        ):
            raise ValueError("permit key must be bounded non-empty text")
        permit_key = (kind, key)
        condition = self._permit_conditions.setdefault(permit_key, asyncio.Condition())
        resolved_deadline = _deadline_or_infinity(deadline)
        if resolved_deadline <= asyncio.get_running_loop().time():
            raise TimeoutError("permit deadline expired")
        async with condition:
            while permit_key in self._permit_active:
                if (
                    self._state is not CoordinatorState.OPEN
                    and not self._current_task_is_admitted()
                ):
                    raise AdmissionClosedError("host admission is not open")
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                wait_deadline = resolved_deadline
                if cancellation is not None:
                    wait_deadline = min(
                        resolved_deadline,
                        asyncio.get_running_loop().time() + 0.05,
                    )
                try:
                    await _wait_condition(condition, wait_deadline)
                except TimeoutError:
                    if cancellation is not None:
                        cancellation.raise_if_cancelled()
                    if wait_deadline < resolved_deadline:
                        continue
                    raise
            if (
                self._state is not CoordinatorState.OPEN
                and not self._current_task_is_admitted()
            ):
                raise AdmissionClosedError("host admission is not open")
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            self._permit_active.add(permit_key)
        return PermitLease(self, kind, key)

    def _current_task_is_admitted(self) -> bool:
        task = asyncio.current_task()
        return task is not None and any(
            lease.owner_task is task for lease in self._active.values()
        )

    async def _release_permit(self, kind: PermitKind, key: str) -> None:
        permit_key = (kind, key)
        condition = self._permit_conditions.get(permit_key)
        if condition is None:
            raise RuntimeError("permit does not belong to this coordinator")
        async with condition:
            if permit_key not in self._permit_active:
                raise RuntimeError("permit is not active")
            self._permit_active.remove(permit_key)
            condition.notify_all()
        async with self._condition:
            self._condition.notify_all()


def _required_deadline(value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("deadline must be finite monotonic time")
    return float(value)


def _deadline_or_infinity(value: float | None) -> float:
    return math.inf if value is None else _required_deadline(value)


async def _wait_condition(condition: asyncio.Condition, deadline: float) -> None:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError
    try:
        await asyncio.wait_for(condition.wait(), timeout=remaining)
    except TimeoutError as error:
        raise TimeoutError from error


__all__ = [
    "AdmissionClosedError",
    "AdmissionDrainTimeout",
    "AdmissionError",
    "AdmissionLease",
    "AdmissionLeaseState",
    "CoordinatorDiagnostics",
    "CoordinatorState",
    "EffectCoordinator",
    "PermitKind",
    "PermitLease",
    "RunAdmissionCoordinator",
    "WorkloadClass",
]
