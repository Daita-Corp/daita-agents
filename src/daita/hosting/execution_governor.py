"""Coordinate admitted host work and conservative keyed resource permits."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Self
from uuid import uuid4

from .._json import FrozenJsonObject
from ..loop.session import RunCancellationToken
from ..observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely


@dataclass(frozen=True, slots=True)
class AdmissionMetrics:
    """Payload-free cumulative measurements; no per-request samples are retained."""

    kind: str
    workload_class: WorkloadClass
    attempts: int
    acquired: int
    completed: int
    waiting: int
    active: int
    high_water: int
    timeouts: int
    cancellations: int
    rejected: int
    queue_seconds: float
    maximum_queue_seconds: float
    execution_seconds: float
    cleanup_seconds: float


@dataclass(slots=True)
class _AdmissionMetrics:
    attempts: int = 0
    acquired: int = 0
    completed: int = 0
    waiting: int = 0
    active: int = 0
    high_water: int = 0
    timeouts: int = 0
    cancellations: int = 0
    rejected: int = 0
    queue_seconds: float = 0.0
    maximum_queue_seconds: float = 0.0
    execution_seconds: float = 0.0
    cleanup_seconds: float = 0.0


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
    GRAPH = "graph"


_BACKGROUND_CLASSES = (
    WorkloadClass.SYSTEM,
    WorkloadClass.ROUTINE,
    WorkloadClass.GRAPH,
)
_BACKGROUND_WEIGHTS = {
    WorkloadClass.SYSTEM: 8,
    WorkloadClass.ROUTINE: 3,
    WorkloadClass.GRAPH: 2,
}
_AGING_QUANTUM_SECONDS = 30.0
_MAX_AGING_QUANTA = 4


class PermitKind(str, Enum):
    PROVIDER = "provider"
    SOURCE_RESOURCE = "source_resource"
    MCP_BINDING = "mcp_binding"
    SQLITE_PRESSURE = "sqlite_pressure"
    EFFECT = "effect"
    ANALYSIS_WORKER = "analysis_worker"
    ANALYSIS_COMPUTE = "analysis_compute"
    ANALYSIS_PARSER = "analysis_parser"


@dataclass(frozen=True, slots=True)
class CoordinatorDiagnostics:
    state: CoordinatorState
    active_leases: int
    waiting_admissions: int
    active_permits: int
    acquisition_sequence: int
    capacities: tuple[tuple[str, int], ...]
    metrics: tuple[AdmissionMetrics, ...]


@dataclass(slots=True)
class _AdmissionWaiter:
    sequence: int
    workload_class: WorkloadClass
    identity: str
    conversation_id: str | None
    enqueued_at: float


@dataclass(slots=True)
class _PermitWaiter:
    sequence: int
    workload_class: WorkloadClass
    enqueued_at: float


class AdmissionLease:
    """Opaque once-releasable host-lifecycle capability."""

    __slots__ = (
        "_coordinator",
        "_owner_task",
        "_state",
        "_context_token",
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
        self._context_token: Token[AdmissionLease | None] | None = None
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
        if self._context_token is not None:
            raise RuntimeError("admission context may be entered exactly once")
        self._context_token = self._coordinator._admitted_context.set(self)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        try:
            await self.release()
        finally:
            if self._context_token is not None:
                self._coordinator._admitted_context.reset(self._context_token)
                self._context_token = None


class PermitLease:
    """One short-lived keyed permit, independent of an execution lease."""

    __slots__ = (
        "_coordinator",
        "_released",
        "_workload_class",
        "_acquired_at",
        "key",
        "kind",
    )

    def __init__(
        self,
        coordinator: RunAdmissionCoordinator,
        kind: PermitKind,
        key: str,
        workload_class: WorkloadClass | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._released = False
        self.kind = kind
        self.key = key
        self._workload_class = workload_class
        self._acquired_at = coordinator._clock()

    async def release(self) -> None:
        if self._released:
            raise RuntimeError("permit lease may be released exactly once")
        self._released = True
        started = self._coordinator._clock()
        worker = asyncio.create_task(
            self._coordinator._release_permit(
                self.kind,
                self.key,
                self._workload_class,
                acquired_at=self._acquired_at,
                started=started,
            )
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
    """Own bounded host admission, lifecycle drain, and keyed permit seams."""

    def __init__(
        self,
        *,
        execution_capacity: int = 1,
        foreground_execution_reserve: int = 0,
        provider_capacity: int = 1,
        foreground_provider_reserve: int = 0,
        source_resource_capacities: Mapping[str, int] | None = None,
        sqlite_pressure_capacity: int = 1,
        clock: Callable[[], float] | None = None,
        id_factory: Callable[[str], str] | None = None,
        cancellation_grace_seconds: float = 5.0,
        analysis_worker_capacity: int = 1,
        observer: AgentObserver | None = None,
    ) -> None:
        if (
            not isinstance(execution_capacity, int)
            or isinstance(execution_capacity, bool)
            or not 1 <= execution_capacity <= 16
        ):
            raise ValueError("execution capacity must be between one and sixteen")
        if (
            not isinstance(foreground_execution_reserve, int)
            or isinstance(foreground_execution_reserve, bool)
            or not 0 <= foreground_execution_reserve < execution_capacity
        ):
            raise ValueError("foreground execution reserve is invalid")
        if (
            not isinstance(provider_capacity, int)
            or isinstance(provider_capacity, bool)
            or not 1 <= provider_capacity <= 8
        ):
            raise ValueError("provider capacity must be between one and eight")
        if (
            not isinstance(foreground_provider_reserve, int)
            or isinstance(foreground_provider_reserve, bool)
            or not 0 <= foreground_provider_reserve < provider_capacity
        ):
            raise ValueError("foreground provider reserve is invalid")
        if (
            not isinstance(sqlite_pressure_capacity, int)
            or isinstance(sqlite_pressure_capacity, bool)
            or not 1 <= sqlite_pressure_capacity <= 4
        ):
            raise ValueError("SQLite pressure capacity must be between one and four")
        source_capacities = dict(source_resource_capacities or {})
        for key, capacity in source_capacities.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(capacity, int)
                or isinstance(capacity, bool)
                or not 1 <= capacity <= 8
            ):
                raise ValueError(
                    "source/resource capacities must be between one and eight"
                )
        if (
            not isinstance(cancellation_grace_seconds, (int, float))
            or isinstance(cancellation_grace_seconds, bool)
            or not 0 < float(cancellation_grace_seconds) <= 30
        ):
            raise ValueError("cancellation grace must be positive and at most 30")
        self._clock = clock or (lambda: asyncio.get_running_loop().time())
        if (
            type(analysis_worker_capacity) is not int
            or not 1 <= analysis_worker_capacity <= 2
        ):
            raise ValueError("Analysis worker capacity must be one or two")
        self._analysis_worker_capacity = analysis_worker_capacity
        self._observer = observer
        self._metrics: dict[tuple[str, WorkloadClass], _AdmissionMetrics] = {}
        self._id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid4().hex}")
        self._cancellation_grace_seconds = float(cancellation_grace_seconds)
        self._execution_capacity = execution_capacity
        self._foreground_execution_reserve = foreground_execution_reserve
        self._provider_capacity = provider_capacity
        self._foreground_provider_reserve = foreground_provider_reserve
        self._source_resource_capacities = source_capacities
        self._sqlite_pressure_capacity = sqlite_pressure_capacity
        self._condition = asyncio.Condition()
        self._state = CoordinatorState.OPEN
        self._sequence = 0
        self._waiters: deque[_AdmissionWaiter] = deque()
        self._active: dict[str, AdmissionLease] = {}
        self._admitted_context: ContextVar[AdmissionLease | None] = ContextVar(
            "daita_admission", default=None
        )
        self._active_conversations: set[str] = set()
        self._execution_deficits = {workload: 0 for workload in _BACKGROUND_CLASSES}
        self._execution_cursor = 0
        self._permit_conditions: dict[tuple[PermitKind, str], asyncio.Condition] = {}
        self._permit_active: dict[tuple[PermitKind, str], int] = {}
        self._provider_background_active: dict[str, int] = {}
        self._provider_waiters: dict[str, deque[_PermitWaiter]] = {}
        self._provider_deficits: dict[str, dict[WorkloadClass, int]] = {}
        self._provider_cursors: dict[str, int] = {}
        self.effect_coordinator = EffectCoordinator(self)

    def configure_source_resource_capacity(self, key: str, capacity: int) -> None:
        """Install the adapter-owned read capacity for one exact source key."""

        if not isinstance(key, str) or not key:
            raise ValueError("source/resource capacity key must be non-empty")
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or not 1 <= capacity <= 8
        ):
            raise ValueError("source/resource capacity must be between one and eight")
        self._source_resource_capacities[key] = capacity

    @property
    def state(self) -> CoordinatorState:
        return self._state

    def current_deadline(self) -> float | None:
        """Read a live inherited scheduling deadline; it grants no authority."""
        lease = self._admitted_context.get()
        return (
            lease.absolute_deadline
            if lease is not None and self._active.get(lease.lease_id) is lease
            else None
        )

    def diagnostics(self) -> CoordinatorDiagnostics:
        return CoordinatorDiagnostics(
            state=self._state,
            active_leases=len(self._active),
            waiting_admissions=len(self._waiters),
            active_permits=sum(self._permit_active.values()),
            acquisition_sequence=self._sequence,
            capacities=(
                ("execution", self._execution_capacity),
                ("foreground_execution_reserve", self._foreground_execution_reserve),
                ("provider", self._provider_capacity),
                ("foreground_provider_reserve", self._foreground_provider_reserve),
                ("sqlite_pressure", self._sqlite_pressure_capacity),
                ("analysis_worker", self._analysis_worker_capacity),
                ("analysis_compute", 1),
                ("analysis_parser", 1),
                (
                    "source_resource_maximum",
                    max(self._source_resource_capacities.values(), default=1),
                ),
                ("mcp_binding", 1),
            ),
            metrics=tuple(
                AdmissionMetrics(
                    kind=kind,
                    workload_class=workload,
                    **asdict(metrics),
                )
                for (kind, workload), metrics in sorted(
                    self._metrics.items(),
                    key=lambda item: (item[0][0], item[0][1].value),
                )
            ),
        )

    def _record_wait(self, kind: str, workload: WorkloadClass) -> _AdmissionMetrics:
        metrics = self._metrics.setdefault((kind, workload), _AdmissionMetrics())
        metrics.attempts += 1
        metrics.waiting += 1
        return metrics

    def _record_acquisition(self, metrics: _AdmissionMetrics, started: float) -> None:
        waited = max(0.0, self._clock() - started)
        metrics.waiting -= 1
        metrics.acquired += 1
        metrics.active += 1
        metrics.high_water = max(metrics.high_water, metrics.active)
        metrics.queue_seconds += waited
        metrics.maximum_queue_seconds = max(metrics.maximum_queue_seconds, waited)

    def _record_failure(self, metrics: _AdmissionMetrics, error: BaseException) -> None:
        metrics.waiting -= 1
        if isinstance(error, TimeoutError):
            metrics.timeouts += 1
        elif isinstance(error, asyncio.CancelledError):
            metrics.cancellations += 1
        else:
            metrics.rejected += 1

    def _record_completion(
        self, kind: str, workload: WorkloadClass, acquired: float, started: float
    ) -> None:
        metrics = self._metrics[(kind, workload)]
        metrics.active -= 1
        metrics.completed += 1
        elapsed = max(0.0, started - acquired)
        cleanup = max(0.0, self._clock() - started)
        metrics.execution_seconds += elapsed
        metrics.cleanup_seconds += cleanup
        _emit_safely(
            self._observer,
            AgentEvent(
                kind=AgentEventKind.ADMISSION_UPDATED,
                occurred_at=datetime.now(UTC),
                run_id="host",
                conversation_id="host",
                run_origin="host",
                data=FrozenJsonObject.from_mapping(
                    {
                        "permit_kind": kind,
                        "workload_class": workload.value,
                        "active": metrics.active,
                        "high_water": metrics.high_water,
                        "queue_seconds": metrics.queue_seconds,
                        "execution_seconds": elapsed,
                        "cleanup_seconds": cleanup,
                    }
                ),
            ),
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
                self._clock(),
            )
            self._waiters.append(waiter)
            metrics = self._record_wait("execution", workload_class)
            try:
                while not self._waiter_ready(waiter):
                    if self._state is not CoordinatorState.OPEN:
                        raise AdmissionClosedError("host admission started draining")
                    await _wait_condition(self._condition, deadline)
                self._waiters.remove(waiter)
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
                if workload_class is not WorkloadClass.FOREGROUND:
                    self._execution_cursor = self._record_weighted_selection(
                        waiter,
                        self._eligible_execution_waiters(),
                        self._execution_deficits,
                        self._execution_cursor,
                    )
                self._condition.notify_all()
                self._record_acquisition(metrics, waiter.enqueued_at)
                return lease
            except BaseException as error:
                self._record_failure(metrics, error)
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    def _waiter_ready(self, waiter: _AdmissionWaiter) -> bool:
        eligible = self._eligible_execution_waiters()
        foreground = tuple(
            candidate
            for candidate in eligible
            if candidate.workload_class is WorkloadClass.FOREGROUND
        )
        selected = (
            foreground[0]
            if foreground
            else self._select_weighted_waiter(
                eligible,
                self._execution_cursor,
            )
        )
        background_active = sum(
            lease.workload_class is not WorkloadClass.FOREGROUND
            for lease in self._active.values()
        )
        return (
            self._state is CoordinatorState.OPEN
            and bool(self._waiters)
            and selected is waiter
            and len(self._active) < self._execution_capacity
            and (
                waiter.workload_class is WorkloadClass.FOREGROUND
                or background_active
                < self._execution_capacity - self._foreground_execution_reserve
            )
            and (
                waiter.conversation_id is None
                or waiter.conversation_id not in self._active_conversations
            )
        )

    def _eligible_execution_waiters(self) -> tuple[_AdmissionWaiter, ...]:
        """Return FIFO conversation heads that are not currently executing."""

        seen_conversations: set[str] = set()
        eligible: list[_AdmissionWaiter] = []
        for candidate in self._waiters:
            conversation_id = candidate.conversation_id
            if conversation_id is None:
                eligible.append(candidate)
                continue
            if conversation_id in seen_conversations:
                continue
            seen_conversations.add(conversation_id)
            if conversation_id not in self._active_conversations:
                eligible.append(candidate)
        return tuple(eligible)

    @staticmethod
    def _select_weighted_waiter(
        waiters: tuple[_AdmissionWaiter, ...] | tuple[_PermitWaiter, ...],
        cursor: int,
    ) -> _AdmissionWaiter | _PermitWaiter | None:
        for offset in range(len(_BACKGROUND_CLASSES)):
            workload = _BACKGROUND_CLASSES[(cursor + offset) % len(_BACKGROUND_CLASSES)]
            selected = next(
                (
                    candidate
                    for candidate in waiters
                    if candidate.workload_class is workload
                ),
                None,
            )
            if selected is not None:
                return selected
        return None

    def _record_weighted_selection(
        self,
        selected: _AdmissionWaiter | _PermitWaiter,
        remaining: tuple[_AdmissionWaiter, ...] | tuple[_PermitWaiter, ...],
        deficits: dict[WorkloadClass, int],
        cursor: int,
    ) -> int:
        """Apply one unit-cost weighted-deficit admission with bounded aging."""

        workload = selected.workload_class
        if workload is WorkloadClass.FOREGROUND:
            return cursor
        index = _BACKGROUND_CLASSES.index(workload)
        deficit = deficits[workload]
        if deficit < 1:
            waited = max(0.0, self._clock() - selected.enqueued_at)
            aging = min(
                _MAX_AGING_QUANTA,
                int(waited // _AGING_QUANTUM_SECONDS),
            )
            deficit += _BACKGROUND_WEIGHTS[workload] + aging
        deficit -= 1
        same_class_ready = any(
            candidate.workload_class is workload for candidate in remaining
        )
        deficits[workload] = deficit if same_class_ready else 0
        if same_class_ready and deficit >= 1:
            return index
        return (index + 1) % len(_BACKGROUND_CLASSES)

    async def _release_execution(self, lease: AdmissionLease) -> None:
        started = self._clock()
        async with self._condition:
            current = self._active.get(lease.lease_id)
            if current is not lease:
                raise RuntimeError("admission lease is not active in this coordinator")
            del self._active[lease.lease_id]
            if lease.conversation_id is not None:
                self._active_conversations.discard(lease.conversation_id)
            self._condition.notify_all()
        self._record_completion(
            "execution", lease.workload_class, lease.acquired_at, started
        )

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

    async def analysis_worker_permit(
        self, *, deadline: float, cancellation: RunCancellationToken
    ) -> PermitLease:
        """Reserve one bounded interpreter and its memory/scratch for its lifetime."""
        return await self._acquire_permit(
            PermitKind.ANALYSIS_WORKER,
            "analysis:worker",
            deadline=deadline,
            cancellation=cancellation,
        )

    async def analysis_compute_permit(
        self, *, deadline: float, cancellation: RunCancellationToken
    ) -> PermitLease:
        return await self._acquire_permit(
            PermitKind.ANALYSIS_COMPUTE,
            "analysis:cpu",
            deadline=deadline,
            cancellation=cancellation,
        )

    async def analysis_parser_permit(
        self, *, deadline: float, cancellation: RunCancellationToken
    ) -> PermitLease:
        """Reserve the independent bounded parser lane while interpreters remain suspended."""
        return await self._acquire_permit(
            PermitKind.ANALYSIS_PARSER,
            "analysis:parser",
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
        capacity = self._permit_capacity(kind, key)
        resolved_deadline = _deadline_or_infinity(deadline)
        if resolved_deadline <= asyncio.get_running_loop().time():
            raise TimeoutError("permit deadline expired")
        workload = self._current_task_workload_class()
        effective_workload = workload or WorkloadClass.SYSTEM
        waiter: _PermitWaiter | None = None
        async with condition:
            started = self._clock()
            metrics = self._record_wait(kind.value, effective_workload)
            if kind is PermitKind.PROVIDER:
                self._sequence += 1
                waiter = _PermitWaiter(
                    self._sequence,
                    effective_workload,
                    self._clock(),
                )
                self._provider_waiters.setdefault(key, deque()).append(waiter)
            try:
                while not self._permit_ready(
                    kind,
                    key,
                    capacity,
                    waiter,
                    effective_workload,
                ):
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
                if waiter is not None:
                    self._provider_waiters[key].remove(waiter)
                    if effective_workload is not WorkloadClass.FOREGROUND:
                        deficits = self._provider_deficits.setdefault(
                            key,
                            {workload: 0 for workload in _BACKGROUND_CLASSES},
                        )
                        cursor = self._provider_cursors.get(key, 0)
                        self._provider_cursors[key] = self._record_weighted_selection(
                            waiter,
                            tuple(self._provider_waiters[key]),
                            deficits,
                            cursor,
                        )
                self._permit_active[permit_key] = (
                    self._permit_active.get(permit_key, 0) + 1
                )
                if (
                    kind is PermitKind.PROVIDER
                    and effective_workload is not WorkloadClass.FOREGROUND
                ):
                    self._provider_background_active[key] = (
                        self._provider_background_active.get(key, 0) + 1
                    )
                self._record_acquisition(metrics, started)
            except BaseException as error:
                self._record_failure(metrics, error)
                if waiter is not None and waiter in self._provider_waiters.get(key, ()):
                    self._provider_waiters[key].remove(waiter)
                    condition.notify_all()
                raise
        return PermitLease(self, kind, key, workload)

    def _permit_ready(
        self,
        kind: PermitKind,
        key: str,
        capacity: int,
        waiter: _PermitWaiter | None,
        workload: WorkloadClass,
    ) -> bool:
        active = self._permit_active.get((kind, key), 0)
        if active >= capacity:
            return False
        if kind is not PermitKind.PROVIDER:
            return True
        waiters = self._provider_waiters.get(key, ())
        foreground = tuple(
            item for item in waiters if item.workload_class is WorkloadClass.FOREGROUND
        )
        selected = (
            foreground[0]
            if foreground
            else self._select_weighted_waiter(
                tuple(waiters),
                self._provider_cursors.get(key, 0),
            )
        )
        if selected is not waiter:
            return False
        if workload is WorkloadClass.FOREGROUND:
            return True
        background_capacity = (
            self._provider_capacity - self._foreground_provider_reserve
        )
        return self._provider_background_active.get(key, 0) < background_capacity

    def _permit_capacity(self, kind: PermitKind, key: str) -> int:
        if kind is PermitKind.ANALYSIS_WORKER:
            return self._analysis_worker_capacity
        if kind is PermitKind.SOURCE_RESOURCE:
            return self._source_resource_capacities.get(key, 1)
        if kind is PermitKind.SQLITE_PRESSURE:
            return self._sqlite_pressure_capacity
        if kind is PermitKind.PROVIDER:
            return self._provider_capacity
        return 1

    def _current_task_workload_class(self) -> WorkloadClass | None:
        inherited = self._admitted_context.get()
        if inherited is not None and self._active.get(inherited.lease_id) is inherited:
            return inherited.workload_class
        task = asyncio.current_task()
        if task is None:
            return None
        return next(
            (
                lease.workload_class
                for lease in self._active.values()
                if lease.owner_task is task
            ),
            None,
        )

    def _current_task_is_admitted(self) -> bool:
        inherited = self._admitted_context.get()
        if inherited is not None and self._active.get(inherited.lease_id) is inherited:
            return True
        task = asyncio.current_task()
        return task is not None and any(
            lease.owner_task is task for lease in self._active.values()
        )

    async def _release_permit(
        self,
        kind: PermitKind,
        key: str,
        workload_class: WorkloadClass | None,
        *,
        acquired_at: float,
        started: float,
    ) -> None:
        permit_key = (kind, key)
        condition = self._permit_conditions.get(permit_key)
        if condition is None:
            raise RuntimeError("permit does not belong to this coordinator")
        async with condition:
            active = self._permit_active.get(permit_key, 0)
            if active < 1:
                raise RuntimeError("permit is not active")
            if active == 1:
                del self._permit_active[permit_key]
            else:
                self._permit_active[permit_key] = active - 1
            if (
                kind is PermitKind.PROVIDER
                and workload_class is not WorkloadClass.FOREGROUND
            ):
                background = self._provider_background_active.get(key, 0)
                if background < 1:
                    raise RuntimeError("provider background permit is not active")
                if background == 1:
                    del self._provider_background_active[key]
                else:
                    self._provider_background_active[key] = background - 1
            condition.notify_all()
        # Settle metrics before drain coordination can yield to a new holder.
        self._record_completion(
            kind.value, workload_class or WorkloadClass.SYSTEM, acquired_at, started
        )
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
