"""Explicit foreground host for one persistent local agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import math
from pathlib import Path
from typing import Self, TypeVar
from uuid import uuid4

from .._json import canonical_json
from ..adapters.models import SourceRegistration
from ..adapters.protocols import ResourceSource
from ..capabilities import CapabilityRegistry
from ..events.models import CommittedEvent, EventCursor
from ..llm.models import ModelProfile
from ..llm.protocols import ModelProvider
from ..loop.driver import ContextBuilder, DomainController
from ..loop.models import LoopBudgets, LoopExit
from ..monitors.scheduler import (
    MonitorOutcomeProjection,
    MonitorOutcomeProjector,
    MonitorScheduler,
    MonitorSchedulerResult,
)
from ..monitors.models import (
    Monitor,
    MonitorConfirmation,
    MonitorDefinition,
    MonitorInspection,
    MonitorProposal,
    MonitorStatus,
)
from ..operations.checkpoints import OperationSnapshot
from ..operations.governance import (
    ApprovalRequest,
    ApprovalStatus,
    DefaultPolicyEvaluator,
)
from ..operations.models import AgentTrigger, TriggerKind
from .embedded import EmbeddedAgent
from .inbox import (
    HostInboxItem,
    HostInboxKind,
    HostInboxStatus,
    HostMutationAdmission,
    host_inbox_request_hash,
    host_mutation_request_hash,
)

_MAX_CADENCE_SECONDS = 3_600.0
_MAX_INBOX_PASS = 1_000
_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _stable_id(prefix: str, *parts: object) -> str:
    digest = sha256(canonical_json({"parts": parts}).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


async def _finish_despite_cancellation(
    factory: Callable[[], Awaitable[_T]],
) -> tuple[_T, bool]:
    """Finish one durability boundary before forwarding caller cancellation."""

    worker = asyncio.ensure_future(factory())
    cancellation_requested = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancellation_requested = True
            continue
        except BaseException:
            break
    try:
        result = worker.result()
    except BaseException as error:
        if cancellation_requested and not isinstance(error, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return result, cancellation_requested


class AgentHostState(str, Enum):
    OPEN = "open"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class AgentHostStatus:
    agent_id: str
    state: AgentHostState
    configured: bool
    pending_inbox: int
    nonterminal_operation_ids: tuple[str, ...]
    started_at: datetime | None = None
    last_pass_at: datetime | None = None
    last_error: str | None = None


class AgentHostStateError(RuntimeError):
    """Raised when a host command is invalid for its lifecycle state."""


class _NoFindingProjector:
    async def project(
        self,
        *,
        definition: object,
        operation: OperationSnapshot,
        checkpoint: object,
    ) -> MonitorOutcomeProjection:
        return MonitorOutcomeProjection(matched=False)


class _HostTriggerRunner:
    def __init__(self, host: AgentHost) -> None:
        self._host = host

    async def run(self, trigger: AgentTrigger) -> object:
        return await self._host._run_trigger(trigger)


class AgentHost:
    """Own foreground cadence and control without creating work on import/open."""

    def __init__(
        self,
        *,
        embedded: EmbeddedAgent,
        monitor_projector: MonitorOutcomeProjector | None = None,
        cadence_seconds: float = 1.0,
        monitor_lease_seconds: float = 300.0,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
        holder_id: str | None = None,
    ) -> None:
        if not isinstance(embedded, EmbeddedAgent):
            raise TypeError("embedded must be an EmbeddedAgent")
        if (
            not isinstance(cadence_seconds, (int, float))
            or isinstance(cadence_seconds, bool)
            or not math.isfinite(float(cadence_seconds))
            or not 0 < float(cadence_seconds) <= _MAX_CADENCE_SECONDS
        ):
            raise ValueError("cadence_seconds must be finite, positive, and bounded")
        if not callable(clock) or not callable(id_factory):
            raise TypeError("clock and id_factory must be callable")
        self._embedded = embedded
        self._store = embedded._store
        self._clock = clock
        self._id_factory = id_factory
        self._cadence_seconds = float(cadence_seconds)
        self._monitor_lease_seconds = float(monitor_lease_seconds)
        self._holder_id = _required_text(
            holder_id or id_factory("agent-host"),
            "host holder_id",
        )
        self._state = AgentHostState.OPEN
        self._started_at: datetime | None = None
        self._last_pass_at: datetime | None = None
        self._last_error: str | None = None
        self._stop_requested = asyncio.Event()
        self._wake_hint = asyncio.Event()
        self._stopped = asyncio.Event()
        self._execution_lock = asyncio.Lock()
        self._cadence_task: asyncio.Task[None] | None = None
        self._active_task: asyncio.Task[LoopExit] | None = None
        self._active_trigger_id: str | None = None
        self._active_operation_id: str | None = None
        self._active_interrupt: LoopExit | None = None
        self._scheduler = MonitorScheduler(
            agent_id=embedded.identity.id,
            store=self._store,
            operations=self._store,
            runner=_HostTriggerRunner(self),
            projector=monitor_projector or _NoFindingProjector(),
            holder_id=self._holder_id,
            lease_seconds=monitor_lease_seconds,
            clock=clock,
            id_factory=id_factory,
        )

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets = LoopBudgets(),
        monitor_projector: MonitorOutcomeProjector | None = None,
        cadence_seconds: float = 1.0,
        monitor_lease_seconds: float = 300.0,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> Self:
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id
        embedded = await EmbeddedAgent.create(
            name,
            root=root,
            model=model,
            model_profile=model_profile,
            context_builder=context_builder,
            domain=domain,
            capabilities=capabilities,
            policy=policy,
            budgets=budgets,
            clock=resolved_clock,
            id_factory=resolved_ids,
        )
        try:
            return cls(
                embedded=embedded,
                monitor_projector=monitor_projector,
                cadence_seconds=cadence_seconds,
                monitor_lease_seconds=monitor_lease_seconds,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
        except BaseException:
            await embedded.close()
            raise

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets = LoopBudgets(),
        monitor_projector: MonitorOutcomeProjector | None = None,
        cadence_seconds: float = 1.0,
        monitor_lease_seconds: float = 300.0,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> Self:
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id
        embedded = await EmbeddedAgent.open(
            name,
            root=root,
            model=model,
            model_profile=model_profile,
            context_builder=context_builder,
            domain=domain,
            capabilities=capabilities,
            policy=policy,
            budgets=budgets,
            clock=resolved_clock,
            id_factory=resolved_ids,
        )
        try:
            return cls(
                embedded=embedded,
                monitor_projector=monitor_projector,
                cadence_seconds=cadence_seconds,
                monitor_lease_seconds=monitor_lease_seconds,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
        except BaseException:
            await embedded.close()
            raise

    @property
    def state(self) -> AgentHostState:
        return self._state

    @property
    def id(self) -> str:
        return self._embedded.identity.id

    @property
    def name(self) -> str:
        return self._embedded.identity.display_name

    @property
    def home(self) -> Path:
        return self._embedded.home

    @property
    def model_profile(self) -> ModelProfile | None:
        return self._embedded.model_profile

    @property
    def configured(self) -> bool:
        return self._embedded._loop is not None

    async def start(self) -> None:
        """Recover durable work, then explicitly start one cadence task."""

        if self._state is AgentHostState.RUNNING:
            return
        if self._state is not AgentHostState.OPEN:
            raise AgentHostStateError(f"cannot start host from {self._state.value}")
        self._state = AgentHostState.RUNNING
        self._started_at = _aware_utc(self._clock(), "host clock")
        try:
            if self.configured:
                async with self._execution_lock:
                    await self._recover_startup_locked()
                    await self._run_once_locked(self._started_at)
            self._cadence_task = asyncio.create_task(
                self._cadence(),
                name=f"daita-host:{self._embedded.identity.id}",
            )
        except BaseException:
            self._state = AgentHostState.OPEN
            raise

    async def serve(self) -> None:
        await self.start()
        await self._stopped.wait()

    async def run_once(
        self,
        now: datetime | None = None,
    ) -> tuple[MonitorSchedulerResult, ...]:
        self._require_running()
        tick = _aware_utc(now or self._clock(), "host pass time")
        async with self._execution_lock:
            return await self._run_once_locked(tick)

    async def admit_mutation(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        idempotency_key: str,
    ) -> HostMutationAdmission:
        """Durably bind one client key before dispatching mutable host work."""

        self._require_running()
        request = HostMutationAdmission(
            agent_id=self.id,
            idempotency_key=idempotency_key,
            method=method,
            request_hash=host_mutation_request_hash(
                method=method,
                params=params,
            ),
            created_at=_aware_utc(self._clock(), "host mutation time"),
        )
        return await self._store.admit_host_mutation(request)

    async def attach(
        self,
        source: ResourceSource,
        *,
        idempotency_key: str,
    ) -> SourceRegistration:
        self._require_running()
        async with self._execution_lock:
            return await self._embedded.attach(
                source,
                idempotency_key=idempotency_key,
            )

    async def inspect_operation(self, operation_id: str) -> OperationSnapshot:
        self._require_running()
        return await self._embedded.inspect(operation_id)

    async def submit(
        self,
        message: str,
        *,
        idempotency_key: str,
        session_id: str | None = None,
    ) -> HostInboxItem:
        """Durably enqueue and process one stable user trigger."""

        self._require_running()
        message = _required_text(message, "host message")
        idempotency_key = _required_text(idempotency_key, "idempotency_key")
        if session_id is not None:
            session_id = _required_text(session_id, "session_id")
        payload: dict[str, object] = {
            "message": message,
            "session_id": session_id,
            "source_id": f"host:{idempotency_key}",
        }
        trigger_id = _stable_id(
            "host-trigger",
            self._embedded.identity.id,
            idempotency_key,
        )
        item = self._pending_item(
            HostInboxKind.TRIGGER,
            idempotency_key=idempotency_key,
            payload=payload,
            trigger_id=trigger_id,
        )
        stored = await self._store.enqueue_host_inbox(item)
        self._wake_hint.set()
        async with self._execution_lock:
            return await self._process_item(stored)

    async def decide_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        decided_by: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> ApprovalRequest:
        """Persist a decision, then durably wake the same operation."""

        self._require_running()
        decision, decision_cancelled = await _finish_despite_cancellation(
            lambda: self._embedded.decide_approval(
                approval_id,
                status=status,
                decided_by=decided_by,
                reason=reason,
            )
        )
        wake_key = idempotency_key or (f"approval:{approval_id}:{status.value}")
        item = self._pending_item(
            HostInboxKind.APPROVAL_WAKE,
            idempotency_key=wake_key,
            payload={
                "approval_id": approval_id,
                "operation_id": decision.operation_id,
            },
        )
        stored, enqueue_cancelled = await _finish_despite_cancellation(
            lambda: self._store.enqueue_host_inbox(item)
        )
        self._wake_hint.set()
        if decision_cancelled or enqueue_cancelled:
            raise asyncio.CancelledError
        async with self._execution_lock:
            await self._process_item(stored)
        return decision

    async def approve(
        self,
        approval_id: str,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> ApprovalRequest:
        return await self.decide_approval(
            approval_id,
            status=ApprovalStatus.APPROVED,
            decided_by=decided_by,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    async def reject(
        self,
        approval_id: str,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> ApprovalRequest:
        return await self.decide_approval(
            approval_id,
            status=ApprovalStatus.DENIED,
            decided_by=decided_by,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    async def cancel(
        self,
        operation_id: str,
        *,
        reason: str = "user_cancelled",
    ) -> LoopExit:
        """Interrupt through the runtime control lane and stop matching live I/O."""

        self._require_running()
        result = await self._embedded.interrupt(operation_id, reason)
        active_task = self._active_task
        active_operation_id = self._active_operation_id
        active_trigger_id = self._active_trigger_id
        matches = active_operation_id == operation_id
        if not matches and active_task is not None and active_trigger_id is not None:
            active = await self._store.load_by_trigger(active_trigger_id)
            matches = (
                active is not None and active.snapshot.operation.id == operation_id
            )
        if matches and active_task is not None and not active_task.done():
            self._active_interrupt = result
            active_task.cancel()
        return result

    async def propose_monitor(
        self,
        monitor_id: str,
        definition: MonitorDefinition,
        *,
        idempotency_key: str,
        source_operation_id: str | None = None,
    ) -> MonitorProposal:
        self._require_running()
        return await self._embedded.propose_monitor(
            monitor_id,
            definition,
            idempotency_key=idempotency_key,
            source_operation_id=source_operation_id,
        )

    async def confirm_monitor(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorInspection:
        self._require_running()
        return await self._embedded.confirm_monitor(
            proposal_id,
            candidate_hash=candidate_hash,
            actor_id=actor_id,
            reason=reason,
        )

    async def reject_monitor(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorConfirmation:
        self._require_running()
        return await self._embedded.reject_monitor(
            proposal_id,
            candidate_hash=candidate_hash,
            actor_id=actor_id,
            reason=reason,
        )

    async def list_monitors(
        self,
        *,
        statuses: tuple[MonitorStatus, ...] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> tuple[Monitor, ...]:
        self._require_running()
        return await self._embedded.list_monitors(
            statuses=statuses,
            include_deleted=include_deleted,
            limit=limit,
        )

    async def list_monitor_proposals(
        self,
        *,
        limit: int = 100,
    ) -> tuple[MonitorProposal, ...]:
        self._require_running()
        return await self._embedded.list_monitor_proposals(limit=limit)

    async def inspect_monitor(self, monitor_id: str) -> MonitorInspection:
        self._require_running()
        return await self._embedded.inspect_monitor(monitor_id)

    async def pause_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        self._require_running()
        return await self._embedded.pause_monitor(
            monitor_id,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def resume_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        self._require_running()
        return await self._embedded.resume_monitor(
            monitor_id,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def delete_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        self._require_running()
        return await self._embedded.delete_monitor(
            monitor_id,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def run_monitor_now(
        self,
        monitor_id: str,
        *,
        idempotency_key: str,
        lease_seconds: float | None = None,
    ) -> MonitorSchedulerResult:
        self._require_running()
        resolved_lease = self._monitor_lease_seconds
        if lease_seconds is not None:
            if (
                not isinstance(lease_seconds, (int, float))
                or isinstance(lease_seconds, bool)
                or not math.isfinite(float(lease_seconds))
                or float(lease_seconds) <= 0
                or float(lease_seconds) > 300
            ):
                raise ValueError("lease_seconds must be finite and from 0 through 300")
            resolved_lease = min(float(lease_seconds), resolved_lease)
        async with self._execution_lock:
            claim = await self._embedded.claim_monitor_run_now(
                monitor_id,
                idempotency_key=idempotency_key,
                holder_id=self._holder_id,
                lease_seconds=resolved_lease,
            )
            return await self._scheduler.run_claimed(claim)

    async def status(self) -> AgentHostStatus:
        pending = await self._store.list_pending_host_inbox(
            self._embedded.identity.id,
            limit=_MAX_INBOX_PASS,
        )
        nonterminal = await self._embedded.inspect_nonterminal()
        return AgentHostStatus(
            agent_id=self._embedded.identity.id,
            state=self._state,
            configured=self.configured,
            pending_inbox=len(pending),
            nonterminal_operation_ids=tuple(
                snapshot.operation.id for snapshot in nonterminal
            ),
            started_at=self._started_at,
            last_pass_at=self._last_pass_at,
            last_error=self._last_error,
        )

    async def read_events(
        self,
        cursor: EventCursor | None = None,
        *,
        limit: int = 100,
    ) -> tuple[CommittedEvent, ...]:
        return await self._embedded.read_events(cursor, limit=limit)

    def subscribe_events(
        self,
        cursor: EventCursor | None = None,
    ) -> AsyncIterator[CommittedEvent]:
        return self._embedded.subscribe_events(cursor)

    async def stop(self, *, drain: bool = True) -> None:
        """Stop cadence, optionally drain durable work, then release the writer."""

        if self._state is AgentHostState.STOPPED:
            return
        if self._state is AgentHostState.OPEN:
            self._state = AgentHostState.STOPPING
            try:
                await self._embedded.close()
            finally:
                self._state = AgentHostState.STOPPED
                self._stopped.set()
            return
        if self._state is AgentHostState.STOPPING:
            await self._stopped.wait()
            return
        self._state = AgentHostState.STOPPING
        self._stop_requested.set()
        self._wake_hint.set()
        try:
            cadence = self._cadence_task
            if cadence is not None and cadence is not asyncio.current_task():
                cadence.cancel()
                try:
                    await cadence
                except asyncio.CancelledError:
                    pass
            if drain and self.configured:
                async with self._execution_lock:
                    await self._drain_inbox_locked()
            elif not drain:
                active = self._active_task
                if active is not None and not active.done():
                    active.cancel()
                    if active is not asyncio.current_task():
                        try:
                            await active
                        except asyncio.CancelledError:
                            pass
        finally:
            try:
                await self._embedded.close()
            finally:
                self._state = AgentHostState.STOPPED
                self._stopped.set()

    async def _cadence(self) -> None:
        try:
            while not self._stop_requested.is_set():
                try:
                    await self.run_once()
                    self._last_error = None
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._last_error = f"{type(error).__name__}: {error}"[:2_000]
                if self._wake_hint.is_set():
                    self._wake_hint.clear()
                    continue
                try:
                    async with asyncio.timeout(self._cadence_seconds):
                        await self._wake_hint.wait()
                except TimeoutError:
                    pass
                finally:
                    self._wake_hint.clear()
        except asyncio.CancelledError:
            raise

    async def _run_once_locked(
        self,
        now: datetime,
    ) -> tuple[MonitorSchedulerResult, ...]:
        await self._drain_inbox_locked()
        results: tuple[MonitorSchedulerResult, ...] = ()
        if self.configured:
            results = await self._scheduler.run_due(now)
        self._last_pass_at = _aware_utc(self._clock(), "host clock")
        return results

    async def _drain_inbox_locked(self) -> None:
        if not self.configured:
            return
        while True:
            pending = await self._store.list_pending_host_inbox(
                self._embedded.identity.id,
                limit=_MAX_INBOX_PASS,
            )
            if not pending:
                return
            for item in pending:
                await self._process_item(item)

    async def _process_item(self, item: HostInboxItem) -> HostInboxItem:
        if item.status is HostInboxStatus.COMPLETED:
            return item
        current = await self._store.enqueue_host_inbox(item)
        if current.status is HostInboxStatus.COMPLETED:
            return current
        if current.kind is HostInboxKind.TRIGGER:
            trigger = self._trigger_from_item(current)
            trigger_result = await self._run_trigger(trigger)
            completed_operation_id = trigger_result.operation_id
        else:
            payload_operation_id = current.payload.get("operation_id")
            if not isinstance(payload_operation_id, str) or not payload_operation_id:
                raise ValueError("approval wake is missing operation_id")
            snapshot = await self._embedded.inspect(payload_operation_id)
            if snapshot.trigger.kind is TriggerKind.MONITOR:
                monitor_result = await self._scheduler.resume_trigger(
                    snapshot.trigger.id
                )
                if monitor_result.operation_id is None:
                    raise AgentHostStateError(
                        "monitor approval wake produced no operation"
                    )
                completed_operation_id = monitor_result.operation_id
            else:
                resumed_result = await self._resume_operation(payload_operation_id)
                completed_operation_id = resumed_result.operation_id
        completed = replace(
            current,
            revision=2,
            status=HostInboxStatus.COMPLETED,
            updated_at=_aware_utc(self._clock(), "host clock"),
            operation_id=completed_operation_id,
        )
        return await self._store.complete_host_inbox(
            completed,
            expected_revision=1,
        )

    async def _run_trigger(self, trigger: AgentTrigger) -> LoopExit:
        return await self._run_active(
            lambda: self._embedded.run_trigger(trigger),
            trigger_id=trigger.id,
        )

    async def _resume_operation(self, operation_id: str) -> LoopExit:
        return await self._run_active(
            lambda: self._embedded.resume(operation_id),
            operation_id=operation_id,
        )

    async def _run_active(
        self,
        factory: Callable[[], Awaitable[LoopExit]],
        *,
        trigger_id: str | None = None,
        operation_id: str | None = None,
    ) -> LoopExit:
        if self._active_task is not None:
            raise AgentHostStateError("host already owns active execution")
        label = trigger_id or operation_id or "unknown"

        async def run_factory() -> LoopExit:
            return await factory()

        task: asyncio.Task[LoopExit] = asyncio.create_task(
            run_factory(),
            name=f"daita-host-operation:{label}",
        )
        self._active_task = task
        self._active_trigger_id = trigger_id
        self._active_operation_id = operation_id
        self._active_interrupt = None
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            interrupted = self._active_interrupt
            if current is not None and current.cancelling():
                raise
            if interrupted is not None:
                return interrupted
            raise
        finally:
            if self._active_task is task:
                self._active_task = None
                self._active_trigger_id = None
                self._active_operation_id = None
                self._active_interrupt = None

    async def _recover_startup_locked(self) -> None:
        snapshots = await self._embedded.inspect_nonterminal()
        for snapshot in snapshots:
            await self._resume_operation(snapshot.operation.id)

    def _pending_item(
        self,
        kind: HostInboxKind,
        *,
        idempotency_key: str,
        payload: dict[str, object],
        trigger_id: str | None = None,
    ) -> HostInboxItem:
        now = _aware_utc(self._clock(), "host clock")
        item_id = _stable_id(
            "host-inbox",
            self._embedded.identity.id,
            idempotency_key,
        )
        return HostInboxItem(
            id=item_id,
            agent_id=self._embedded.identity.id,
            kind=kind,
            idempotency_key=idempotency_key,
            request_hash=host_inbox_request_hash(
                kind=kind,
                payload=payload,
                trigger_id=trigger_id,
            ),
            payload=payload,
            revision=1,
            status=HostInboxStatus.PENDING,
            created_at=now,
            updated_at=now,
            trigger_id=trigger_id,
        )

    def _trigger_from_item(self, item: HostInboxItem) -> AgentTrigger:
        message = item.payload.get("message")
        source_id = item.payload.get("source_id")
        session_id = item.payload.get("session_id")
        if not isinstance(message, str) or not isinstance(source_id, str):
            raise ValueError("host trigger inbox payload is malformed")
        if session_id is not None and not isinstance(session_id, str):
            raise ValueError("host trigger session_id is malformed")
        assert item.trigger_id is not None
        return AgentTrigger(
            id=item.trigger_id,
            agent_id=self._embedded.identity.id,
            kind=TriggerKind.USER,
            source_id=source_id,
            session_id=session_id,
            payload={"message": message},
            created_at=item.created_at,
        )

    def _require_running(self) -> None:
        if self._state is not AgentHostState.RUNNING:
            raise AgentHostStateError(f"host is {self._state.value}")

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop(drain=True)


__all__ = [
    "AgentHost",
    "AgentHostState",
    "AgentHostStateError",
    "AgentHostStatus",
]
