"""Runtime-native worker task handoff and leased execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .kernel import (
    RuntimeKernel,
    RuntimeKernelExecutorFailed,
    RuntimeKernelExecutionError,
    RuntimeKernelLeaseLost,
    RuntimeKernelTaskAlreadyTerminal,
    RuntimeKernelTaskNotRunnable,
    TaskExecutionResult,
    TaskLease,
)
from .primitives import (
    Capability,
    Operation,
    RuntimeEvent,
    RuntimeEventType,
    Task,
    TaskStatus,
    Worker,
)


@dataclass(frozen=True)
class WorkerRuntimeOptions:
    """Configuration for one runtime worker runner."""

    worker_id: str
    owner: str
    queues: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    lease_seconds: float = 60.0
    poll_interval_seconds: float = 1.0
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "queues", tuple(self.queues))
        object.__setattr__(self, "domains", tuple(self.domains))
        object.__setattr__(self, "capability_ids", tuple(self.capability_ids))
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds cannot be negative")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")


@dataclass(frozen=True)
class TaskHandoff:
    """Durable task handoff descriptor for worker-capable runtime work."""

    task_id: str
    operation_id: str
    worker_id: str | None = None
    worker_owner: str | None = None
    reason: str = "handoff"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class WorkerRunResult:
    """Outcome from one worker task attempt."""

    handoff: TaskHandoff
    lease: TaskLease | None = None
    execution: TaskExecutionResult | None = None
    events: tuple[RuntimeEvent, ...] = ()
    skipped: bool = False
    error: BaseException | None = None


class WorkerRuntime:
    """Leased runner for tasks declared as worker-capable by plugins."""

    def __init__(
        self,
        *,
        kernel: RuntimeKernel,
        options: WorkerRuntimeOptions,
    ) -> None:
        self.kernel = kernel
        self.store = kernel.store
        self.registry = kernel.extension_registry
        self.options = options
        self.worker = self.registry.get_worker(options.worker_id, owner=options.owner)
        self._validate_options_against_worker(self.worker)

    async def handoff_task(
        self,
        task_id: str,
        *,
        reason: str = "handoff",
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskHandoff:
        """Record a durable handoff event for an already-persisted task."""
        task = await self.store.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        operation = await self.store.load_operation(task.operation_id)
        if operation is None:
            raise KeyError(task.operation_id)
        capability = self._capability_for_task(task)
        self._validate_task_for_worker(task, capability)
        handoff = TaskHandoff(
            task_id=task.id,
            operation_id=operation.id,
            worker_id=self.worker.id,
            worker_owner=self.worker.owner,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        await self.store.append_event(
            self._event(
                RuntimeEventType.WORKER_HANDOFF,
                operation=operation,
                task=task,
                capability=capability,
                message=(
                    f"Task {task.id} handed off to worker "
                    f"{self.worker.owner}:{self.worker.id}."
                ),
                payload={
                    "handoff": {
                        "reason": handoff.reason,
                        "metadata": handoff.metadata,
                    },
                    "worker": self.worker.to_dict(),
                },
            )
        )
        return handoff

    async def execute_handoff(
        self,
        handoff: TaskHandoff,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> WorkerRunResult:
        """Claim and execute a handed-off task through the runtime kernel."""
        task = await self.store.load_task(handoff.task_id)
        if task is None:
            raise KeyError(handoff.task_id)
        operation = await self.store.load_operation(task.operation_id)
        if operation is None:
            raise KeyError(task.operation_id)
        capability = self._capability_for_task(task)
        self._validate_task_for_worker(task, capability)
        try:
            lease = await self.kernel.claim_task(
                task.id,
                lease_owner=f"worker:{self.worker.owner}:{self.worker.id}",
                lease_seconds=self.options.lease_seconds,
                worker_id=self.worker.id,
                worker_owner=self.worker.owner,
            )
            claim_event = self._event(
                RuntimeEventType.WORKER_LEASE_CLAIMED,
                operation=operation,
                task=task,
                capability=capability,
                message=(
                    f"Worker {self.worker.owner}:{self.worker.id} claimed "
                    f"task {task.id}."
                ),
                payload={
                    "lease_id": lease.lease_id,
                    "lease_owner": lease.lease_owner,
                    "lease_expires_at": lease.lease_expires_at,
                    "attempt_count": lease.attempt_count,
                },
            )
            await self.store.append_event(claim_event)
            execution = await self.kernel.execute_claimed_task(
                lease,
                context={
                    "worker_id": self.worker.id,
                    "worker_owner": self.worker.owner,
                    **dict(context or {}),
                },
            )
            completed_event = self._event(
                RuntimeEventType.WORKER_COMPLETED,
                operation=execution.operation,
                task=execution.task,
                capability=execution.capability,
                message=(
                    f"Worker {self.worker.owner}:{self.worker.id} completed "
                    f"task {task.id}."
                ),
                payload={
                    "lease_id": lease.lease_id,
                    "evidence_ids": [
                        item.id for item in execution.evidence if item.id is not None
                    ],
                },
            )
            await self.store.append_event(completed_event)
            return WorkerRunResult(
                handoff=handoff,
                lease=lease,
                execution=execution,
                events=(claim_event, *execution.events, completed_event),
            )
        except RuntimeKernelTaskAlreadyTerminal as exc:
            return WorkerRunResult(
                handoff=handoff,
                execution=exc.result,
                skipped=True,
                error=exc,
            )
        except RuntimeKernelExecutionError as exc:
            result = exc.result
            current_task = result.task if result is not None else task
            current_operation = result.operation if result is not None else operation
            current_capability = result.capability if result is not None else capability
            failed_event = self._event(
                _worker_failure_type(exc),
                operation=current_operation,
                task=current_task,
                capability=current_capability,
                message=(
                    f"Worker {self.worker.owner}:{self.worker.id} could not "
                    f"complete task {task.id}."
                ),
                payload={"error": {"type": type(exc).__name__, "message": str(exc)}},
            )
            await self.store.append_event(failed_event)
            return WorkerRunResult(
                handoff=handoff,
                execution=result,
                events=((*result.events, failed_event) if result else (failed_event,)),
                error=exc,
            )

    async def heartbeat(self, lease: TaskLease) -> TaskLease:
        """Extend a worker-owned task lease and record a heartbeat event."""
        updated = await self.kernel.heartbeat_task(
            lease,
            extend_by_seconds=self.options.lease_seconds,
        )
        task = await self.store.load_task(updated.task_id)
        operation = (
            await self.store.load_operation(updated.operation_id)
            if task is not None
            else None
        )
        if task is not None and operation is not None:
            await self.store.append_event(
                self._event(
                    RuntimeEventType.WORKER_HEARTBEAT,
                    operation=operation,
                    task=task,
                    capability=self._capability_for_task(task),
                    message=(
                        f"Worker {self.worker.owner}:{self.worker.id} "
                        f"heartbeat for task {task.id}."
                    ),
                    payload={
                        "lease_id": updated.lease_id,
                        "lease_expires_at": updated.lease_expires_at,
                    },
                )
            )
        return updated

    async def run_once(
        self,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> WorkerRunResult | None:
        """Discover one pending worker-capable task and execute it."""
        for task in await self.store.list_tasks():
            if task.status not in {TaskStatus.PENDING, TaskStatus.BLOCKED}:
                continue
            capability = self._capability_for_task(task)
            if not self._task_matches_worker(task, capability):
                continue
            handoff = await self.handoff_task(task.id, reason="worker_poll")
            return await self.execute_handoff(handoff, context=context)
        return None

    async def run_forever(self, *, context: Mapping[str, Any] | None = None) -> None:
        """Poll for worker-capable tasks until cancelled."""
        semaphore = asyncio.Semaphore(self.options.max_concurrency)
        running: set[asyncio.Task[Any]] = set()
        try:
            while True:
                running = {task for task in running if not task.done()}
                if semaphore.locked():
                    await asyncio.sleep(self.options.poll_interval_seconds)
                    continue
                await semaphore.acquire()

                async def _attempt() -> None:
                    try:
                        await self.run_once(context=context)
                    finally:
                        semaphore.release()

                running.add(asyncio.create_task(_attempt()))
                await asyncio.sleep(self.options.poll_interval_seconds)
        finally:
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

    def _validate_options_against_worker(self, worker: Worker) -> None:
        if self.options.max_concurrency > worker.max_concurrency:
            raise ValueError("worker runtime max_concurrency exceeds declaration")
        for capability_id in self.options.capability_ids:
            if capability_id not in worker.capability_ids:
                raise ValueError(
                    f"worker {worker.owner}:{worker.id} cannot handle "
                    f"{capability_id!r}"
                )

    def _validate_task_for_worker(self, task: Task, capability: Capability) -> None:
        if not self._task_matches_worker(task, capability):
            raise RuntimeKernelTaskNotRunnable(
                f"Task {task.id} is not handled by worker "
                f"{self.worker.owner}:{self.worker.id}."
            )

    def _task_matches_worker(self, task: Task, capability: Capability) -> bool:
        if task.capability_id not in self.worker.capability_ids:
            return False
        if self.options.capability_ids and task.capability_id not in set(
            self.options.capability_ids
        ):
            return False
        if self.options.domains and not set(self.options.domains).intersection(
            capability.domains
        ):
            return False
        if self.options.queues:
            queue = task.metadata.get("queue")
            if queue not in set(self.options.queues):
                return False
        owner = task.metadata.get("owner")
        return owner is None or owner == self.worker.owner

    def _capability_for_task(self, task: Task) -> Capability:
        owner = task.metadata.get("owner") if task.metadata else None
        if owner:
            return self.registry.get_capability(task.capability_id, owner=str(owner))
        return self.registry.get_capability(task.capability_id)

    def _event(
        self,
        type: RuntimeEventType,
        *,
        operation: Operation,
        task: Task,
        capability: Capability,
        message: str,
        payload: Mapping[str, Any] | None = None,
    ) -> RuntimeEvent:
        return RuntimeEvent(
            type=type,
            runtime_id=self.kernel.runtime_id,
            runtime_kind=self.kernel.runtime_kind,
            operation_id=operation.id,
            task_id=task.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            plugin_id=capability.owner,
            trace_id=_latest_trace_id(operation, task),
            span_id=_latest_span_id(operation, task),
            message=message,
            payload=dict(payload or {}),
        )


def _worker_failure_type(exc: RuntimeKernelExecutionError) -> RuntimeEventType:
    if isinstance(exc, RuntimeKernelLeaseLost):
        return RuntimeEventType.WORKER_TIMEOUT
    if isinstance(exc, RuntimeKernelExecutorFailed):
        return RuntimeEventType.WORKER_FAILED
    if isinstance(exc, RuntimeKernelTaskNotRunnable):
        return RuntimeEventType.WORKER_FAILED
    return RuntimeEventType.WORKER_FAILED


def _latest_trace_id(operation: Operation, task: Task) -> str | None:
    return (
        task.metadata.get("trace_id") or operation.metadata.get("trace_id")
        if task.metadata or operation.metadata
        else None
    )


def _latest_span_id(operation: Operation, task: Task) -> str | None:
    return (
        task.metadata.get("span_id") or operation.metadata.get("span_id")
        if task.metadata or operation.metadata
        else None
    )
