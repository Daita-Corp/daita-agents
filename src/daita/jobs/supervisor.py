"""Claim, execute, fence, recover, and reconcile durable job attempts within limits."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from functools import partial
from hashlib import sha256
from typing import TypeVar

from .._json import canonical_json
from ..adapters.job_profiles import (
    ConnectedJobProfile,
    ExternalCancelRequest,
    ExternalResultRequest,
    ExternalStartRequest,
    ExternalStatusRequest,
)
from ..artifacts.models import (
    ArtifactAuthorship,
    ArtifactError,
    ArtifactRef,
    artifact_ref_to_mapping,
)
from ..artifacts.store import AgentHomeArtifactStore
from ..capabilities import CapabilityInputError
from ..capability_runtime import CapabilityRuntime, InternalCapabilityRequest
from ..distribution.models import (
    ArtifactRequirement,
    OutcomeContract,
    outcome_artifact_reference,
    validate_outcome_artifact_references,
)
from ..distribution.owner import DistributionOwner, construct_graph_job_delivery
from ..hosting.execution_governor import RunAdmissionCoordinator, WorkloadClass
from ..llm.models import ModelSensitivity
from ..loop.models import RunInput
from ..storage.sqlite import SQLiteStateStore
from .graph.models import (
    BudgetAmount,
    GraphInspection,
    GraphTask,
    TaskAttempt,
    TaskCheckpoint,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
    canonical_digest,
    reserved_artifact_id,
)
from .models import (
    MAX_JOB_EXTERNAL_OBSERVATIONS,
    MAX_RUNNING_JOBS_GLOBAL,
    ExternalIntent,
    ExternalIntentDisposition,
    ExternalIntentKind,
    ExternalObservation,
    ExternalObservedStatus,
    JobAttempt,
    JobAttemptStatus,
    JobDesiredState,
    JobExecutionMode,
    JobResult,
    JobRun,
    JobStatus,
)
from .owner import JobError, JobOwner

_ARTIFACT_ID = re.compile(r"artifact-[0-9a-f]{32}\Z")
_RUN_ID = re.compile(r"run-[0-9a-f]{32}\Z")
_DEFAULT_POLL_SECONDS = 0.05
_PROFILE_WORK_KIND = "data_profile_work"
_PROFILE_FINALIZER_KIND = "data_profile_finalizer"
_T = TypeVar("_T")


class _ProcessJobCapacity:
    """The exact process-local global running-job bound."""

    _lock = threading.Lock()
    _in_use = 0

    @classmethod
    def acquire(cls) -> bool:
        with cls._lock:
            if cls._in_use >= MAX_RUNNING_JOBS_GLOBAL:
                return False
            cls._in_use += 1
            return True

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            if cls._in_use < 1:
                raise RuntimeError("global job capacity was released without a claim")
            cls._in_use -= 1


class JobSupervisor:
    """Claim, revalidate, execute, reconcile, and fence the one JobRun aggregate."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: SQLiteStateStore,
        owner: JobOwner,
        runtime: CapabilityRuntime,
        revalidate_external: Callable[[JobRun], Awaitable[None]],
        artifacts: AgentHomeArtifactStore,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        on_terminal: Callable[[JobRun], None] | None = None,
        admission_coordinator: RunAdmissionCoordinator | None = None,
        distribution: DistributionOwner | None = None,
        graph_parallelism: int = 1,
        graph_heartbeat_seconds: float = 10.0,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("job supervisor agent_id must be non-empty text")
        if not isinstance(poll_seconds, (int, float)) or not 0 < poll_seconds <= 1:
            raise ValueError("job supervisor poll interval is outside its bound")
        self._agent_id = agent_id
        self._store = store
        self._owner = owner
        self._runtime = runtime
        if not callable(revalidate_external):
            raise TypeError("external job revalidator must be callable")
        self._revalidate_external = revalidate_external
        self._artifacts = artifacts
        self._clock = clock
        self._id_factory = id_factory
        if on_terminal is not None and not callable(on_terminal):
            raise TypeError("terminal observer must be callable or None")
        self._on_terminal = on_terminal
        if admission_coordinator is not None and not isinstance(
            admission_coordinator, RunAdmissionCoordinator
        ):
            raise TypeError("graph admission coordinator is invalid")
        if distribution is not None and not isinstance(distribution, DistributionOwner):
            raise TypeError("graph distribution owner is invalid")
        if (
            not isinstance(graph_parallelism, int)
            or isinstance(graph_parallelism, bool)
            or not 1 <= graph_parallelism <= 4
        ):
            raise ValueError("graph parallelism must be between one and four")
        if (
            not isinstance(graph_heartbeat_seconds, (int, float))
            or isinstance(graph_heartbeat_seconds, bool)
            or not 10 <= float(graph_heartbeat_seconds) <= 30
        ):
            raise ValueError("graph heartbeat interval is outside its bound")
        self._admission_coordinator = admission_coordinator
        self._distribution = distribution
        self._graph_parallelism = graph_parallelism
        self._graph_heartbeat_seconds = float(graph_heartbeat_seconds)
        self._poll_seconds = float(poll_seconds)
        self._wake = asyncio.Event()
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._worker_modes: dict[str, JobExecutionMode] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._driver: asyncio.Task[None] | None = None
        self._graph_driver: asyncio.Task[None] | None = None
        self._graph_workers: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._graph_wake = asyncio.Event()
        self._closing = False

    async def start(self) -> None:
        if self._driver is not None:
            raise RuntimeError("job supervisor is already started")
        await self._recover_daita_claims()
        self._driver = asyncio.create_task(
            self._drive(),
            name=f"daita-job-supervisor:{self._agent_id}",
        )

    async def start_graph_integration(self) -> None:
        """Start only the explicit draft-graph supervisor path used by integration."""

        if self._graph_driver is not None or self._driver is not None:
            raise RuntimeError("job supervisor is already started")
        if self._admission_coordinator is None or self._distribution is None:
            raise RuntimeError("draft graph integration dependencies are unavailable")
        await self._recover_graph_attempts()
        self._graph_driver = asyncio.create_task(
            self._drive_graph(),
            name=f"daita-graph-supervisor:{self._agent_id}",
        )

    def wake(self, job_id: str | None = None) -> None:
        if self._closing:
            return
        if job_id is not None:
            cancel_event = self._cancel_events.get(job_id)
            if cancel_event is not None:
                cancel_event.set()
            worker = self._workers.get(job_id)
            if worker is not None and not worker.done():
                # Daita-owned reads are cancellable. Connected attempts must remain
                # alive long enough to persist and reconcile a remote cancel intent.
                worker_job = self._worker_modes.get(job_id)
                if worker_job is JobExecutionMode.DAITA:
                    worker.cancel("job_cancel_requested")
        self._wake.set()
        self._graph_wake.set()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        current_loop = asyncio.get_running_loop()
        all_tasks = tuple(
            item
            for item in (
                self._driver,
                self._graph_driver,
                *self._workers.values(),
                *self._graph_workers.values(),
            )
            if item is not None
        )
        tasks = tuple(item for item in all_tasks if item.get_loop() is current_loop)
        if tasks:
            self._wake.set()
        for task in tasks:
            task.cancel("host_closing")
        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
        self._driver = None
        self._graph_driver = None
        self._workers.clear()
        self._graph_workers.clear()
        self._worker_modes.clear()
        self._cancel_events.clear()

    async def _recover_daita_claims(self) -> None:
        running = await self._store.list_jobs(
            self._agent_id,
            statuses=frozenset({JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}),
        )
        now = self._clock()
        for job in running:
            if job.specification.execution_mode is not JobExecutionMode.DAITA:
                continue
            attempt = _claimed_attempt(job)
            try:
                artifact = await self._artifacts.recover_reserved(
                    attempt.execution_run_id,
                    attempt.reserved_artifact_id,
                )
            except ArtifactError:
                await self._finalize(
                    job,
                    JobAttemptStatus.NEEDS_ATTENTION,
                    failure_code="job_artifact_reconciliation_failed",
                )
                continue
            if artifact is not None:
                result = _recovered_artifact_result(
                    job, artifact, now, self._id_factory
                )
                await self._finalize(job, JobAttemptStatus.SUCCEEDED, result=result)
                continue
            await self._store.recover_stale_job(
                self._agent_id,
                job.job_id,
                recovered_at=now,
                restart_safe=True,
            )

    async def _graph_store_call(
        self,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        coordinator = self._required_graph_coordinator()
        permit = await coordinator.sqlite_pressure_permit()
        async with permit:
            return await operation()

    def _required_graph_coordinator(self) -> RunAdmissionCoordinator:
        if self._admission_coordinator is None:
            raise RuntimeError("draft graph coordinator is unavailable")
        return self._admission_coordinator

    def _required_graph_distribution(self) -> DistributionOwner:
        if self._distribution is None:
            raise RuntimeError("draft graph distribution owner is unavailable")
        return self._distribution

    async def _recover_graph_attempts(self) -> None:
        attempts = await self._graph_store_call(
            lambda: self._store.list_active_graph_attempts(self._agent_id)
        )
        for attempt in attempts:
            inspection = await self._graph_store_call(
                partial(self._store.inspect_graph, self._agent_id, attempt.job_id)
            )
            if inspection is None:
                continue
            task = _inspection_task(inspection, attempt.task_id)
            try:
                _internal_task_contract(task)
                payload = await self._artifacts.read_reserved(
                    attempt.run_id,
                    reserved_artifact_id(attempt.attempt_id),
                )
            except (
                ArtifactError,
                CapabilityInputError,
                TypeError,
                ValueError,
            ) as error:
                await self._graph_store_call(
                    partial(
                        self._store.fail_graph_attempt,
                        attempt.agent_id,
                        attempt.job_id,
                        attempt.task_id,
                        attempt.attempt_id,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        failed_at=self._clock(),
                        retryable=False,
                        reason_code=_safe_graph_failure_code(error),
                    )
                )
                continue
            if payload is not None:
                started = await self._graph_store_call(
                    partial(
                        self._store.start_graph_attempt,
                        attempt.agent_id,
                        attempt.job_id,
                        attempt.task_id,
                        attempt.attempt_id,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        started_at=attempt.started_at or payload.ref.created_at,
                    )
                )
                if started is not None:
                    try:
                        await self._promote_graph_artifact(
                            inspection,
                            task,
                            started,
                            payload.ref,
                            payload.content,
                        )
                    except (
                        ArtifactError,
                        CapabilityInputError,
                        TypeError,
                        ValueError,
                    ) as error:
                        await self._graph_store_call(
                            partial(
                                self._store.fail_graph_attempt,
                                started.agent_id,
                                started.job_id,
                                started.task_id,
                                started.attempt_id,
                                claim_token=started.claim_token,
                                fencing_epoch=started.fencing_epoch,
                                failed_at=self._clock(),
                                retryable=False,
                                reason_code=_safe_graph_failure_code(error),
                            )
                        )
                    continue
            await self._graph_store_call(
                partial(
                    self._store.fence_graph_attempt,
                    attempt.agent_id,
                    attempt.job_id,
                    attempt.task_id,
                    attempt.attempt_id,
                    fencing_epoch=attempt.fencing_epoch,
                    fenced_at=self._clock(),
                    requeue=True,
                    reason_code="host_restarted",
                )
            )

    async def _drive_graph(self) -> None:
        try:
            while not self._closing:
                self._graph_wake.clear()
                await self._recover_stale_graph_attempts()
                await self._graph_store_call(
                    lambda: self._store.expire_due_graphs(
                        self._agent_id,
                        expired_at=self._clock(),
                    )
                )
                await self._launch_ready_graph_tasks()
                try:
                    await asyncio.wait_for(
                        self._graph_wake.wait(),
                        timeout=self._poll_seconds,
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    async def _recover_stale_graph_attempts(self) -> None:
        stale = await self._graph_store_call(
            lambda: self._store.list_stale_graph_attempts(
                self._agent_id,
                now=self._clock(),
            )
        )
        for attempt in stale:
            key = (attempt.job_id, attempt.task_id)
            fenced = await self._graph_store_call(
                partial(
                    self._store.fence_graph_attempt,
                    attempt.agent_id,
                    attempt.job_id,
                    attempt.task_id,
                    attempt.attempt_id,
                    fencing_epoch=attempt.fencing_epoch,
                    fenced_at=self._clock(),
                    requeue=True,
                    reason_code="lease_expired",
                )
            )
            if fenced is None:
                continue
            worker = self._graph_workers.get(key)
            if worker is not None and not worker.done():
                worker.cancel("graph_attempt_lease_expired")
                await asyncio.gather(worker, return_exceptions=True)

    async def _launch_ready_graph_tasks(self) -> None:
        available = self._graph_parallelism - len(self._graph_workers)
        if available <= 0:
            return
        ready = await self._graph_store_call(
            lambda: self._store.list_ready_graph_tasks(
                self._agent_id,
                now=self._clock(),
                limit=64,
            )
        )
        for task in ready:
            key = (task.job_id, task.task_id)
            if key in self._graph_workers:
                continue
            worker = asyncio.create_task(
                self._run_graph_task(task),
                name=f"daita-graph-task:{task.job_id}:{task.task_id}",
            )
            self._graph_workers[key] = worker

            def done(completed: asyncio.Task[None], *, key=key) -> None:
                self._graph_workers.pop(key, None)
                if not completed.cancelled():
                    completed.exception()
                self._graph_wake.set()

            worker.add_done_callback(done)
            available -= 1
            if available == 0:
                return

    async def _run_graph_task(self, selected: GraphTask) -> None:
        coordinator = self._required_graph_coordinator()
        lease = await coordinator.admit_execution(
            WorkloadClass.GRAPH,
            f"{selected.job_id}:{selected.task_id}",
        )
        async with lease:
            inspection = await self._graph_store_call(
                lambda: self._store.inspect_graph(self._agent_id, selected.job_id)
            )
            if inspection is None:
                return
            current_task = _inspection_task(inspection, selected.task_id)
            contract = _internal_task_contract(current_task)
            attempt_id = self._id_factory("attempt")
            claim_token = self._id_factory("claim")
            run_id = _required_generated_id(
                self._id_factory("run"),
                _RUN_ID,
                "graph execution run",
            )
            claimed_at = self._clock()
            absolute_deadline = min(
                inspection.job.deadline_at,
                claimed_at
                + timedelta(seconds=current_task.specification.max_wall_time_seconds),
            )
            try:
                attempt = await self._graph_store_call(
                    lambda: self._store.claim_graph_task(
                        self._agent_id,
                        current_task.job_id,
                        current_task.task_id,
                        attempt_id=attempt_id,
                        claim_token=claim_token,
                        run_id=run_id,
                        executor_id=str(contract["capability_id"]),
                        claimed_at=claimed_at,
                        lease_seconds=30,
                        absolute_deadline_at=absolute_deadline,
                        budget_reservations=_attempt_budget_reservations(current_task),
                    )
                )
            except Exception:
                attempt = await self._find_graph_attempt(
                    current_task.job_id,
                    current_task.task_id,
                    attempt_id,
                )
                if attempt is None:
                    raise
            if attempt is None:
                return
            started = await self._graph_store_call(
                lambda: self._store.start_graph_attempt(
                    attempt.agent_id,
                    attempt.job_id,
                    attempt.task_id,
                    attempt.attempt_id,
                    claim_token=attempt.claim_token,
                    fencing_epoch=attempt.fencing_epoch,
                    started_at=self._clock(),
                )
            )
            if started is None:
                return
            await self._checkpoint_graph_start(started)
            heartbeat = asyncio.create_task(
                self._heartbeat_graph_attempt(started),
                name=f"daita-graph-heartbeat:{started.attempt_id}",
            )
            try:
                await self._execute_graph_attempt(inspection, current_task, started)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                current = await self._graph_store_call(
                    lambda: self._store.inspect_graph(
                        self._agent_id, current_task.job_id
                    )
                )
                if (
                    current is not None
                    and _inspection_task(current, current_task.task_id).state
                    is TaskState.SUCCEEDED
                ):
                    return
                await self._graph_store_call(
                    lambda: self._store.fail_graph_attempt(
                        started.agent_id,
                        started.job_id,
                        started.task_id,
                        started.attempt_id,
                        claim_token=started.claim_token,
                        fencing_epoch=started.fencing_epoch,
                        failed_at=self._clock(),
                        retryable=_graph_failure_is_retryable(error),
                        reason_code=_safe_graph_failure_code(error),
                    )
                )
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _find_graph_attempt(
        self,
        job_id: str,
        task_id: str,
        attempt_id: str,
    ) -> TaskAttempt | None:
        inspection = await self._graph_store_call(
            lambda: self._store.inspect_graph(self._agent_id, job_id)
        )
        if inspection is None:
            return None
        return next(
            (
                item
                for item in inspection.attempts
                if item.task_id == task_id and item.attempt_id == attempt_id
            ),
            None,
        )

    async def _checkpoint_graph_start(self, attempt: TaskAttempt) -> None:
        payload = {
            "state": "running",
            "executor_id": attempt.executor_id,
            "reserved_budgets": {
                item.dimension: item.amount for item in attempt.reserved_budgets
            },
        }
        checkpoint = TaskCheckpoint(
            agent_id=attempt.agent_id,
            job_id=attempt.job_id,
            task_id=attempt.task_id,
            attempt_id=attempt.attempt_id,
            checkpoint_id=f"{attempt.attempt_id}:started",
            fencing_epoch=attempt.fencing_epoch,
            ordinal=1,
            milestone="execution_started",
            payload=payload,
            created_at=self._clock(),
            payload_digest=canonical_digest(payload),
        )
        try:
            await self._graph_store_call(
                lambda: self._store.checkpoint_graph_attempt(
                    checkpoint,
                    claim_token=attempt.claim_token,
                )
            )
        except Exception:
            inspection = await self._graph_store_call(
                lambda: self._store.inspect_graph(self._agent_id, attempt.job_id)
            )
            current = (
                None
                if inspection is None
                else next(
                    (
                        item
                        for item in inspection.attempts
                        if item.attempt_id == attempt.attempt_id
                    ),
                    None,
                )
            )
            if (
                current is None
                or checkpoint.checkpoint_id not in current.checkpoint_ids
            ):
                raise

    async def _heartbeat_graph_attempt(self, attempt: TaskAttempt) -> None:
        try:
            while True:
                await asyncio.sleep(self._graph_heartbeat_seconds)
                renewed = await self._graph_store_call(
                    lambda: self._store.heartbeat_graph_attempt(
                        attempt.agent_id,
                        attempt.job_id,
                        attempt.task_id,
                        attempt.attempt_id,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        heartbeat_at=self._clock(),
                    )
                )
                if renewed is None:
                    return
        except asyncio.CancelledError:
            return

    async def _execute_graph_attempt(
        self,
        inspection: GraphInspection,
        task: GraphTask,
        attempt: TaskAttempt,
    ) -> None:
        contract = _internal_task_contract(task)
        reservation = reserved_artifact_id(attempt.attempt_id)
        recovered = await self._artifacts.read_reserved(attempt.run_id, reservation)
        if recovered is not None:
            await self._promote_graph_artifact(
                inspection,
                task,
                attempt,
                recovered.ref,
                recovered.content,
            )
            return
        arguments = self._graph_execution_arguments(inspection, task, contract)
        run = RunInput(
            id=attempt.run_id,
            agent_id=attempt.agent_id,
            message="Execute the exact frozen internal graph task.",
            created_at=attempt.started_at or self._clock(),
            conversation_id=inspection.job.conversation_id,
            source_scope_ids=task.specification.authority.source_ids,
        )
        request = InternalCapabilityRequest(
            run=run,
            call_id=f"graph-call-{attempt.attempt_id}",
            capability_id=str(contract["capability_id"]),
            contract_digest=str(contract["contract_digest"]),
            arguments=arguments,
            sensitivity=task.specification.authority.sensitivity,
            reserved_artifact_id=reservation,
        )
        permit_key = contract.get("source_permit_key")
        try:
            if permit_key is None:
                outcome = await self._runtime.execute_internal(request)
            else:
                if not isinstance(permit_key, str) or not permit_key:
                    raise ValueError("graph source permit key is invalid")
                permit = (
                    await self._required_graph_coordinator().source_resource_permit(
                        permit_key
                    )
                )
                async with permit:
                    outcome = await self._runtime.execute_internal(request)
        except BaseException:
            recovered = await self._artifacts.read_reserved(attempt.run_id, reservation)
            if recovered is None:
                raise
            await self._promote_graph_artifact(
                inspection,
                task,
                attempt,
                recovered.ref,
                recovered.content,
            )
            return
        if outcome.artifact_ref is None:
            raise ValueError("internal graph capability returned no required artifact")
        if outcome.output.kind != contract["output_kind"]:
            raise ValueError("internal graph capability returned the wrong output kind")
        payload = await self._artifacts.read_reserved(attempt.run_id, reservation)
        if payload is None or payload.ref != outcome.artifact_ref:
            raise ValueError("internal graph artifact cannot be authenticated")
        await self._promote_graph_artifact(
            inspection,
            task,
            attempt,
            payload.ref,
            payload.content,
        )

    def _graph_execution_arguments(
        self,
        inspection: GraphInspection,
        task: GraphTask,
        contract: Mapping[str, object],
    ) -> Mapping[str, object]:
        if contract["kind"] == _PROFILE_WORK_KIND:
            arguments = contract.get("arguments")
            if not isinstance(arguments, Mapping):
                raise ValueError("graph work arguments are malformed")
            return arguments
        if task.role is not TaskRole.FINALIZER:
            raise ValueError("only the reserved finalizer can aggregate graph results")
        required_task_ids = tuple(
            sorted(
                edge.upstream_task_id
                for edge in inspection.dependencies
                if edge.downstream_task_id == task.task_id
            )
        )
        accepted = {item.task_id: item for item in inspection.results}
        if set(accepted) - {item.task_id for item in inspection.tasks}:
            raise ValueError("graph accepted results reference unknown tasks")
        work_results: list[dict[str, object]] = []
        for task_id in required_task_ids:
            result = accepted.get(task_id)
            if result is None or len(result.artifact_ids) != 1:
                raise ValueError("the finalizer barrier has an incomplete result")
            raw_refs = result.provenance.get("artifact_refs")
            if not isinstance(raw_refs, tuple) or len(raw_refs) != 1:
                raise ValueError("a required accepted artifact is unauthenticated")
            work_results.append(
                {
                    "task_id": task_id,
                    "result_id": result.result_id,
                    "result_digest": result.result_digest,
                    "artifact_ref": raw_refs[0],
                }
            )
        resource_ids = contract.get("resource_ids")
        sample_rows = contract.get("sample_rows")
        if not isinstance(resource_ids, tuple) or not isinstance(sample_rows, int):
            raise ValueError("graph finalizer contract is malformed")
        return {
            "job_id": inspection.job.job_id,
            "resource_ids": resource_ids,
            "sample_rows": sample_rows,
            "work_results": tuple(work_results),
        }

    async def _promote_graph_artifact(
        self,
        inspection: GraphInspection,
        task: GraphTask,
        attempt: TaskAttempt,
        artifact: ArtifactRef,
        content: bytes,
    ) -> TaskResult:
        contract = _internal_task_contract(task)
        _validate_graph_artifact(
            inspection,
            task,
            attempt,
            artifact,
            content,
            capability_id=str(contract["capability_id"]),
        )
        payload = _profile_result_payload(
            content,
            job_id=inspection.job.job_id,
            expected_resource_ids=(
                task.specification.authority.resource_ids
                if contract["kind"] == _PROFILE_WORK_KIND
                else _contract_resource_ids(contract)
            ),
        )
        completed_at = self._clock()
        result = _graph_task_result(
            task=task,
            attempt=attempt,
            result_id=self._id_factory("result"),
            result_kind=str(contract["output_kind"]),
            payload=payload,
            artifact=artifact,
            completed_at=completed_at,
        )
        if task.role is TaskRole.FINALIZER:
            await self._finalize_graph_result(inspection, attempt, result, artifact)
        else:
            try:
                await self._graph_store_call(
                    lambda: self._store.complete_graph_attempt(
                        result,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        usage=(BudgetAmount("work_units", 1),),
                    )
                )
            except Exception:
                current = await self._graph_store_call(
                    lambda: self._store.inspect_graph(result.agent_id, result.job_id)
                )
                accepted = (
                    None
                    if current is None
                    else next(
                        (
                            item
                            for item in current.results
                            if item.task_id == result.task_id
                        ),
                        None,
                    )
                )
                if accepted != result:
                    raise
        return result

    async def _finalize_graph_result(
        self,
        inspection: GraphInspection,
        attempt: TaskAttempt,
        result: TaskResult,
        artifact: ArtifactRef,
    ) -> None:
        contract = _internal_task_contract(
            _inspection_task(inspection, attempt.task_id)
        )
        requirement = ArtifactRequirement(
            required=True,
            minimum_count=1,
            maximum_count=1,
            allowed_media_types=("application/json",),
            allowed_authorships=(ArtifactAuthorship.EXACT_SOURCE_DATA,),
            allowed_producer_capability_ids=(str(contract["capability_id"]),),
            maximum_artifact_bytes=1 * 1024 * 1024,
            maximum_total_bytes=1 * 1024 * 1024,
            maximum_sensitivity=inspection.job.specification.authority.sensitivity,
        )
        outcome_contract = OutcomeContract(
            require_terminal_conclusion=True,
            artifact_requirements=(requirement,),
            maximum_total_artifact_bytes=1 * 1024 * 1024,
            maximum_effective_sensitivity=(
                inspection.job.specification.authority.sensitivity
            ),
            require_current_run_provenance=True,
            require_exact_source_bindings=True,
        )
        artifact_references = validate_outcome_artifact_references(
            (outcome_artifact_reference(artifact),),
            contract=outcome_contract,
            resulting_run_id=attempt.run_id,
        )
        distribution = self._required_graph_distribution()
        target = distribution.resolve_conversation_inbox(
            inspection.job.conversation_id,
            sensitivity_ceiling=inspection.job.specification.authority.sensitivity,
        )
        plan = distribution.resolve_plan(
            inspection.job.conversation_id,
            destination_id=target.destination_id,
            sensitivity_ceiling=inspection.job.specification.authority.sensitivity,
        )
        if plan.plan_digest != inspection.job.specification.distribution_plan_digest:
            raise ValueError("graph distribution plan changed before finalization")
        delivery = construct_graph_job_delivery(
            delivery_id=self._id_factory("delivery"),
            agent_id=result.agent_id,
            conversation_id=inspection.job.conversation_id,
            job_id=result.job_id,
            target=target,
            result_id=result.result_id,
            result_digest=result.result_digest,
            result_summary=result.summary,
            artifact_references=artifact_references,
            effective_sensitivity=result.sensitivity,
            provenance_digest=canonical_digest(
                {
                    "job_id": result.job_id,
                    "result_digest": result.result_digest,
                    "artifact_ids": result.artifact_ids,
                }
            ),
            observed_at=result.completed_at,
        )
        try:
            await self._graph_store_call(
                lambda: self._store.finalize_graph_attempt(
                    result,
                    delivery,
                    claim_token=attempt.claim_token,
                    fencing_epoch=attempt.fencing_epoch,
                    usage=(BudgetAmount("work_units", 1),),
                )
            )
        except Exception:
            current, deliveries = await asyncio.gather(
                self._graph_store_call(
                    lambda: self._store.inspect_graph(result.agent_id, result.job_id)
                ),
                self._graph_store_call(
                    lambda: self._store.list_draft_graph_deliveries(
                        result.agent_id,
                        job_id=result.job_id,
                        limit=2,
                    )
                ),
            )
            accepted = (
                None
                if current is None
                else next(
                    (
                        item
                        for item in current.results
                        if item.task_id == result.task_id
                    ),
                    None,
                )
            )
            if accepted != result or deliveries != (delivery,):
                raise

    async def _drive(self) -> None:
        try:
            while not self._closing:
                self._wake.clear()
                expired = await self._store.expire_due_jobs(
                    self._agent_id,
                    expired_at=self._clock(),
                )
                for job in expired:
                    self._notify_terminal(job)
                await self._adopt_external_claims()
                await self._claim_available()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._poll_seconds,
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    async def _adopt_external_claims(self) -> None:
        running = await self._store.list_jobs(
            self._agent_id,
            statuses=frozenset({JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}),
        )
        for job in reversed(running):
            if (
                job.job_id in self._workers
                or job.specification.execution_mode
                is not JobExecutionMode.CONNECTED_EXECUTOR
            ):
                continue
            if not _ProcessJobCapacity.acquire():
                return
            self._launch(job)

    async def _claim_available(self) -> None:
        while not self._closing:
            queued = await self._store.list_jobs(
                self._agent_id,
                statuses=frozenset({JobStatus.QUEUED}),
                limit=1,
            )
            if not queued or not _ProcessJobCapacity.acquire():
                return
            claimed: JobRun | None = None
            try:
                claimed = await self._store.claim_next_job(
                    self._agent_id,
                    claim_token=self._id_factory("claim"),
                    execution_run_id=_required_generated_id(
                        self._id_factory("run"),
                        _RUN_ID,
                        "execution run",
                    ),
                    reserved_artifact_id=_required_generated_id(
                        self._id_factory("artifact"),
                        _ARTIFACT_ID,
                        "reserved artifact",
                    ),
                    claimed_at=self._clock(),
                    lease_seconds=300.0,
                )
            except BaseException:
                _ProcessJobCapacity.release()
                raise
            if claimed is None:
                _ProcessJobCapacity.release()
                return
            self._launch(claimed)

    def _launch(self, job: JobRun) -> None:
        cancel_event = asyncio.Event()
        self._cancel_events[job.job_id] = cancel_event
        task = asyncio.create_task(
            self._run_claimed(job, cancel_event),
            name=f"daita-job:{job.job_id}",
        )
        self._workers[job.job_id] = task
        self._worker_modes[job.job_id] = job.specification.execution_mode

        def done(completed: asyncio.Task[None]) -> None:
            self._workers.pop(job.job_id, None)
            self._worker_modes.pop(job.job_id, None)
            self._cancel_events.pop(job.job_id, None)
            _ProcessJobCapacity.release()
            if not completed.cancelled():
                completed.exception()
            self._wake.set()

        task.add_done_callback(done)

    async def _run_claimed(
        self,
        job: JobRun,
        cancel_event: asyncio.Event,
    ) -> None:
        remaining = (job.specification.deadline_at - self._clock()).total_seconds()
        external = (
            job.specification.execution_mode is JobExecutionMode.CONNECTED_EXECUTOR
        )
        if remaining <= 0 and not external:
            await self._finalize(
                job,
                JobAttemptStatus.FAILED,
                failure_code="job_deadline_exceeded",
            )
            return
        timeout = (
            job.specification.max_wall_time_seconds
            if remaining <= 0
            else min(job.specification.max_wall_time_seconds, remaining)
        )
        try:
            async with asyncio.timeout(timeout):
                if job.specification.execution_mode is JobExecutionMode.DAITA:
                    await self._run_daita(job)
                else:
                    await self._run_external(
                        job,
                        cancel_event,
                        allow_start=remaining > 0,
                    )
        except asyncio.CancelledError:
            if self._closing:
                return
            await self._finish_cancelled_daita(job)
        except TimeoutError:
            await self._finalize(
                job,
                JobAttemptStatus.NEEDS_ATTENTION,
                failure_code="job_wall_time_exceeded",
            )
        except (JobError, ArtifactError, ValueError, TypeError) as error:
            await self._finalize(
                job,
                JobAttemptStatus.NEEDS_ATTENTION,
                failure_code=_safe_failure_code(error),
            )
        except Exception:
            await self._finalize(
                job,
                (
                    JobAttemptStatus.NEEDS_ATTENTION
                    if job.specification.execution_mode
                    is JobExecutionMode.CONNECTED_EXECUTOR
                    else JobAttemptStatus.FAILED
                ),
                failure_code=(
                    "external_reconciliation_unavailable"
                    if job.specification.execution_mode
                    is JobExecutionMode.CONNECTED_EXECUTOR
                    else "job_execution_failed"
                ),
            )

    async def _run_daita(self, job: JobRun) -> None:
        current = await self._store.load_job(self._agent_id, job.job_id)
        if current is None:
            return
        if current.desired_state is JobDesiredState.CANCEL:
            await self._finalize(job, JobAttemptStatus.CANCELLED)
            return
        attempt = _claimed_attempt(job)
        specification = job.specification
        arguments = dict(specification.arguments)
        arguments.update(
            {
                "job_id": job.job_id,
                "specification_digest": job.specification_digest,
                "resource_bindings": tuple(
                    _resource_binding_payload(item)
                    for item in specification.resource_bindings
                ),
            }
        )
        source_ids = job.source_ids
        run = RunInput(
            id=attempt.execution_run_id,
            agent_id=job.agent_id,
            message="Execute the exact frozen durable data job.",
            created_at=attempt.claimed_at,
            conversation_id=job.conversation_id,
            source_scope_ids=source_ids,
        )
        outcome = await self._runtime.execute_internal(
            InternalCapabilityRequest(
                run=run,
                call_id=f"job-call-{job.job_id}",
                capability_id=specification.execution_capability_id,
                contract_digest=specification.execution_contract_digest,
                arguments=arguments,
                sensitivity=specification.sensitivity,
                reserved_artifact_id=attempt.reserved_artifact_id,
            )
        )
        current = await self._store.load_job(self._agent_id, job.job_id)
        if current is None:
            return
        result = JobResult(
            result_id=self._id_factory("result"),
            summary=outcome.output.data,
            sensitivity=outcome.output.sensitivity or specification.sensitivity,
            provenance=outcome.output.sensitivity_provenance,
            artifact_refs=(
                () if outcome.artifact_ref is None else (outcome.artifact_ref,)
            ),
            completed_at=self._clock(),
        )
        # A successfully published and validated result wins a cancellation race.
        await self._finalize(job, JobAttemptStatus.SUCCEEDED, result=result)

    async def _finish_cancelled_daita(self, job: JobRun) -> None:
        attempt = _claimed_attempt(job)
        try:
            artifact = await self._artifacts.recover_reserved(
                attempt.execution_run_id,
                attempt.reserved_artifact_id,
            )
        except ArtifactError:
            await self._finalize(
                job,
                JobAttemptStatus.NEEDS_ATTENTION,
                failure_code="job_artifact_reconciliation_failed",
            )
            return
        if artifact is not None:
            result = _recovered_artifact_result(
                job,
                artifact,
                self._clock(),
                self._id_factory,
            )
            await self._finalize(job, JobAttemptStatus.SUCCEEDED, result=result)
        else:
            await self._finalize(job, JobAttemptStatus.CANCELLED)

    async def _run_external(
        self,
        job: JobRun,
        cancel_event: asyncio.Event,
        *,
        allow_start: bool,
    ) -> None:
        attempt = _claimed_attempt(job)
        start_key = f"job-start:{job.job_id}:{job.specification_digest}"
        current = await self._store.load_job(self._agent_id, job.job_id)
        if current is None:
            return
        current_attempt = _claimed_attempt(current)
        start_intent = next(
            (
                item
                for item in current_attempt.external_intents
                if item.kind is ExternalIntentKind.START
            ),
            None,
        )
        external_job_id = None
        if start_intent is None:
            if not allow_start:
                await self._finalize(
                    job,
                    JobAttemptStatus.FAILED,
                    failure_code="job_deadline_exceeded",
                )
                return
            pending = ExternalIntent(
                kind=ExternalIntentKind.START,
                idempotency_key=start_key,
                requested_at=self._clock(),
                disposition=ExternalIntentDisposition.PENDING,
            )
            recorded = await self._store.record_external_intent(
                self._agent_id,
                job.job_id,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                intent=pending,
            )
            if recorded is None:
                return
            try:
                profile = await self._current_external_profile(recorded)
            except (JobError, CapabilityInputError) as error:
                await self._settle_revalidation_rejection(
                    job,
                    attempt,
                    ExternalIntentKind.START,
                    error,
                )
                raise
            try:
                receipt = await profile.start(
                    ExternalStartRequest(
                        job_id=job.job_id,
                        specification_digest=job.specification_digest,
                        idempotency_key=start_key,
                        arguments=_external_start_arguments(job),
                    )
                )
            except asyncio.CancelledError:
                if self._closing:
                    raise
                receipt = None
            except Exception:
                receipt = None
            disposition = (
                ExternalIntentDisposition.OUTCOME_UNKNOWN
                if receipt is None
                else receipt.disposition
            )
            external_job_id = None if receipt is None else receipt.external_job_id
            settled = await self._store.settle_external_intent(
                self._agent_id,
                job.job_id,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                kind=ExternalIntentKind.START,
                disposition=disposition,
                completed_at=self._clock() if receipt is None else receipt.observed_at,
                external_job_id=external_job_id,
                reason_code=(
                    "external_start_response_lost"
                    if receipt is None
                    else receipt.reason_code
                ),
            )
            if settled is None:
                return
            if disposition is ExternalIntentDisposition.REJECTED:
                await self._finalize(
                    job,
                    JobAttemptStatus.FAILED,
                    failure_code="external_start_rejected",
                )
                return
        else:
            external_job_id = start_intent.external_job_id
            if start_intent.disposition is ExternalIntentDisposition.REJECTED:
                await self._finalize(
                    job,
                    JobAttemptStatus.FAILED,
                    failure_code="external_start_rejected",
                )
                return

        current = await self._store.load_job(self._agent_id, job.job_id)
        if current is None:
            return
        first_sequence = len(_claimed_attempt(current).external_observations) + 1
        for sequence in range(first_sequence, MAX_JOB_EXTERNAL_OBSERVATIONS + 1):
            current = await self._store.load_job(self._agent_id, job.job_id)
            if current is None:
                return
            if current.desired_state is JobDesiredState.CANCEL:
                external_job_id = await self._request_external_cancel(
                    current,
                    external_job_id,
                )
            profile = await self._current_external_profile(
                current,
                revalidate_data_scope=False,
            )
            status = await profile.status(
                ExternalStatusRequest(
                    job_id=job.job_id,
                    specification_digest=job.specification_digest,
                    idempotency_key=start_key,
                    external_job_id=external_job_id,
                )
            )
            external_job_id = status.external_job_id
            observation = ExternalObservation(
                sequence=sequence,
                observed_at=status.observed_at,
                status=status.status,
                observation_digest=_observation_digest(status.observation),
                external_job_id=status.external_job_id,
            )
            recorded = await self._store.record_external_observation(
                self._agent_id,
                job.job_id,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                observation=observation,
            )
            if recorded is None:
                return
            if status.status is ExternalObservedStatus.SUCCEEDED:
                profile = await self._current_external_profile(recorded)
                payload = await profile.read_result(
                    ExternalResultRequest(
                        job_id=job.job_id,
                        specification_digest=job.specification_digest,
                        external_job_id=status.external_job_id,
                    )
                )
                _validate_external_result(job, payload.sensitivity)
                result = JobResult(
                    result_id=self._id_factory("result"),
                    summary=payload.summary,
                    sensitivity=payload.sensitivity,
                    provenance={
                        "authority": "connected_executor",
                        "profile_id": _external_binding(job).profile_id,
                        "external_job_id": status.external_job_id,
                        "external": payload.provenance,
                    },
                    artifact_refs=(),
                    completed_at=payload.observed_at,
                )
                await self._finalize(job, JobAttemptStatus.SUCCEEDED, result=result)
                return
            if status.status is ExternalObservedStatus.FAILED:
                await self._finalize(
                    job,
                    JobAttemptStatus.FAILED,
                    failure_code="external_job_failed",
                )
                return
            if status.status is ExternalObservedStatus.CANCELLED:
                await self._finalize(job, JobAttemptStatus.CANCELLED)
                return
            cancel_event.clear()
            try:
                await asyncio.wait_for(
                    cancel_event.wait(),
                    timeout=self._poll_seconds,
                )
            except TimeoutError:
                pass
        await self._finalize(
            job,
            JobAttemptStatus.NEEDS_ATTENTION,
            failure_code="external_observation_limit_exceeded",
        )

    async def _request_external_cancel(
        self,
        job: JobRun,
        external_job_id: str | None,
    ) -> str | None:
        attempt = _claimed_attempt(job)
        existing = next(
            (
                item
                for item in attempt.external_intents
                if item.kind is ExternalIntentKind.CANCEL
            ),
            None,
        )
        if existing is not None or external_job_id is None:
            return external_job_id
        cancel_key = f"job-cancel:{job.job_id}:{job.specification_digest}"
        pending = ExternalIntent(
            kind=ExternalIntentKind.CANCEL,
            idempotency_key=cancel_key,
            requested_at=self._clock(),
            disposition=ExternalIntentDisposition.PENDING,
        )
        recorded = await self._store.record_external_intent(
            self._agent_id,
            job.job_id,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            intent=pending,
        )
        if recorded is None:
            return external_job_id
        try:
            profile = await self._current_external_profile(recorded)
        except (JobError, CapabilityInputError) as error:
            await self._settle_revalidation_rejection(
                job,
                attempt,
                ExternalIntentKind.CANCEL,
                error,
            )
            raise
        try:
            receipt = await profile.cancel(
                ExternalCancelRequest(
                    job_id=job.job_id,
                    external_job_id=external_job_id,
                    specification_digest=job.specification_digest,
                    idempotency_key=cancel_key,
                )
            )
        except asyncio.CancelledError:
            if self._closing:
                raise
            receipt = None
        except Exception:
            receipt = None
        await self._store.settle_external_intent(
            self._agent_id,
            job.job_id,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            kind=ExternalIntentKind.CANCEL,
            disposition=(
                ExternalIntentDisposition.OUTCOME_UNKNOWN
                if receipt is None
                else receipt.disposition
            ),
            completed_at=self._clock() if receipt is None else receipt.observed_at,
            reason_code=(
                "external_cancel_response_lost"
                if receipt is None
                else receipt.reason_code
            ),
        )
        return external_job_id

    async def _current_external_profile(
        self,
        job: JobRun,
        *,
        revalidate_data_scope: bool = True,
    ) -> ConnectedJobProfile:
        if revalidate_data_scope:
            await self._revalidate_external(job)
        return self._owner.connected_profile_for(job)

    async def _settle_revalidation_rejection(
        self,
        job: JobRun,
        attempt: JobAttempt,
        kind: ExternalIntentKind,
        error: JobError | CapabilityInputError,
    ) -> None:
        await self._store.settle_external_intent(
            self._agent_id,
            job.job_id,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            kind=kind,
            disposition=ExternalIntentDisposition.REJECTED,
            completed_at=self._clock(),
            reason_code=error.code,
        )

    async def _finalize(
        self,
        job: JobRun,
        status: JobAttemptStatus,
        *,
        result: JobResult | None = None,
        failure_code: str | None = None,
    ) -> JobRun | None:
        attempt = _claimed_attempt(job)
        finalized = await self._store.finalize_job_attempt(
            self._agent_id,
            job.job_id,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            attempt_status=status,
            completed_at=self._clock(),
            result=result,
            failure_code=failure_code,
        )
        if finalized is not None and finalized.terminal:
            self._notify_terminal(finalized)
        return finalized

    def _notify_terminal(self, job: JobRun) -> None:
        if self._on_terminal is None:
            return
        try:
            self._on_terminal(job)
        except Exception:
            return


def _claimed_attempt(job: JobRun) -> JobAttempt:
    attempt = job.current_attempt
    if attempt is None or attempt.status is not JobAttemptStatus.CLAIMED:
        raise ValueError("job does not carry one exact claimed attempt")
    return attempt


def _inspection_task(inspection: GraphInspection, task_id: str) -> GraphTask:
    task = next((item for item in inspection.tasks if item.task_id == task_id), None)
    if task is None:
        raise ValueError("graph task is absent from its inspection")
    return task


def _internal_task_contract(task: GraphTask) -> Mapping[str, object]:
    if task.execution_kind is not TaskExecutionKind.INTERNAL_CAPABILITY:
        raise ValueError("the Phase 3 graph slice admits only internal tasks")
    contract = task.specification.expected_result_contract
    kind = contract.get("kind")
    capability_id = contract.get("capability_id")
    contract_digest = contract.get("contract_digest")
    output_kind = contract.get("output_kind")
    if kind not in {_PROFILE_WORK_KIND, _PROFILE_FINALIZER_KIND}:
        raise ValueError("the graph task kind is outside the static slice")
    if any(
        not isinstance(value, str) or not value
        for value in (capability_id, contract_digest, output_kind)
    ):
        raise ValueError("the internal graph task contract is malformed")
    assert isinstance(capability_id, str)
    assert isinstance(contract_digest, str)
    if kind == _PROFILE_FINALIZER_KIND and task.role is not TaskRole.FINALIZER:
        raise ValueError("the profile finalizer contract is not reserved")
    if kind == _PROFILE_WORK_KIND and task.role is TaskRole.FINALIZER:
        raise ValueError("the reserved finalizer cannot execute work")
    if capability_id not in task.specification.authority.capability_ids:
        raise ValueError("the task capability exceeds its immutable authority")
    binding = task.specification.authority.contract_bindings.get(capability_id)
    if binding != contract_digest:
        raise ValueError("the task capability digest differs from its authority")
    return contract


def _attempt_budget_reservations(task: GraphTask) -> tuple[BudgetAmount, ...]:
    raw = task.specification.expected_result_contract.get("attempt_budgets")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("the internal task attempt budget is malformed")
    task_limits = {item.dimension: item.amount for item in task.specification.budgets}
    reservations: list[BudgetAmount] = []
    for dimension, amount in raw.items():
        if not isinstance(dimension, str) or not isinstance(amount, int):
            raise ValueError("the internal task attempt budget is malformed")
        if dimension not in task_limits or amount > task_limits[dimension]:
            raise ValueError("the attempt budget exceeds the immutable task budget")
        reservations.append(BudgetAmount(dimension, amount))
    return tuple(sorted(reservations))


def _contract_resource_ids(contract: Mapping[str, object]) -> tuple[str, ...]:
    raw = contract.get("resource_ids")
    if not isinstance(raw, tuple) or any(not isinstance(item, str) for item in raw):
        raise ValueError("the graph finalizer resource contract is malformed")
    return raw


def _validate_graph_artifact(
    inspection: GraphInspection,
    task: GraphTask,
    attempt: TaskAttempt,
    artifact: ArtifactRef,
    content: bytes,
    *,
    capability_id: str,
) -> None:
    if (
        artifact.run_id != attempt.run_id
        or artifact.conversation_id != inspection.job.conversation_id
        or artifact.call_id != f"graph-call-{attempt.attempt_id}"
        or artifact.capability_id != capability_id
        or artifact.artifact_id != reserved_artifact_id(attempt.attempt_id)
        or artifact.media_type != "application/json"
        or artifact.provenance.authorship is not ArtifactAuthorship.EXACT_SOURCE_DATA
        or artifact.byte_size != len(content)
    ):
        raise ValueError("the internal graph artifact differs from its attempt")
    expected_resources = set(task.specification.authority.resource_ids)
    actual_resources = {
        item.resource_id for item in artifact.provenance.resource_bindings
    }
    if expected_resources and actual_resources != expected_resources:
        raise ValueError("the internal graph artifact differs from its task scope")
    if task.role is TaskRole.FINALIZER and actual_resources != set(
        inspection.job.specification.authority.resource_ids
    ):
        raise ValueError("the finalizer artifact lacks exact root resource bindings")


def _profile_result_payload(
    content: bytes,
    *,
    job_id: str,
    expected_resource_ids: tuple[str, ...],
) -> Mapping[str, object]:
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("the profile artifact is not valid JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("kind") != "data_profile"
        or document.get("job_id") != job_id
        or not isinstance(document.get("resources"), list)
    ):
        raise ValueError("the profile artifact has the wrong result contract")
    resources = document["resources"]
    resource_ids: list[str] = []
    sampled_rows = 0
    truncated_resources = 0
    for resource in resources:
        if not isinstance(resource, dict):
            raise ValueError("the profile artifact contains a malformed resource")
        resource_id = resource.get("resource_id")
        sampled = resource.get("sampled_rows")
        if (
            not isinstance(resource_id, str)
            or not isinstance(sampled, int)
            or isinstance(sampled, bool)
        ):
            raise ValueError("the profile artifact contains invalid result evidence")
        resource_ids.append(resource_id)
        sampled_rows += sampled
        if resource.get("truncated") is True:
            truncated_resources += 1
    if (
        document.get("resource_count") != len(resources)
        or tuple(sorted(resource_ids)) != tuple(sorted(expected_resource_ids))
        or len(resource_ids) != len(set(resource_ids))
    ):
        raise ValueError("the profile artifact does not cover its exact resources")
    return {
        "job_id": job_id,
        "profiled_resources": len(resources),
        "sampled_rows": sampled_rows,
        "truncated_resources": truncated_resources,
    }


def _graph_task_result(
    *,
    task: GraphTask,
    attempt: TaskAttempt,
    result_id: str,
    result_kind: str,
    payload: Mapping[str, object],
    artifact: ArtifactRef,
    completed_at: datetime,
) -> TaskResult:
    schema_digest = canonical_digest(
        {
            "result_kind": result_kind,
            "task_contract": task.specification.expected_result_contract,
        }
    )
    sensitivity = ModelSensitivity(artifact.sensitivity.value)
    provenance = {
        "authority": "authenticated_internal_capability_artifact",
        "task_spec_digest": task.task_spec_digest,
        "artifact_refs": (artifact_ref_to_mapping(artifact),),
    }
    verification = {
        "artifact_sha256": artifact.sha256,
        "capability_id": artifact.capability_id,
        "artifact_authenticated": True,
    }
    digest_material = {
        "agent_id": task.agent_id,
        "job_id": task.job_id,
        "task_id": task.task_id,
        "result_id": result_id,
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "result_kind": result_kind,
        "schema_digest": schema_digest,
        "payload": payload,
        "summary": (
            f"Profiled {payload['profiled_resources']} exact resource(s) with "
            f"{payload['sampled_rows']} sampled row(s)."
        ),
        "sensitivity": sensitivity.value,
        "provenance": provenance,
        "artifact_ids": (artifact.artifact_id,),
        "effect_receipt_ids": (),
        "verification": verification,
        "residual_risk": None,
        "downstream_constraints": {},
        "completed_at": completed_at.isoformat(),
    }
    return TaskResult(
        agent_id=task.agent_id,
        job_id=task.job_id,
        task_id=task.task_id,
        result_id=result_id,
        attempt_id=attempt.attempt_id,
        run_id=attempt.run_id,
        result_kind=result_kind,
        schema_digest=schema_digest,
        payload=payload,
        summary=str(digest_material["summary"]),
        sensitivity=sensitivity,
        provenance=provenance,
        artifact_ids=(artifact.artifact_id,),
        effect_receipt_ids=(),
        verification=verification,
        residual_risk=None,
        downstream_constraints={},
        completed_at=completed_at,
        result_digest=canonical_digest(digest_material),
    )


def _safe_graph_failure_code(error: BaseException) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9._:/-]{1,256}", code):
        return code
    if isinstance(error, ArtifactError):
        return error.code
    if isinstance(error, CapabilityInputError):
        return error.code
    return "graph_internal_execution_failed"


def _graph_failure_is_retryable(error: BaseException) -> bool:
    return not isinstance(
        error,
        (ArtifactError, CapabilityInputError, TypeError, ValueError),
    )


def _resource_binding_payload(value) -> dict[str, object]:
    return {
        "source_id": value.source_id,
        "source_revision": value.source_revision,
        "resource_id": value.resource_id,
        "resource_revision": value.resource_revision,
        "adapter_id": value.adapter_id,
        "sensitivity": value.sensitivity.value,
    }


def _external_start_arguments(job: JobRun) -> dict[str, object]:
    return {
        "job_kind": job.specification.job_kind,
        "arguments": job.specification.arguments,
        "resource_bindings": tuple(
            _resource_binding_payload(item)
            for item in job.specification.resource_bindings
        ),
    }


def _observation_digest(value: Mapping[str, object]) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _recovered_artifact_result(
    job: JobRun,
    artifact: ArtifactRef,
    completed_at: datetime,
    id_factory: Callable[[str], str],
) -> JobResult:
    return JobResult(
        result_id=id_factory("result"),
        summary={"recovered_artifact_id": artifact.artifact_id},
        sensitivity=ModelSensitivity(artifact.sensitivity.value),
        provenance={
            "authority": "reserved_artifact_reconciliation",
            "artifact_sha256": artifact.sha256,
            "specification_digest": job.specification_digest,
        },
        artifact_refs=(artifact,),
        completed_at=completed_at,
    )


def _required_generated_id(value: str, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"job {name} identity factory returned an invalid identity")
    return value


def _safe_failure_code(error: BaseException) -> str:
    if isinstance(error, JobError):
        return error.code
    if isinstance(error, ArtifactError):
        return error.code
    if isinstance(error, CapabilityInputError):
        return error.code
    return "job_contract_revalidation_failed"


def _sensitivity_rank(value: ModelSensitivity) -> int:
    return {
        ModelSensitivity.PUBLIC: 0,
        ModelSensitivity.INTERNAL: 1,
        ModelSensitivity.CONFIDENTIAL: 2,
        ModelSensitivity.RESTRICTED: 3,
    }[value]


def _validate_external_result(job: JobRun, sensitivity: ModelSensitivity) -> None:
    binding = _external_binding(job)
    if _sensitivity_rank(sensitivity) < _sensitivity_rank(
        job.specification.sensitivity
    ) or _sensitivity_rank(sensitivity) > _sensitivity_rank(
        binding.maximum_sensitivity
    ):
        raise ValueError("external result sensitivity is outside its frozen ceiling")


def _external_binding(job: JobRun):
    binding = job.specification.external_executor
    if binding is None:
        raise ValueError("external job has no exact executor binding")
    return binding


__all__ = ["JobSupervisor"]
