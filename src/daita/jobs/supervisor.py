"""Claim, execute, fence, recover, and reconcile durable job attempts within limits."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from hashlib import sha256

from .._json import canonical_json
from ..adapters.job_profiles import (
    ConnectedJobProfile,
    ExternalCancelRequest,
    ExternalResultRequest,
    ExternalStartRequest,
    ExternalStatusRequest,
)
from ..artifacts.models import ArtifactError, ArtifactRef
from ..artifacts.store import AgentHomeArtifactStore
from ..capabilities import CapabilityInputError
from ..capability_runtime import CapabilityRuntime, InternalCapabilityRequest
from ..llm.models import ModelSensitivity
from ..loop.models import RunInput
from ..storage.sqlite import SQLiteStateStore
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
        self._poll_seconds = float(poll_seconds)
        self._wake = asyncio.Event()
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._worker_modes: dict[str, JobExecutionMode] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._driver: asyncio.Task[None] | None = None
        self._closing = False

    async def start(self) -> None:
        if self._driver is not None:
            raise RuntimeError("job supervisor is already started")
        await self._recover_daita_claims()
        self._driver = asyncio.create_task(
            self._drive(),
            name=f"daita-job-supervisor:{self._agent_id}",
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

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        current_loop = asyncio.get_running_loop()
        all_tasks = tuple(
            item for item in (self._driver, *self._workers.values()) if item is not None
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
        self._workers.clear()
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
            source_id=source_ids[0] if len(source_ids) == 1 else None,
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
