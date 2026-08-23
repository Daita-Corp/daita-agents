"""Admit durable jobs and own their lifecycle transitions and public projections."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from ..adapters.job_profiles import ConnectedJobProfile
from ..errors import DaitaError, ErrorRetryability
from ..llm.models import ModelSensitivity
from ..loop.models import RunInput
from .models import (
    ConnectedExecutorBinding,
    JobDesiredState,
    JobExecutionMode,
    JobInspection,
    JobResultView,
    JobRun,
    JobSpecification,
    JobStatus,
    JobSummary,
    MAX_JOB_DEADLINE_SECONDS,
    MAX_JOB_LIST_PAGE_SIZE,
    job_inspection,
    job_result_view,
    job_summary,
)


class JobError(DaitaError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(
            message,
            error_code=code,
            retryability=ErrorRetryability.PERMANENT,
        )


class JobStore(Protocol):
    async def admit_job(self, job: JobRun) -> JobRun: ...

    async def load_job(self, agent_id: str, job_id: str) -> JobRun | None: ...

    async def list_jobs(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = None,
        statuses: frozenset[JobStatus] = frozenset(),
        limit: int = MAX_JOB_LIST_PAGE_SIZE,
    ) -> tuple[JobRun, ...]: ...

    async def request_job_cancel(
        self,
        agent_id: str,
        job_id: str,
        *,
        requested_at: datetime,
    ) -> JobRun | None: ...


class JobOwner:
    """Admit frozen jobs and expose only bounded owner-scoped lifecycle views."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: JobStore,
        connected_profiles: tuple[ConnectedJobProfile, ...] = (),
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("job owner agent_id must be non-empty text")
        for method in (
            "admit_job",
            "load_job",
            "list_jobs",
            "request_job_cancel",
        ):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"job store must provide {method}")
        if not callable(clock) or not callable(id_factory):
            raise TypeError("job owner clock and id_factory must be callable")
        profiles: dict[str, ConnectedJobProfile] = {}
        for profile in tuple(connected_profiles):
            profile_id = profile.profile_id
            if not isinstance(profile_id, str) or not profile_id:
                raise ValueError("connected profile ID must be non-empty text")
            if profile_id in profiles:
                raise ValueError(f"duplicate connected job profile: {profile_id}")
            state = profile.current_state()
            if state.binding.profile_id != profile_id:
                raise ValueError(
                    "connected job profile identity changed at composition"
                )
            profiles[profile_id] = profile
        self.agent_id = agent_id
        self._store = store
        self._profiles = profiles
        self._clock = clock
        self._id_factory = id_factory
        self._wake: Callable[[str | None], None] | None = None

    def bind_wake(self, wake: Callable[[str | None], None]) -> None:
        if self._wake is not None:
            raise RuntimeError("job owner wake callback is already bound")
        if not callable(wake):
            raise TypeError("job wake callback must be callable")
        self._wake = wake

    def resolve_external_selection(
        self,
        profile_id: str,
        *,
        job_kind: str,
        sensitivity: ModelSensitivity,
    ) -> ConnectedExecutorBinding:
        if not isinstance(profile_id, str) or not profile_id:
            raise JobError(
                "external_executor_not_connected",
                "The explicitly selected connected executor is unavailable.",
            )
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise JobError(
                "external_executor_not_connected",
                "The explicitly selected connected executor is unavailable.",
            )
        state = profile.current_state()
        if (
            state.binding.profile_id != profile_id
            or not state.active
            or job_kind not in state.supported_job_kinds
        ):
            raise JobError(
                "external_executor_not_admitted",
                "The explicitly selected connected executor is not currently admitted for this job kind.",
            )
        if _sensitivity_rank(sensitivity) > _sensitivity_rank(
            state.binding.maximum_sensitivity
        ):
            raise JobError(
                "external_executor_sensitivity_blocked",
                "The explicitly selected connected executor cannot receive this job sensitivity.",
            )
        return state.binding

    def connected_profile_for(self, job: JobRun) -> ConnectedJobProfile:
        binding = job.specification.external_executor
        if (
            job.specification.execution_mode is not JobExecutionMode.CONNECTED_EXECUTOR
            or binding is None
        ):
            raise JobError(
                "external_executor_not_selected",
                "This job does not have an explicitly selected connected executor.",
            )
        profile = self._profiles.get(binding.profile_id)
        if profile is None:
            raise JobError(
                "external_executor_revoked",
                "The connected executor was revoked before execution.",
            )
        state = profile.current_state()
        if not state.active or state.binding != binding:
            raise JobError(
                "external_executor_drifted",
                "The exact connected executor identity or contract changed.",
            )
        if job.specification.job_kind not in state.supported_job_kinds:
            raise JobError(
                "external_executor_unsupported",
                "The exact connected executor no longer supports this job kind.",
            )
        if _sensitivity_rank(job.specification.sensitivity) > _sensitivity_rank(
            state.binding.maximum_sensitivity
        ):
            raise JobError(
                "external_executor_sensitivity_blocked",
                "The exact connected executor no longer admits this job sensitivity.",
            )
        return profile

    async def admit(
        self,
        *,
        run: RunInput,
        call_id: str,
        specification: JobSpecification,
    ) -> JobRun:
        if run.agent_id != self.agent_id:
            raise JobError("job_owner_mismatch", "The job owner identity changed.")
        if run.conversation_id is None:
            raise JobError(
                "job_conversation_required",
                "A durable job requires one exact conversation identity.",
            )
        now = self._clock()
        if specification.deadline_at <= now:
            raise JobError(
                "job_deadline_expired",
                "The requested job deadline has already expired.",
            )
        if (specification.deadline_at - now).total_seconds() > MAX_JOB_DEADLINE_SECONDS:
            raise JobError(
                "job_deadline_limit_exceeded",
                "The requested job deadline exceeds the fixed admission horizon.",
            )
        job = JobRun(
            job_id=self._id_factory("job"),
            agent_id=self.agent_id,
            conversation_id=run.conversation_id,
            origin_run_id=run.id,
            origin_call_id=call_id,
            specification=specification,
            specification_digest=specification.digest,
            status=JobStatus.QUEUED,
            desired_state=JobDesiredState.RUN,
            created_at=now,
            updated_at=now,
        )
        try:
            stored = await self._store.admit_job(job)
        except ValueError as error:
            code = str(error)
            if not code.startswith("job_"):
                code = "job_admission_failed"
            raise JobError(
                code,
                "The durable job could not be admitted within its fixed limits.",
            ) from error
        self._notify(None)
        return stored

    async def list(
        self,
        *,
        origin_conversation_id: str | None = None,
        statuses: frozenset[JobStatus] = frozenset(),
        limit: int = MAX_JOB_LIST_PAGE_SIZE,
    ) -> tuple[JobSummary, ...]:
        jobs = await self._store.list_jobs(
            self.agent_id,
            conversation_id=origin_conversation_id,
            statuses=statuses,
            limit=limit,
        )
        return tuple(job_summary(item) for item in jobs)

    async def inspect(self, job_id: str) -> JobInspection | None:
        job = await self._load_owned(job_id)
        return None if job is None else job_inspection(job)

    async def read_result(self, job_id: str) -> JobResultView | None:
        job = await self._load_owned(job_id)
        return None if job is None else job_result_view(job)

    async def cancel(self, job_id: str) -> JobInspection | None:
        current = await self._load_owned(job_id)
        if current is None:
            return None
        updated = await self._store.request_job_cancel(
            self.agent_id,
            job_id,
            requested_at=self._clock(),
        )
        if updated is None:
            return None
        self._notify(job_id)
        return job_inspection(updated)

    async def _load_owned(self, job_id: str) -> JobRun | None:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be non-empty text")
        return await self._store.load_job(self.agent_id, job_id)

    def _notify(self, job_id: str | None) -> None:
        if self._wake is not None:
            self._wake(job_id)


def _sensitivity_rank(value: ModelSensitivity) -> int:
    return {
        ModelSensitivity.PUBLIC: 0,
        ModelSensitivity.INTERNAL: 1,
        ModelSensitivity.CONFIDENTIAL: 2,
        ModelSensitivity.RESTRICTED: 3,
    }[value]


__all__ = ["JobError", "JobOwner", "JobStore"]
