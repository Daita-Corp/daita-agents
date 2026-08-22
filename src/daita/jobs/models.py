"""Bounded records for the one Stage B durable-job lifecycle."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256

from .._json import FrozenJsonObject, canonical_json
from ..artifacts.models import ArtifactRef
from ..llm.models import ModelSensitivity

MAX_JOBS_PER_AGENT = 256
# The active bound must be reachable through the only two nonterminal capacity
# states admitted by Stage B: queued and running.
MAX_ACTIVE_JOBS_PER_AGENT = 52
MAX_QUEUED_JOBS_PER_AGENT = 48
MAX_RUNNING_JOBS_PER_AGENT = 4
MAX_RUNNING_JOBS_GLOBAL = 8
MAX_RUNNING_JOBS_PER_SOURCE = 2
MAX_JOB_SPECIFICATION_BYTES = 64 * 1024
MAX_JOB_SPECIFICATION_DEPTH = 12
MAX_JOB_RESOURCE_BINDINGS = 16
MAX_JOB_ATTEMPTS = 3
MAX_JOB_RENEWALS = 4
MAX_JOB_EXTERNAL_OBSERVATIONS = 32
MAX_JOB_WALL_TIME_SECONDS = 300.0
MAX_JOB_DEADLINE_SECONDS = 3_600.0
MAX_JOB_LIST_PAGE_SIZE = 50
MAX_JOB_SUMMARY_BYTES = 8 * 1024
MAX_JOB_INLINE_RESULT_BYTES = 64 * 1024
MAX_JOB_RESULT_DEPTH = 12
MAX_JOB_ARTIFACTS = 1
MAX_EXTERNAL_REQUEST_BYTES = 64 * 1024
MAX_EXTERNAL_RESPONSE_BYTES = 256 * 1024

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _text(value: str, name: str, *, maximum: int = 2_048) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or any(character in "\r\n\x00" for character in value)
    ):
        raise ValueError(f"{name} must be bounded non-empty single-line text")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 digest")


def _utc(value: datetime, name: str) -> None:
    offset = value.utcoffset() if isinstance(value, datetime) else None
    if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _optional_utc(value: datetime | None, name: str) -> None:
    if value is not None:
        _utc(value, name)


def _json_depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (tuple, list)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _bounded_json(
    value: Mapping[str, object],
    *,
    name: str,
    maximum_bytes: int,
    maximum_depth: int,
) -> FrozenJsonObject:
    frozen = FrozenJsonObject.from_mapping(value)
    if len(canonical_json(frozen).encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte bound")
    if _json_depth(frozen) > maximum_depth:
        raise ValueError(f"{name} exceeds its depth bound")
    return frozen


class JobExecutionMode(str, Enum):
    DAITA = "daita"
    CONNECTED_EXECUTOR = "connected_executor"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_ATTENTION = "needs_attention"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.NEEDS_ATTENTION,
    }
)


class JobDesiredState(str, Enum):
    RUN = "run"
    CANCEL = "cancel"


class JobAttemptStatus(str, Enum):
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_ATTENTION = "needs_attention"
    FENCED = "fenced"


class ExternalIntentKind(str, Enum):
    START = "start"
    CANCEL = "cancel"


class ExternalIntentDisposition(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    OUTCOME_UNKNOWN = "outcome_unknown"


class ExternalObservedStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class JobResourceBinding:
    source_id: str
    source_revision: str
    resource_id: str
    resource_revision: str
    adapter_id: str
    sensitivity: ModelSensitivity

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "job source_id"),
            (self.source_revision, "job source_revision"),
            (self.resource_id, "job resource_id"),
            (self.adapter_id, "job adapter_id"),
        ):
            _text(value, name, maximum=1_024)
        _digest(self.resource_revision, "job resource_revision")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("job resource sensitivity must be ModelSensitivity")


@dataclass(frozen=True, slots=True)
class ConnectedExecutorBinding:
    profile_id: str
    binding_id: str
    execution_identity: str
    contract_digest: str
    revision: int
    maximum_sensitivity: ModelSensitivity

    def __post_init__(self) -> None:
        for value, name in (
            (self.profile_id, "connected executor profile_id"),
            (self.binding_id, "connected executor binding_id"),
            (self.execution_identity, "connected executor identity"),
        ):
            _text(value, name, maximum=1_024)
        _digest(self.contract_digest, "connected executor contract_digest")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("connected executor revision must be positive")
        if not isinstance(self.maximum_sensitivity, ModelSensitivity):
            raise TypeError(
                "connected executor maximum_sensitivity must be ModelSensitivity"
            )


@dataclass(frozen=True, slots=True)
class JobSpecification:
    job_kind: str
    arguments: Mapping[str, object]
    resource_bindings: tuple[JobResourceBinding, ...]
    execution_capability_id: str
    execution_contract_digest: str
    execution_mode: JobExecutionMode
    sensitivity: ModelSensitivity
    deadline_at: datetime
    max_wall_time_seconds: float
    external_executor: ConnectedExecutorBinding | None = None

    def __post_init__(self) -> None:
        _text(self.job_kind, "job kind", maximum=128)
        _text(
            self.execution_capability_id,
            "job execution capability_id",
            maximum=256,
        )
        _digest(
            self.execution_contract_digest,
            "job execution contract_digest",
        )
        bindings = tuple(self.resource_bindings)
        if not 1 <= len(bindings) <= MAX_JOB_RESOURCE_BINDINGS:
            raise ValueError("job resource bindings exceed their bound")
        if any(not isinstance(item, JobResourceBinding) for item in bindings):
            raise TypeError("job resource bindings must be JobResourceBinding records")
        if len({item.resource_id for item in bindings}) != len(bindings):
            raise ValueError("job resource bindings cannot duplicate resources")
        if bindings != tuple(
            sorted(bindings, key=lambda item: (item.source_id, item.resource_id))
        ):
            raise ValueError("job resource bindings must be sorted")
        if not isinstance(self.execution_mode, JobExecutionMode):
            raise TypeError("job execution_mode must be JobExecutionMode")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("job sensitivity must be ModelSensitivity")
        _utc(self.deadline_at, "job deadline_at")
        if (
            not isinstance(self.max_wall_time_seconds, (int, float))
            or isinstance(self.max_wall_time_seconds, bool)
            or not math.isfinite(float(self.max_wall_time_seconds))
            or not 0 < float(self.max_wall_time_seconds) <= MAX_JOB_WALL_TIME_SECONDS
        ):
            raise ValueError("job max_wall_time_seconds is outside its bound")
        external = self.external_executor
        if self.execution_mode is JobExecutionMode.DAITA and external is not None:
            raise ValueError("Daita-mode job cannot contain an external executor")
        if (
            self.execution_mode is JobExecutionMode.CONNECTED_EXECUTOR
            and not isinstance(external, ConnectedExecutorBinding)
        ):
            raise ValueError("connected-executor job requires its exact binding")
        arguments = _bounded_json(
            self.arguments,
            name="job specification",
            maximum_bytes=MAX_JOB_SPECIFICATION_BYTES,
            maximum_depth=MAX_JOB_SPECIFICATION_DEPTH,
        )
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "resource_bindings", bindings)
        object.__setattr__(
            self,
            "max_wall_time_seconds",
            float(self.max_wall_time_seconds),
        )

    @property
    def digest(self) -> str:
        return (
            "sha256:"
            + sha256(canonical_json(self.digest_material()).encode("utf-8")).hexdigest()
        )

    def digest_material(self) -> dict[str, object]:
        return {
            "job_kind": self.job_kind,
            "arguments": self.arguments,
            "resource_bindings": [
                {
                    "source_id": item.source_id,
                    "source_revision": item.source_revision,
                    "resource_id": item.resource_id,
                    "resource_revision": item.resource_revision,
                    "adapter_id": item.adapter_id,
                    "sensitivity": item.sensitivity.value,
                }
                for item in self.resource_bindings
            ],
            "execution_capability_id": self.execution_capability_id,
            "execution_contract_digest": self.execution_contract_digest,
            "execution_mode": self.execution_mode.value,
            "sensitivity": self.sensitivity.value,
            "deadline_at": self.deadline_at.isoformat(),
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "external_executor": (
                None
                if self.external_executor is None
                else {
                    "profile_id": self.external_executor.profile_id,
                    "binding_id": self.external_executor.binding_id,
                    "execution_identity": (self.external_executor.execution_identity),
                    "contract_digest": self.external_executor.contract_digest,
                    "revision": self.external_executor.revision,
                    "maximum_sensitivity": (
                        self.external_executor.maximum_sensitivity.value
                    ),
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class ExternalIntent:
    kind: ExternalIntentKind
    idempotency_key: str
    requested_at: datetime
    disposition: ExternalIntentDisposition
    completed_at: datetime | None = None
    external_job_id: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ExternalIntentKind):
            raise TypeError("external intent kind must be ExternalIntentKind")
        _text(self.idempotency_key, "external idempotency key", maximum=256)
        _utc(self.requested_at, "external intent requested_at")
        if not isinstance(self.disposition, ExternalIntentDisposition):
            raise TypeError(
                "external intent disposition must be ExternalIntentDisposition"
            )
        _optional_utc(self.completed_at, "external intent completed_at")
        if self.external_job_id is not None:
            _text(self.external_job_id, "external job_id", maximum=1_024)
        if self.reason_code is not None:
            _text(self.reason_code, "external intent reason", maximum=128)
        if self.disposition is ExternalIntentDisposition.PENDING:
            if self.completed_at is not None:
                raise ValueError("pending external intent cannot be completed")
        elif self.completed_at is None:
            raise ValueError("settled external intent requires completed_at")


@dataclass(frozen=True, slots=True)
class ExternalObservation:
    sequence: int
    observed_at: datetime
    status: ExternalObservedStatus
    observation_digest: str
    external_job_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or not 1 <= self.sequence <= MAX_JOB_EXTERNAL_OBSERVATIONS
        ):
            raise ValueError("external observation sequence is outside its bound")
        _utc(self.observed_at, "external observation observed_at")
        if not isinstance(self.status, ExternalObservedStatus):
            raise TypeError("external observation status is invalid")
        _digest(self.observation_digest, "external observation digest")
        _text(self.external_job_id, "external observation job_id", maximum=1_024)


@dataclass(frozen=True, slots=True)
class JobAttempt:
    number: int
    fencing_epoch: int
    claim_token: str
    execution_run_id: str
    reserved_artifact_id: str
    status: JobAttemptStatus
    claimed_at: datetime
    lease_expires_at: datetime
    renewals: int = 0
    completed_at: datetime | None = None
    error_code: str | None = None
    external_intents: tuple[ExternalIntent, ...] = ()
    external_observations: tuple[ExternalObservation, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.number, int)
            or isinstance(self.number, bool)
            or not 1 <= self.number <= MAX_JOB_ATTEMPTS
        ):
            raise ValueError("job attempt number is outside its bound")
        if (
            not isinstance(self.fencing_epoch, int)
            or isinstance(self.fencing_epoch, bool)
            or self.fencing_epoch < 1
        ):
            raise ValueError("job fencing_epoch must be positive")
        for value, name in (
            (self.claim_token, "job claim_token"),
            (self.execution_run_id, "job execution_run_id"),
            (self.reserved_artifact_id, "job reserved_artifact_id"),
        ):
            _text(value, name, maximum=256)
        if not isinstance(self.status, JobAttemptStatus):
            raise TypeError("job attempt status must be JobAttemptStatus")
        _utc(self.claimed_at, "job attempt claimed_at")
        _utc(self.lease_expires_at, "job attempt lease_expires_at")
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("job attempt lease must expire after claim")
        if (
            not isinstance(self.renewals, int)
            or isinstance(self.renewals, bool)
            or not 0 <= self.renewals <= MAX_JOB_RENEWALS
        ):
            raise ValueError("job attempt renewals are outside their bound")
        _optional_utc(self.completed_at, "job attempt completed_at")
        if self.error_code is not None:
            _text(self.error_code, "job attempt error_code", maximum=128)
        if self.status is JobAttemptStatus.CLAIMED:
            if self.completed_at is not None:
                raise ValueError("claimed job attempt cannot have completed_at")
        elif self.completed_at is None:
            raise ValueError("settled job attempt requires completed_at")
        intents = tuple(self.external_intents)
        observations = tuple(self.external_observations)
        if len(intents) > 2 or any(
            not isinstance(item, ExternalIntent) for item in intents
        ):
            raise ValueError("job external intents exceed their embedded bound")
        if len({item.kind for item in intents}) != len(intents):
            raise ValueError("job external intent kind cannot repeat")
        if len(observations) > MAX_JOB_EXTERNAL_OBSERVATIONS or any(
            not isinstance(item, ExternalObservation) for item in observations
        ):
            raise ValueError("job external observations exceed their bound")
        if tuple(item.sequence for item in observations) != tuple(
            range(1, len(observations) + 1)
        ):
            raise ValueError("job external observation sequence is invalid")
        object.__setattr__(self, "external_intents", intents)
        object.__setattr__(self, "external_observations", observations)


@dataclass(frozen=True, slots=True)
class JobResult:
    result_id: str
    summary: Mapping[str, object]
    sensitivity: ModelSensitivity
    provenance: Mapping[str, object]
    artifact_refs: tuple[ArtifactRef, ...]
    completed_at: datetime

    def __post_init__(self) -> None:
        _text(self.result_id, "job result_id", maximum=256)
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("job result sensitivity must be ModelSensitivity")
        _utc(self.completed_at, "job result completed_at")
        summary = _bounded_json(
            self.summary,
            name="job result summary",
            maximum_bytes=MAX_JOB_INLINE_RESULT_BYTES,
            maximum_depth=MAX_JOB_RESULT_DEPTH,
        )
        provenance = _bounded_json(
            self.provenance,
            name="job result provenance",
            maximum_bytes=MAX_JOB_SUMMARY_BYTES,
            maximum_depth=MAX_JOB_RESULT_DEPTH,
        )
        refs = tuple(self.artifact_refs)
        if len(refs) > MAX_JOB_ARTIFACTS or any(
            not isinstance(item, ArtifactRef) for item in refs
        ):
            raise ValueError("job artifact references exceed their bound")
        if any(item.sensitivity.value != self.sensitivity.value for item in refs):
            raise ValueError("job artifact sensitivity must match the job result")
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "artifact_refs", refs)


@dataclass(frozen=True, slots=True)
class JobRun:
    job_id: str
    agent_id: str
    conversation_id: str
    origin_run_id: str
    origin_call_id: str
    specification: JobSpecification
    specification_digest: str
    status: JobStatus
    desired_state: JobDesiredState
    created_at: datetime
    updated_at: datetime
    revision: int = 1
    fencing_epoch: int = 0
    attempts: tuple[JobAttempt, ...] = ()
    cancel_requested_at: datetime | None = None
    terminal_at: datetime | None = None
    terminal_observed_at: datetime | None = None
    result: JobResult | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.job_id, "job_id"),
            (self.agent_id, "job agent_id"),
            (self.conversation_id, "job conversation_id"),
            (self.origin_run_id, "job origin_run_id"),
            (self.origin_call_id, "job origin_call_id"),
        ):
            _text(value, name, maximum=256)
        if not isinstance(self.specification, JobSpecification):
            raise TypeError("job specification must be JobSpecification")
        _digest(self.specification_digest, "job specification_digest")
        if self.specification_digest != self.specification.digest:
            raise ValueError("job specification digest does not match its content")
        if not isinstance(self.status, JobStatus):
            raise TypeError("job status must be JobStatus")
        if not isinstance(self.desired_state, JobDesiredState):
            raise TypeError("job desired_state must be JobDesiredState")
        _utc(self.created_at, "job created_at")
        _utc(self.updated_at, "job updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("job updated_at cannot precede created_at")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("job revision must be positive")
        if (
            not isinstance(self.fencing_epoch, int)
            or isinstance(self.fencing_epoch, bool)
            or self.fencing_epoch < 0
        ):
            raise ValueError("job fencing_epoch must be non-negative")
        attempts = tuple(self.attempts)
        if len(attempts) > MAX_JOB_ATTEMPTS or any(
            not isinstance(item, JobAttempt) for item in attempts
        ):
            raise ValueError("job attempts exceed their embedded bound")
        if tuple(item.number for item in attempts) != tuple(
            range(1, len(attempts) + 1)
        ):
            raise ValueError("job attempt numbering is invalid")
        epochs = tuple(item.fencing_epoch for item in attempts)
        if epochs != tuple(sorted(set(epochs))):
            raise ValueError("job attempt fencing epochs must be strictly increasing")
        if attempts and attempts[-1].fencing_epoch > self.fencing_epoch:
            raise ValueError("job aggregate fencing epoch trails its attempt")
        _optional_utc(self.cancel_requested_at, "job cancel_requested_at")
        _optional_utc(self.terminal_at, "job terminal_at")
        _optional_utc(self.terminal_observed_at, "job terminal_observed_at")
        terminal = self.status in TERMINAL_JOB_STATUSES
        if terminal != (self.terminal_at is not None):
            raise ValueError("job terminal state must agree with terminal_at")
        if self.terminal_observed_at is not None and not terminal:
            raise ValueError("only a terminal job can be observed terminal")
        if self.status is JobStatus.SUCCEEDED:
            if not isinstance(self.result, JobResult):
                raise ValueError("successful job requires its exact result")
        elif self.result is not None:
            raise ValueError("non-successful job cannot expose a result")
        if self.failure_code is not None:
            _text(self.failure_code, "job failure_code", maximum=128)
        if self.status in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            if not attempts or attempts[-1].status is not JobAttemptStatus.CLAIMED:
                raise ValueError("running job requires one current claimed attempt")
        if (
            self.status is JobStatus.QUEUED
            and attempts
            and (attempts[-1].status is JobAttemptStatus.CLAIMED)
        ):
            raise ValueError("queued job cannot retain a live claim")
        if self.desired_state is JobDesiredState.CANCEL and (
            self.cancel_requested_at is None
        ):
            raise ValueError("cancel-desired job requires cancel_requested_at")
        object.__setattr__(self, "attempts", attempts)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.source_id for item in self.specification.resource_bindings})
        )

    @property
    def current_attempt(self) -> JobAttempt | None:
        return self.attempts[-1] if self.attempts else None


@dataclass(frozen=True, slots=True)
class JobAttemptView:
    number: int
    fencing_epoch: int
    status: JobAttemptStatus
    claimed_at: datetime
    completed_at: datetime | None
    error_code: str | None
    external_intents: tuple[ExternalIntent, ...]
    external_observations: tuple[ExternalObservation, ...]


@dataclass(frozen=True, slots=True)
class JobSummary:
    job_id: str
    origin_conversation_id: str
    job_kind: str
    status: JobStatus
    execution_mode: JobExecutionMode
    source_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    sensitivity: ModelSensitivity
    created_at: datetime
    updated_at: datetime
    result_available: bool


@dataclass(frozen=True, slots=True)
class JobInspection:
    summary: JobSummary
    origin_run_id: str
    specification_digest: str
    execution_capability_id: str
    execution_contract_digest: str
    desired_state: JobDesiredState
    deadline_at: datetime
    attempts: tuple[JobAttemptView, ...]
    cancel_requested_at: datetime | None
    terminal_at: datetime | None
    failure_code: str | None
    external_executor: ConnectedExecutorBinding | None


@dataclass(frozen=True, slots=True)
class JobResultView:
    job_id: str
    result_id: str
    summary: FrozenJsonObject
    sensitivity: ModelSensitivity
    provenance: FrozenJsonObject
    artifact_refs: tuple[ArtifactRef, ...]
    completed_at: datetime


def job_summary(job: JobRun) -> JobSummary:
    return JobSummary(
        job_id=job.job_id,
        origin_conversation_id=job.conversation_id,
        job_kind=job.specification.job_kind,
        status=job.status,
        execution_mode=job.specification.execution_mode,
        source_ids=job.source_ids,
        resource_ids=tuple(
            item.resource_id for item in job.specification.resource_bindings
        ),
        sensitivity=job.specification.sensitivity,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result_available=job.result is not None,
    )


def job_inspection(job: JobRun) -> JobInspection:
    return JobInspection(
        summary=job_summary(job),
        origin_run_id=job.origin_run_id,
        specification_digest=job.specification_digest,
        execution_capability_id=job.specification.execution_capability_id,
        execution_contract_digest=job.specification.execution_contract_digest,
        desired_state=job.desired_state,
        deadline_at=job.specification.deadline_at,
        attempts=tuple(
            JobAttemptView(
                number=item.number,
                fencing_epoch=item.fencing_epoch,
                status=item.status,
                claimed_at=item.claimed_at,
                completed_at=item.completed_at,
                error_code=item.error_code,
                external_intents=item.external_intents,
                external_observations=item.external_observations,
            )
            for item in job.attempts
        ),
        cancel_requested_at=job.cancel_requested_at,
        terminal_at=job.terminal_at,
        failure_code=job.failure_code,
        external_executor=job.specification.external_executor,
    )


def job_result_view(job: JobRun) -> JobResultView | None:
    result = job.result
    if result is None:
        return None
    return JobResultView(
        job_id=job.job_id,
        result_id=result.result_id,
        summary=FrozenJsonObject.from_mapping(result.summary),
        sensitivity=result.sensitivity,
        provenance=FrozenJsonObject.from_mapping(result.provenance),
        artifact_refs=result.artifact_refs,
        completed_at=result.completed_at,
    )


__all__ = [
    "ConnectedExecutorBinding",
    "ExternalIntent",
    "ExternalIntentDisposition",
    "ExternalIntentKind",
    "ExternalObservation",
    "ExternalObservedStatus",
    "JobAttempt",
    "JobAttemptStatus",
    "JobAttemptView",
    "JobDesiredState",
    "JobExecutionMode",
    "JobInspection",
    "JobResourceBinding",
    "JobResult",
    "JobResultView",
    "JobRun",
    "JobSpecification",
    "JobStatus",
    "JobSummary",
    "MAX_ACTIVE_JOBS_PER_AGENT",
    "MAX_EXTERNAL_REQUEST_BYTES",
    "MAX_EXTERNAL_RESPONSE_BYTES",
    "MAX_JOB_ATTEMPTS",
    "MAX_JOB_DEADLINE_SECONDS",
    "MAX_JOB_EXTERNAL_OBSERVATIONS",
    "MAX_JOB_INLINE_RESULT_BYTES",
    "MAX_JOB_LIST_PAGE_SIZE",
    "MAX_JOB_RENEWALS",
    "MAX_JOB_SPECIFICATION_BYTES",
    "MAX_JOB_WALL_TIME_SECONDS",
    "MAX_JOBS_PER_AGENT",
    "MAX_QUEUED_JOBS_PER_AGENT",
    "MAX_RUNNING_JOBS_GLOBAL",
    "MAX_RUNNING_JOBS_PER_AGENT",
    "MAX_RUNNING_JOBS_PER_SOURCE",
    "TERMINAL_JOB_STATUSES",
    "job_inspection",
    "job_result_view",
    "job_summary",
]
