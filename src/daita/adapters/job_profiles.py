"""Define connected job-executor contracts and the offline conformance profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .._json import FrozenJsonObject, canonical_json
from ..jobs.models import (
    MAX_EXTERNAL_REQUEST_BYTES,
    MAX_EXTERNAL_RESPONSE_BYTES,
    MAX_JOB_RESULT_DEPTH,
    ConnectedExecutorBinding,
    ExternalIntentDisposition,
    ExternalObservedStatus,
)
from ..llm.models import ModelSensitivity


def _text(value: str, name: str, *, maximum: int = 1_024) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be bounded non-empty text")


def _utc(value: datetime, name: str) -> None:
    offset = value.utcoffset() if isinstance(value, datetime) else None
    if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_depth(item) for item in value.values()), default=0)
    if isinstance(value, (tuple, list)):
        return 1 + max((_depth(item) for item in value), default=0)
    return 0


def _payload(
    value: Mapping[str, object],
    *,
    name: str,
    maximum_bytes: int,
) -> FrozenJsonObject:
    result = FrozenJsonObject.from_mapping(value)
    if len(canonical_json(result).encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte bound")
    if _depth(result) > MAX_JOB_RESULT_DEPTH:
        raise ValueError(f"{name} exceeds its depth bound")
    return result


@dataclass(frozen=True, slots=True)
class ConnectedJobProfileState:
    binding: ConnectedExecutorBinding
    supported_job_kinds: frozenset[str]
    active: bool

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConnectedExecutorBinding):
            raise TypeError("connected job profile binding is invalid")
        kinds = frozenset(self.supported_job_kinds)
        if not kinds or len(kinds) > 16:
            raise ValueError("connected job profile kinds exceed their bound")
        for value in kinds:
            _text(value, "connected job kind", maximum=128)
        if not isinstance(self.active, bool):
            raise TypeError("connected job profile active must be boolean")
        object.__setattr__(self, "supported_job_kinds", kinds)


@dataclass(frozen=True, slots=True)
class ExternalStartRequest:
    job_id: str
    specification_digest: str
    idempotency_key: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        for value, name in (
            (self.job_id, "external start job_id"),
            (self.specification_digest, "external start specification_digest"),
            (self.idempotency_key, "external start idempotency_key"),
        ):
            _text(value, name)
        object.__setattr__(
            self,
            "arguments",
            _payload(
                self.arguments,
                name="external start request",
                maximum_bytes=MAX_EXTERNAL_REQUEST_BYTES,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExternalStartReceipt:
    disposition: ExternalIntentDisposition
    observed_at: datetime
    external_job_id: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, ExternalIntentDisposition):
            raise TypeError("external start disposition is invalid")
        if self.disposition is ExternalIntentDisposition.PENDING:
            raise ValueError("external start adapter cannot return pending")
        _utc(self.observed_at, "external start observed_at")
        if self.external_job_id is not None:
            _text(self.external_job_id, "external job_id")
        if self.reason_code is not None:
            _text(self.reason_code, "external start reason", maximum=128)
        if (
            self.disposition is ExternalIntentDisposition.ACCEPTED
            and self.external_job_id is None
        ):
            raise ValueError("accepted external start requires external_job_id")


@dataclass(frozen=True, slots=True)
class ExternalCancelRequest:
    job_id: str
    external_job_id: str
    specification_digest: str
    idempotency_key: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.job_id, "external cancel job_id"),
            (self.external_job_id, "external cancel external_job_id"),
            (self.specification_digest, "external cancel specification_digest"),
            (self.idempotency_key, "external cancel idempotency_key"),
        ):
            _text(value, name)


@dataclass(frozen=True, slots=True)
class ExternalCancelReceipt:
    disposition: ExternalIntentDisposition
    observed_at: datetime
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, ExternalIntentDisposition):
            raise TypeError("external cancel disposition is invalid")
        if self.disposition is ExternalIntentDisposition.PENDING:
            raise ValueError("external cancel adapter cannot return pending")
        _utc(self.observed_at, "external cancel observed_at")
        if self.reason_code is not None:
            _text(self.reason_code, "external cancel reason", maximum=128)


@dataclass(frozen=True, slots=True)
class ExternalStatusRequest:
    job_id: str
    specification_digest: str
    idempotency_key: str
    external_job_id: str | None

    def __post_init__(self) -> None:
        for value, name in (
            (self.job_id, "external status job_id"),
            (self.specification_digest, "external status specification_digest"),
            (self.idempotency_key, "external status idempotency_key"),
        ):
            _text(value, name)
        if self.external_job_id is not None:
            _text(self.external_job_id, "external status external_job_id")


@dataclass(frozen=True, slots=True)
class ExternalStatusReceipt:
    status: ExternalObservedStatus
    external_job_id: str
    observed_at: datetime
    observation: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExternalObservedStatus):
            raise TypeError("external status receipt status is invalid")
        _text(self.external_job_id, "external status job_id")
        _utc(self.observed_at, "external status observed_at")
        object.__setattr__(
            self,
            "observation",
            _payload(
                self.observation,
                name="external status response",
                maximum_bytes=MAX_EXTERNAL_RESPONSE_BYTES,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExternalResultRequest:
    job_id: str
    specification_digest: str
    external_job_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.job_id, "external result job_id"),
            (self.specification_digest, "external result specification_digest"),
            (self.external_job_id, "external result external_job_id"),
        ):
            _text(value, name)


@dataclass(frozen=True, slots=True)
class ExternalResultPayload:
    summary: Mapping[str, object]
    sensitivity: ModelSensitivity
    provenance: Mapping[str, object]
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("external result sensitivity must be ModelSensitivity")
        _utc(self.observed_at, "external result observed_at")
        object.__setattr__(
            self,
            "summary",
            _payload(
                self.summary,
                name="external result summary",
                maximum_bytes=MAX_EXTERNAL_RESPONSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _payload(
                self.provenance,
                name="external result provenance",
                maximum_bytes=MAX_EXTERNAL_REQUEST_BYTES,
            ),
        )


class ConnectedJobProfile(Protocol):
    """One separately admitted exact connected executor profile.

    ``current_state`` is the connector-owned current-admission boundary. Each
    call must re-resolve authentication and secret references and report the
    exact active binding; callers do not cache its answer across external I/O.
    """

    @property
    def profile_id(self) -> str: ...

    def current_state(self) -> ConnectedJobProfileState: ...

    async def start(self, request: ExternalStartRequest) -> ExternalStartReceipt: ...

    async def status(self, request: ExternalStatusRequest) -> ExternalStatusReceipt: ...

    async def cancel(
        self,
        request: ExternalCancelRequest,
    ) -> ExternalCancelReceipt: ...

    async def read_result(
        self,
        request: ExternalResultRequest,
    ) -> ExternalResultPayload: ...


__all__ = [
    "ConnectedJobProfile",
    "ConnectedJobProfileState",
    "ExternalCancelReceipt",
    "ExternalCancelRequest",
    "ExternalResultPayload",
    "ExternalResultRequest",
    "ExternalStartReceipt",
    "ExternalStartRequest",
    "ExternalStatusReceipt",
    "ExternalStatusRequest",
]
