"""Shared DB monitor scheduler result and error types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from daita.runtime import Evidence

from ..monitors import DbMonitorRun


@dataclass(frozen=True)
class DbMonitorSchedulerResult:
    """One scheduler decision for a durable DB monitor."""

    monitor_id: str
    run: DbMonitorRun
    claimed: bool = False


@dataclass(frozen=True)
class DbMonitorObservationResult:
    """Compact observed facts and runtime provenance for one monitor tick."""

    value: Any
    evidence: Evidence
    task_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    summary: dict[str, Any] | None = None


class DbMonitorObservationBlocked(RuntimeError):
    """Raised when a persisted observation plan cannot safely execute."""

    status = "blocked"

    def __init__(
        self,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
        task_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details or {})
        self.task_ids = tuple(task_ids)
        self.evidence_ids = tuple(evidence_ids)


class DbMonitorObservationFailed(DbMonitorObservationBlocked):
    """Raised when observation execution fails after a tick has started."""

    status = "failed"
