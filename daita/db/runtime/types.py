"""Runtime-local types for the DB runtime facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daita.runtime import (
    ApprovalRequest,
    Evidence,
    GovernanceAuditRecord,
    GovernanceResult,
    Operation,
    RuntimeEvent,
    Task,
    TaskStatus,
)

from ..analysis import DbAnalysisPlan

_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }
)
_DEFAULT_TASK_LEASE_SECONDS = 300.0


@dataclass(frozen=True)
class _GovernancePersistence:
    result: GovernanceResult
    audit_record: GovernanceAuditRecord
    approvals_to_request: tuple[ApprovalRequest, ...]
    events: tuple[RuntimeEvent, ...]


@dataclass(frozen=True)
class _MonitorEffectGovernanceDecision:
    status: str
    reason: str | None
    result: GovernanceResult
    audit_record: GovernanceAuditRecord

    @property
    def allowed(self) -> bool:
        return self.status == "allowed"


@dataclass(frozen=True)
class _AnalysisPlanState:
    plan: DbAnalysisPlan
    plan_evidence: Evidence
    validation_evidence: Evidence
    revision_evidence: Evidence | None = None

    @property
    def selected_plan_evidence(self) -> Evidence:
        return self.revision_evidence or self.plan_evidence


@dataclass(frozen=True)
class _SourcePreparationSnapshot:
    evidence: Evidence
    store_id: str
    schema_fingerprint: str
    cached_at: float


class DbRuntimeGovernanceBlocked(PermissionError):
    """Raised when persisted runtime governance blocks executor invocation."""

    def __init__(
        self,
        *,
        operation: Operation,
        task: Task | None,
        governance: GovernanceResult,
    ) -> None:
        self.operation = operation
        self.task = task
        self.governance = governance
        super().__init__(_governance_blocked_answer(governance))


class DbRuntimeTaskNotRunnable(RuntimeError):
    """Raised when a persisted task is not currently executable."""

    def __init__(
        self,
        task: Task,
        message: str,
        *,
        readiness: dict[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.readiness = readiness or {}
        super().__init__(message)


def _governance_blocked_answer(governance: GovernanceResult) -> str:
    if governance.blocked:
        return "This operation was denied by governance policy."
    if governance.pending_approval:
        return "This operation requires approval before execution."
    return "This operation was blocked by governance policy."


def _governance_blocked_warning(governance: GovernanceResult) -> str:
    if governance.blocked:
        return "db_runtime_governance_denied"
    if governance.pending_approval:
        return "db_runtime_approval_required"
    return "db_runtime_governance_blocked"
