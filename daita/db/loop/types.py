"""Data structures for the DB agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daita.runtime import TaskDependency

from ..runtime.tasks.models import DbTaskSpec
from .utils import _json_dict


@dataclass(frozen=True)
class DbActionCompilation:
    """Runtime compilation of planner actions to governed DB task specs."""

    accepted_action_summaries: tuple[dict[str, Any], ...] = ()
    rejected_action_summaries: tuple[dict[str, Any], ...] = ()
    task_specs: tuple[DbTaskSpec, ...] = ()
    compiled_contract_snapshot: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_action_summaries",
            tuple(_json_dict(item) for item in self.accepted_action_summaries),
        )
        object.__setattr__(
            self,
            "rejected_action_summaries",
            tuple(_json_dict(item) for item in self.rejected_action_summaries),
        )
        object.__setattr__(self, "task_specs", tuple(self.task_specs))
        object.__setattr__(
            self,
            "compiled_contract_snapshot",
            _json_dict(self.compiled_contract_snapshot),
        )
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_action_summaries": list(self.accepted_action_summaries),
            "rejected_action_summaries": list(self.rejected_action_summaries),
            "task_specs": [spec.to_dict() for spec in self.task_specs],
            "compiled_contract_snapshot": self.compiled_contract_snapshot,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class DbLoopResult:
    """Terminal result returned by the DB agent loop."""

    status: str
    evidence_refs: tuple[dict[str, Any], ...] = ()
    task_refs: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status:
            raise ValueError("status is required")
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(_json_dict(item) for item in self.evidence_refs),
        )
        object.__setattr__(
            self, "task_refs", tuple(_json_dict(item) for item in self.task_refs)
        )
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "task_refs": list(self.task_refs),
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class _ResolvedSqlInput:
    sql: str
    provenance: str
    query_plan_dependency: TaskDependency | None = None
    plan_validation_dependency: TaskDependency | None = None
    sql_validation_dependency: TaskDependency | None = None
    source_evidence_id: str | None = None
    source_evidence_kind: str | None = None
    source_evidence_owner: str | None = None
    source_task_id: str | None = None
    source_payload_fingerprint: str | None = None
