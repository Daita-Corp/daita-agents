"""Progress guard primitives for the DB agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..planner_protocol import DbPlannerDecision
from ..verification import DB_FINALIZATION_CONTROL_EVIDENCE_KINDS
from .utils import _stable_hash


@dataclass(frozen=True)
class _LoopProgressSnapshot:
    task_statuses: dict[str, str]
    accepted_evidence: tuple[dict[str, Any], ...] = ()
    rejected_evidence: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _LoopProgressDecision:
    terminal_status: str | None = None
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    retry_facts: tuple[dict[str, Any], ...] = ()


@dataclass
class _LoopProgressGuard:
    seen_no_progress_fingerprints: set[str] = field(default_factory=set)
    failed_action_counts: dict[str, int] = field(default_factory=dict)
    sql_error_counts: dict[str, int] = field(default_factory=dict)
    no_progress_count: int = 0

    def evaluate(self, facts: Mapping[str, Any]) -> _LoopProgressDecision:
        retry_facts: list[dict[str, Any]] = []
        sql_terminal = False
        for fingerprint in facts.get("sql_error_fingerprints") or ():
            previous = self.sql_error_counts.get(str(fingerprint), 0)
            count = previous + 1
            self.sql_error_counts[str(fingerprint)] = count
            if previous == 1:
                retry_facts.append(
                    {
                        "warning": "db_agent_loop_repeated_sql_failure",
                        "fingerprint": str(fingerprint),
                        "count": count,
                        "message": (
                            "The same SQL validation or execution failure "
                            "repeated; choose a different action or repair the SQL."
                        ),
                    }
                )
                sql_terminal = True
            elif previous >= 2:
                sql_terminal = True

        repeated_failed_action = False
        if facts.get("failed_action") and not facts.get("new_accepted_evidence_refs"):
            for fingerprint in facts.get("compiled_action_fingerprints") or ():
                previous = self.failed_action_counts.get(str(fingerprint), 0)
                self.failed_action_counts[str(fingerprint)] = previous + 1
                if previous >= 1:
                    repeated_failed_action = True

        repeated_no_progress = False
        progress_fingerprint = str(facts.get("progress_fingerprint") or "")
        if facts.get("no_progress"):
            self.no_progress_count += 1
            repeated_no_progress = (
                progress_fingerprint in self.seen_no_progress_fingerprints
                or self.no_progress_count >= 2
            )
            if progress_fingerprint:
                self.seen_no_progress_fingerprints.add(progress_fingerprint)
        else:
            self.no_progress_count = 0

        if retry_facts and not sql_terminal:
            return _LoopProgressDecision(retry_facts=tuple(retry_facts))

        warnings: list[str] = []
        terminal_status: str | None = None
        if sql_terminal:
            terminal_status = "failed"
            warnings.append("db_agent_loop_repeated_sql_failure")
        if repeated_failed_action:
            terminal_status = terminal_status or "failed"
            warnings.append("db_agent_loop_repeated_action")
        if repeated_no_progress:
            terminal_status = terminal_status or "blocked"
            warnings.append("db_agent_loop_no_progress")

        if terminal_status is None:
            return _LoopProgressDecision()

        return _LoopProgressDecision(
            terminal_status=terminal_status,
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics={
                "sql_terminal": sql_terminal,
                "repeated_failed_action": repeated_failed_action,
                "repeated_no_progress": repeated_no_progress,
                "no_progress_count": self.no_progress_count,
            },
            retry_facts=tuple(retry_facts),
        )


def _compiled_action_fingerprints(
    decision: DbPlannerDecision,
) -> tuple[str, ...]:
    return tuple(
        _stable_hash(
            {
                "kind": action.kind.value,
                "input": action.input,
                "depends_on": list(action.depends_on),
                "metadata": action.metadata,
            }
        )
        for action in decision.actions
    )


def _execution_error_fingerprint(error: Mapping[str, Any]) -> str:
    return _stable_hash(
        {
            "capability_id": error.get("capability_id"),
            "error": error.get("error"),
            "error_type": error.get("error_type"),
            "readiness": error.get("readiness"),
        }
    )


def _is_sql_execution_error(error: Mapping[str, Any]) -> bool:
    capability_id = str(error.get("capability_id") or "")
    if capability_id in {
        "db.sql.validate",
        "db.sql.execute_read",
        "db.sql.execute_write",
    }:
        return True
    text = str(error.get("error") or "").lower()
    return "sql" in text or "validation_failed" in text


def _blocked_resource_execution_errors(
    errors: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(error) for error in errors if _is_blocked_resource_error(error))


def _is_blocked_resource_error(error: Mapping[str, Any]) -> bool:
    if str(error.get("capability_id") or "") != "db.sql.validate":
        return False
    text = str(error.get("error") or "").lower()
    return any(
        marker in text
        for marker in (
            "sql guardrail rejected blocked table",
            "sql guardrail rejected blocked column",
            "sql guardrail rejected table(s) outside allowlist",
        )
    )


def _new_evidence_refs(
    before: tuple[dict[str, Any], ...],
    after: tuple[dict[str, Any], ...],
    *,
    include_loop_control: bool,
) -> tuple[dict[str, Any], ...]:
    before_ids = {item.get("id") for item in before}
    refs = tuple(item for item in after if item.get("id") not in before_ids)
    if include_loop_control:
        return refs
    return tuple(
        item
        for item in refs
        if item.get("kind") not in DB_FINALIZATION_CONTROL_EVIDENCE_KINDS
    )
