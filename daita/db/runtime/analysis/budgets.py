"""Analysis budget accounting helpers."""

from __future__ import annotations

import time
from typing import Any

from daita.runtime import Evidence

from ...analysis import DbAnalysisPlan


class DbRuntimeAnalysisBudgetMixin:
    def _analysis_budget_failure(
        self,
        plan: DbAnalysisPlan,
        evidence: tuple[Evidence, ...],
        *,
        started_at: float,
    ) -> dict[str, Any] | None:
        usage = _analysis_budget_usage(evidence, started_at=started_at)
        failures = []
        budgets = plan.budgets
        if usage["total_rows"] > budgets.max_total_rows:
            failures.append("budget_max_total_rows_exceeded")
        if usage["llm_calls"] > budgets.max_llm_calls:
            failures.append("budget_max_llm_calls_exceeded")
        if usage["context_chars"] > budgets.max_context_chars:
            failures.append("budget_max_context_chars_exceeded")
        if usage["duration_seconds"] > budgets.max_duration_seconds:
            failures.append("budget_max_duration_seconds_exceeded")
        if not failures:
            return None
        return {
            "budget_exceeded": True,
            "failures": failures,
            "budget_usage": usage,
            "budgets": budgets.to_dict(),
        }

    @staticmethod
    def _analysis_step_budget_failure(
        step: Any,
        evidence: tuple[Evidence, ...],
    ) -> dict[str, Any] | None:
        rows = sum(
            _analysis_query_result_rows(item)
            for item in evidence
            if item.kind == "query.result"
        )
        if rows <= step.budgets.max_rows:
            return None
        return {
            "budget_exceeded": True,
            "failures": ["step_max_rows_exceeded"],
            "budget_usage": {"step_rows": rows},
            "budgets": step.budgets.to_dict(),
        }


def _analysis_budget_usage(
    evidence: tuple[Evidence, ...],
    *,
    started_at: float,
) -> dict[str, Any]:
    return {
        "total_rows": sum(
            _analysis_query_result_rows(item)
            for item in evidence
            if item.kind == "query.result" and item.accepted
        ),
        "llm_calls": sum(1 for item in evidence if _analysis_evidence_used_llm(item)),
        "context_chars": sum(
            len(str(item.payload.get("rendered_context") or ""))
            for item in evidence
            if item.kind == "planning.context" and item.accepted
        ),
        "duration_seconds": time.monotonic() - started_at,
    }


def _analysis_query_result_rows(evidence: Evidence) -> int:
    rows = evidence.payload.get("rows")
    if isinstance(rows, list):
        return len(rows)
    try:
        return int(evidence.payload.get("total_rows") or 0)
    except (TypeError, ValueError):
        return 0


def _analysis_evidence_used_llm(evidence: Evidence) -> bool:
    if not evidence.accepted:
        return False
    diagnostics = evidence.payload.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("mode") == "llm":
        return True
    planner_diagnostics = evidence.payload.get("planner_diagnostics")
    return isinstance(planner_diagnostics, dict) and bool(
        planner_diagnostics.get("model") or planner_diagnostics.get("provider")
    )
