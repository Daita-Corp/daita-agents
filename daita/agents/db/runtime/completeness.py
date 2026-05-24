"""Evidence-driven DB run completeness diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ...runtime.state import FinalAnswerReadiness
from ..config.policies import ANSWER_EVIDENCE_DB_TOOLS
from ..query.requirements import AnswerRequirement, output_satisfies_requirement
from .state import DbRunState

INCOMPLETE_FINAL_MARKERS = (
    "i need to",
    "we need to",
    "i will",
    "i'll",
    "let's find out",
    "will write",
    "will run",
    "run a corrected",
    "write a corrected",
)


def attach_db_completeness(run_state) -> Dict[str, Any] | None:
    """Attach a DB completeness summary to runtime diagnostics when applicable."""
    db_state = run_state.domains.get("db")
    has_db_evidence = any(record.domain == "db" for record in run_state.evidence)
    if not isinstance(db_state, DbRunState) and not has_db_evidence:
        return None

    summary = summarize_db_completeness(run_state)
    run_state.diagnostics["db_completeness"] = summary
    if isinstance(db_state, DbRunState):
        db_state.final_completeness_status = summary
    return summary


def evaluate_db_final_answer_readiness(
    run_state, final_text: str, available_tools: Iterable[Any]
) -> FinalAnswerReadiness | None:
    """Evaluate DB-specific final-answer readiness behind the generic hook."""
    db_state = run_state.domains.get("db")
    has_db_domain = isinstance(db_state, DbRunState) or any(
        record.domain == "db" for record in run_state.evidence
    )
    if not has_db_domain:
        return None

    summary = summarize_db_completeness(run_state)
    diagnostics = {"db_completeness": summary}
    if isinstance(db_state, DbRunState):
        db_state.final_completeness_status = summary

    incomplete_final = _looks_incomplete_final_answer(final_text)
    can_answer = bool(summary.get("can_answer"))
    if can_answer and not incomplete_final:
        return FinalAnswerReadiness(diagnostics=diagnostics)

    if not final_text:
        return FinalAnswerReadiness(diagnostics=diagnostics)

    available_names = _tool_names(available_tools)
    if not available_names.intersection(ANSWER_EVIDENCE_DB_TOOLS):
        return FinalAnswerReadiness(diagnostics=diagnostics)

    warning = (
        "db_final_answer_incomplete"
        if can_answer and incomplete_final
        else "db_final_answer_without_query_evidence"
    )
    return FinalAnswerReadiness(
        allow_final=False,
        guidance=_final_answer_guidance(db_state, summary),
        warning=warning,
        diagnostics=diagnostics,
    )


def summarize_db_completeness(run_state) -> Dict[str, Any]:
    """Summarize whether DB evidence looks sufficient to answer safely.

    This helper is diagnostic-only. It does not block final answers and does not
    alter model prompts.
    """
    db_state = run_state.domains.get("db")
    db_records = [record for record in run_state.evidence if record.domain == "db"]
    by_kind = _records_by_kind(db_records)
    executed_records = by_kind.get("executed_query", [])
    rejected_records = by_kind.get("rejected_sql", [])
    validated_records = by_kind.get("validated_sql", [])
    plan_records = by_kind.get("query_plan", [])
    latest_db_record = db_records[-1] if db_records else None
    unresolved_repair = _has_unresolved_repair(latest_db_record)

    requirements = (
        list(getattr(db_state, "answer_requirements", []) or [])
        if isinstance(db_state, DbRunState)
        else []
    )
    required_fields = [requirement.display_name for requirement in requirements]
    row_counts = [
        record.payload.get("row_count")
        for record in executed_records
        if isinstance(record.payload.get("row_count"), int)
    ]
    returned_columns = _returned_columns(executed_records)
    evidence_columns = _evidence_columns(executed_records)
    missing_required_fields = _missing_required_fields(
        requirements, evidence_columns, executed_records
    )
    truncated_count = sum(
        1 for record in executed_records if bool(record.payload.get("truncated"))
    )
    status = _status(
        executed_count=len(executed_records),
        rejected_count=len(rejected_records),
        truncated_count=truncated_count,
        row_counts=row_counts,
        missing_required_fields=missing_required_fields,
        unresolved_repair=unresolved_repair,
    )

    return {
        "status": status,
        "can_answer": (
            len(executed_records) > 0
            and not missing_required_fields
            and not unresolved_repair
        ),
        "plans_created": len(plan_records),
        "sql_validated": len(validated_records),
        "sql_rejected": len(rejected_records),
        "queries_executed": len(executed_records),
        "row_counts": row_counts,
        "total_rows_observed": sum(row_counts),
        "returned_columns": sorted(returned_columns),
        "evidence_columns": sorted(evidence_columns),
        "latest_db_evidence_kind": latest_db_record.kind if latest_db_record else None,
        "latest_db_evidence_tool": (
            latest_db_record.source_tool if latest_db_record else None
        ),
        "unresolved_repair": unresolved_repair,
        "truncated_result_count": truncated_count,
        "required_answer_fields": required_fields,
        "missing_required_fields": missing_required_fields,
        "evidence_count": len(db_records),
        "warnings": _warnings(
            rejected_records,
            truncated_count,
            required_fields,
            missing_required_fields,
            row_counts,
            unresolved_repair,
        ),
    }


def _records_by_kind(records: List[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for record in records:
        grouped.setdefault(record.kind, []).append(record)
    return grouped


def _tool_names(tools: Iterable[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if isinstance(tool, str):
            names.add(tool)
        else:
            name = getattr(tool, "name", None)
            if name:
                names.add(str(name))
    return names


def _looks_incomplete_final_answer(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return any(marker in normalized for marker in INCOMPLETE_FINAL_MARKERS)


def _status(
    *,
    executed_count: int,
    rejected_count: int,
    truncated_count: int,
    row_counts: List[int],
    missing_required_fields: List[str],
    unresolved_repair: bool,
) -> str:
    if unresolved_repair:
        return "blocked"
    if executed_count > 0 and missing_required_fields:
        return "missing_required_fields"
    if executed_count > 0 and row_counts and sum(row_counts) == 0:
        return "answerable_empty"
    if executed_count > 0 and truncated_count == 0:
        return "answerable"
    if executed_count > 0:
        return "answerable_with_caveats"
    if rejected_count > 0:
        return "blocked"
    return "insufficient_evidence"


def _returned_columns(records: List[Any]) -> set[str]:
    columns: set[str] = set()
    for record in records:
        for column in record.payload.get("columns") or []:
            if column:
                columns.add(str(column))
    return columns


def _evidence_columns(records: List[Any]) -> set[str]:
    columns = _returned_columns(records)
    for record in records:
        for column in record.payload.get("selected_columns") or []:
            if column:
                columns.add(str(column))
    return columns


def _has_unresolved_repair(latest_db_record: Any) -> bool:
    if latest_db_record is None:
        return False
    if latest_db_record.kind == "rejected_sql":
        return True
    payload = getattr(latest_db_record, "payload", {}) or {}
    return bool(payload.get("repair_required") or payload.get("preflight_failed"))


def _missing_required_fields(
    requirements: List[AnswerRequirement],
    evidence_columns: set[str],
    executed_records: List[Any],
) -> List[str]:
    if not requirements:
        return []
    if not executed_records:
        return [requirement.display_name for requirement in requirements]
    if not evidence_columns:
        return [requirement.display_name for requirement in requirements]
    return [
        requirement.display_name
        for requirement in requirements
        if not _requirement_is_covered_by_columns(requirement, evidence_columns)
    ]


def _requirement_is_covered_by_columns(
    requirement: AnswerRequirement, returned_columns: set[str]
) -> bool:
    return any(
        output_satisfies_requirement(requirement, column) for column in returned_columns
    )


def _final_answer_guidance(db_state: Any, summary: Dict[str, Any]) -> str:
    plan_instruction = ""
    if isinstance(db_state, DbRunState) and db_state.planned_queries:
        latest_plan = db_state.planned_queries[-1]
        plan_id = latest_plan.get("plan_id")
        compiled_sql = (latest_plan.get("result") or {}).get("compiled_sql")
        if plan_id and compiled_sql:
            plan_instruction = (
                f" Use db_query with plan_id={plan_id!r} if that plan answers "
                "the question."
            )
    missing = summary.get("missing_required_fields") or []
    missing_instruction = (
        " Missing fields: " + ", ".join(str(field) for field in missing[:5]) + "."
        if missing
        else ""
    )
    return (
        "Do not provide a final answer yet. If the executed rows answer the "
        "question, answer directly from those rows. If they do not contain the "
        "requested fields, use db_query or db_compile_and_query with corrected "
        f"SQL to gather the missing evidence.{missing_instruction}{plan_instruction}"
    )


def _warnings(
    rejected_records: List[Any],
    truncated_count: int,
    required_fields: List[str],
    missing_required_fields: List[str],
    row_counts: List[int],
    unresolved_repair: bool,
) -> List[str]:
    warnings: List[str] = []
    if rejected_records:
        warnings.append("sql_rejections_present")
    if unresolved_repair:
        warnings.append("unresolved_sql_repair")
    if truncated_count:
        warnings.append("truncated_results_present")
    if required_fields:
        warnings.append("required_fields_tracked")
    if missing_required_fields:
        warnings.append("missing_required_fields")
    if row_counts and sum(row_counts) == 0:
        warnings.append("empty_results_present")
    return warnings
