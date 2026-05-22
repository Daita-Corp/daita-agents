"""
Structured per-run state for agents created by ``Agent.from_db()``.

The state object is intentionally small and transient. It tracks execution
facts that help a single run avoid duplicate work and recover deterministically;
durable database semantics still belong in DB memory, while structural facts
belong in the catalog profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DbQueryPlan:
    """Structured representation of a database question before SQL exists."""

    goal: str
    required_fields: List[str] = field(default_factory=list)
    candidate_tables: List[str] = field(default_factory=list)
    required_joins: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    aggregations: List[str] = field(default_factory=list)
    grouping: List[str] = field(default_factory=list)
    ordering: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    assumptions: List[str] = field(default_factory=list)
    answer_checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DbRunState:
    """Transient state reset for each ``from_db`` agent run."""

    inspected_tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    candidate_columns: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    known_join_paths: List[Dict[str, Any]] = field(default_factory=list)
    failed_sql_fingerprints: Dict[str, int] = field(default_factory=dict)
    validated_sql: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    executed_queries: List[Dict[str, Any]] = field(default_factory=list)
    required_answer_fields: List[str] = field(default_factory=list)
    planned_queries: List[Dict[str, Any]] = field(default_factory=list)
    plans_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    next_plan_number: int = 1
    final_completeness_status: Optional[Dict[str, Any]] = None

    def record_candidate_columns(
        self, field_name: str, candidates: List[Dict[str, Any]]
    ) -> None:
        if candidates:
            self.candidate_columns[field_name] = list(candidates)

    def record_join_paths(self, result: Dict[str, Any]) -> None:
        for path in result.get("paths") or []:
            if path not in self.known_join_paths:
                self.known_join_paths.append(path)

    def record_failed_sql(
        self,
        fingerprint: str,
        validation: Optional[Dict[str, Any]] = None,
        *,
        source_tool: str = "db_query",
    ) -> int:
        self.failed_sql_fingerprints[fingerprint] = (
            self.failed_sql_fingerprints.get(fingerprint, 0) + 1
        )
        _record_db_evidence(
            "rejected_sql",
            source_tool=source_tool,
            payload={
                "sql_fingerprint": fingerprint,
                "attempt_count": self.failed_sql_fingerprints[fingerprint],
                **_compact_validation(validation or {}),
            },
        )
        return self.failed_sql_fingerprints[fingerprint]

    def record_validated_sql(
        self,
        fingerprint: str,
        validation: Dict[str, Any],
        *,
        source_tool: str = "db_query",
    ) -> None:
        self.validated_sql[fingerprint] = dict(validation)
        _record_db_evidence(
            "validated_sql",
            source_tool=source_tool,
            payload={
                "sql_fingerprint": fingerprint,
                **_compact_validation(validation),
            },
        )

    def record_executed_query(
        self, metadata: Dict[str, Any], *, source_tool: str = "db_query"
    ) -> None:
        self.executed_queries.append(dict(metadata))
        _record_db_evidence(
            "executed_query",
            source_tool=source_tool,
            payload=_compact_query_metadata(metadata),
        )

    def record_plan(self, plan: DbQueryPlan, result: Dict[str, Any]) -> str:
        plan_id = f"plan_{self.next_plan_number}"
        self.next_plan_number += 1
        stored = {"plan_id": plan_id, "plan": plan.to_dict(), "result": dict(result)}
        self.planned_queries.append(stored)
        self.plans_by_id[plan_id] = stored
        for field_name in plan.required_fields:
            if field_name not in self.required_answer_fields:
                self.required_answer_fields.append(field_name)
        _record_db_evidence(
            "query_plan",
            source_tool="db_plan_query",
            payload={
                "plan_id": plan_id,
                "plan": plan.to_dict(),
                "route": result.get("route"),
                "resolved_tables": list(result.get("resolved_tables") or []),
                "compiled_sql": result.get("compiled_sql"),
                "validation": _compact_validation(result.get("validation") or {}),
            },
        )
        return plan_id

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        return self.plans_by_id.get(plan_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "inspected_table_count": len(self.inspected_tables),
            "candidate_field_count": len(self.candidate_columns),
            "known_join_path_count": len(self.known_join_paths),
            "failed_sql_count": sum(self.failed_sql_fingerprints.values()),
            "validated_sql_count": len(self.validated_sql),
            "executed_query_count": len(self.executed_queries),
            "required_answer_fields": list(self.required_answer_fields),
            "planned_query_count": len(self.planned_queries),
        }


def get_db_run_state(owner: Any) -> Optional[DbRunState]:
    try:
        from daita.agents.runtime.contextvars import get_active_run_state

        run_state = get_active_run_state()
        if run_state is not None:
            state = run_state.domains.get("db")
            if isinstance(state, DbRunState):
                return state
    except Exception:
        pass

    state = getattr(owner, "_daita_db_run_state", None)
    return state if isinstance(state, DbRunState) else None


def set_db_run_state(owner: Any, state: DbRunState) -> None:
    setattr(owner, "_daita_db_run_state", state)
    _register_db_final_answer_readiness(owner)
    try:
        from daita.agents.runtime.contextvars import get_active_run_state

        run_state = get_active_run_state()
        if run_state is not None:
            run_state.domains["db"] = state
            hook = _db_final_answer_readiness_hook()
            if hook is not None and hook not in run_state.final_answer_readiness_hooks:
                run_state.final_answer_readiness_hooks.append(hook)
    except Exception:
        pass


def _record_db_evidence(
    kind: str,
    *,
    source_tool: str,
    payload: Dict[str, Any],
    notes: Optional[List[str]] = None,
) -> None:
    try:
        from daita.agents.runtime.evidence import add_active_evidence

        add_active_evidence(
            domain="db",
            kind=kind,
            source_tool=source_tool,
            payload=payload,
            notes=notes,
        )
    except Exception:
        pass


def _register_db_final_answer_readiness(owner: Any) -> None:
    hook = _db_final_answer_readiness_hook()
    if hook is None:
        return
    hooks = getattr(owner, "_daita_final_answer_readiness_hooks", None)
    if not isinstance(hooks, list):
        hooks = []
        setattr(owner, "_daita_final_answer_readiness_hooks", hooks)
    if hook not in hooks:
        hooks.append(hook)


def _db_final_answer_readiness_hook() -> Any:
    try:
        from .completeness import evaluate_db_final_answer_readiness

        return evaluate_db_final_answer_readiness
    except Exception:
        return None


def _compact_validation(validation: Dict[str, Any]) -> Dict[str, Any]:
    allowed = (
        "ok",
        "plan_id",
        "source",
        "sql",
        "sql_fingerprint",
        "error",
        "error_type",
        "missing_tables",
        "missing_columns",
        "available_columns",
        "referenced_tables",
        "referenced_columns",
        "selected_columns",
        "repair_required",
        "blocked_repeat",
        "status",
        "message",
    )
    return {key: validation[key] for key in allowed if key in validation}


def _compact_query_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    allowed = (
        "plan_id",
        "sql_fingerprint",
        "sql",
        "tables",
        "columns",
        "row_count",
        "returned_rows",
        "truncated",
        "assumptions",
    )
    return {key: metadata[key] for key in allowed if key in metadata}
