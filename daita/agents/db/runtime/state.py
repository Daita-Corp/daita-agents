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

from ..query.requirements import AnswerRequirement
from ..utils import string_list as _string_list


@dataclass
class DbQueryPlan:
    """Structured representation of a database question before SQL exists."""

    goal: str
    required_fields: List[str] = field(default_factory=list)
    answer_requirements: List[AnswerRequirement] = field(default_factory=list)
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

    intent_kind: Optional[str] = None
    inspected_tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    candidate_columns: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    catalog_search_count: int = 0
    catalog_evidence: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {"tables": [], "columns": [], "joins": []}
    )
    known_join_paths: List[Dict[str, Any]] = field(default_factory=list)
    failed_sql_fingerprints: Dict[str, int] = field(default_factory=dict)
    validated_sql: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    executed_queries: List[Dict[str, Any]] = field(default_factory=list)
    answer_requirements: List[AnswerRequirement] = field(default_factory=list)
    planned_queries: List[Dict[str, Any]] = field(default_factory=list)
    plans_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    next_plan_number: int = 1
    final_completeness_status: Optional[Dict[str, Any]] = None
    run_contract: Any = None

    def record_candidate_columns(
        self, field_name: str, candidates: List[Dict[str, Any]]
    ) -> None:
        if candidates:
            self.candidate_columns[field_name] = list(candidates)

    @property
    def required_answer_fields(self) -> List[str]:
        return [requirement.display_name for requirement in self.answer_requirements]

    def record_join_paths(self, result: Dict[str, Any]) -> None:
        for path in result.get("paths") or []:
            if path not in self.known_join_paths:
                self.known_join_paths.append(path)

    def record_catalog_tool_result(
        self, tool_name: str, arguments: Dict[str, Any], result: Any
    ) -> None:
        if not isinstance(result, dict) or result.get("error"):
            return
        if tool_name == "catalog_search_schema":
            self._record_catalog_search(arguments, result)
        elif tool_name == "catalog_inspect_table":
            self._record_catalog_inspection(arguments, result)
        elif tool_name == "catalog_find_join_paths":
            self._record_catalog_join_paths(arguments, result)

    def catalog_planner_evidence(self) -> Dict[str, Any]:
        tables = list(self.catalog_evidence.get("tables") or [])
        joins = list(self.catalog_evidence.get("joins") or [])
        for join in joins:
            for table_name in (join.get("from_tables") or []) + (
                join.get("to_tables") or []
            ):
                table = {
                    "table": table_name,
                    "confidence": join.get("confidence", 0.95),
                    "source": "catalog",
                    "provenance": join.get("provenance", "catalog_evidence"),
                }
                _append_unique_evidence(tables, table, key=("table",))
        return {
            "tables": tables,
            "columns": list(self.catalog_evidence.get("columns") or []),
            "joins": joins,
            "sources": ["catalog"] if tables or joins else [],
            "confidence": _evidence_confidence(tables + joins),
        }

    def _record_catalog_search(
        self, arguments: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        self.catalog_search_count += 1
        for table in result.get("tables") or []:
            if not isinstance(table, dict):
                continue
            table_name = str(table.get("name") or table.get("asset_ref") or "").strip()
            if not table_name:
                continue
            _append_unique_evidence(
                self.catalog_evidence["tables"],
                {
                    "table": table_name,
                    "confidence": _score_to_confidence(table.get("score"), 0.75),
                    "source": "catalog",
                    "provenance": "catalog_search_schema",
                    "query": arguments.get("query"),
                },
                key=("table",),
            )
            for matched_field in table.get("matched_fields") or []:
                if not isinstance(matched_field, dict):
                    continue
                column_name = str(matched_field.get("name") or "").strip()
                if not column_name:
                    continue
                _append_unique_evidence(
                    self.catalog_evidence["columns"],
                    {
                        "table": table_name,
                        "column": column_name,
                        "confidence": _score_to_confidence(
                            matched_field.get("score"), 0.7
                        ),
                        "source": "catalog",
                        "provenance": "catalog_search_schema",
                    },
                    key=("table", "column"),
                )

    def _record_catalog_inspection(
        self, arguments: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        table_name = str(
            result.get("table_name") or arguments.get("table_name") or ""
        ).strip()
        if not table_name:
            return
        self.inspected_tables[table_name] = dict(result)
        _append_unique_evidence(
            self.catalog_evidence["tables"],
            {
                "table": table_name,
                "confidence": 0.9,
                "source": "catalog",
                "provenance": "catalog_inspect_table",
            },
            key=("table",),
        )
        for column in result.get("columns") or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name") or "").strip()
            if not column_name:
                continue
            _append_unique_evidence(
                self.catalog_evidence["columns"],
                {
                    "table": table_name,
                    "column": column_name,
                    "type": column.get("type"),
                    "confidence": 0.9,
                    "source": "catalog",
                    "provenance": "catalog_inspect_table",
                },
                key=("table", "column"),
            )

    def _record_catalog_join_paths(
        self, arguments: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        from_tables = _string_list(
            result.get("from_tables")
            or result.get("from_assets")
            or arguments.get("from_tables")
        )
        to_tables = _string_list(
            result.get("to_tables")
            or result.get("to_assets")
            or arguments.get("to_tables")
        )
        join_result = {
            **result,
            "from_tables": from_tables,
            "to_tables": to_tables,
            "source": "catalog",
            "provenance": "catalog_find_join_paths",
            "confidence": 0.95 if result.get("reachable") else 0.4,
        }
        _append_unique_evidence(
            self.catalog_evidence["joins"],
            join_result,
            key=("from_tables", "to_tables", "reachable"),
        )
        self.record_join_paths(result)
        _record_db_evidence(
            "catalog_join_path",
            source_tool="catalog_find_join_paths",
            payload={
                "from_tables": from_tables,
                "to_tables": to_tables,
                "reachable": bool(result.get("reachable")),
                "path_count": result.get("path_count", len(result.get("paths") or [])),
            },
        )

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
        self.record_answer_requirements(plan.answer_requirements)
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

    def record_answer_requirements(self, requirements: List[AnswerRequirement]) -> None:
        seen = {requirement.raw for requirement in self.answer_requirements}
        for requirement in requirements:
            if requirement.raw in seen:
                continue
            self.answer_requirements.append(requirement)
            seen.add(requirement.raw)

    def tool_retry_fingerprint(
        self, tool_call: Dict[str, Any], raw_result: Any, *, kind: str
    ) -> Optional[str]:
        """Return a DB-domain retry identity for generic loop guardrails."""

        if not isinstance(raw_result, dict):
            return None
        tool_name = str(tool_call.get("name") or "")
        sql_fp = str(raw_result.get("sql_fingerprint") or "").strip()
        if kind == "error" and tool_name in {
            "db_query",
            "db_validate_sql",
            "db_compile_and_query",
        }:
            if sql_fp and (
                raw_result.get("repair_required")
                or raw_result.get("preflight_failed")
                or raw_result.get("blocked_repeat")
            ):
                error_type = str(raw_result.get("error_type") or "sql_repair")
                return f"db:sql_error:{sql_fp}:{error_type}"
        if kind == "result" and tool_name == "db_plan_query":
            sql = str(raw_result.get("compiled_sql") or "").strip()
            if sql:
                from ..query.sql_validator import sql_fingerprint

                validation = raw_result.get("validation") or {}
                status = "valid" if validation.get("ok") else "invalid"
                return f"db:plan:{sql_fingerprint(sql)}:{status}"
        return None

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        return self.plans_by_id.get(plan_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "inspected_table_count": len(self.inspected_tables),
            "intent_kind": self.intent_kind,
            "candidate_field_count": len(self.candidate_columns),
            "catalog_search_count": self.catalog_search_count,
            "catalog_table_evidence_count": len(
                self.catalog_evidence.get("tables") or []
            ),
            "catalog_join_evidence_count": len(
                self.catalog_evidence.get("joins") or []
            ),
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


def set_db_run_state(
    owner: Any,
    state: DbRunState,
    *,
    register_final_answer_readiness: bool = True,
) -> None:
    setattr(owner, "_daita_db_run_state", state)
    if register_final_answer_readiness:
        _register_db_final_answer_readiness(owner)
    try:
        from daita.agents.runtime.contextvars import get_active_run_state

        run_state = get_active_run_state()
        if run_state is not None:
            run_state.domains["db"] = state
            if register_final_answer_readiness:
                hook = _db_final_answer_readiness_hook()
                if (
                    hook is not None
                    and hook not in run_state.final_answer_readiness_hooks
                ):
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
        "selected_columns",
        "row_count",
        "returned_rows",
        "truncated",
        "assumptions",
    )
    return {key: metadata[key] for key in allowed if key in metadata}


def _append_unique_evidence(
    items: List[Dict[str, Any]], item: Dict[str, Any], *, key: tuple[str, ...]
) -> None:
    item_key = _evidence_key(item, key)
    for index, existing in enumerate(items):
        if _evidence_key(existing, key) != item_key:
            continue
        if _score(item) > _score(existing):
            items[index] = item
        return
    items.append(item)


def _evidence_key(item: Dict[str, Any], key: tuple[str, ...]) -> tuple[Any, ...]:
    values = []
    for part in key:
        value = item.get(part)
        if isinstance(value, list):
            values.append(tuple(str(entry).lower() for entry in value))
        else:
            values.append(str(value or "").lower())
    return tuple(values)


def _score_to_confidence(value: Any, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score <= 0:
        return default
    return round(min(0.95, score / 10), 3)


def _evidence_confidence(items: List[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return round(min(1.0, max(_score(item) for item in items)), 3)


def _score(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
