"""
Structured per-run state for agents created by ``Agent.from_db()``.

The state object is intentionally small and transient. It tracks execution
facts that help a single run avoid duplicate work and recover deterministically;
durable database semantics still belong in DB memory or the schema cache.
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

    def get_inspected_table(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self.inspected_tables.get(cache_key)

    def record_inspected_table(self, cache_key: str, result: Dict[str, Any]) -> None:
        self.inspected_tables[cache_key] = dict(result)

    def record_candidate_columns(
        self, field_name: str, candidates: List[Dict[str, Any]]
    ) -> None:
        if candidates:
            self.candidate_columns[field_name] = list(candidates)

    def record_join_paths(self, result: Dict[str, Any]) -> None:
        for path in result.get("paths") or []:
            if path not in self.known_join_paths:
                self.known_join_paths.append(path)

    def record_failed_sql(self, fingerprint: str) -> int:
        self.failed_sql_fingerprints[fingerprint] = (
            self.failed_sql_fingerprints.get(fingerprint, 0) + 1
        )
        return self.failed_sql_fingerprints[fingerprint]

    def record_validated_sql(
        self, fingerprint: str, validation: Dict[str, Any]
    ) -> None:
        self.validated_sql[fingerprint] = dict(validation)

    def record_executed_query(self, metadata: Dict[str, Any]) -> None:
        self.executed_queries.append(dict(metadata))

    def record_plan(self, plan: DbQueryPlan, result: Dict[str, Any]) -> str:
        plan_id = f"plan_{self.next_plan_number}"
        self.next_plan_number += 1
        stored = {"plan_id": plan_id, "plan": plan.to_dict(), "result": dict(result)}
        self.planned_queries.append(stored)
        self.plans_by_id[plan_id] = stored
        for field_name in plan.required_fields:
            if field_name not in self.required_answer_fields:
                self.required_answer_fields.append(field_name)
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
    state = getattr(owner, "_daita_db_run_state", None)
    return state if isinstance(state, DbRunState) else None


def set_db_run_state(owner: Any, state: DbRunState) -> None:
    setattr(owner, "_daita_db_run_state", state)
