"""Shared intent classifiers for deterministic query planning."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .metadata import split_identifier
from .ir import QueryPlan
from .requirements import AnswerRequirement

COUNT_ENTITY_TOKENS = {
    "event",
    "events",
    "execution",
    "executions",
    "item",
    "items",
    "operation",
    "operations",
    "record",
    "records",
    "row",
    "rows",
    "run",
    "runs",
    "transaction",
    "transactions",
}


@dataclass
class QueryIntent:
    """Normalized unresolved query intent from the LLM-facing tool args."""

    goal: str
    required_fields: list[str] = field(default_factory=list)
    answer_requirements: list[AnswerRequirement] = field(default_factory=list)
    candidate_tables: list[str] = field(default_factory=list)
    required_joins: list[dict[str, Any]] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    grouping: list[str] = field(default_factory=list)
    ordering: list[str] = field(default_factory=list)
    limit: Optional[int] = None
    assumptions: list[str] = field(default_factory=list)
    answer_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlanRecord:
    """Canonical stored plan record for one resolved query attempt."""

    plan_id: Optional[str]
    intent: QueryIntent
    query_ir: Optional[QueryPlan]
    validation: dict[str, Any]
    compiled_sql: Optional[str] = None
    route: str = ""
    resolved_tables: list[str] = field(default_factory=list)
    ambiguous_tables: list[dict[str, Any]] = field(default_factory=list)
    unknown_tables: list[dict[str, Any]] = field(default_factory=list)
    plan_warnings: list[dict[str, Any]] = field(default_factory=list)
    best_join_path: Optional[dict[str, Any]] = None
    knowledge_used: Optional[dict[str, Any]] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self, *, include_diagnostics: bool = True) -> dict[str, Any]:
        result = {
            "ok": True,
            "plan_id": self.plan_id,
            "route": self.route,
            "resolved_tables": list(self.resolved_tables),
            "ambiguous_tables": list(self.ambiguous_tables),
            "unknown_tables": list(self.unknown_tables),
            "compiled_sql": self.compiled_sql,
            "validation": dict(self.validation),
            "plan_warnings": list(self.plan_warnings),
            "best_join_path": self.best_join_path,
            "knowledge_used": self.knowledge_used,
            "next_steps": list(self.next_steps),
        }
        result["next_step"] = result["next_steps"][0] if result["next_steps"] else None
        if self.compiled_sql:
            result["suggested_next_tool"] = "db_query"
            result["suggested_next_arguments"] = (
                {"plan_id": self.plan_id}
                if self.plan_id
                else {"sql": self.compiled_sql}
            )
        if include_diagnostics:
            result["intent"] = self.intent.to_dict()
            result["query_ir"] = (
                self.query_ir.to_dict() if self.query_ir is not None else None
            )
            result.update(self.diagnostics)
        return result


def looks_like_count_intent(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(
        re.search(r"\b(count|how many|number of|row count|rows)\b", lowered)
        or re.search(r"\b(most|top|fewest|least)\b", lowered)
        or re.search(
            r"\btotal[_\s-]?"
            r"(events|executions|items|operations|records|rows|runs|transactions)\b",
            lowered,
        )
    )


def is_count_metric_name(name: str) -> bool:
    tokens = set(split_identifier(name))
    return bool(
        tokens & {"count", "number", "quantity"}
        or ("total" in tokens and bool(tokens & COUNT_ENTITY_TOKENS))
    )
