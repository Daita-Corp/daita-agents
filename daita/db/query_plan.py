"""Typed query plan records for DB runtime planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal, Mapping

_PLAN_OPERATION_ALIASES = {
    "query": "read",
    "query_planning": "read",
    "data_query": "read",
    "data.query": "read",
    "read_query": "read",
    "read_only": "read",
    "select": "read",
    "sql_select": "read",
    "schema_query": "schema",
    "schema.query": "schema",
    "write": "write_propose",
    "write_proposal": "write_propose",
    "write.propose": "write_propose",
}


def _tuple_strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    items = tuple(str(item) for item in values if item is not None)
    return items


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    json.dumps(copied, sort_keys=True, default=str)
    return copied


def _join_columns_from_text(
    text: str | None,
    *,
    left_table: str,
    right_table: str,
) -> tuple[str, str] | None:
    if not text or not left_table or not right_table:
        return None
    normalized = text.replace("->", "=")
    if normalized.count("=") != 1:
        return None
    left_ref, right_ref = (
        _parse_column_ref(part) for part in normalized.split("=", maxsplit=1)
    )
    if left_ref is None or right_ref is None:
        return None
    left_ref_table, left_ref_column = left_ref
    right_ref_table, right_ref_column = right_ref
    left_key = left_table.lower()
    right_key = right_table.lower()
    if left_ref_table == left_key and right_ref_table == right_key:
        return left_ref_column, right_ref_column
    if left_ref_table == right_key and right_ref_table == left_key:
        return right_ref_column, left_ref_column
    return None


def _parse_column_ref(value: str) -> tuple[str, str] | None:
    cleaned = value.strip().strip(";")
    for token in ('"', "`", "[", "]"):
        cleaned = cleaned.replace(token, "")
    parts = [part.strip().lower() for part in cleaned.split(".") if part.strip()]
    if len(parts) < 2:
        return None
    return parts[-2], parts[-1]


def _column_from_join_key(value: Any) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip()
    if not cleaned:
        return ""
    for token in ('"', "`", "[", "]"):
        cleaned = cleaned.replace(token, "")
    return cleaned.split(".")[-1].strip()


@dataclass(frozen=True)
class DbJoinSpec:
    """Join relationship proposed by a planner."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    relationship_ref: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DbJoinSpec":
        left_table = str(value.get("left_table") or "")
        right_table = str(value.get("right_table") or "")
        left_column = str(value.get("left_column") or "") or _column_from_join_key(
            value.get("left_key")
        )
        right_column = str(value.get("right_column") or "") or _column_from_join_key(
            value.get("right_key")
        )
        relationship_ref = (
            str(value["relationship_ref"])
            if value.get("relationship_ref")
            else str(value["relationship"]) if value.get("relationship") else None
        )
        if not left_column or not right_column:
            inferred = _join_columns_from_text(
                relationship_ref or str(value.get("condition") or ""),
                left_table=left_table,
                right_table=right_table,
            )
            if inferred is not None:
                left_column = left_column or inferred[0]
                right_column = right_column or inferred[1]
        return cls(
            left_table=left_table,
            left_column=left_column,
            right_table=right_table,
            right_column=right_column,
            relationship_ref=relationship_ref,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DbFilterSpec:
    """Filter proposed by a planner."""

    column: str
    operator: str
    value: Any = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DbFilterSpec":
        return cls(
            column=str(value.get("column") or ""),
            operator=str(value.get("operator") or ""),
            value=value.get("value"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DbAggregationSpec:
    """Aggregation proposed by a planner."""

    function: str
    column: str | None = None
    alias: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DbAggregationSpec":
        return cls(
            function=str(value.get("function") or ""),
            column=str(value["column"]) if value.get("column") else None,
            alias=str(value["alias"]) if value.get("alias") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DbQueryPlanCandidate:
    """One SQL candidate inside a structured plan proposal."""

    sql: str
    purpose: str
    confidence: float
    tables: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    expected_columns: tuple[str, ...] = ()
    expected_row_count: str | None = None
    assumptions: tuple[str, ...] = ()
    risk_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("candidate confidence must be between 0 and 1")
        object.__setattr__(self, "tables", _tuple_strings(self.tables))
        object.__setattr__(self, "columns", _tuple_strings(self.columns))
        object.__setattr__(
            self, "expected_columns", _tuple_strings(self.expected_columns)
        )
        object.__setattr__(self, "assumptions", _tuple_strings(self.assumptions))
        object.__setattr__(self, "risk_notes", _tuple_strings(self.risk_notes))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DbQueryPlanCandidate":
        return cls(
            sql=str(value.get("sql") or ""),
            purpose=str(value.get("purpose") or ""),
            confidence=float(value.get("confidence", 0)),
            tables=_tuple_strings(value.get("tables")),
            columns=_tuple_strings(value.get("columns")),
            expected_columns=_tuple_strings(value.get("expected_columns")),
            expected_row_count=(
                str(value["expected_row_count"])
                if value.get("expected_row_count") is not None
                else None
            ),
            assumptions=_tuple_strings(value.get("assumptions")),
            risk_notes=_tuple_strings(value.get("risk_notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DbQueryPlan:
    """Structured query plan proposal accepted by the DB runtime."""

    operation: Literal["read", "write_propose", "schema", "analysis"] = "read"
    selected_sql: str | None = None
    candidates: tuple[DbQueryPlanCandidate, ...] = ()
    selected_tables: tuple[str, ...] = ()
    joins: tuple[DbJoinSpec, ...] = ()
    filters: tuple[DbFilterSpec, ...] = ()
    aggregations: tuple[DbAggregationSpec, ...] = ()
    group_by: tuple[str, ...] = ()
    order_by: tuple[str, ...] = ()
    limit: int | None = None
    assumptions: tuple[str, ...] = ()
    clarification_question: str | None = None
    confidence: float = 0.0
    planner: str = "unknown"

    def __post_init__(self) -> None:
        if self.operation not in {"read", "write_propose", "schema", "analysis"}:
            raise ValueError("unsupported query plan operation")
        if not 0 <= self.confidence <= 1:
            raise ValueError("plan confidence must be between 0 and 1")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self, "selected_tables", _tuple_strings(self.selected_tables)
        )
        object.__setattr__(self, "joins", tuple(self.joins))
        object.__setattr__(self, "filters", tuple(self.filters))
        object.__setattr__(self, "aggregations", tuple(self.aggregations))
        object.__setattr__(self, "group_by", _tuple_strings(self.group_by))
        object.__setattr__(self, "order_by", _tuple_strings(self.order_by))
        object.__setattr__(self, "assumptions", _tuple_strings(self.assumptions))

    @classmethod
    def deterministic(
        cls,
        *,
        sql: str | None,
        tables: tuple[str, ...] = (),
        columns: tuple[str, ...] = (),
        filters: tuple[DbFilterSpec, ...] = (),
        confidence: float = 0.85,
        strategy: str = "deterministic",
        assumptions: tuple[str, ...] = (),
    ) -> "DbQueryPlan":
        candidates = (
            (
                DbQueryPlanCandidate(
                    sql=sql,
                    purpose=strategy,
                    confidence=confidence,
                    tables=tables,
                    columns=columns,
                    assumptions=assumptions,
                ),
            )
            if sql
            else ()
        )
        return cls(
            operation="read",
            selected_sql=sql,
            candidates=candidates,
            selected_tables=tables,
            filters=filters,
            assumptions=assumptions,
            confidence=confidence if sql else 0.0,
            planner="deterministic",
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DbQueryPlan":
        candidates = tuple(
            DbQueryPlanCandidate.from_mapping(item)
            for item in value.get("candidates", ()) or ()
            if isinstance(item, Mapping)
        )
        return cls(
            operation=_normalize_operation(value.get("operation")),  # type: ignore[arg-type]
            selected_sql=(
                str(value["selected_sql"]) if value.get("selected_sql") else None
            ),
            candidates=candidates,
            selected_tables=_tuple_strings(value.get("selected_tables")),
            joins=tuple(
                DbJoinSpec.from_mapping(item)
                for item in value.get("joins", ()) or ()
                if isinstance(item, Mapping)
            ),
            filters=tuple(
                DbFilterSpec.from_mapping(item)
                for item in value.get("filters", ()) or ()
                if isinstance(item, Mapping)
            ),
            aggregations=tuple(
                DbAggregationSpec.from_mapping(item)
                for item in value.get("aggregations", ()) or ()
                if isinstance(item, Mapping)
            ),
            group_by=_tuple_strings(value.get("group_by")),
            order_by=_tuple_strings(value.get("order_by")),
            limit=int(value["limit"]) if value.get("limit") is not None else None,
            assumptions=_tuple_strings(value.get("assumptions")),
            clarification_question=(
                str(value["clarification_question"])
                if value.get("clarification_question")
                else None
            ),
            confidence=float(value.get("confidence", 0)),
            planner=str(value.get("planner") or "unknown"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "selected_sql": self.selected_sql,
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_tables": list(self.selected_tables),
            "joins": [item.to_dict() for item in self.joins],
            "filters": [item.to_dict() for item in self.filters],
            "aggregations": [item.to_dict() for item in self.aggregations],
            "group_by": list(self.group_by),
            "order_by": list(self.order_by),
            "limit": self.limit,
            "assumptions": list(self.assumptions),
            "clarification_question": self.clarification_question,
            "confidence": self.confidence,
            "planner": self.planner,
        }


def _normalize_operation(value: Any) -> str:
    operation = str(value or "read").strip().lower()
    return _PLAN_OPERATION_ALIASES.get(operation, operation)


@dataclass(frozen=True)
class DbQueryPlanValidation:
    """Deterministic validation result for a plan proposal."""

    valid: bool
    accepted_sql: str | None = None
    sql_fingerprint: str | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    validation_facts: tuple[dict[str, Any], ...] = ()
    plan_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", _tuple_strings(self.errors))
        object.__setattr__(self, "warnings", _tuple_strings(self.warnings))
        object.__setattr__(
            self,
            "validation_facts",
            tuple(_json_dict(item) for item in self.validation_facts),
        )
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "accepted_sql": self.accepted_sql,
            "sql": self.accepted_sql,
            "sql_fingerprint": self.sql_fingerprint,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "validation_facts": [dict(item) for item in self.validation_facts],
            "plan_fingerprint": self.plan_fingerprint,
            "metadata": self.metadata,
        }
