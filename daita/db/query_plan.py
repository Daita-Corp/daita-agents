"""Typed query plan records for DB runtime planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal, Mapping


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
        return cls(
            left_table=str(value.get("left_table") or ""),
            left_column=str(value.get("left_column") or ""),
            right_table=str(value.get("right_table") or ""),
            right_column=str(value.get("right_column") or ""),
            relationship_ref=(
                str(value["relationship_ref"])
                if value.get("relationship_ref")
                else None
            ),
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
            operation=str(value.get("operation") or "read"),  # type: ignore[arg-type]
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


@dataclass(frozen=True)
class DbQueryPlanValidation:
    """Deterministic validation result for a plan proposal."""

    valid: bool
    accepted_sql: str | None = None
    sql_fingerprint: str | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    plan_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", _tuple_strings(self.errors))
        object.__setattr__(self, "warnings", _tuple_strings(self.warnings))
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "accepted_sql": self.accepted_sql,
            "sql": self.accepted_sql,
            "sql_fingerprint": self.sql_fingerprint,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "plan_fingerprint": self.plan_fingerprint,
            "metadata": self.metadata,
        }
