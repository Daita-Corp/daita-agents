"""Typed query intent model for deterministic ``from_db`` SQL compilation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

MetricKind = Literal["count", "distinct_count", "sum", "avg", "min", "max"]
FilterOperator = Literal["=", "!=", ">", ">=", "<", "<=", "between", "in", "like"]
OrderDirection = Literal["asc", "desc"]


@dataclass(frozen=True)
class FieldRef:
    table: str
    column: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FieldRef":
        return cls(
            table=str(value.get("table") or ""), column=str(value.get("column") or "")
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Metric:
    name: str
    kind: MetricKind
    table: str
    column: Optional[str] = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Metric":
        column = value.get("column")
        return cls(
            name=str(value.get("name") or ""),
            kind=value.get("kind") or "count",
            table=str(value.get("table") or ""),
            column=str(column) if column is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Filter:
    field: FieldRef
    operator: FilterOperator
    value: Any

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Filter":
        return cls(
            field=FieldRef.from_dict(value.get("field") or {}),
            operator=value.get("operator") or "=",
            value=value.get("value"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field.to_dict(),
            "operator": self.operator,
            "value": self.value,
        }


@dataclass(frozen=True)
class Join:
    left: FieldRef
    right: FieldRef

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Join":
        return cls(
            left=FieldRef.from_dict(value.get("left") or {}),
            right=FieldRef.from_dict(value.get("right") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass(frozen=True)
class OrderBy:
    field: str | FieldRef
    direction: OrderDirection = "asc"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OrderBy":
        raw_field = value.get("field")
        field: str | FieldRef
        if isinstance(raw_field, dict):
            field = FieldRef.from_dict(raw_field)
        else:
            field = str(raw_field or "")
        direction = str(value.get("direction") or "asc").lower()
        return cls(field=field, direction="desc" if direction == "desc" else "asc")

    def to_dict(self) -> dict[str, Any]:
        field: Any = (
            self.field.to_dict() if isinstance(self.field, FieldRef) else self.field
        )
        return {"field": field, "direction": self.direction}


@dataclass
class QueryPlan:
    grain: list[FieldRef] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    joins: list[Join] = field(default_factory=list)
    order_by: list[OrderBy] = field(default_factory=list)
    limit: Optional[int] = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryPlan":
        return cls(
            grain=[FieldRef.from_dict(item) for item in value.get("grain") or []],
            metrics=[Metric.from_dict(item) for item in value.get("metrics") or []],
            filters=[Filter.from_dict(item) for item in value.get("filters") or []],
            joins=[Join.from_dict(item) for item in value.get("joins") or []],
            order_by=[OrderBy.from_dict(item) for item in value.get("order_by") or []],
            limit=_optional_int(value.get("limit")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grain": [item.to_dict() for item in self.grain],
            "metrics": [item.to_dict() for item in self.metrics],
            "filters": [item.to_dict() for item in self.filters],
            "joins": [item.to_dict() for item in self.joins],
            "order_by": [item.to_dict() for item in self.order_by],
            "limit": self.limit,
        }


def _optional_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
