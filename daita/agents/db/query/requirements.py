"""Structured answer requirements for ``from_db`` query planning."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from .metadata import normalize_identifier, required_field_matches_output

RequirementKind = Literal["field", "aggregate", "semantic"]


@dataclass(frozen=True)
class SourceColumn:
    table: str = ""
    column: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceColumn":
        return cls(
            table=str(value.get("table") or ""),
            column=str(value.get("column") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerRequirement:
    raw: str
    kind: RequirementKind = "semantic"
    source_expression: str = ""
    source_columns: tuple[SourceColumn, ...] = field(default_factory=tuple)
    output_name: str = ""
    acceptable_outputs: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnswerRequirement":
        return cls(
            raw=str(value.get("raw") or ""),
            kind=value.get("kind") or "semantic",
            source_expression=str(value.get("source_expression") or ""),
            source_columns=tuple(
                SourceColumn.from_dict(item)
                for item in value.get("source_columns") or []
                if isinstance(item, dict)
            ),
            output_name=str(value.get("output_name") or ""),
            acceptable_outputs=tuple(
                str(item)
                for item in value.get("acceptable_outputs") or []
                if str(item).strip()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def display_name(self) -> str:
        if self.kind == "aggregate" and self.output_name:
            return self.output_name
        return self.raw


@dataclass(frozen=True)
class ParsedAggregation:
    kind: str
    table: str
    column: Optional[str]
    alias: str = ""
    expression: str = ""


def parse_answer_requirements(
    values: list[str], *, aggregations: list[str] | None = None
) -> list[AnswerRequirement]:
    """Parse raw tool ``required_fields`` into the only downstream contract."""

    requirements: list[AnswerRequirement] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        requirement = _parse_requirement(raw)
        if not _contains_requirement(requirements, requirement):
            requirements.append(requirement)
    return _attach_aggregation_sources(requirements, aggregations or [])


def metric_alias_from_requirements(
    *,
    expression: str,
    kind: str,
    table: str,
    column: Optional[str],
    requirements: list[AnswerRequirement],
) -> str:
    parsed = parse_aggregation(expression)
    normalized_expression = _normalize_expression(
        parsed.expression if parsed else expression
    )
    for requirement in requirements:
        if requirement.kind != "aggregate" or not requirement.output_name:
            continue
        if (
            _normalize_expression(requirement.source_expression)
            == normalized_expression
        ):
            return requirement.output_name
        if _aggregate_sources_match(requirement, kind=kind, table=table, column=column):
            return requirement.output_name
    return ""


def requirement_covers_metric(requirement: AnswerRequirement, metric: Any) -> bool:
    if requirement.kind != "aggregate":
        return False
    if requirement.output_name and required_field_matches_output(
        requirement.output_name, getattr(metric, "name", "")
    ):
        return True
    return _aggregate_sources_match(
        requirement,
        kind=str(getattr(metric, "kind", "") or ""),
        table=str(getattr(metric, "table", "") or ""),
        column=getattr(metric, "column", None),
    )


def requirement_covers_field(requirement: AnswerRequirement, field_ref: Any) -> bool:
    table = str(getattr(field_ref, "table", "") or "")
    column = str(getattr(field_ref, "column", "") or "")
    if requirement.kind == "field":
        return any(
            _source_column_matches(source, table=table, column=column)
            for source in requirement.source_columns
        )
    return required_field_matches_output(requirement.raw, column) or (
        table and required_field_matches_output(requirement.raw, f"{table}.{column}")
    )


def output_satisfies_requirement(
    requirement: AnswerRequirement, output_name: str
) -> bool:
    candidates = [requirement.output_name, *requirement.acceptable_outputs]
    for source in requirement.source_columns:
        if source.column:
            candidates.append(source.column)
        if source.table and source.column:
            candidates.append(f"{source.table}.{source.column}")
    candidates.append(requirement.raw)
    return any(
        candidate and required_field_matches_output(candidate, output_name)
        for candidate in candidates
    )


def parse_aggregation(expression: str) -> ParsedAggregation | None:
    match = re.search(
        r"\b(count|sum|avg|min|max)\s*\(\s*(distinct\s+)?(?:(?P<table>[A-Za-z_][\w.]*)\.)?(?P<column>[A-Za-z_][\w]*|\*)?\s*\)(?:\s+as\s+(?P<alias>[A-Za-z_][\w]*))?",
        expression or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw_kind = match.group(1).lower()
    distinct = bool(match.group(2))
    kind = "distinct_count" if raw_kind == "count" and distinct else raw_kind
    column = match.group("column")
    column = None if column in (None, "*") else column
    table = match.group("table") or ""
    alias = match.group("alias") or ""
    source = "*" if column is None else f"{table + '.' if table else ''}{column}"
    func = "COUNT" if kind == "distinct_count" else raw_kind.upper()
    inner = f"DISTINCT {source}" if kind == "distinct_count" else source
    return ParsedAggregation(
        kind=kind,
        table=table,
        column=column,
        alias=alias,
        expression=f"{func}({inner})",
    )


def _parse_requirement(raw: str) -> AnswerRequirement:
    aggregate = parse_aggregation(raw)
    if aggregate is not None:
        output_name = aggregate.alias or _trailing_alias(raw)
        outputs = _unique_strings([output_name, _snake_case(output_name)])
        source_columns = (
            (SourceColumn(aggregate.table, aggregate.column),)
            if aggregate.column
            else tuple()
        )
        return AnswerRequirement(
            raw=raw,
            kind="aggregate",
            source_expression=aggregate.expression,
            source_columns=source_columns,
            output_name=output_name,
            acceptable_outputs=tuple(outputs),
        )

    field_ref = _parse_field_ref(raw)
    if field_ref is not None:
        table, column = field_ref
        return AnswerRequirement(
            raw=raw,
            kind="field",
            source_columns=(SourceColumn(table, column),),
            output_name=column,
            acceptable_outputs=tuple(
                _unique_strings([column, raw, f"{table}.{column}" if table else ""])
            ),
        )

    output_name = _snake_case(raw)
    return AnswerRequirement(
        raw=raw,
        kind="semantic",
        output_name=output_name,
        acceptable_outputs=tuple(_unique_strings([raw, output_name])),
    )


def _attach_aggregation_sources(
    requirements: list[AnswerRequirement], aggregations: list[str]
) -> list[AnswerRequirement]:
    if not requirements or not aggregations:
        return requirements

    resolved = list(requirements)
    for expression in aggregations:
        aggregate = parse_aggregation(expression)
        if aggregate is None or not aggregate.column:
            continue
        for index, requirement in enumerate(resolved):
            if requirement.kind not in {"field", "semantic"}:
                continue
            if not _requirement_matches_aggregation(requirement, aggregate):
                continue
            outputs = _unique_strings(
                [
                    requirement.raw,
                    requirement.output_name,
                    aggregate.alias,
                    _snake_case(aggregate.alias),
                ]
            )
            resolved[index] = AnswerRequirement(
                raw=requirement.raw,
                kind="aggregate",
                source_expression=aggregate.expression,
                source_columns=(SourceColumn(aggregate.table, aggregate.column),),
                output_name=aggregate.alias or requirement.output_name,
                acceptable_outputs=tuple(outputs),
            )
            break
    return resolved


def _requirement_matches_aggregation(
    requirement: AnswerRequirement, aggregate: ParsedAggregation
) -> bool:
    if requirement.kind == "field":
        return any(
            _source_column_matches(
                source, table=aggregate.table, column=aggregate.column or ""
            )
            for source in requirement.source_columns
        )
    candidates = [aggregate.alias, aggregate.column or "", aggregate.expression]
    return any(
        candidate and required_field_matches_output(requirement.raw, candidate)
        for candidate in candidates
    )


def _parse_field_ref(raw: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"\s*(?:(?P<table>[A-Za-z_][\w.]*)\.)?(?P<column>[A-Za-z_][\w]*)\s*",
        raw,
    )
    if not match:
        return None
    table = match.group("table") or ""
    if not table:
        return None
    column = match.group("column")
    if column.lower() in {"sum", "avg", "min", "max", "count"}:
        return None
    return table, column


def _trailing_alias(raw: str) -> str:
    match = re.search(r"\bas\s+([A-Za-z_][\w]*)\s*$", raw or "", re.IGNORECASE)
    return match.group(1) if match else ""


def _aggregate_sources_match(
    requirement: AnswerRequirement,
    *,
    kind: str,
    table: str,
    column: Optional[str],
) -> bool:
    if requirement.kind != "aggregate":
        return False
    if requirement.source_expression:
        parsed = parse_aggregation(requirement.source_expression)
        if parsed and parsed.kind and parsed.kind != kind:
            return False
    if not requirement.source_columns:
        return True
    return any(
        _source_column_matches(source, table=table, column=column or "")
        for source in requirement.source_columns
    )


def _source_column_matches(source: SourceColumn, *, table: str, column: str) -> bool:
    if normalize_identifier(source.column) != normalize_identifier(column):
        return False
    if not source.table:
        return True
    source_table = source.table.lower()
    table_key = table.lower()
    return source_table == table_key or source_table == table_key.split(".")[-1]


def _contains_requirement(
    requirements: list[AnswerRequirement], requirement: AnswerRequirement
) -> bool:
    normalized_raw = normalize_identifier(requirement.raw)
    return any(
        normalize_identifier(item.raw) == normalized_raw for item in requirements
    )


def _normalize_expression(expression: str) -> str:
    return normalize_identifier(re.sub(r"\bas\s+[A-Za-z_][\w]*\s*$", "", expression))


def _snake_case(value: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", str(value or "").lower())
    return "_".join(tokens)


def _unique_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out
