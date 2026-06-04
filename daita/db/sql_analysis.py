"""Dialect-aware SQL analysis for database guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


class SqlAnalysisError(ValueError):
    """Raised when SQL cannot be parsed or analyzed safely."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "dialect_parse_error",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class SqlTableRef:
    name: str
    parts: tuple[str, ...] = field(default_factory=tuple)
    alias: str = ""
    is_cte: bool = False

    @property
    def key(self) -> str:
        return ".".join(self.parts or (self.name,)).lower()

    @property
    def short_key(self) -> str:
        return (self.parts[-1] if self.parts else self.name).lower()


@dataclass(frozen=True)
class SqlColumnRef:
    name: str
    table: str = ""
    parts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def qualifier_key(self) -> str:
        return self.table.lower()


@dataclass(frozen=True)
class SqlSelectItem:
    alias: str
    expression_sql: str
    is_count: bool = False


@dataclass(frozen=True)
class SqlAnalysis:
    sql: str
    dialect: str
    statement_type: str
    statement_count: int
    is_read: bool
    has_limit: bool
    tables: tuple[SqlTableRef, ...]
    columns: tuple[SqlColumnRef, ...]
    select_items: tuple[SqlSelectItem, ...]
    mutating_statement_types: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_multiple_statements(self) -> bool:
        return self.statement_count > 1

    @property
    def referenced_column_names(self) -> set[str]:
        return {column.key for column in self.columns}


def analyze_sql(sql: str, *, dialect: str = "") -> SqlAnalysis:
    """Parse SQL and return a structured, dialect-aware analysis.

    ``sqlglot`` is intentionally imported lazily so base installs that never use
    SQL database features do not pay for SQL parsing dependencies.
    """

    try:
        import sqlglot
        from sqlglot import exp
        from sqlglot.errors import ParseError
    except ImportError as exc:
        raise ImportError(
            "sqlglot is required for SQL validation. Install with: "
            "pip install 'daita-agents[databases]' or the matching database extra."
        ) from exc

    normalized_dialect = normalize_sqlglot_dialect(dialect)
    parse_sql = _sql_for_parse(sql, normalized_dialect)
    try:
        expressions = sqlglot.parse(parse_sql, read=normalized_dialect)
    except ParseError as exc:
        raise SqlAnalysisError(str(exc), error_type="dialect_parse_error") from exc

    expressions = [expression for expression in expressions if expression is not None]
    if not expressions:
        raise SqlAnalysisError("Missing SQL query", error_type="dialect_parse_error")

    root = expressions[0]
    explain_inner = _explain_inner_sql(sql, root, exp)
    if explain_inner:
        inner = analyze_sql(explain_inner, dialect=dialect)
        return SqlAnalysis(
            sql=sql,
            dialect=normalized_dialect,
            statement_type="explain",
            statement_count=len(expressions),
            is_read=inner.is_read,
            has_limit=inner.has_limit,
            tables=inner.tables,
            columns=inner.columns,
            select_items=inner.select_items,
            mutating_statement_types=inner.mutating_statement_types,
        )

    cte_names = _cte_names(root, exp)
    tables = tuple(_table_refs(root, exp, cte_names))
    alias_to_table = _alias_to_table(tables)
    columns = tuple(_column_refs(root, exp))
    select_items = tuple(_select_items(root, exp, normalized_dialect))
    mutating_types = tuple(sorted(_mutating_statement_types(root, exp)))

    return SqlAnalysis(
        sql=sql,
        dialect=normalized_dialect,
        statement_type=_statement_type(root),
        statement_count=len(expressions),
        is_read=_is_read_statement(root, exp) and not mutating_types,
        has_limit=bool(root.find(exp.Limit)),
        tables=tables,
        columns=_resolve_column_qualifiers(columns, alias_to_table, cte_names),
        select_items=select_items,
        mutating_statement_types=mutating_types,
    )


def normalize_sqlglot_dialect(dialect: str) -> str:
    value = (dialect or "").lower()
    if value in {"postgresql", "postgres"}:
        return "postgres"
    if value in {"sqlite", "mysql", "snowflake", "bigquery"}:
        return value
    return value or "postgres"


def _sql_for_parse(sql: str, dialect: str) -> str:
    """Return a parser-friendly SQL string without changing executed SQL."""
    if dialect != "mysql" or "%s" not in sql:
        return sql

    pieces: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(sql):
        char = sql[i]
        if quote is not None:
            pieces.append(char)
            if char == "\\" and i + 1 < len(sql):
                i += 1
                pieces.append(sql[i])
            elif char == quote:
                quote = None
            i += 1
            continue

        if char in {"'", '"', "`"}:
            quote = char
            pieces.append(char)
            i += 1
            continue

        if sql.startswith("%s", i):
            pieces.append("?")
            i += 2
            continue

        pieces.append(char)
        i += 1

    return "".join(pieces)


def _statement_type(root: Any) -> str:
    return root.__class__.__name__.lower()


def _is_read_statement(root: Any, exp: Any) -> bool:
    if isinstance(root, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return True
    explain = getattr(exp, "Explain", None)
    if explain is not None and isinstance(root, explain):
        target = root.this
        return target is not None and _is_read_statement(target, exp)
    if isinstance(root, (exp.Show, exp.Describe)):
        return True
    return False


def _mutating_statement_types(root: Any, exp: Any) -> set[str]:
    mutating_classes = tuple(
        cls
        for cls in (
            getattr(exp, "Insert", None),
            getattr(exp, "Update", None),
            getattr(exp, "Delete", None),
            getattr(exp, "Merge", None),
            getattr(exp, "Create", None),
            getattr(exp, "Drop", None),
            getattr(exp, "Alter", None),
            getattr(exp, "TruncateTable", None),
        )
        if cls is not None
    )
    out: set[str] = set()
    for node in root.walk():
        if isinstance(node, mutating_classes):
            out.add(node.__class__.__name__.upper())
    return out


def _explain_inner_sql(sql: str, root: Any, exp: Any) -> str:
    if not isinstance(root, getattr(exp, "Command", ())):
        return ""
    match = re.match(
        r"\s*EXPLAIN(?:\s+ANALYZE)?\s+(.+)\s*$", sql, re.IGNORECASE | re.DOTALL
    )
    return match.group(1) if match else ""


def _cte_names(root: Any, exp: Any) -> set[str]:
    names: set[str] = set()
    for cte in root.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            names.add(alias.lower())
    return names


def _table_refs(root: Any, exp: Any, cte_names: set[str]) -> list[SqlTableRef]:
    refs: list[SqlTableRef] = []
    for table in root.find_all(exp.Table):
        parts = tuple(
            str(part.name) for part in table.parts if getattr(part, "name", "")
        )
        name = parts[-1] if parts else str(table.name or "")
        if not name:
            continue
        alias = table.alias or ""
        refs.append(
            SqlTableRef(
                name=name,
                parts=parts or (name,),
                alias=alias,
                is_cte=name.lower() in cte_names,
            )
        )
    return refs


def _alias_to_table(tables: tuple[SqlTableRef, ...]) -> dict[str, SqlTableRef]:
    out: dict[str, SqlTableRef] = {}
    for table in tables:
        if table.is_cte:
            out[table.short_key] = table
            if table.alias:
                out[table.alias.lower()] = table
            continue
        out[table.short_key] = table
        out[table.key] = table
        if table.alias:
            out[table.alias.lower()] = table
    return out


def _column_refs(root: Any, exp: Any) -> list[SqlColumnRef]:
    refs: list[SqlColumnRef] = []
    for column in root.find_all(exp.Column):
        parts = tuple(
            str(part.name) for part in column.parts if getattr(part, "name", "")
        )
        name = parts[-1] if parts else str(column.name or "")
        if not name:
            continue
        qualifier = str(column.table or "")
        refs.append(SqlColumnRef(name=name, table=qualifier, parts=parts or (name,)))
    return refs


def _resolve_column_qualifiers(
    columns: tuple[SqlColumnRef, ...],
    alias_to_table: dict[str, SqlTableRef],
    cte_names: set[str],
) -> tuple[SqlColumnRef, ...]:
    resolved: list[SqlColumnRef] = []
    for column in columns:
        qualifier = column.qualifier_key
        if not qualifier:
            resolved.append(column)
            continue
        table = alias_to_table.get(qualifier)
        if table is None or table.is_cte or qualifier in cte_names:
            resolved.append(column)
            continue
        resolved.append(
            SqlColumnRef(name=column.name, table=table.key, parts=column.parts)
        )
    return tuple(resolved)


def _select_items(root: Any, exp: Any, dialect: str) -> list[SqlSelectItem]:
    select = root.find(exp.Select)
    if select is None:
        return []
    items: list[SqlSelectItem] = []
    for expression in select.expressions:
        alias = expression.alias if isinstance(expression, exp.Alias) else ""
        items.append(
            SqlSelectItem(
                alias=alias,
                expression_sql=expression.sql(dialect=dialect),
                is_count=bool(expression.find(exp.Count)),
            )
        )
    return items
