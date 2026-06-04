"""
Deterministic query planning for the first DB runtime slice.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from daita.runtime import Evidence, Operation

from .models import DbIntent, DbIntentKind, DbRequest


@dataclass(frozen=True)
class DbQueryPlan:
    """A planned SQL query plus runtime evidence describing the plan."""

    sql: str | None
    evidence: Evidence
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...] = ()


class DbQueryPlanner:
    """Plan simple read SQL from a request and schema evidence."""

    def plan_read_query(
        self,
        request: DbRequest,
        intent: DbIntent,
        operation: Operation,
        schema: dict[str, Any],
        *,
        relationship_payload: dict[str, Any] | None = None,
    ) -> DbQueryPlan:
        """Build a deterministic read query for the current vertical slice."""
        warnings: list[str] = []
        explicit = _explicit_sql(request)
        if explicit:
            sql = explicit
            strategy = "explicit_sql"
        elif intent.kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY:
            from_table, to_table = self.relationship_tables_for_prompt(
                request.prompt, schema
            )
            sql = (
                _join_sql_from_schema(
                    schema,
                    from_table,
                    to_table,
                    relationship_payload or {},
                )
                if from_table and to_table
                else None
            )
            strategy = "catalog_relationship_join"
        else:
            table = self.best_table_for_prompt(request.prompt, schema)
            sql = _single_table_sql(request.prompt, schema, table) if table else None
            strategy = "single_table"

        if not sql:
            warnings.append("db_runtime_sql_planning_failed")

        diagnostics = {
            "planner": "deterministic",
            "strategy": strategy,
            "sql": sql,
            "schema_table_count": len(schema.get("tables", []) or []),
        }
        return DbQueryPlan(
            sql=sql,
            evidence=Evidence(
                kind="query.plan",
                owner="db.runtime",
                operation_id=operation.id,
                payload={
                    "sql": sql,
                    "strategy": strategy,
                    "valid": sql is not None,
                    "warnings": warnings,
                },
            ),
            diagnostics=diagnostics,
            warnings=tuple(warnings),
        )

    @staticmethod
    def best_table_for_prompt(prompt: str, schema: dict[str, Any]) -> str | None:
        """Return the most likely table referenced by a prompt."""
        return _best_table_for_prompt(prompt, schema)

    @staticmethod
    def relationship_tables_for_prompt(
        prompt: str, schema: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Return likely relationship endpoints for a join-style prompt."""
        return _relationship_tables_for_prompt(prompt, schema)


def _explicit_sql(request: DbRequest) -> str | None:
    for mapping in (request.metadata, request.constraints):
        value = mapping.get("sql") or mapping.get("query")
        if value:
            return str(value)
    return None


def _table_map(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(table.get("name")): table
        for table in schema.get("tables", []) or []
        if table.get("name")
    }


def _best_table_for_prompt(prompt: str, schema: dict[str, Any]) -> str | None:
    tables = _table_map(schema)
    if not tables:
        return None
    lowered = prompt.lower()
    ranked: list[tuple[int, str]] = []
    for table, table_schema in tables.items():
        table_lower = table.lower()
        singular = table_lower[:-1] if table_lower.endswith("s") else table_lower
        score = 0
        if table_lower in lowered:
            score += 4
        if singular and singular in lowered:
            score += 2
        for term in _table_metadata_terms(table_schema):
            if term in lowered:
                score += 3
        for column in table_schema.get("columns", []) or []:
            name = str(column.get("name") or "").lower()
            if name and name in lowered:
                score += 1
        ranked.append((score, table))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def _table_metadata_terms(table: dict[str, Any]) -> tuple[str, ...]:
    metadata = table.get("metadata") or {}
    terms: list[str] = []
    for key in ("business_name", "label", "description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            terms.extend(_term_variants(value))
    for key in ("aliases", "tags"):
        values = metadata.get(key) or ()
        if isinstance(values, str):
            values = (values,)
        for value in values:
            if isinstance(value, str) and value.strip():
                terms.extend(_term_variants(value))
    return tuple(dict.fromkeys(terms))


def _term_variants(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower().replace("_", " ")
    collapsed = " ".join(normalized.split())
    if not collapsed:
        return ()
    variants = [collapsed]
    if " " in collapsed:
        variants.append(collapsed.replace(" ", "_"))
    singular = collapsed[:-1] if collapsed.endswith("s") else collapsed
    if singular != collapsed:
        variants.append(singular)
    return tuple(variants)


def _relationship_tables_for_prompt(
    prompt: str, schema: dict[str, Any]
) -> tuple[str | None, str | None]:
    tables = _table_map(schema)
    matches = [
        table
        for table in tables
        if table.lower() in prompt.lower()
        or table.lower().rstrip("s") in prompt.lower()
    ]
    if len(matches) >= 2:
        return matches[0], matches[1]
    for fk in schema.get("foreign_keys", []) or []:
        source = str(fk.get("source_table") or "")
        target = str(fk.get("target_table") or "")
        if source in tables and target in tables:
            return source, target
    return None, None


def _single_table_sql(
    prompt: str, schema: dict[str, Any], table: str | None
) -> str | None:
    if not table:
        return None
    if _is_count_prompt(prompt):
        return f"SELECT COUNT(*) AS count FROM {_quote_ident(table)}"
    columns = _selected_columns_for_prompt(prompt, schema, table)
    where_clause = _where_clause_for_prompt(prompt, schema, table)
    sql = f"SELECT {columns} FROM {_quote_ident(table)}"
    if where_clause:
        sql = f"{sql} WHERE {where_clause}"
    return sql


def _is_count_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "count" in lowered or "how many" in lowered or "number of" in lowered


def _selected_columns_for_prompt(
    prompt: str, schema: dict[str, Any], table: str
) -> str:
    columns = _columns_for_table(schema, table)
    lowered = prompt.lower().split(" where ", 1)[0]
    selected = [
        name
        for name in columns
        if name.lower() in lowered or name.lower().replace("_", " ") in lowered
    ]
    if not selected:
        return "*"
    return ", ".join(_quote_ident(name) for name in selected)


def _where_clause_for_prompt(
    prompt: str, schema: dict[str, Any], table: str
) -> str | None:
    columns = {column.lower(): column for column in _columns_for_table(schema, table)}
    pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|=|>|<)\s*"
        r"('([^']*)'|\"([^\"]*)\"|[0-9]+(?:\.[0-9]+)?|[A-Za-z0-9_.@-]+)"
    )
    for match in pattern.finditer(prompt):
        column = columns.get(match.group(1).lower())
        if column is None:
            continue
        op = match.group(2)
        raw_value = match.group(3).strip("\"'")
        return f"{_quote_ident(column)} {op} {_sql_literal(raw_value)}"
    return None


def _join_sql_from_schema(
    schema: dict[str, Any],
    from_table: str,
    to_table: str,
    relationship_payload: dict[str, Any],
) -> str | None:
    join = _first_join(relationship_payload)
    if join is None:
        join = _foreign_key_join(schema, from_table, to_table)
    if join is None:
        return None
    left_table = join["left_table"]
    right_table = join["right_table"]
    select_list = _aliased_select_list(schema, (left_table, right_table))
    predicate = (
        f"{_quote_ident(left_table)}.{_quote_ident(join['left_column'])} = "
        f"{_quote_ident(right_table)}.{_quote_ident(join['right_column'])}"
    )
    return (
        f"SELECT {select_list} FROM {_quote_ident(left_table)} "
        f"JOIN {_quote_ident(right_table)} ON {predicate}"
    )


def _first_join(payload: dict[str, Any]) -> dict[str, str] | None:
    for path in payload.get("paths", []) or []:
        joins = path.get("joins") or []
        if joins:
            return {
                "left_table": str(joins[0]["left_table"]),
                "left_column": str(joins[0]["left_column"]),
                "right_table": str(joins[0]["right_table"]),
                "right_column": str(joins[0]["right_column"]),
            }
    return None


def _foreign_key_join(
    schema: dict[str, Any], from_table: str, to_table: str
) -> dict[str, str] | None:
    for fk in schema.get("foreign_keys", []) or []:
        source = str(fk.get("source_table") or "")
        target = str(fk.get("target_table") or "")
        if {source, target} == {from_table, to_table}:
            return {
                "left_table": source,
                "left_column": str(fk.get("source_column") or ""),
                "right_table": target,
                "right_column": str(fk.get("target_column") or ""),
            }
    return None


def _aliased_select_list(schema: dict[str, Any], tables: tuple[str, ...]) -> str:
    selected = []
    for table in tables:
        for column in _columns_for_table(schema, table):
            selected.append(
                f"{_quote_ident(table)}.{_quote_ident(column)} "
                f"AS {_quote_ident(f'{table}_{column}')}"
            )
    return ", ".join(selected) if selected else "*"


def _columns_for_table(schema: dict[str, Any], table_name: str) -> list[str]:
    table = _table_map(schema).get(table_name) or {}
    return [
        str(column.get("name"))
        for column in table.get("columns", []) or []
        if column.get("name")
    ]


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
        return value
    return "'" + value.replace("'", "''") + "'"
