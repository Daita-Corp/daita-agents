"""
Deterministic query planning for the first DB runtime slice.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from daita.runtime import Evidence, Operation

from .models import DbIntent, DbIntentKind, DbRequest
from .planning_context import planner_eligible_column_value_hint
from .query_plan import DbFilterSpec
from .query_plan import DbQueryPlan as StructuredDbQueryPlan
from .query_sql_validation import sql_fingerprint


@dataclass(frozen=True)
class DbQueryPlan:
    """A planned SQL query plus runtime evidence describing the plan."""

    sql: str | None
    evidence: Evidence
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _QueryIntent:
    """Shared deterministic query intent before SQL shape rendering."""

    tables: tuple[str, ...]
    filters: tuple[DbFilterSpec, ...]
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
        planning_context: dict[str, Any] | None = None,
    ) -> DbQueryPlan:
        """Build a deterministic read query for the current vertical slice."""
        warnings: list[str] = []
        explicit = _explicit_sql(request)
        if explicit:
            sql = explicit
            filters: tuple[DbFilterSpec, ...] = ()
            strategy = "explicit_sql"
        elif intent.kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY:
            from_table, to_table = self.relationship_tables_for_prompt(
                request.prompt, schema
            )
            query_intent = _query_intent_for_join(
                request.prompt,
                schema,
                from_table,
                to_table,
                planning_context=planning_context,
            )
            filters = query_intent.filters
            warnings.extend(query_intent.warnings)
            sql = (
                _join_sql_from_schema(
                    schema,
                    from_table,
                    to_table,
                    relationship_payload or {},
                    filters=filters,
                )
                if from_table and to_table and not query_intent.warnings
                else None
            )
            strategy = "catalog_relationship_join"
        else:
            table = self.best_table_for_prompt(request.prompt, schema)
            if table:
                query_intent = _query_intent_for_table(
                    request.prompt,
                    schema,
                    table,
                    planning_context=planning_context,
                )
                sql = _single_table_sql(request.prompt, schema, table, query_intent)
                filters = query_intent.filters
                warnings.extend(query_intent.warnings)
            else:
                sql = None
                filters = ()
            strategy = "single_table"

        if not sql:
            warnings.append("db_runtime_sql_planning_failed")

        tables = _tables_referenced_by_sql(sql, schema) if sql else ()
        structured = StructuredDbQueryPlan.deterministic(
            sql=sql,
            tables=tables,
            filters=filters,
            confidence=0.9 if sql else 0.0,
            strategy=strategy,
        )
        diagnostics = {
            "planner": "deterministic",
            "strategy": strategy,
            "sql": sql,
            "schema_table_count": len(schema.get("tables", []) or []),
        }
        return DbQueryPlan(
            sql=sql,
            evidence=Evidence(
                kind="query.plan.proposal",
                owner="db_runtime",
                operation_id=operation.id,
                payload={
                    "sql": sql,
                    "structured_plan": structured.to_dict(),
                    "strategy": strategy,
                    "valid": sql is not None,
                    "warnings": warnings,
                    "plan_fingerprint": _stable_hash(structured.to_dict()),
                    "sql_fingerprint": sql_fingerprint(sql) if sql else None,
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

    @staticmethod
    def needs_value_hint_context(prompt: str, schema: dict[str, Any]) -> bool:
        """Return whether natural-language value hints could change SQL rendering."""
        return _needs_value_hint_context(prompt, schema)


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
    lowered = _normalize_phrase(prompt)
    ranked: list[tuple[int, str]] = []
    for table, table_schema in tables.items():
        score = 0
        for term in _table_terms(table, table_schema):
            if _term_in_prompt(term, lowered):
                score += 4
        for term in _table_metadata_terms(table_schema):
            if _term_in_prompt(term, lowered):
                score += 3
        for column in table_schema.get("columns", []) or []:
            name = _normalize_phrase(str(column.get("name") or ""))
            if name and _term_in_prompt(name, lowered):
                score += 1
        ranked.append((score, table))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def _table_terms(table: str, table_schema: dict[str, Any]) -> tuple[str, ...]:
    terms = list(_term_variants(table))
    terms.extend(_table_metadata_terms(table_schema))
    return tuple(dict.fromkeys(terms))


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
    collapsed = _normalize_phrase(value)
    if not collapsed:
        return ()
    variants = [collapsed]
    if " " in collapsed:
        variants.append(collapsed.replace(" ", "_"))
    singular = collapsed[:-1] if collapsed.endswith("s") else collapsed
    if singular != collapsed:
        variants.append(singular)
    return tuple(variants)


def _normalize_phrase(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _term_in_prompt(term: str, normalized_prompt: str) -> bool:
    normalized = _normalize_phrase(term)
    if not normalized:
        return False
    return re.search(rf"\b{re.escape(normalized)}\b", normalized_prompt) is not None


def _relationship_tables_for_prompt(
    prompt: str, schema: dict[str, Any]
) -> tuple[str | None, str | None]:
    tables = _table_map(schema)
    normalized_prompt = _normalize_phrase(prompt)
    matches: list[tuple[int, str]] = []
    for table, table_schema in tables.items():
        positions = [
            normalized_prompt.find(_normalize_phrase(term))
            for term in _table_terms(table, table_schema)
            if _term_in_prompt(term, normalized_prompt)
        ]
        if positions:
            matches.append(
                (min(position for position in positions if position >= 0), table)
            )
    matches.sort(key=lambda item: (item[0], item[1]))
    if len(matches) >= 2:
        return matches[0][1], matches[1][1]
    if _has_relationship_intent(prompt) or matches:
        return None, None
    for fk in schema.get("foreign_keys", []) or []:
        source = str(fk.get("source_table") or "")
        target = str(fk.get("target_table") or "")
        if source in tables and target in tables:
            return source, target
    return None, None


def _has_relationship_intent(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(token in lowered for token in ("join", "relationship", "related"))


def _needs_value_hint_context(prompt: str, schema: dict[str, Any]) -> bool:
    tables = _hint_context_tables_for_prompt(prompt, schema)
    if not tables:
        return False
    if _filters_for_prompt(prompt, schema, tables):
        return False
    normalized_prompt = _normalize_phrase(prompt)
    for table in tables:
        table_schema = _table_map(schema).get(table) or {}
        for column in table_schema.get("columns", []) or []:
            if not _value_hint_candidate_column(column):
                continue
            name = str(column.get("name") or "")
            if any(
                _term_in_prompt(term, normalized_prompt)
                for term in _term_variants(name)
            ):
                return True
    return False


def _hint_context_tables_for_prompt(
    prompt: str, schema: dict[str, Any]
) -> tuple[str, ...]:
    if _has_relationship_intent(prompt):
        from_table, to_table = _relationship_tables_for_prompt(prompt, schema)
        return tuple(table for table in (from_table, to_table) if table)
    table = _best_table_for_prompt(prompt, schema)
    return (table,) if table else ()


def _value_hint_candidate_column(column: dict[str, Any]) -> bool:
    if column.get("is_primary_key"):
        return False
    data_type = str(column.get("type") or column.get("data_type") or "").lower()
    if not data_type:
        return True
    return any(
        token in data_type
        for token in (
            "char",
            "text",
            "enum",
            "bool",
            "int",
        )
    )


def _query_intent_for_table(
    prompt: str,
    schema: dict[str, Any],
    table: str | None,
    *,
    planning_context: dict[str, Any] | None,
) -> _QueryIntent:
    if not table:
        return _QueryIntent(tables=(), filters=())
    filters = _filters_for_prompt(
        prompt,
        schema,
        (table,),
        planning_context=planning_context,
    )
    return _QueryIntent(tables=(table,), filters=filters)


def _single_table_sql(
    prompt: str, schema: dict[str, Any], table: str | None, query_intent: _QueryIntent
) -> str | None:
    if not table:
        return None
    filters = query_intent.filters
    where_clause = _where_clause_from_filters(filters, qualify=False)
    if _is_count_prompt(prompt):
        sql = f"SELECT COUNT(*) AS count FROM {_quote_ident(table)}"
        if where_clause:
            sql = f"{sql} WHERE {where_clause}"
        return sql
    columns = _selected_columns_for_prompt(prompt, schema, table)
    sql = f"SELECT {columns} FROM {_quote_ident(table)}"
    if where_clause:
        sql = f"{sql} WHERE {where_clause}"
    return sql


def _query_intent_for_join(
    prompt: str,
    schema: dict[str, Any],
    from_table: str | None,
    to_table: str | None,
    *,
    planning_context: dict[str, Any] | None,
) -> _QueryIntent:
    tables = tuple(table for table in (from_table, to_table) if table)
    filters = _filters_for_prompt(
        prompt,
        schema,
        tables,
        planning_context=planning_context,
    )
    warnings = tuple(
        f"db_runtime_ambiguous_filter_column:{filter_spec.column}"
        for filter_spec in filters
        if "." not in str(filter_spec.column)
    )
    return _QueryIntent(tables=tables, filters=filters, warnings=warnings)


def _tables_referenced_by_sql(sql: str, schema: dict[str, Any]) -> tuple[str, ...]:
    lowered = sql.lower()
    tables = []
    for table in _table_map(schema):
        quoted = f'"{table.lower()}"'
        if table.lower() in lowered or quoted in lowered:
            tables.append(table)
    return tuple(tables)


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


def _filters_for_prompt(
    prompt: str,
    schema: dict[str, Any],
    tables: tuple[str, ...],
    *,
    planning_context: dict[str, Any] | None = None,
) -> tuple[DbFilterSpec, ...]:
    table_columns = _columns_by_table(schema, tables)
    pattern = re.compile(
        r"\b(?:(?P<table>[A-Za-z_][A-Za-z0-9_]*)\.)?"
        r"(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<operator>>=|<=|!=|=|>|<)\s*"
        r"(?P<value>'([^']*)'|\"([^\"]*)\"|[0-9]+(?:\.[0-9]+)?|[A-Za-z0-9_.@-]+)"
    )
    filters: list[DbFilterSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for match in pattern.finditer(prompt):
        resolved = _resolve_filter_column(
            table_columns,
            match.group("column"),
            explicit_table=match.group("table"),
        )
        if resolved is None:
            continue
        table, column = resolved
        op = match.group("operator")
        raw_value = match.group("value").strip("\"'")
        column_ref = f"{table}.{column}" if table else column
        key = (column_ref.lower(), op, raw_value.lower())
        if key in seen:
            continue
        seen.add(key)
        filters.append(
            DbFilterSpec(
                column=column_ref,
                operator=op,
                value=raw_value,
            )
        )
    for filter_spec in _natural_language_filters_from_hints(
        prompt,
        table_columns,
        planning_context,
    ):
        key = (
            str(filter_spec.column).lower(),
            filter_spec.operator,
            str(filter_spec.value).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        filters.append(filter_spec)
    return tuple(filters)


def _columns_by_table(
    schema: dict[str, Any], tables: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    return {
        table: {column.lower(): column for column in _columns_for_table(schema, table)}
        for table in tables
        if table
    }


def _resolve_filter_column(
    table_columns: dict[str, dict[str, str]],
    column_name: str,
    *,
    explicit_table: str | None = None,
) -> tuple[str | None, str] | None:
    column_key = str(column_name or "").lower()
    if not column_key:
        return None
    if explicit_table:
        table_key = explicit_table.lower()
        for table, columns in table_columns.items():
            if table.lower() == table_key and column_key in columns:
                return table, columns[column_key]
        return None
    matches = [
        (table, columns[column_key])
        for table, columns in table_columns.items()
        if column_key in columns
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return None, matches[0][1]
    return None


def _natural_language_filters_from_hints(
    prompt: str,
    table_columns: dict[str, dict[str, str]],
    planning_context: dict[str, Any] | None,
) -> tuple[DbFilterSpec, ...]:
    if not planning_context:
        return ()
    normalized_prompt = _normalize_phrase(prompt)
    filters: list[DbFilterSpec] = []
    for hint in planning_context.get("column_value_hints", []) or []:
        if not isinstance(hint, dict) or not planner_eligible_column_value_hint(hint):
            continue
        table = str(hint.get("table") or "")
        column = str(hint.get("column") or "")
        if not table or not column:
            continue
        resolved = _resolve_filter_column(table_columns, column, explicit_table=table)
        if resolved is None:
            continue
        resolved_table, resolved_column = resolved
        for item in hint.get("observed_values", []) or []:
            value = item.get("value") if isinstance(item, dict) else item
            if value is None:
                continue
            if not _term_in_prompt(str(value), normalized_prompt):
                continue
            filters.append(
                DbFilterSpec(
                    column=f"{resolved_table}.{resolved_column}",
                    operator="=",
                    value=str(value),
                )
            )
            break
    return tuple(filters)


def _where_clause_from_filters(
    filters: tuple[DbFilterSpec, ...], *, qualify: bool
) -> str | None:
    if not filters:
        return None
    predicates = []
    for filter_spec in filters:
        table, column = _split_filter_column(filter_spec.column)
        column_sql = (
            f"{_quote_ident(table)}.{_quote_ident(column)}"
            if qualify and table
            else _quote_ident(column)
        )
        predicates.append(
            f"{column_sql} {filter_spec.operator} {_sql_literal(str(filter_spec.value))}"
        )
    return " AND ".join(predicates)


def _split_filter_column(value: Any) -> tuple[str | None, str]:
    parts = [part for part in str(value).split(".") if part]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, parts[-1] if parts else ""


def _join_sql_from_schema(
    schema: dict[str, Any],
    from_table: str,
    to_table: str,
    relationship_payload: dict[str, Any],
    *,
    filters: tuple[DbFilterSpec, ...] = (),
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
    sql = (
        f"SELECT {select_list} FROM {_quote_ident(left_table)} "
        f"JOIN {_quote_ident(right_table)} ON {predicate}"
    )
    where_clause = _where_clause_from_filters(filters, qualify=True)
    if where_clause:
        sql = f"{sql} WHERE {where_clause}"
    return sql


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


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
