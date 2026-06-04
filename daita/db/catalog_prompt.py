"""Prompt read model for DB agents backed by catalog profile facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from daita.db.config.policies import SchemaPromptPolicy
from daita.db.query_tool_views import CATALOG_RELATIONSHIP_TOOL_VIEW


@dataclass(frozen=True)
class DBPromptReadModel:
    """Rendered schema facts and budget metadata for the DB system prompt."""

    database_type: str
    strategy: str
    table_count: int
    column_count: int
    schema_lines: List[str] = field(default_factory=list)
    relationship_lines: List[str] = field(default_factory=list)
    omitted_tables: List[str] = field(default_factory=list)


def build_db_prompt_read_model(
    schema: Dict[str, Any],
    *,
    policy: Optional[SchemaPromptPolicy] = None,
    relationship_tool: str = CATALOG_RELATIONSHIP_TOOL_VIEW,
) -> DBPromptReadModel:
    """Render bounded DB schema facts for prompt assembly."""
    policy = policy or SchemaPromptPolicy()
    tables = schema.get("tables", []) or []
    table_count = int(schema.get("table_count") or len(tables))
    foreign_keys = schema.get("foreign_keys", []) or []
    column_count = sum(len(table.get("columns", []) or []) for table in tables)
    strategy = _select_schema_strategy(schema, policy)
    schema_lines, omitted_tables = _render_schema_tables(
        tables,
        table_count=table_count,
        strategy=strategy,
        policy=policy,
    )
    return DBPromptReadModel(
        database_type=str(schema.get("database_type", "database")),
        strategy=strategy,
        table_count=table_count,
        column_count=column_count,
        schema_lines=schema_lines,
        relationship_lines=_render_relationships(
            foreign_keys, policy, relationship_tool=relationship_tool
        ),
        omitted_tables=omitted_tables,
    )


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic estimate suitable for prompt-budget tests."""
    return max(1, (len(text) + 3) // 4)


def _render_schema_tables(
    tables: List[Dict[str, Any]],
    *,
    table_count: int,
    strategy: str,
    policy: SchemaPromptPolicy,
) -> tuple[List[str], List[str]]:
    if table_count == 0:
        return ["Database is empty. No tables found."], []
    if strategy == "full":
        return _render_full_schema(tables, policy), []
    if strategy == "compact":
        shown_tables = tables[: policy.compact_table_limit]
        omitted_tables = [str(t.get("name")) for t in tables[len(shown_tables) :]]
        lines = _render_compact_schema(shown_tables)
        if omitted_tables:
            lines.append(
                f"... {len(omitted_tables)} additional tables omitted from prompt summary."
            )
        return lines, omitted_tables

    shown_tables = tables[: policy.summary_table_limit]
    omitted_tables = [str(t.get("name")) for t in tables[len(shown_tables) :]]
    lines = _render_retrieval_schema(shown_tables)
    if omitted_tables:
        lines.append(
            f"- ... {len(omitted_tables)} additional tables omitted from prompt summary"
        )
    return lines, omitted_tables


def _render_full_schema(
    tables: List[Dict[str, Any]], policy: SchemaPromptPolicy
) -> List[str]:
    lines: List[str] = []
    for table in tables:
        lines.append(f"### {table['name']}{_row_count_suffix(table.get('row_count'))}")
        has_comments = (
            any(col.get("column_comment") for col in table.get("columns", []))
            and policy.include_column_comments
        )
        if has_comments:
            lines.append("| Column | Type | PK | Nullable | Comment |")
            lines.append("|--------|------|----|----------|---------|")
        else:
            lines.append("| Column | Type | PK | Nullable |")
            lines.append("|--------|------|----|----------|")
        for col in table.get("columns", []):
            lines.append(
                _render_column_row(col, has_comments=has_comments, policy=policy)
            )
        lines.append("")
    return lines


def _render_column_row(
    col: Dict[str, Any],
    *,
    has_comments: bool,
    policy: SchemaPromptPolicy,
) -> str:
    pk_flag = "Yes" if col.get("is_primary_key") else ""
    nullable_flag = "No" if not col.get("nullable", True) else "Yes"
    type_str = col["type"]
    if policy.include_sample_values and col.get("_samples"):
        type_str += f" (samples: {', '.join(str(v) for v in col['_samples'])})"
    if has_comments:
        comment = col.get("column_comment") or ""
        return (
            f"| {col['name']} | {type_str} | {pk_flag} | {nullable_flag} | {comment} |"
        )
    return f"| {col['name']} | {type_str} | {pk_flag} | {nullable_flag} |"


def _render_compact_schema(tables: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for table in tables:
        col_names = ", ".join(c["name"] for c in table.get("columns", []))
        lines.append(f"### {table['name']}{_row_count_suffix(table.get('row_count'))}")
        lines.append(f"Columns: {col_names}")
        lines.append("")
    return lines


def _render_retrieval_schema(tables: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for table in tables:
        row_label = _compact_row_count_label(table.get("row_count"))
        col_count = len(table.get("columns", []))
        if row_label is not None:
            lines.append(f"- {table['name']} ({row_label}, {col_count} columns)")
        else:
            lines.append(f"- {table['name']} ({col_count} columns)")
    return lines


def _render_relationships(
    foreign_keys: List[Dict[str, Any]],
    policy: SchemaPromptPolicy,
    *,
    relationship_tool: str,
) -> List[str]:
    lines = ["## Relationship Hints"]
    if not foreign_keys:
        lines.append("No foreign key relationships discovered.")
        return lines

    relationship_limit = (
        policy.max_inline_relationships
        if policy.relationship_mode == "summary"
        else len(foreign_keys)
    )
    for fk in foreign_keys[:relationship_limit]:
        lines.append(
            f"- {fk['source_table']}.{fk['source_column']} "
            f"-> {fk['target_table']}.{fk['target_column']}"
        )
    if len(foreign_keys) > relationship_limit:
        lines.append(
            f"- ... {len(foreign_keys) - relationship_limit} additional relationships available via {relationship_tool}"
        )
    return lines


def _row_count_suffix(row_count: Any) -> str:
    return f" ({row_count:,} rows)" if row_count is not None else ""


def _compact_row_count_label(row_count: Any) -> Optional[str]:
    if row_count is None:
        return None
    if row_count >= 1_000_000:
        return f"{row_count / 1_000_000:.0f}M rows"
    if row_count >= 1_000:
        return f"{row_count / 1_000:.0f}K rows"
    return f"{row_count} rows"


def _select_schema_strategy(schema: Dict[str, Any], policy: SchemaPromptPolicy) -> str:
    tables = schema.get("tables", []) or []
    table_count = int(schema.get("table_count") or len(tables))
    column_count = sum(len(table.get("columns", []) or []) for table in tables)
    if table_count == 0:
        return "full"
    if policy.preferred_strategy == "retrieval":
        return "retrieval"
    if policy.preferred_strategy == "compact":
        return "compact"
    if (
        table_count <= policy.max_inline_tables
        and column_count <= policy.max_inline_columns
    ):
        probe = _estimate_full_schema_tokens(schema, policy)
        if probe <= policy.max_inline_schema_tokens:
            return "full"

    if policy.preferred_strategy == "full":
        return "compact"

    if column_count > policy.max_inline_columns:
        return "retrieval"

    if table_count <= policy.compact_table_limit:
        return "compact"
    return "retrieval"


def _estimate_full_schema_tokens(
    schema: Dict[str, Any], policy: SchemaPromptPolicy
) -> int:
    parts: List[str] = []
    for table in schema.get("tables", []) or []:
        parts.append(str(table.get("name", "")))
        for col in table.get("columns", []) or []:
            parts.append(str(col.get("name", "")))
            parts.append(str(col.get("type", "")))
            if policy.include_column_comments:
                parts.append(str(col.get("column_comment", "")))
            if policy.include_sample_values:
                parts.extend(str(v) for v in col.get("_samples", []) or [])
    return estimate_tokens("\n".join(parts))
