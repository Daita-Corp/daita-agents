"""
System prompt generation and domain inference.
"""

from typing import Any, Dict, List, Optional

from .policies import PromptBuildResult, SchemaPromptPolicy

DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "e-commerce": [
        "order",
        "product",
        "cart",
        "customer",
        "payment",
        "shipping",
        "inventory",
        "sku",
    ],
    "CRM": [
        "contact",
        "lead",
        "opportunity",
        "account",
        "campaign",
        "deal",
        "pipeline",
    ],
    "analytics": [
        "event",
        "metric",
        "dimension",
        "fact",
        "session",
        "pageview",
        "funnel",
    ],
    "content management": [
        "post",
        "article",
        "page",
        "comment",
        "tag",
        "media",
        "author",
    ],
    "financial": [
        "transaction",
        "ledger",
        "balance",
        "invoice",
        "journal",
        "budget",
    ],
    "healthcare": [
        "patient",
        "diagnosis",
        "prescription",
        "appointment",
        "provider",
        "claim",
    ],
    "HR": [
        "employee",
        "department",
        "salary",
        "leave",
        "payroll",
        "position",
    ],
}


def infer_domain(schema: Dict[str, Any]) -> str:
    """
    Keyword-based domain detection from table names.

    Returns the best-matching domain label, or ``"general-purpose"`` when no
    domain scores at least 2 keyword matches.
    """
    table_names = [t["name"].lower() for t in schema.get("tables", [])]

    def _strip_plural(name: str) -> str:
        if name.endswith("ies"):
            return name[:-3] + "y"
        if name.endswith("es") and len(name) > 3:
            return name[:-2]
        if name.endswith("s") and len(name) > 2:
            return name[:-1]
        return name

    # Check both original and singularized forms
    simplified = table_names + [_strip_plural(n) for n in table_names]

    best_domain = "general-purpose"
    best_score = 1  # Minimum threshold: >1 (i.e. >=2) to win

    for domain, keywords in DOMAIN_SIGNALS.items():
        score = sum(1 for kw in keywords if any(kw in name for name in simplified))
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def build_prompt(
    schema: Dict[str, Any],
    domain: str,
    user_prompt: Optional[str],
    *,
    analyst_tools: Optional[List[str]] = None,
    schema_navigation_enabled: bool = False,
    policy: Optional[SchemaPromptPolicy] = None,
) -> str:
    return build_prompt_result(
        schema,
        domain,
        user_prompt,
        analyst_tools=analyst_tools,
        schema_navigation_enabled=schema_navigation_enabled,
        policy=policy,
    ).prompt


def build_prompt_result(
    schema: Dict[str, Any],
    domain: str,
    user_prompt: Optional[str],
    *,
    analyst_tools: Optional[List[str]] = None,
    schema_navigation_enabled: bool = False,
    policy: Optional[SchemaPromptPolicy] = None,
) -> PromptBuildResult:
    """
    Generate a system prompt from the normalized schema.

    Tiered by table count, column count, and estimated token budget:

    * full — full column detail for small schemas
    * compact — table sections with column names only
    * retrieval — small table/relationship index; schema tools carry detail
    """
    policy = policy or SchemaPromptPolicy()
    db_type = schema.get("database_type", "database")
    table_count = schema.get("table_count", 0)
    tables = schema.get("tables", [])
    foreign_keys = schema.get("foreign_keys", [])
    column_count = sum(len(table.get("columns", []) or []) for table in tables)
    strategy = _select_schema_strategy(schema, policy)

    lines: List[str] = []

    lines.append(
        f"You are a data analyst agent connected to a {domain} {db_type} database."
    )

    if user_prompt:
        lines.append("")
        lines.append(user_prompt)

    lines.append("")
    lines.append(f"## Database Schema ({table_count} tables)")
    lines.append("")

    schema_lines, omitted_tables = _render_schema_tables(
        tables,
        table_count=table_count,
        strategy=strategy,
        policy=policy,
    )
    lines.extend(schema_lines)

    lines.append("")
    lines.extend(_render_relationships(foreign_keys, policy))

    lines.append("")
    lines.append("## Guidelines")
    lines.append("- Use the database query tools to answer questions.")
    lines.append(
        "- Do not ask the user to confirm routine read-only steps. Inspect schema, "
        "find join paths, run SELECT queries, and repair recoverable SQL errors "
        "autonomously."
    )
    lines.append("- Always use LIMIT to keep result sets manageable.")
    lines.append(
        "- Before writing SQL that joins tables, verify the relevant tables, columns, "
        "and join path. Use db_find_join_path when the relationship is not direct."
    )
    lines.append(
        "- If SQL fails because a table or column is missing, inspect the schema and "
        "retry with corrected SQL before giving a final answer."
    )
    lines.append(
        "- When multiple business interpretations are valid, choose the most direct "
        "one, state the assumption, and proceed unless the choice would change data."
    )
    if schema_navigation_enabled or strategy == "retrieval":
        lines.append(
            "- This schema is large. Use db_search_schema, db_list_tables, "
            "db_inspect_table, and db_describe_relationships to find omitted "
            "or ambiguous tables before writing SQL."
        )
    lines.append(
        "- Monetary and quantity columns (e.g. price, amount, total, fee) may be stored "
        "in smallest units (cents, pence, basis points). Check sample values and column "
        "comments before formatting numbers as currency."
    )
    lines.append(
        "- When presenting numeric results, include units where known "
        '(e.g. "$12.50" not "1250", "1,500 kg" not "1500").'
    )
    if analyst_tools:
        lines.append("")
        lines.append("## Analyst Toolkit")
        lines.append(
            "Statistical analysis tools may be available for correlation, anomaly, "
            "pivot, similarity, comparison, and forecasting tasks. Use raw SQL for "
            "straightforward retrieval and simple aggregation."
        )

    prompt = "\n".join(lines)
    estimated_tokens = estimate_tokens(prompt)
    budget_exceeded = estimated_tokens > policy.max_inline_schema_tokens
    return PromptBuildResult(
        prompt=prompt,
        strategy=strategy,
        estimated_tokens=estimated_tokens,
        table_count=int(table_count or len(tables)),
        column_count=column_count,
        omitted_tables=omitted_tables,
        budget_exceeded=budget_exceeded,
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
    foreign_keys: List[Dict[str, Any]], policy: SchemaPromptPolicy
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
            f"→ {fk['target_table']}.{fk['target_column']}"
        )
    if len(foreign_keys) > relationship_limit:
        lines.append(
            f"- ... {len(foreign_keys) - relationship_limit} additional relationships available via db_describe_relationships"
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
