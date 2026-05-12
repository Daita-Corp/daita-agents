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
    * retrieval — table index plus relationship summary; schema tools carry detail
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

    omitted_tables: List[str] = []

    if table_count == 0:
        lines.append("Database is empty. No tables found.")
    elif strategy == "full":
        for t in tables:
            row_info = (
                f" ({t['row_count']:,} rows)" if t.get("row_count") is not None else ""
            )
            lines.append(f"### {t['name']}{row_info}")
            has_comments = (
                any(col.get("column_comment") for col in t.get("columns", []))
                and policy.include_column_comments
            )
            if has_comments:
                lines.append("| Column | Type | PK | Nullable | Comment |")
                lines.append("|--------|------|----|----------|---------|")
            else:
                lines.append("| Column | Type | PK | Nullable |")
                lines.append("|--------|------|----|----------|")
            for col in t.get("columns", []):
                pk_flag = "Yes" if col.get("is_primary_key") else ""
                nullable_flag = "No" if not col.get("nullable", True) else "Yes"
                type_str = col["type"]
                if policy.include_sample_values and col.get("_samples"):
                    type_str += (
                        f" (samples: {', '.join(str(v) for v in col['_samples'])})"
                    )
                if has_comments:
                    comment = col.get("column_comment") or ""
                    lines.append(
                        f"| {col['name']} | {type_str} | {pk_flag} | {nullable_flag} | {comment} |"
                    )
                else:
                    lines.append(
                        f"| {col['name']} | {type_str} | {pk_flag} | {nullable_flag} |"
                    )
            lines.append("")
    elif strategy == "compact":
        shown_tables = tables[: policy.compact_table_limit]
        for t in shown_tables:
            row_info = (
                f" ({t['row_count']:,} rows)" if t.get("row_count") is not None else ""
            )
            col_names = ", ".join(c["name"] for c in t.get("columns", []))
            lines.append(f"### {t['name']}{row_info}")
            lines.append(f"Columns: {col_names}")
            lines.append("")
        omitted_tables = [str(t.get("name")) for t in tables[len(shown_tables) :]]
        if omitted_tables:
            lines.append(
                f"... {len(omitted_tables)} additional tables omitted from prompt summary."
            )
    else:
        shown_tables = tables[: policy.summary_table_limit]
        for t in shown_tables:
            row_count = t.get("row_count")
            col_count = len(t.get("columns", []))
            if row_count is not None:
                if row_count >= 1_000_000:
                    row_str = f"{row_count / 1_000_000:.0f}M rows"
                elif row_count >= 1_000:
                    row_str = f"{row_count / 1_000:.0f}K rows"
                else:
                    row_str = f"{row_count} rows"
                lines.append(f"- {t['name']} ({row_str}, {col_count} columns)")
            else:
                lines.append(f"- {t['name']} ({col_count} columns)")
        omitted_tables = [str(t.get("name")) for t in tables[len(shown_tables) :]]
        if omitted_tables:
            lines.append(
                f"- ... {len(omitted_tables)} additional tables omitted from prompt summary"
            )

    lines.append("")
    lines.append("## Relationships")
    if foreign_keys:
        relationship_limit = (
            50 if policy.relationship_mode == "summary" else len(foreign_keys)
        )
        for fk in foreign_keys[:relationship_limit]:
            lines.append(
                f"- {fk['source_table']}.{fk['source_column']} "
                f"→ {fk['target_table']}.{fk['target_column']}"
            )
        if len(foreign_keys) > relationship_limit:
            lines.append(
                f"- ... {len(foreign_keys) - relationship_limit} additional relationships omitted"
            )
    else:
        lines.append("No foreign key relationships discovered.")

    lines.append("")
    lines.append("## Guidelines")
    lines.append("- Use the database query tools to answer questions.")
    lines.append("- Always use LIMIT to keep result sets manageable.")
    lines.append("- When joining tables, reference the relationships above.")
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
            "When analyst tools are available, prefer them over manual SQL for statistical "
            "and pattern-based questions. Use raw SQL for straightforward data retrieval and "
            "simple aggregation."
        )
        lines.append("")
        lines.append("| Tool | When to use |")
        lines.append("|------|-------------|")
        tool_help = _analyst_tool_help()
        for tool_name in analyst_tools:
            if tool_name in tool_help:
                lines.append(f"| `{tool_name}` | {tool_help[tool_name]} |")
        lines.append("")
        lines.append(
            "Do not reach for memory/recall tools to answer data similarity or pattern questions."
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


def _select_schema_strategy(schema: Dict[str, Any], policy: SchemaPromptPolicy) -> str:
    tables = schema.get("tables", []) or []
    table_count = int(schema.get("table_count") or len(tables))
    column_count = sum(len(table.get("columns", []) or []) for table in tables)
    if table_count == 0:
        return "full"
    if policy.preferred_strategy == "retrieval":
        return "retrieval"
    if (
        table_count <= policy.max_inline_tables
        and column_count <= policy.max_inline_columns
    ):
        probe = _estimate_full_schema_tokens(schema, policy)
        if probe <= policy.max_inline_schema_tokens:
            return "full"
    if policy.preferred_strategy == "full":
        return "compact"
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


def _analyst_tool_help() -> Dict[str, str]:
    return {
        "pivot_table": (
            "Cross-tabulate data (e.g. revenue by product and month). Write the SQL "
            "first, then pass it with rows/columns/values."
        ),
        "correlate": (
            "Find which columns move together. Write SQL returning candidate columns, "
            "then call correlate."
        ),
        "detect_anomalies": (
            "Spot unusual values. Write SQL for relevant rows, specify the column to test."
        ),
        "compare_entities": (
            "Side-by-side entity comparison. Pass entity_table + IDs; dimensions are "
            "auto-inferred from FK relationships."
        ),
        "find_similar": (
            "Find entities like a reference. Pass entity_table + entity_id, and scope "
            "large searches with candidate_sql or candidate_limit."
        ),
        "forecast_trend": (
            "Project a metric forward. Write SQL returning date and metric columns; "
            "frequency is auto-detected."
        ),
    }
