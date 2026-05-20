"""
System prompt generation and domain inference.
"""

from typing import Any, Dict, List, Optional

from .catalog_prompt import build_db_prompt_read_model, estimate_tokens
from .config.policies import PromptBuildResult, SchemaPromptPolicy

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
    catalog_tools_enabled: bool = True,
    catalog_store_id: Optional[str] = None,
    policy: Optional[SchemaPromptPolicy] = None,
) -> str:
    return build_prompt_result(
        schema,
        domain,
        user_prompt,
        analyst_tools=analyst_tools,
        catalog_tools_enabled=catalog_tools_enabled,
        catalog_store_id=catalog_store_id,
        policy=policy,
    ).prompt


def build_prompt_result(
    schema: Dict[str, Any],
    domain: str,
    user_prompt: Optional[str],
    *,
    analyst_tools: Optional[List[str]] = None,
    catalog_tools_enabled: bool = True,
    catalog_store_id: Optional[str] = None,
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
    prompt_model = build_db_prompt_read_model(schema, policy=policy)

    lines: List[str] = []

    lines.append(
        f"You are a data analyst agent connected to a {domain} {prompt_model.database_type} database."
    )

    if user_prompt:
        lines.append("")
        lines.append(user_prompt)

    lines.append("")
    lines.append(f"## Database Schema ({prompt_model.table_count} tables)")
    if catalog_store_id:
        lines.append(f"Active catalog store_id: {catalog_store_id}")
    lines.append("")

    lines.extend(prompt_model.schema_lines)

    lines.append("")
    lines.extend(prompt_model.relationship_lines)

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
        "- For clear count, top-N, grouped aggregation, and simple filtered "
        "questions, prefer db_compile_and_query first. Use db_plan_query for "
        "ambiguous, multi-table, join-heavy, validation, or debugging workflows."
    )
    lines.append(
        "- Use catalog/schema tools to traverse tables, columns, and relationships; "
        "use SQL only for retrieving or aggregating specific values."
    )
    lines.append(
        "- For count questions, use COUNT(*) or COUNT(primary_key) and alias the "
        "result; do not SUM a similarly named column unless the schema confirms "
        "that column stores the requested metric."
    )
    lines.append(
        "- Before writing SQL that joins tables, verify the relevant tables, columns, "
        "and join path. Use catalog_find_join_paths when the relationship is not direct."
    )
    lines.append(
        "- If SQL fails because a table or column is missing, inspect the schema and "
        "retry with corrected SQL before giving a final answer."
    )
    lines.append(
        "- If db_query returns repair_required or preflight_failed, never call "
        "db_query again with the same SQL. Use catalog_inspect_table, "
        "catalog_search_schema, or catalog_find_join_paths, then call "
        "db_validate_sql or db_query with corrected SQL."
    )
    lines.append(
        "- When multiple business interpretations are valid, choose the most direct "
        "one, state the assumption, and proceed unless the choice would change data."
    )
    if catalog_tools_enabled or prompt_model.strategy == "retrieval":
        lines.append(
            "- Use the active catalog store ID with catalog_search_schema, "
            "catalog_inspect_table, and catalog_find_join_paths to find omitted "
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
        strategy=prompt_model.strategy,
        estimated_tokens=estimated_tokens,
        table_count=prompt_model.table_count,
        column_count=prompt_model.column_count,
        omitted_tables=prompt_model.omitted_tables,
        budget_exceeded=budget_exceeded,
    )
