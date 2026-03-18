"""
System prompt generation and domain inference.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "e-commerce": [
        "order", "product", "cart", "customer", "payment",
        "shipping", "inventory", "sku",
    ],
    "CRM": [
        "contact", "lead", "opportunity", "account", "campaign", "deal", "pipeline",
    ],
    "analytics": [
        "event", "metric", "dimension", "fact", "session", "pageview", "funnel",
    ],
    "content management": [
        "post", "article", "page", "comment", "tag", "media", "author",
    ],
    "financial": [
        "transaction", "ledger", "balance", "invoice", "journal", "budget",
    ],
    "healthcare": [
        "patient", "diagnosis", "prescription", "appointment", "provider", "claim",
    ],
    "HR": [
        "employee", "department", "salary", "leave", "payroll", "position",
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
        score = sum(
            1 for kw in keywords if any(kw in name for name in simplified)
        )
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def build_prompt(
    schema: Dict[str, Any],
    domain: str,
    user_prompt: Optional[str],
) -> str:
    """
    Generate a system prompt from the normalized schema.

    Tiered by table count to prevent context bloat:

    * ≤30 tables — full column detail (markdown table per table)
    * 31–80 tables — compact (column names only)
    * 81+ tables — summary only (name + row count + column count)
    """
    db_type = schema.get("database_type", "database")
    table_count = schema.get("table_count", 0)
    tables = schema.get("tables", [])
    foreign_keys = schema.get("foreign_keys", [])

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

    if table_count == 0:
        lines.append("Database is empty. No tables found.")
    elif table_count <= 30:
        for t in tables:
            row_info = (
                f" ({t['row_count']:,} rows)"
                if t.get("row_count") is not None
                else ""
            )
            lines.append(f"### {t['name']}{row_info}")
            has_comments = any(col.get("column_comment") for col in t.get("columns", []))
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
                if col.get("_samples"):
                    type_str += f" (samples: {', '.join(str(v) for v in col['_samples'])})"
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
    elif table_count <= 80:
        for t in tables:
            row_info = (
                f" ({t['row_count']:,} rows)"
                if t.get("row_count") is not None
                else ""
            )
            col_names = ", ".join(c["name"] for c in t.get("columns", []))
            lines.append(f"### {t['name']}{row_info}")
            lines.append(f"Columns: {col_names}")
            lines.append("")
    else:
        for t in tables:
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

    lines.append("")
    lines.append("## Relationships")
    if foreign_keys:
        for fk in foreign_keys:
            lines.append(
                f"- {fk['source_table']}.{fk['source_column']} "
                f"→ {fk['target_table']}.{fk['target_column']}"
            )
    else:
        lines.append("No foreign key relationships discovered.")

    lines.append("")
    lines.append("## Guidelines")
    lines.append("- Use the database query tools to answer questions.")
    lines.append("- Always use LIMIT to keep result sets manageable.")
    lines.append("- When joining tables, reference the relationships above.")
    lines.append(
        "- Monetary and quantity columns (e.g. price, amount, total, fee) may be stored "
        "in smallest units (cents, pence, basis points). Check sample values and column "
        "comments before formatting numbers as currency."
    )
    lines.append(
        "- When presenting numeric results, include units where known "
        "(e.g. \"$12.50\" not \"1250\", \"1,500 kg\" not \"1500\")."
    )
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
    lines.append(
        "| `pivot_table` | Cross-tabulate data (e.g. revenue by product and month). "
        "Write the SQL first, then pass it to pivot_table with rows/columns/values. |"
    )
    lines.append(
        "| `correlate` | Find which columns move together (e.g. 'what drives order value?'). "
        "Write SQL returning the candidate columns, then call correlate. |"
    )
    lines.append(
        "| `detect_anomalies` | Spot unusual values (e.g. 'any suspicious orders?'). "
        "Write SQL for the relevant rows, specify the column to test. |"
    )
    lines.append(
        "| `compare_entities` | Side-by-side entity comparison (e.g. 'compare Alice and Bob'). "
        "Pass entity_table + IDs; dimensions are auto-inferred from FK relationships. |"
    )
    lines.append(
        "| `find_similar` | Find entities like a reference (e.g. 'customers like Alice'). "
        "Pass entity_table + entity_id; uses normalised distance across FK-derived dimensions. |"
    )
    lines.append(
        "| `forecast_trend` | Project a metric forward (e.g. 'where is revenue heading?'). "
        "Write SQL returning date and metric columns; frequency is auto-detected. |"
    )
    lines.append("")
    lines.append(
        "Do not reach for memory/recall tools to answer data similarity or pattern questions."
    )

    return "\n".join(lines)
