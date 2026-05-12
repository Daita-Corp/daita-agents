"""
Per-run tool selection for ``from_db()`` agents.
"""

from __future__ import annotations

from typing import Any, List

CORE_QUERY_TOOLS = ("db_query", "db_count", "db_sample", "db_find", "db_aggregate")
WRITE_QUERY_TOOLS = ("db_execute",)
SCHEMA_TOOLS = (
    "db_search_schema",
    "db_inspect_table",
    "db_describe_relationships",
    "db_list_tables",
)
ANALYST_TOOL_INTENTS = {
    "correlate": ("correlate", "correlation", "relationship between"),
    "detect_anomalies": ("anomaly", "anomalies", "outlier", "spike"),
    "pivot_table": ("pivot", "cross tab", "crosstab", "breakdown by"),
    "compare_entities": ("compare", "versus", "vs ", "difference between"),
    "find_similar": ("similar", "similarity", "nearest", "lookalike"),
    "forecast_trend": ("forecast", "predict", "trend", "projection"),
}
QUALITY_KEYWORDS = ("quality", "freshness", "completeness", "null rate", "profile")
LINEAGE_KEYWORDS = ("lineage", "dependency", "depends on", "upstream", "downstream")
WRITE_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "upsert",
    "write",
    "mutate",
    "create row",
)


def select_db_tools_for_prompt(agent: Any, prompt: str) -> List[str]:
    available = set(getattr(agent.tool_registry, "tool_names", []))
    selected: List[str] = []

    _extend_available(selected, available, CORE_QUERY_TOOLS)
    _extend_available(selected, available, SCHEMA_TOOLS)

    text = prompt.lower()
    selected.extend(_explicitly_mentioned_tools(available, text))

    for tool_name, keywords in ANALYST_TOOL_INTENTS.items():
        if tool_name in available and any(keyword in text for keyword in keywords):
            selected.append(tool_name)

    if any(keyword in text for keyword in QUALITY_KEYWORDS):
        selected.extend(name for name in available if name.startswith("dq_"))

    if any(keyword in text for keyword in WRITE_KEYWORDS):
        _extend_available(selected, available, WRITE_QUERY_TOOLS)

    if any(keyword in text for keyword in LINEAGE_KEYWORDS):
        selected.extend(
            name
            for name in available
            if name in {"trace_lineage", "find_lineage_paths", "export_lineage"}
        )

    if _has_vector_columns(getattr(agent, "_db_schema", {}) or {}):
        selected.extend(name for name in available if name.endswith("_vector_search"))

    return _dedupe([name for name in selected if name in available])


def _explicitly_mentioned_tools(available: set[str], text: str) -> List[str]:
    return [name for name in sorted(available) if name.lower() in text]


def _extend_available(
    selected: List[str], available: set[str], candidates: tuple[str, ...]
) -> None:
    selected.extend(name for name in candidates if name in available)


def _has_vector_columns(schema: dict) -> bool:
    for table in schema.get("tables", []) or []:
        for column in table.get("columns", []) or []:
            col_type = str(column.get("type", "")).lower()
            if (
                "vector" in col_type
                or "embedding" in str(column.get("name", "")).lower()
            ):
                return True
    return False


def _dedupe(names: List[str]) -> List[str]:
    seen = set()
    out = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out
