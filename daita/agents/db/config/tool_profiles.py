"""
Per-run tool selection for ``from_db()`` agents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

from ..query.catalog_adapter import catalog_schema_snapshot, has_likely_catalog_match
from ..utils import unique_preserving_order
from .policies import (
    DB_MEMORY_TOOLS,
    SCHEMA_NAVIGATION_TOOLS,
    WRITE_DB_TOOLS,
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
MEMORY_KEYWORDS = (
    "remember",
    "store memory",
    "save memory",
    "note for later",
    "update_memory",
    "memory",
)
MEMORY_WRITE_KEYWORDS = (
    "remember",
    "store memory",
    "save memory",
    "note for later",
    "update_memory",
)
SCHEMA_KEYWORDS = (
    "schema",
    "table",
    "tables",
    "column",
    "columns",
    "describe",
    "structure",
    "relationship",
    "relationships",
)
QUERY_KEYWORDS = (
    "how many",
    "count",
    "total",
    "sum",
    "average",
    "avg",
    "minimum",
    "maximum",
    "min ",
    "max ",
    "list",
    "show",
    "find",
    "lookup",
    "which",
    "what is",
    "who is",
)
DATA_QUERY_KEYWORDS = (
    "row",
    "rows",
    "record",
    "records",
    "data",
    "value",
    "values",
    "where",
    "filter",
)
INFERENCE_ONLY_KEYWORDS = (
    "what kind",
    "what type",
    "describe this database",
    "tell me about this database",
    "company",
    "business",
)
WRITE_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "upsert",
    "write",
    "mutate",
    "create row",
)
SCHEMA_REPAIR_KEYWORDS = (
    "join",
    "joins",
    "foreign key",
    "foreign keys",
    "relationship",
    "relationships",
    "related",
    "connect",
    "path",
)
QUERY_SHAPE_KEYWORDS = (
    "after",
    "before",
    "between",
    "breakdown",
    "group",
    "grouped",
    "highest",
    "include",
    "lowest",
    "matching",
    "most",
    "least",
    "per ",
    "top",
    "where",
)
DATA_EXECUTION_KEYWORDS = (
    "calculate",
    "compute",
    "execute",
    "query",
    "run sql",
    "run a query",
    "by month",
    "by week",
    "by day",
    "per ",
    "group by",
)
SCHEMA_SEARCH_PATTERNS = (
    r"\bsearch\s+(?:the\s+)?schema\b",
    r"\binspect\s+(?:the\s+)?schema\b",
    r"\bscan\s+(?:the\s+)?schema\b",
    r"\bschema\s+(?:for|to find|search)\b",
    r"\b(?:which|what)\s+columns?\b",
    r"\bcolumns?\s+(?:that|which|might|may|could|look|represent|related|safe|safest)\b",
    r"\bfind\s+(?:tables?|columns?)\b",
    r"\bfind\b.*\bcolumns?\b",
    r"\btables?\s+related\b",
    r"\bdo not query data\b",
    r"\bjust inspect\s+(?:the\s+)?schema\b",
)


@dataclass(frozen=True)
class DbToolProfile:
    """Deterministic tool profile for one ``from_db`` prompt."""

    intent: str
    tools: List[str]
    required_phases: List[str] = field(default_factory=list)


def select_db_tool_profile(agent: Any, prompt: str) -> DbToolProfile:
    available = set(getattr(agent.tool_registry, "tool_names", []))
    selected: List[str] = []
    text = prompt.lower()

    explicit_tools = _explicitly_mentioned_tools(available, text)
    if explicit_tools and _is_strict_explicit_tool_request(text):
        return DbToolProfile(
            intent=_classify_db_intent(text, likely_catalog_match=False),
            tools=unique_preserving_order(explicit_tools),
            required_phases=[],
        )

    intent = _classify_db_intent(text, likely_catalog_match=False)
    needs_query_tools = intent in {
        "manual_sql",
        "admin_or_write",
        "data_query_simple",
        "data_query_catalog_assisted",
    }
    likely_catalog_match = (
        has_likely_catalog_match(agent, text) if needs_query_tools else False
    )
    intent = _classify_db_intent(text, likely_catalog_match=likely_catalog_match)
    needs_query_tools = intent in {
        "manual_sql",
        "admin_or_write",
        "data_query_simple",
        "data_query_catalog_assisted",
    }
    needs_schema_tools = _profile_needs_schema_tools(
        text, intent, needs_query_tools, likely_catalog_match=likely_catalog_match
    )

    compile_first = _should_start_with_compile_first(
        available,
        text,
        needs_query_tools=needs_query_tools,
    )

    if intent == "manual_sql":
        _extend_available(selected, available, ("db_validate_sql", "db_query"))
    elif intent == "admin_or_write":
        _extend_available(selected, available, WRITE_DB_TOOLS)
        _extend_available(selected, available, ("db_validate_sql", "db_query"))
    elif intent == "data_query_simple":
        if compile_first:
            _extend_available(selected, available, ("db_compile_and_query",))
        else:
            _extend_available(selected, available, ("db_plan_query", "db_query"))
    elif intent == "data_query_catalog_assisted":
        _extend_available(selected, available, ("db_plan_query", "db_query"))
        if any(
            keyword in text for keyword in ("validate", "validation", "debug", "repair")
        ):
            _extend_available(selected, available, ("db_validate_sql",))

    if needs_schema_tools:
        if intent == "data_query_catalog_assisted":
            _extend_available(selected, available, SCHEMA_NAVIGATION_TOOLS)
        elif compile_first and not likely_catalog_match:
            _extend_available(selected, available, ("catalog_search_schema",))
        else:
            _extend_available(selected, available, SCHEMA_NAVIGATION_TOOLS)

    selected.extend(explicit_tools)

    for tool_name, keywords in ANALYST_TOOL_INTENTS.items():
        if tool_name in available and any(keyword in text for keyword in keywords):
            selected.append(tool_name)

    if any(keyword in text for keyword in QUALITY_KEYWORDS):
        selected.extend(name for name in available if name.startswith("dq_"))

    if any(keyword in text for keyword in WRITE_KEYWORDS):
        _extend_available(selected, available, WRITE_DB_TOOLS)

    if any(keyword in text for keyword in LINEAGE_KEYWORDS):
        selected.extend(
            name
            for name in available
            if name in {"trace_lineage", "find_lineage_paths", "export_lineage"}
        )

    if hasattr(agent, "_db_memory_semantics") and (
        intent == "memory_only"
        or any(keyword in text for keyword in MEMORY_WRITE_KEYWORDS)
    ):
        _extend_available(selected, available, DB_MEMORY_TOOLS)

    if _has_vector_columns(catalog_schema_snapshot(agent)):
        selected.extend(name for name in available if name.endswith("_vector_search"))

    tools = unique_preserving_order([name for name in selected if name in available])
    return DbToolProfile(
        intent=intent,
        tools=tools,
        required_phases=_required_phases(intent, tools),
    )


def select_db_tools_for_prompt(agent: Any, prompt: str) -> List[str]:
    return select_db_tool_profile(agent, prompt).tools


def _should_start_with_compile_first(
    available: set[str], text: str, *, needs_query_tools: bool
) -> bool:
    return (
        needs_query_tools
        and "db_compile_and_query" in available
        and not _needs_advanced_db_tools(available, text)
    )


def _needs_advanced_db_tools(available: set[str], text: str) -> bool:
    if _explicitly_mentioned_tools(available, text):
        return True
    if any(keyword in text for keyword in WRITE_KEYWORDS):
        return True
    if any(keyword in text for keyword in SCHEMA_REPAIR_KEYWORDS):
        return True
    if any(keyword in text for keyword in QUALITY_KEYWORDS + LINEAGE_KEYWORDS):
        return True
    if any(
        any(keyword in text for keyword in keywords)
        for keywords in ANALYST_TOOL_INTENTS.values()
    ):
        return True
    if any(
        keyword in text for keyword in ("validate", "validation", "debug", "repair")
    ):
        return True
    if _needs_staged_query_path(text):
        return True
    return _has_explicit_sql(text)


def _needs_staged_query_path(text: str) -> bool:
    if _is_simple_count_question(text):
        return False
    return any(keyword in text for keyword in QUERY_SHAPE_KEYWORDS)


def _is_simple_count_question(text: str) -> bool:
    return (
        any(keyword in text for keyword in ("how many", "count"))
        and not any(keyword in text for keyword in QUERY_SHAPE_KEYWORDS)
        and not _has_explicit_sql(text)
    )


def _has_explicit_sql(text: str) -> bool:
    return bool(re.search(r"\b(select|with|explain|sql)\b", text))


def _profile_needs_schema_tools(
    text: str,
    intent: str,
    needs_query_tools: bool,
    *,
    likely_catalog_match: bool,
) -> bool:
    explicit_schema = any(tool.lower() in text for tool in SCHEMA_NAVIGATION_TOOLS)
    if explicit_schema:
        return True
    if intent == "data_query_catalog_assisted":
        return True
    if intent == "schema_only":
        return True
    if any(keyword in text for keyword in SCHEMA_REPAIR_KEYWORDS):
        return True
    if _is_schema_question(text):
        return True
    if not needs_query_tools:
        return False
    return not likely_catalog_match


def _classify_db_intent(text: str, *, likely_catalog_match: bool) -> str:
    if _has_explicit_sql(text):
        return "manual_sql"
    if any(keyword in text for keyword in WRITE_KEYWORDS):
        return "admin_or_write"
    if _is_memory_only_intent(text):
        return "memory_only"
    if _is_schema_only_intent(text):
        return "schema_only"
    if _has_catalog_assisted_data_intent(text, likely_catalog_match):
        return "data_query_catalog_assisted"
    return "data_query_simple"


def _is_schema_only_intent(text: str) -> bool:
    if _has_schema_search_phrase(text) and not _has_data_execution_intent(text):
        return True
    if any(keyword in text for keyword in INFERENCE_ONLY_KEYWORDS):
        return True
    if any(keyword in text for keyword in SCHEMA_KEYWORDS) and not any(
        keyword in text
        for keyword in DATA_QUERY_KEYWORDS
        + (
            "how many",
            "count",
            "total",
            "sum",
            "average",
            "avg",
            "minimum",
            "maximum",
            "highest",
            "lowest",
            "top",
        )
    ):
        return True
    return _is_schema_question(text) and not _has_data_answer_intent(text)


def _has_catalog_assisted_data_intent(text: str, likely_catalog_match: bool) -> bool:
    if not _has_data_answer_intent(text) and not any(
        keyword in text for keyword in QUERY_KEYWORDS
    ):
        return False
    if _has_schema_search_phrase(text):
        return True
    if any(tool in text for tool in SCHEMA_NAVIGATION_TOOLS):
        return True
    if any(keyword in text for keyword in SCHEMA_REPAIR_KEYWORDS):
        return True
    if "catalog" in text or "graph" in text:
        return True
    if not likely_catalog_match and _needs_staged_query_path(text):
        return True
    return False


def _required_phases(intent: str, tools: List[str]) -> List[str]:
    phases = []
    if intent == "data_query_catalog_assisted" and any(
        name.startswith("catalog_") for name in tools
    ):
        phases.append("catalog")
    if any(name in tools for name in ("db_plan_query", "db_compile_and_query")):
        phases.append("plan")
    if any(name in tools for name in ("db_query", "db_compile_and_query")):
        phases.append("execute")
    return phases


def _has_data_answer_intent(text: str) -> bool:
    if _has_data_execution_intent(text):
        return True
    if any(keyword in text for keyword in DATA_QUERY_KEYWORDS):
        return True
    if _needs_staged_query_path(text):
        return True
    return any(
        keyword in text
        for keyword in (
            "how many",
            "count",
            "total",
            "sum",
            "average",
            "avg",
            "minimum",
            "maximum",
        )
    )


def _is_schema_question(text: str) -> bool:
    return any(keyword in text for keyword in SCHEMA_KEYWORDS) and not any(
        keyword in text for keyword in DATA_QUERY_KEYWORDS
    )


def _has_schema_search_phrase(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in SCHEMA_SEARCH_PATTERNS)


def _has_data_execution_intent(text: str) -> bool:
    if (
        "do not query" in text
        or "without querying" in text
        or "just inspect the schema" in text
    ):
        return False
    return any(keyword in text for keyword in DATA_EXECUTION_KEYWORDS)


def _is_memory_only_intent(text: str) -> bool:
    if not any(keyword in text for keyword in MEMORY_KEYWORDS):
        return False
    return not any(
        keyword in text
        for keyword in (
            SCHEMA_KEYWORDS
            + QUERY_KEYWORDS
            + DATA_QUERY_KEYWORDS
            + WRITE_KEYWORDS
            + DATA_EXECUTION_KEYWORDS
        )
    )


def _explicitly_mentioned_tools(available: set[str], text: str) -> List[str]:
    return [name for name in sorted(available) if name.lower() in text]


def _is_strict_explicit_tool_request(text: str) -> bool:
    return any(marker in text for marker in ("exactly once", "only", "just"))


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
