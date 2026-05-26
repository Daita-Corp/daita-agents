"""Deterministic prompt classification for ``Agent.from_db()``.

This module is the single owner for the current phrase-based fallback
classifier. The rest of the DB runtime should consume ``DbIntent`` and prompt
signals rather than embedding language heuristics directly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .intent import DbIntent, DbIntentKind
from .tool_selection import CATALOG_NAVIGATION_TOOL_NAMES


@dataclass(frozen=True)
class DbPromptClassification:
    """Structured prompt signals used by DB tool selection."""

    intent: DbIntent
    likely_catalog_match: bool = False
    query_shape: str | None = None
    aggregation_kind: str | None = None
    explicit_tools: tuple[str, ...] = ()
    strict_explicit_tool_request: bool = False
    needs_schema_tools: bool = False
    compile_first: bool = False
    requested_analyst_tools: tuple[str, ...] = ()
    needs_semantic_memory: bool = False
    semantic_memory_terms: tuple[str, ...] = ()
    needs_quality_tools: bool = False
    needs_write_tools: bool = False
    needs_lineage_tools: bool = False
    needs_memory_tools: bool = False
    needs_validation_tool: bool = False


ANALYST_TOOL_INTENTS = {
    "correlate": ("correlate", "correlation", "relationship between"),
    "detect_anomalies": ("anomaly", "anomalies", "outlier", "spike"),
    "pivot_table": ("pivot", "cross tab", "crosstab", "breakdown by"),
    "compare_entities": ("compare", "versus", "vs ", "difference between"),
    "find_similar": ("similar", "similarity", "nearest", "lookalike"),
    "forecast_trend": ("forecast", "predict", "trend", "projection"),
}
QUALITY_TERMS = ("quality", "freshness", "completeness", "null rate", "profile")
LINEAGE_TERMS = ("lineage", "dependency", "depends on", "upstream", "downstream")
MEMORY_TERMS = (
    "remember",
    "store memory",
    "save memory",
    "note for later",
    "update_memory",
    "memory",
)
MEMORY_WRITE_TERMS = (
    "remember",
    "store memory",
    "save memory",
    "note for later",
    "update_memory",
)
SEMANTIC_MEMORY_TERMS = (
    "business rule",
    "business rules",
    "calculate",
    "calculation",
    "caveat",
    "caveats",
    "definition",
    "definitions",
    "exclude",
    "excludes",
    "include",
    "includes",
    "known issue",
    "known issues",
    "meaning",
    "metric",
    "metrics",
    "remember",
    "stored rule",
    "unit",
    "units",
    "what does",
    "you said",
)
SCHEMA_TERMS = (
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
QUERY_TERMS = (
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
DATA_QUERY_TERMS = (
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
INFERENCE_ONLY_TERMS = (
    "what kind",
    "what type",
    "describe this database",
    "tell me about this database",
    "company",
    "business",
)
WRITE_TERMS = (
    "insert",
    "update",
    "delete",
    "upsert",
    "write",
    "mutate",
    "create row",
)
SCHEMA_REPAIR_TERMS = (
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
QUERY_SHAPE_TERMS = (
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
DATA_EXECUTION_TERMS = (
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
VALIDATION_TERMS = ("validate", "validation", "debug", "repair")
SIMPLE_COUNT_TERMS = ("how many", "count")
DATA_AGGREGATION_TERMS = (
    "how many",
    "count",
    "total",
    "sum",
    "average",
    "avg",
    "minimum",
    "maximum",
)
SCHEMA_EXCLUSION_TERMS = DATA_AGGREGATION_TERMS + ("highest", "lowest", "top")
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
NO_DATA_QUERY_TERMS = (
    "do not query",
    "without querying",
    "just inspect the schema",
)
STRICT_EXPLICIT_TOOL_TERMS = ("exactly once", "only", "just")
EXPLICIT_SQL_PATTERN = re.compile(r"\b(select|with|explain|sql)\b")


def classify_db_prompt(
    prompt: str,
    *,
    available_tools: Iterable[str] = (),
    likely_catalog_match: bool,
) -> DbPromptClassification:
    """Classify a DB prompt into structured intent and selection signals."""

    text = str(prompt or "").lower()
    available_set = set(available_tools)
    intent = _classify_db_intent(text, likely_catalog_match=likely_catalog_match)
    needs_query_tools = intent.needs_sql_execution
    query_shape = _query_shape(text)
    aggregation_kind = _aggregation_kind(text, query_shape=query_shape)
    explicit_tools = tuple(
        name for name in sorted(available_set) if name.lower() in text
    )
    compile_first = _should_start_with_compile_first(
        available_set,
        text,
        needs_query_tools=needs_query_tools,
    )
    requested_analyst_tools = tuple(
        tool_name
        for tool_name, terms in ANALYST_TOOL_INTENTS.items()
        if tool_name in available_set and _has_any(text, terms)
    )
    needs_schema_tools = _needs_schema_tools(
        text,
        intent,
        needs_query_tools,
        likely_catalog_match=likely_catalog_match,
    )
    needs_memory_tools = intent.kind == DbIntentKind.MEMORY_ONLY or _has_any(
        text, MEMORY_WRITE_TERMS
    )
    semantic_memory_terms = tuple(
        term for term in SEMANTIC_MEMORY_TERMS if term in text
    )
    return DbPromptClassification(
        intent=intent,
        likely_catalog_match=likely_catalog_match,
        query_shape=query_shape,
        aggregation_kind=aggregation_kind,
        explicit_tools=explicit_tools,
        strict_explicit_tool_request=bool(explicit_tools)
        and _has_any(text, STRICT_EXPLICIT_TOOL_TERMS),
        needs_schema_tools=needs_schema_tools,
        compile_first=compile_first,
        requested_analyst_tools=requested_analyst_tools,
        needs_semantic_memory=bool(semantic_memory_terms),
        semantic_memory_terms=semantic_memory_terms,
        needs_quality_tools=_has_any(text, QUALITY_TERMS),
        needs_write_tools=_has_any(text, WRITE_TERMS),
        needs_lineage_tools=_has_any(text, LINEAGE_TERMS),
        needs_memory_tools=needs_memory_tools,
        needs_validation_tool=_has_any(text, VALIDATION_TERMS),
    )


def _classify_db_intent(text: str, *, likely_catalog_match: bool) -> DbIntent:
    if _has_explicit_sql(text):
        return DbIntent.from_kind(DbIntentKind.MANUAL_SQL)
    if _has_any(text, WRITE_TERMS):
        return DbIntent.from_kind(DbIntentKind.ADMIN_OR_WRITE)
    if _is_memory_only_intent(text):
        return DbIntent.from_kind(DbIntentKind.MEMORY_ONLY)
    if _is_schema_only_intent(text):
        return DbIntent.from_kind(DbIntentKind.SCHEMA_ONLY)
    if _has_catalog_assisted_data_intent(text, likely_catalog_match):
        return DbIntent.from_kind(DbIntentKind.DATA_QUERY_CATALOG_ASSISTED)
    return DbIntent.from_kind(DbIntentKind.DATA_QUERY_SIMPLE)


def _should_start_with_compile_first(
    available: set[str], text: str, *, needs_query_tools: bool
) -> bool:
    return (
        needs_query_tools
        and "db_compile_and_query" in available
        and not _needs_advanced_db_tools(available, text)
    )


def _needs_advanced_db_tools(available: set[str], text: str) -> bool:
    if any(name.lower() in text for name in available):
        return True
    if _has_any(text, WRITE_TERMS):
        return True
    if _has_any(text, SCHEMA_REPAIR_TERMS):
        return True
    if _has_any(text, QUALITY_TERMS + LINEAGE_TERMS):
        return True
    if any(_has_any(text, terms) for terms in ANALYST_TOOL_INTENTS.values()):
        return True
    if _has_any(text, VALIDATION_TERMS):
        return True
    if _needs_staged_query_path(text):
        return True
    return _has_explicit_sql(text)


def _needs_staged_query_path(text: str) -> bool:
    if _is_simple_count_question(text):
        return False
    return _has_any(text, QUERY_SHAPE_TERMS)


def _query_shape(text: str) -> str | None:
    if _is_simple_count_question(text):
        return "simple_count"
    if _needs_staged_query_path(text):
        return "staged_query"
    return None


def _aggregation_kind(text: str, *, query_shape: str | None) -> str | None:
    if query_shape == "simple_count":
        return "count"
    if _has_any(text, ("count", "how many", "number of", "row count")):
        return "count"
    if _has_any(text, ("sum", "total")):
        return "sum"
    if _has_any(text, ("average", "avg")):
        return "avg"
    if _has_any(text, ("minimum", "min ")):
        return "min"
    if _has_any(text, ("maximum", "max ")):
        return "max"
    return None


def _is_simple_count_question(text: str) -> bool:
    return (
        _has_any(text, SIMPLE_COUNT_TERMS + ("number of", "row count"))
        and not _has_any(text, QUERY_SHAPE_TERMS)
        and not _has_explicit_sql(text)
    )


def _needs_schema_tools(
    text: str,
    intent: DbIntent,
    needs_query_tools: bool,
    *,
    likely_catalog_match: bool,
) -> bool:
    if any(tool.lower() in text for tool in CATALOG_NAVIGATION_TOOL_NAMES):
        return True
    if intent.kind == DbIntentKind.DATA_QUERY_CATALOG_ASSISTED:
        return True
    if intent.kind == DbIntentKind.SCHEMA_ONLY:
        return True
    if _has_any(text, SCHEMA_REPAIR_TERMS):
        return True
    if _is_schema_question(text):
        return True
    if not needs_query_tools:
        return False
    return not likely_catalog_match


def _is_schema_only_intent(text: str) -> bool:
    if _has_schema_search_phrase(text) and not _has_data_execution_intent(text):
        return True
    if _has_any(text, INFERENCE_ONLY_TERMS):
        return True
    if _has_any(text, SCHEMA_TERMS) and not _has_any(
        text, DATA_QUERY_TERMS + SCHEMA_EXCLUSION_TERMS
    ):
        return True
    return _is_schema_question(text) and not _has_data_answer_intent(text)


def _has_catalog_assisted_data_intent(text: str, likely_catalog_match: bool) -> bool:
    if not _has_data_answer_intent(text) and not _has_any(text, QUERY_TERMS):
        return False
    if _has_schema_search_phrase(text):
        return True
    if any(tool in text for tool in CATALOG_NAVIGATION_TOOL_NAMES):
        return True
    if _has_any(text, SCHEMA_REPAIR_TERMS):
        return True
    if _has_any(text, ("catalog", "graph")):
        return True
    if not likely_catalog_match and _needs_staged_query_path(text):
        return True
    return False


def _has_data_answer_intent(text: str) -> bool:
    if _has_data_execution_intent(text):
        return True
    if _has_any(text, DATA_QUERY_TERMS):
        return True
    if _needs_staged_query_path(text):
        return True
    return _has_any(text, DATA_AGGREGATION_TERMS)


def _is_schema_question(text: str) -> bool:
    return _has_any(text, SCHEMA_TERMS) and not _has_any(text, DATA_QUERY_TERMS)


def _has_schema_search_phrase(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in SCHEMA_SEARCH_PATTERNS)


def _has_data_execution_intent(text: str) -> bool:
    if _has_any(text, NO_DATA_QUERY_TERMS):
        return False
    return _has_any(text, DATA_EXECUTION_TERMS)


def _is_memory_only_intent(text: str) -> bool:
    if not _has_any(text, MEMORY_TERMS):
        return False
    return not _has_any(
        text,
        SCHEMA_TERMS
        + QUERY_TERMS
        + DATA_QUERY_TERMS
        + WRITE_TERMS
        + DATA_EXECUTION_TERMS,
    )


def _has_explicit_sql(text: str) -> bool:
    return bool(EXPLICIT_SQL_PATTERN.search(text))


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)
