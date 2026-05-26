"""DB-local tool capabilities and selection helpers for ``Agent.from_db()``.

This module centralizes the tool-name knowledge that belongs to the DB runtime.
It does not change the generic ``AgentTool`` contract and should not grow into
a framework-wide capability registry.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

DB_PLAN_CAPABILITY = "db.plan"
DB_VALIDATE_SQL_CAPABILITY = "db.validate_sql"
DB_EXECUTE_CAPABILITY = "db.execute"
DB_COMPILE_AND_EXECUTE_CAPABILITY = "db.compile_and_execute"
DB_WRITE_CAPABILITY = "db.write"
CATALOG_SEARCH_CAPABILITY = "catalog.search"
CATALOG_INSPECT_CAPABILITY = "catalog.inspect"
CATALOG_RELATIONSHIP_PATHS_CAPABILITY = "catalog.relationship_paths"
DB_MEMORY_WRITE_CAPABILITY = "db.memory.write"
DB_QUALITY_PROFILE_CAPABILITY = "db.quality.profile"
DB_LINEAGE_TRACE_CAPABILITY = "db.lineage.trace"
VECTOR_SEARCH_CAPABILITY = "vector.search"
VECTOR_UPSERT_CAPABILITY = "vector.upsert"
ANALYST_CAPABILITY_PREFIX = "db.analyst."

GENERIC_MEMORY_WRITE_TOOLS = (
    "remember",
    "update_memory",
)
CATALOG_NAVIGATION_TOOL_NAMES = (
    "catalog_search_schema",
    "catalog_inspect_table",
    "catalog_find_join_paths",
)
GENERIC_CATALOG_TOOLS = (
    "search_catalog",
    "inspect_asset",
    "find_relationship_paths",
)
RELATIONAL_CATALOG_ALIASES = {
    "search_catalog": "catalog_search_schema",
    "inspect_asset": "catalog_inspect_table",
    "find_relationship_paths": "catalog_find_join_paths",
}
ANALYST_TOOL_NAMES = frozenset(
    (
        "pivot_table",
        "correlate",
        "detect_anomalies",
        "compare_entities",
        "find_similar",
        "forecast_trend",
    )
)
LINEAGE_TOOL_NAMES = (
    "trace_lineage",
    "find_lineage_paths",
    "export_lineage",
)

DB_TOOL_CAPABILITIES = {
    "db_plan_query": (DB_PLAN_CAPABILITY,),
    "db_compile_and_query": (DB_COMPILE_AND_EXECUTE_CAPABILITY,),
    "db_validate_sql": (DB_VALIDATE_SQL_CAPABILITY,),
    "db_query": (DB_EXECUTE_CAPABILITY,),
    "db_count": (DB_EXECUTE_CAPABILITY,),
    "db_sample": (DB_EXECUTE_CAPABILITY,),
    "db_find": (DB_EXECUTE_CAPABILITY,),
    "db_aggregate": (DB_EXECUTE_CAPABILITY,),
    "db_execute": (DB_WRITE_CAPABILITY, DB_EXECUTE_CAPABILITY),
    "db_remember": (DB_MEMORY_WRITE_CAPABILITY,),
    "catalog_search_schema": (CATALOG_SEARCH_CAPABILITY,),
    "catalog_inspect_table": (CATALOG_INSPECT_CAPABILITY,),
    "catalog_find_join_paths": (CATALOG_RELATIONSHIP_PATHS_CAPABILITY,),
    "search_catalog": (CATALOG_SEARCH_CAPABILITY,),
    "inspect_asset": (CATALOG_INSPECT_CAPABILITY,),
    "find_relationship_paths": (CATALOG_RELATIONSHIP_PATHS_CAPABILITY,),
    "trace_lineage": (DB_LINEAGE_TRACE_CAPABILITY,),
    "find_lineage_paths": (DB_LINEAGE_TRACE_CAPABILITY,),
    "export_lineage": (DB_LINEAGE_TRACE_CAPABILITY,),
    "dq_profile": (DB_QUALITY_PROFILE_CAPABILITY,),
    "dq_detect_anomaly": (DB_QUALITY_PROFILE_CAPABILITY,),
    "dq_check_freshness": (DB_QUALITY_PROFILE_CAPABILITY,),
    "dq_report": (DB_QUALITY_PROFILE_CAPABILITY,),
    "postgres_vector_search": (VECTOR_SEARCH_CAPABILITY,),
    "postgres_vector_upsert": (VECTOR_UPSERT_CAPABILITY,),
    "remember": (DB_MEMORY_WRITE_CAPABILITY,),
    "update_memory": (DB_MEMORY_WRITE_CAPABILITY,),
    **{name: (f"{ANALYST_CAPABILITY_PREFIX}{name}",) for name in ANALYST_TOOL_NAMES},
}

ANSWER_EVIDENCE_CAPABILITIES = frozenset(
    {DB_EXECUTE_CAPABILITY, DB_COMPILE_AND_EXECUTE_CAPABILITY}
)
CATALOG_NAVIGATION_CAPABILITIES = frozenset(
    {
        CATALOG_SEARCH_CAPABILITY,
        CATALOG_INSPECT_CAPABILITY,
        CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
    }
)
DB_REPAIR_CAPABILITIES = frozenset(
    {
        DB_PLAN_CAPABILITY,
        DB_VALIDATE_SQL_CAPABILITY,
        DB_EXECUTE_CAPABILITY,
        DB_COMPILE_AND_EXECUTE_CAPABILITY,
    }
)
PROVIDER_VISIBLE_CAPABILITIES = frozenset(
    {
        DB_EXECUTE_CAPABILITY,
        VECTOR_SEARCH_CAPABILITY,
        VECTOR_UPSERT_CAPABILITY,
    }
)
REQUIRED_PHASE_CAPABILITIES = {
    "catalog": CATALOG_NAVIGATION_CAPABILITIES,
    "plan": frozenset({DB_PLAN_CAPABILITY, DB_COMPILE_AND_EXECUTE_CAPABILITY}),
    "execute": ANSWER_EVIDENCE_CAPABILITIES,
}


def tool_capabilities(tool_name: str) -> tuple[str, ...]:
    """Return DB-local capabilities declared for a tool name."""

    return DB_TOOL_CAPABILITIES.get(tool_name, ())


def tool_has_any_capability(tool_name: str, capabilities: Iterable[str]) -> bool:
    """Return True when a tool declares at least one capability from ``capabilities``."""

    requested = set(capabilities)
    return bool(requested.intersection(tool_capabilities(tool_name)))


def select_tools_for_capabilities(
    available: Iterable[str], capabilities: Sequence[str]
) -> list[str]:
    """Return available tools matching requested capabilities in request order."""

    available_set = set(available)
    selected: list[str] = []
    for capability in capabilities:
        for tool_name in _capability_tool_order(capability, available_set):
            if tool_name not in selected:
                selected.append(tool_name)
    return selected


def _capability_tool_order(capability: str, available: set[str]) -> list[str]:
    candidates = [
        name
        for name in _KNOWN_TOOL_ORDER
        if name in available and capability in tool_capabilities(name)
    ]
    candidates.extend(
        sorted(
            name
            for name in available
            if name not in _KNOWN_TOOL_ORDER and capability in tool_capabilities(name)
        )
    )
    return candidates


def has_analyst_tool(tool_names: Iterable[str]) -> bool:
    """Return True when any analyst toolkit tool is present."""

    return any(
        any(
            capability.startswith(ANALYST_CAPABILITY_PREFIX)
            for capability in tool_capabilities(name)
        )
        for name in tool_names
    )


def is_schema_navigation_tool(tool_name: str) -> bool:
    return tool_has_any_capability(tool_name, CATALOG_NAVIGATION_CAPABILITIES)


def is_generic_catalog_tool(tool_name: str) -> bool:
    return tool_name in GENERIC_CATALOG_TOOLS


def relational_catalog_alias(tool_name: str) -> str | None:
    return RELATIONAL_CATALOG_ALIASES.get(tool_name)


def is_db_repair_tool(tool_name: str) -> bool:
    return tool_has_any_capability(tool_name, DB_REPAIR_CAPABILITIES)


def should_keep_provider_db_tool(tool_name: str) -> bool:
    """Return True for provider DB tools that remain visible in ``from_db``."""

    return tool_has_any_capability(tool_name, PROVIDER_VISIBLE_CAPABILITIES)


def supports_required_phase(phase: str, tool_names: Iterable[str]) -> bool:
    capabilities = REQUIRED_PHASE_CAPABILITIES.get(phase)
    if capabilities is not None:
        return any(tool_has_any_capability(name, capabilities) for name in tool_names)
    return True


def supported_required_phases(
    required_phases: Sequence[str], tool_names: Iterable[str]
) -> tuple[str, ...]:
    """Return required phases supported by the selected tools."""

    return tuple(
        phase for phase in required_phases if supports_required_phase(phase, tool_names)
    )


_KNOWN_TOOL_ORDER = (
    "db_compile_and_query",
    "db_plan_query",
    "db_query",
    "db_validate_sql",
    "db_count",
    "db_sample",
    "db_find",
    "db_aggregate",
    "db_execute",
    "catalog_search_schema",
    "catalog_inspect_table",
    "catalog_find_join_paths",
    "search_catalog",
    "inspect_asset",
    "find_relationship_paths",
    "db_remember",
    "remember",
    "update_memory",
    "dq_profile",
    "dq_detect_anomaly",
    "dq_check_freshness",
    "dq_report",
    "postgres_vector_search",
    "postgres_vector_upsert",
    *sorted(ANALYST_TOOL_NAMES),
    *LINEAGE_TOOL_NAMES,
)
