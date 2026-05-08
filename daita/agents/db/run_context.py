"""
Compact per-run context for agents created by ``Agent.from_db()``.
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


DEFAULT_CONTEXT_MAX_CHARS = 1600
_ANALYST_TOOL_PREFIXES = (
    "pivot_",
    "correlate",
    "detect_",
    "compare_",
    "find_",
    "forecast_",
)


def build_db_run_context(
    agent: "Agent",
    *,
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    memory_snippets: Optional[List[str]] = None,
) -> str:
    """Build a small DB runtime context block without querying external systems."""
    schema = getattr(agent, "_db_schema", {}) or {}
    plugin = getattr(agent, "_db_plugin", None)
    mode = getattr(agent, "_db_mode", None)
    tool_names = list(getattr(agent.tool_registry, "tool_names", []))
    drift = getattr(agent, "_db_schema_drift", None)
    summary = getattr(agent, "_db_summary", {}) or {}
    tables = schema.get("tables", [])
    foreign_keys = schema.get("foreign_keys", [])

    lines = [
        "<db_runtime_context>",
        "Use this compact DB context for the current question; full schema is in the system prompt.",
        (
            "Database: "
            f"type={schema.get('database_type', getattr(plugin, 'sql_dialect', 'unknown'))}, "
            f"name={schema.get('database_name') or _plugin_database_name(plugin) or 'unknown'}, "
            f"mode={mode or 'unknown'}, "
            f"tables={schema.get('table_count', len(tables))}, "
            f"columns={sum(len(table.get('columns', [])) for table in tables)}, "
            f"relationships={len(foreign_keys)}"
        ),
        "Query policy: " + _query_policy_summary(plugin),
        "Data health: " + _summary_line(summary),
        "Candidate metrics: " + _candidate_metrics_summary(summary),
        "Capabilities: " + _capability_summary(agent, tool_names),
        "Memory: " + _memory_summary(agent, memory_snippets or []),
        "Drift: " + _drift_summary(drift),
        "</db_runtime_context>",
    ]
    return _truncate_context("\n".join(lines), max_chars)


def make_db_context_run(agent: "Agent", original_run: Callable) -> Callable:
    """Return a run wrapper that adds compact DB context to each prompt."""

    async def _db_context_run(prompt: str, **kwargs: Any):
        from .memory import recall_db_memory_context

        memory_snippets = await recall_db_memory_context(agent, prompt)
        context = build_db_run_context(agent, memory_snippets=memory_snippets)
        return await original_run(_augment_prompt(prompt, context), **kwargs)

    return _db_context_run


def make_db_context_stream(agent: "Agent", original_stream: Callable) -> Callable:
    """Return a stream wrapper that adds compact DB context to each prompt."""

    async def _db_context_stream(prompt: str, **kwargs: Any):
        from .memory import recall_db_memory_context

        memory_snippets = await recall_db_memory_context(agent, prompt)
        context = build_db_run_context(agent, memory_snippets=memory_snippets)
        async for event in original_stream(_augment_prompt(prompt, context), **kwargs):
            yield event

    return _db_context_stream


def _augment_prompt(prompt: str, context: str) -> str:
    return f"{context}\n\nUser question:\n{prompt}"


def _query_policy_summary(plugin: Any) -> str:
    if plugin is None:
        return "unknown"

    parts = [
        "read_only=true" if getattr(plugin, "read_only", True) else "read_only=false",
        f"default_limit={getattr(plugin, 'query_default_limit', None)}",
        f"max_rows={getattr(plugin, 'query_max_rows', None)}",
        f"max_chars={getattr(plugin, 'query_max_chars', None)}",
        f"timeout={getattr(plugin, 'query_timeout', None)}",
    ]
    if getattr(plugin, "allowed_tables", None):
        parts.append("table_allowlist=true")
    if getattr(plugin, "blocked_tables", None):
        parts.append("table_blocklist=true")
    if getattr(plugin, "blocked_columns", None):
        parts.append("column_blocklist=true")
    return ", ".join(parts)


def _capability_summary(agent: "Agent", tool_names: List[str]) -> str:
    capabilities = ["sql", "schema"]
    if any(name.startswith(_ANALYST_TOOL_PREFIXES) for name in tool_names):
        capabilities.append("analyst_tools")
    if hasattr(agent, "_db_memory"):
        capabilities.append("memory")
    if hasattr(agent, "_db_lineage"):
        capabilities.append("lineage")
    if hasattr(agent, "_db_history"):
        capabilities.append("history")
    return ", ".join(capabilities)


def _memory_summary(agent: "Agent", snippets: List[str]) -> str:
    if not hasattr(agent, "_db_memory"):
        return "disabled"
    if not snippets:
        return "enabled; use for business rules, metric definitions, and unit conventions; not row lookup"
    joined = "; ".join(snippets[:5])
    return _truncate_line(f"relevant={joined}", 500)


def _drift_summary(drift: Any) -> str:
    if not drift:
        return "none"
    if isinstance(drift, dict):
        items = []
        for key, value in drift.items():
            if not value:
                continue
            if isinstance(value, list):
                items.append(f"{key}={len(value)}")
            else:
                items.append(f"{key}=changed")
            if len(items) >= 5:
                break
        return ", ".join(items) if items else "changed"
    return _truncate_line(str(drift), 240)


def _summary_line(summary: Dict[str, Any]) -> str:
    if not summary:
        return "unknown"
    parts = []
    signals = summary.get("signals") or []
    if signals:
        parts.append(f"signals={len(signals)}")
    if summary.get("fact_tables"):
        parts.append("fact_tables=" + ", ".join(summary["fact_tables"][:3]))
    if summary.get("entity_tables"):
        parts.append("entity_tables=" + ", ".join(summary["entity_tables"][:3]))
    if summary.get("timestamp_columns"):
        parts.append(f"timestamp_columns={len(summary['timestamp_columns'])}")
    return _truncate_line("; ".join(parts) if parts else "ok", 300)


def _candidate_metrics_summary(summary: Dict[str, Any]) -> str:
    metrics = summary.get("candidate_metrics") or []
    if not metrics:
        return "none"
    names = [m.get("name", "") for m in metrics if m.get("name")]
    return _truncate_line(", ".join(names[:5]), 240)


def _plugin_database_name(plugin: Any) -> Any:
    if plugin is None:
        return None
    return (
        getattr(plugin, "database_name", None)
        or getattr(plugin, "database", None)
        or getattr(plugin, "db", None)
        or getattr(plugin, "path", None)
    )


def _truncate_line(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _truncate_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    suffix = "\n...context truncated\n</db_runtime_context>"
    return context[: max_chars - len(suffix)].rstrip() + suffix
