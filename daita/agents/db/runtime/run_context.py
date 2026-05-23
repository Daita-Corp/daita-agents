"""
Compact per-run context for agents created by ``Agent.from_db()``.
"""

import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..config.policies import SCHEMA_NAVIGATION_TOOLS, TERMINAL_DB_TOOLS
from ..catalog_read_model import build_db_catalog_read_model
from ..utils import ANALYST_TOOL_PREFIXES

if TYPE_CHECKING:
    from ...agent import Agent


DEFAULT_CONTEXT_MAX_CHARS = 1600
SCHEMA_HINT_MAX_TABLES = 5
SCHEMA_HINT_MAX_RELATIONSHIPS = 5


def build_db_run_context(
    agent: "Agent",
    *,
    prompt: str = "",
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    memory_snippets: Optional[List[str]] = None,
) -> str:
    """Build a small DB runtime context block without querying external systems."""
    read_model = build_db_catalog_read_model(
        agent, summary=getattr(agent, "_db_summary", {}) or {}
    )
    plugin = getattr(agent, "_db_plugin", None)
    mode = getattr(agent, "_db_mode", None)
    tool_names = list(getattr(agent.tool_registry, "tool_names", []))
    drift = getattr(agent, "_db_drift", None)
    summary = read_model.db_summary

    lines = [
        "<db_runtime_context>",
        "Use DB tools autonomously. Plan open-ended analytic, ranking, label, filtered, or multi-table work with db_plan_query, then execute validated plans with db_query using suggested_next_arguments. Do not repeat db_plan_query for the same goal after it returns a plan_id.",
        (
            "Database: "
            f"type={read_model.database_type}, "
            f"name={read_model.database_name}, "
            f"mode={mode or 'unknown'}, "
            f"tables={read_model.table_count}, "
            f"columns={read_model.column_count}, "
            f"relationships={read_model.relationship_count}"
        ),
        "Query policy: " + _query_policy_summary(plugin),
        (
            "Answer policy: execute an answer-bearing DB query before final "
            "answers; when identifying an entity, include its human-readable "
            "name or label with the id when available."
        ),
        "Capabilities: " + _capability_summary(agent, tool_names),
        "Memory: " + _memory_summary(agent, memory_snippets or []),
        "Data health: " + _summary_line(summary),
        "Candidate metrics: " + _candidate_metrics_summary(summary),
        "Schema hints: " + _catalog_hint_summary(agent, prompt),
        "Drift: " + _drift_summary(drift),
        "</db_runtime_context>",
    ]
    return _truncate_context("\n".join(lines), max_chars)


def make_db_context_run(agent: "Agent", original_run: Callable) -> Callable:
    """Return a run wrapper that adds compact DB context to each prompt."""

    async def _db_context_run(prompt: str, **kwargs: Any):
        from .fast_path import try_db_fast_path

        fast_result = await try_db_fast_path(agent, prompt, kwargs)
        if fast_result is not None:
            history = kwargs.get("history")
            if history is not None and hasattr(history, "add_turn"):
                await history.add_turn(prompt, fast_result.get("result", ""))
            return fast_result
        augmented_prompt, kwargs = await _prepare_db_runtime_call(agent, prompt, kwargs)
        return await original_run(augmented_prompt, **kwargs)

    return _db_context_run


def make_db_context_stream(agent: "Agent", original_stream: Callable) -> Callable:
    """Return a stream wrapper that adds compact DB context to each prompt."""

    async def _db_context_stream(prompt: str, **kwargs: Any):
        augmented_prompt, kwargs = await _prepare_db_runtime_call(agent, prompt, kwargs)
        async for event in original_stream(augmented_prompt, **kwargs):
            yield event

    return _db_context_stream


def _augment_prompt(prompt: str, context: str) -> str:
    return f"{context}\n\nUser question:\n{prompt}"


async def _prepare_db_runtime_call(
    agent: "Agent", prompt: str, kwargs: Dict[str, Any]
) -> tuple[str, Dict[str, Any]]:
    from ..memory import recall_db_memory_context
    from ..config.tool_profiles import select_db_tool_profile
    from .state import DbRunState, set_db_run_state
    from .tracing import db_trace_span

    async with db_trace_span(
        agent,
        "from_db.prepare_runtime_context",
        prompt=prompt[:200],
    ):
        plugin = getattr(agent, "_db_plugin", None)
        run_state = DbRunState()
        set_db_run_state(agent, run_state)
        if plugin is not None:
            set_db_run_state(plugin, run_state)
            setattr(plugin, "_daita_sql_preflight_failures", {})

        async with db_trace_span(agent, "from_db.memory_recall") as (
            trace_manager,
            span_id,
        ):
            memory_snippets = await recall_db_memory_context(agent, prompt)
            decision = getattr(agent, "_db_last_memory_recall_decision", None) or {}
            trace_manager.record_output(
                span_id,
                {
                    "skipped": not bool(decision.get("recall", True)),
                    "reason": decision.get("reason"),
                    "matched_terms": decision.get("matched_terms", []),
                    "snippet_count": len(memory_snippets),
                },
            )

        async with db_trace_span(agent, "from_db.build_runtime_context") as (
            trace_manager,
            span_id,
        ):
            context = build_db_run_context(
                agent, prompt=prompt, memory_snippets=memory_snippets
            )
            trace_manager.record_output(
                span_id,
                {
                    "runtime_context_chars": len(context),
                    "memory_snippet_count": len(memory_snippets),
                },
            )

        async with db_trace_span(agent, "from_db.select_tools") as (
            trace_manager,
            span_id,
        ):
            tool_profile = select_db_tool_profile(agent, prompt)
            selected_tools = tool_profile.tools
            run_state.configure_workflow(
                selected_tools, intent_kind=tool_profile.intent
            )
            trace_manager.record_output(
                span_id,
                {
                    "selected_tools": selected_tools,
                    "selected_tool_count": len(selected_tools),
                    "required_phases": tool_profile.required_phases,
                    "intent_kind": run_state.intent_kind,
                },
            )
    agent._db_last_context_metadata = _context_metadata(context, selected_tools, prompt)
    if kwargs.get("tools") is None:
        kwargs["tools"] = selected_tools
    if kwargs.get("initial_messages"):
        kwargs["initial_messages"] = _compact_initial_messages(
            kwargs["initial_messages"]
        )
    kwargs.setdefault("final_synthesis_without_tools", True)
    kwargs.setdefault("terminal_tools", TERMINAL_DB_TOOLS)
    return _augment_prompt(prompt, context), kwargs


def _context_metadata(
    context: str, selected_tools: List[str], prompt: str
) -> Dict[str, Any]:
    return {
        "runtime_context_chars": len(context),
        "runtime_context_tokens_estimate": max(1, (len(context) + 3) // 4),
        "selected_tools": list(selected_tools),
        "selected_tool_count": len(selected_tools),
        "prompt_terms": _prompt_terms(prompt),
    }


def _compact_initial_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    compacted: List[Dict[str, Any]] = []
    for message in messages[-6:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            compacted.append(message)
            continue
        compacted.append({"role": role, "content": _compact_history_content(content)})
    return compacted


def _compact_history_content(content: str) -> str:
    content = _strip_runtime_context(content).strip()
    if len(content) <= 900:
        return content
    return content[:897].rstrip() + "..."


def _strip_runtime_context(content: str) -> str:
    return re.sub(
        r"<db_runtime_context>.*?</db_runtime_context>\s*",
        "",
        content,
        flags=re.DOTALL,
    )


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
    if any(name.startswith(ANALYST_TOOL_PREFIXES) for name in tool_names):
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


def _catalog_hint_summary(agent: "Agent", prompt: str) -> str:
    store_id = getattr(agent, "_db_catalog_store_id", None)
    catalog = getattr(agent, "_db_catalog", None)
    catalog_suffix = f" with store_id={store_id}" if store_id else ""
    if catalog is None or not store_id or not prompt:
        return "use catalog tools as needed" + catalog_suffix

    result = catalog.catalog_search_schema(
        store_id, prompt, limit=SCHEMA_HINT_MAX_TABLES
    )
    tables = result.get("tables") or []
    if not tables:
        return "use catalog_search_schema for relevant tables" + catalog_suffix

    parts = []
    selected_names = {str(table.get("name")) for table in tables}
    for table in tables[:SCHEMA_HINT_MAX_TABLES]:
        cols = [
            str(col.get("name"))
            for col in table.get("matched_columns", []) or []
            if col.get("name")
        ][:4]
        if cols:
            parts.append(f"{table.get('name')}({', '.join(cols)})")
        else:
            parts.append(str(table.get("name")))

    relationships = []
    for table_name in list(selected_names)[:2]:
        inspected = catalog.get_table_schema(
            store_id,
            table_name,
            include_indexes=False,
            include_foreign_keys=True,
            limit=1,
        )
        for fk in inspected.get("foreign_keys", []) or []:
            relationships.append(
                f"{fk.get('source_asset')}.{fk.get('source_field')}->{fk.get('target_asset')}.{fk.get('target_field')}"
            )
            if len(relationships) >= SCHEMA_HINT_MAX_RELATIONSHIPS:
                break
        if len(relationships) >= SCHEMA_HINT_MAX_RELATIONSHIPS:
            break
    if relationships:
        parts.append("rels=" + ", ".join(relationships))
    return _truncate_line("; ".join(parts), 500)


def _prompt_terms(prompt: str) -> List[str]:
    raw_terms = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", prompt.lower())
    stopwords = {
        "about",
        "also",
        "can",
        "column",
        "columns",
        "could",
        "database",
        "for",
        "from",
        "how",
        "into",
        "kind",
        "tell",
        "table",
        "tables",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "you",
        "your",
    }
    terms = []
    for term in raw_terms:
        singular = term[:-1] if term.endswith("s") and len(term) > 3 else term
        for candidate in (term, singular):
            if candidate not in stopwords and candidate not in terms:
                terms.append(candidate)
    return terms[:12]


def _truncate_line(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _truncate_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    suffix = "\n...context truncated\n</db_runtime_context>"
    return context[: max_chars - len(suffix)].rstrip() + suffix
