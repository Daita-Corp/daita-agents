"""
Compact per-run context for agents created by ``Agent.from_db()``.
"""

import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..catalog_read_model import build_db_catalog_read_model
from ..config.policies import workflow_policy_for_intent
from ..config.route_decision import build_db_route_decision
from ..config.tool_selection import has_analyst_tool
from .orchestrator import (
    DbRunContract,
    DbRunOrchestrator,
)

if TYPE_CHECKING:
    from ...agent import Agent


DEFAULT_CONTEXT_MAX_CHARS = 1600


def build_db_run_context(
    agent: "Agent",
    *,
    prompt: str = "",
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    memory_snippets: Optional[List[str]] = None,
    intent_kind: Optional[str] = None,
    selected_tools: Optional[List[str]] = None,
    contract: Optional[DbRunContract] = None,
) -> str:
    """Build a small DB runtime context block without querying external systems."""
    read_model = build_db_catalog_read_model(
        agent, summary=getattr(agent, "_db_summary", {}) or {}
    )
    plugin = getattr(agent, "_db_plugin", None)
    mode = getattr(agent, "_db_mode", None)
    if contract is not None:
        intent_kind = contract.intent
        tool_names = list(contract.tools)
    else:
        tool_names = list(
            selected_tools or getattr(agent.tool_registry, "tool_names", [])
        )
    drift = getattr(agent, "_db_drift", None)
    summary = read_model.db_summary

    lines = [
        "<db_runtime_context>",
        "Workflow: " + _workflow_policy_summary(intent_kind, contract=contract),
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
        "Answer policy: " + _answer_policy_summary(intent_kind, contract=contract),
        "Capabilities: " + _capability_summary(agent, tool_names),
        "Selected tools: " + _selected_tools_summary(tool_names),
        "Memory: " + _memory_summary(agent, memory_snippets or []),
        "Data health: " + _summary_line(summary),
        "Candidate metrics: " + _candidate_metrics_summary(summary),
        "Drift: " + _drift_summary(drift),
        "</db_runtime_context>",
    ]
    return _truncate_context("\n".join(lines), max_chars)


def make_db_context_run(agent: "Agent", original_run: Callable) -> Callable:
    """Return a run wrapper that adds compact DB context to each prompt."""

    async def _db_context_run(prompt: str, **kwargs: Any):
        from .fast_path import try_db_fast_path

        run_state = DbRunOrchestrator.start_state(agent)
        route_decision = build_db_route_decision(agent, prompt)
        fast_result = await try_db_fast_path(
            agent,
            prompt,
            kwargs,
            run_state=run_state,
            route_decision=route_decision,
        )
        if fast_result is not None:
            history = kwargs.get("history")
            if history is not None and hasattr(history, "add_turn"):
                await history.add_turn(prompt, fast_result.get("result", ""))
            return fast_result
        augmented_prompt, kwargs = await _prepare_db_runtime_call(
            agent,
            prompt,
            kwargs,
            run_state=run_state,
            route_decision=route_decision,
        )
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
    agent: "Agent",
    prompt: str,
    kwargs: Dict[str, Any],
    *,
    run_state: Any = None,
    classification: Any = None,
    route_decision: Any = None,
) -> tuple[str, Dict[str, Any]]:
    if route_decision is None:
        route_decision = build_db_route_decision(
            agent,
            prompt,
            classification=classification,
        )
    orchestrator = DbRunOrchestrator(
        agent,
        prompt,
        state=run_state,
        classification=route_decision.classification,
        route_decision=route_decision,
    )
    prepared = await orchestrator.prepare()
    context = prepared.context
    selected_tools = list(prepared.contract.tools)
    agent._db_last_context_metadata = _context_metadata(context, selected_tools, prompt)
    agent._db_last_context_metadata["contract"] = orchestrator.contract_summary()
    if kwargs.get("tools") is None:
        kwargs["tools"] = selected_tools
    if kwargs.get("initial_messages"):
        kwargs["initial_messages"] = _compact_initial_messages(
            kwargs["initial_messages"]
        )
    kwargs.setdefault("final_synthesis_without_tools", True)
    kwargs.setdefault("terminal_tools", prepared.contract.terminal_tools)
    kwargs.setdefault("partial_exit", True)
    existing_max_iterations = kwargs.get("max_iterations")
    if existing_max_iterations is None:
        kwargs["max_iterations"] = prepared.contract.max_model_turns
    else:
        kwargs["max_iterations"] = min(
            int(existing_max_iterations), prepared.contract.max_model_turns
        )
    kwargs["run_orchestrator"] = orchestrator
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


def _workflow_policy_summary(
    intent_kind: Optional[str], *, contract: Optional[DbRunContract] = None
) -> str:
    if contract is not None and contract.workflow_guidance:
        return contract.workflow_guidance
    return workflow_policy_for_intent(intent_kind).workflow_guidance


def _answer_policy_summary(
    intent_kind: Optional[str], *, contract: Optional[DbRunContract] = None
) -> str:
    if contract is not None and contract.answer_guidance:
        return contract.answer_guidance
    return workflow_policy_for_intent(intent_kind).answer_guidance


def _capability_summary(agent: "Agent", tool_names: List[str]) -> str:
    capabilities = ["sql", "schema"]
    if has_analyst_tool(tool_names):
        capabilities.append("analyst_tools")
    if hasattr(agent, "_db_memory"):
        capabilities.append("memory")
    if hasattr(agent, "_db_lineage"):
        capabilities.append("lineage")
    if hasattr(agent, "_db_history"):
        capabilities.append("history")
    return ", ".join(capabilities)


def _selected_tools_summary(tool_names: List[str]) -> str:
    if not tool_names:
        return "none"
    return ", ".join(tool_names[:12])


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
    lines = context.splitlines()
    if len(lines) < 3:
        suffix = "\n...context truncated\n</db_runtime_context>"
        return context[: max_chars - len(suffix)].rstrip() + suffix

    priority_prefixes = (
        "Workflow:",
        "Database:",
        "Query policy:",
        "Answer policy:",
        "Capabilities:",
        "Candidate metrics:",
        "Memory:",
        "Data health:",
        "Selected tools:",
        "Drift:",
    )
    body = [
        _truncate_line(line, 220)
        for prefix in priority_prefixes
        for line in lines[1:-1]
        if line.startswith(prefix)
    ]
    output = [lines[0]]
    truncated_marker = "...context truncated"
    for line in body:
        candidate = "\n".join(output + [line, truncated_marker, lines[-1]])
        if len(candidate) > max_chars:
            continue
        output.append(line)
    return "\n".join(output + [truncated_marker, lines[-1]])
