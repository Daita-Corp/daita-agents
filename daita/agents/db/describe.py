"""
Platform-safe metadata for agents created by ``Agent.from_db()``.
"""

from typing import Any, Dict, List, TYPE_CHECKING

from .catalog_read_model import build_db_catalog_read_model
from .config.tool_selection import has_analyst_tool

if TYPE_CHECKING:
    from ..agent import Agent


def attach_db_describe(agent: "Agent") -> None:
    """Attach a DB-only ``describe()`` method to a from_db-created agent."""

    def describe() -> Dict[str, Any]:
        return describe_db_agent(agent)

    agent.describe = describe


def describe_db_agent(agent: "Agent") -> Dict[str, Any]:
    """Return compact, side-effect-free metadata for a DB-backed agent."""
    tool_names = list(getattr(agent.tool_registry, "tool_names", []))
    return {
        "kind": "database",
        "name": agent.name,
        "agent_id": agent.agent_id,
        "capabilities": _capabilities(agent, tool_names),
        "llm": _llm_metadata(agent),
        "tools": tool_names,
        "tool_count": len(tool_names),
        "db": _db_metadata(agent, tool_names),
    }


def _capabilities(agent: "Agent", tool_names: List[str]) -> List[str]:
    capabilities = ["sql", "schema"]
    if has_analyst_tool(tool_names):
        capabilities.append("analyst_tools")
    if hasattr(agent, "_db_quality"):
        capabilities.append("data_quality")
    if hasattr(agent, "_db_memory"):
        capabilities.append("memory")
    if hasattr(agent, "_db_lineage"):
        capabilities.append("lineage")
    if hasattr(agent, "_db_audit_log"):
        capabilities.append("audit")
    return capabilities


def _llm_metadata(agent: "Agent") -> Dict[str, Any]:
    return {
        "provider": getattr(agent, "_llm_provider_name", None)
        or getattr(getattr(agent, "_llm", None), "provider_name", None),
        "model": getattr(agent, "_llm_model", None)
        or getattr(getattr(agent, "_llm", None), "model", None),
    }


def _db_metadata(agent: "Agent", tool_names: List[str]) -> Dict[str, Any]:
    read_model = build_db_catalog_read_model(
        agent, summary=getattr(agent, "_db_summary", {}) or {}
    )
    plugin = getattr(agent, "_db_plugin", None)
    drift = getattr(agent, "_db_drift", None)
    summary = read_model.db_summary
    candidate_metrics = summary.get("candidate_metrics", [])
    signals = summary.get("signals", [])
    findings = getattr(agent, "_db_findings", []) or []
    open_findings = [f for f in findings if f.get("status") == "open"]

    metadata: Dict[str, Any] = {
        "database_type": read_model.database_type,
        "database_name": read_model.database_name,
        "catalog_store_id": read_model.store_id,
        "mode": getattr(agent, "_db_mode", None),
        "read_only": bool(getattr(plugin, "read_only", True)),
        "table_count": read_model.table_count,
        "column_count": read_model.column_count,
        "relationship_count": read_model.relationship_count,
        "catalog_status": read_model.catalog_status,
        "drift_status": "changed" if drift else "none",
        "memory_enabled": hasattr(agent, "_db_memory"),
        "lineage_enabled": hasattr(agent, "_db_lineage"),
        "history_enabled": hasattr(agent, "_db_history"),
        "quality_enabled": hasattr(agent, "_db_quality"),
        "audit_enabled": hasattr(agent, "_db_audit_log"),
        "metric_count": len(candidate_metrics),
        "finding_count": len(findings),
        "open_finding_count": len(open_findings),
        "quality_status": "warning" if signals else "ok",
        "tools": tool_names,
    }
    if summary:
        metadata["summary"] = summary
    prompt_metadata = getattr(agent, "_db_prompt_metadata", None)
    if prompt_metadata:
        metadata["prompt"] = prompt_metadata
    if drift:
        metadata["drift"] = drift
    if plugin is not None:
        metadata["query_policy"] = _query_policy(plugin)
    return metadata


def _query_policy(plugin: Any) -> Dict[str, Any]:
    return {
        "read_only": bool(getattr(plugin, "read_only", True)),
        "default_limit": getattr(plugin, "query_default_limit", None),
        "max_rows": getattr(plugin, "query_max_rows", None),
        "max_chars": getattr(plugin, "query_max_chars", None),
        "timeout": getattr(plugin, "query_timeout", None),
        "has_table_allowlist": bool(getattr(plugin, "allowed_tables", None)),
        "has_table_blocklist": bool(getattr(plugin, "blocked_tables", None)),
        "has_column_blocklist": bool(getattr(plugin, "blocked_columns", None)),
    }
