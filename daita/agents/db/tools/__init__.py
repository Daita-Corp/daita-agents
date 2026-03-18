"""
Analyst toolkit — 6 analyst-grade tools auto-registered by from_db().

Each tool combines SQL data fetching with pandas/numpy in-process computation.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ....agents.agent import Agent
    from ....plugins.base_db import BaseDatabasePlugin


def register_analyst_tools(
    agent: "Agent",
    plugin: "BaseDatabasePlugin",
    schema: Dict[str, Any],
) -> None:
    """
    Register analyst tools into agent.tool_registry.

    All 6 tools are registered for SQL databases. For MongoDB, only the two
    query-agnostic tools (correlate, detect_anomalies) are registered because
    the others rely on SQL JOINs that MongoDB does not support.
    """
    from .pivot_table import create_pivot_table_tool
    from .correlate import create_correlate_tool
    from .detect_anomalies import create_detect_anomalies_tool
    from .compare_entities import create_compare_entities_tool
    from .find_similar import create_find_similar_tool
    from .forecast_trend import create_forecast_trend_tool

    is_mongodb = schema.get("database_type", "").lower() == "mongodb"

    tools = [
        create_correlate_tool(plugin, schema),
        create_detect_anomalies_tool(plugin, schema),
    ]

    if not is_mongodb:
        tools += [
            create_pivot_table_tool(plugin, schema),
            create_compare_entities_tool(plugin, schema),
            create_find_similar_tool(plugin, schema),
            create_forecast_trend_tool(plugin, schema),
        ]

    for tool in tools:
        agent.tool_registry.register(tool)
