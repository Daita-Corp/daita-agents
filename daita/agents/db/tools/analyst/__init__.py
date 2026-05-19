"""Optional analyst tools for ``from_db`` agents."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .....agents.agent import Agent
    from .....plugins.base_db import BaseDatabasePlugin


def register_analyst_tools(
    agent: "Agent",
    plugin: "BaseDatabasePlugin",
    schema: Dict[str, Any],
) -> None:
    """Register the optional analyst toolkit into an agent."""
    from ._helpers import AnalystCatalogContext
    from .pivot_table import create_pivot_table_tool
    from .correlate import create_correlate_tool
    from .detect_anomalies import create_detect_anomalies_tool
    from .compare_entities import create_compare_entities_tool
    from .find_similar import create_find_similar_tool
    from .forecast_trend import create_forecast_trend_tool

    catalog_context = AnalystCatalogContext.from_plugin(plugin)
    database_type = (
        catalog_context.database_type or schema.get("database_type", "")
    ).lower()
    is_mongodb = database_type == "mongodb"
    tools = [
        create_correlate_tool(plugin, catalog_context),
        create_detect_anomalies_tool(plugin, catalog_context),
    ]

    if not is_mongodb:
        tools += [
            create_pivot_table_tool(plugin, catalog_context),
            create_compare_entities_tool(plugin, catalog_context),
            create_find_similar_tool(plugin, catalog_context),
            create_forecast_trend_tool(plugin, catalog_context),
        ]

    for tool in tools:
        agent.tool_registry.register(tool)


__all__ = ["register_analyst_tools"]
