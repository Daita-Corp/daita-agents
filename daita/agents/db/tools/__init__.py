"""Tool registration entry points for ``from_db`` agents."""

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
    """Register optional analyst tools."""
    from .analyst import register_analyst_tools as _register

    _register(agent, plugin, schema)


__all__ = ["register_analyst_tools"]
