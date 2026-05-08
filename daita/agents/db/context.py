"""
Public DB context for agents created by ``Agent.from_db()``.
"""

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent
    from ...core.watch import WatchEvent


def attach_db_context(agent: "Agent") -> None:
    """Attach the DB-only ``agent.db`` context object."""
    agent.db = DBContext(agent)


class DBAudit:
    """Read-only convenience wrapper around the DB-agent audit log."""

    def __init__(self, agent: "Agent"):
        self._agent = agent

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(getattr(self._agent, "_db_audit_log", []))

    def last(self) -> Optional[Dict[str, Any]]:
        entries = getattr(self._agent, "_db_audit_log", [])
        return entries[-1] if entries else None

    def export_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.entries, indent=indent, default=str)


class DBContext:
    """Stable SDK/debug context for a from_db-created agent."""

    def __init__(self, agent: "Agent"):
        self._agent = agent
        self.audit = DBAudit(agent)
        from .findings import DBFindings

        self.findings = DBFindings(agent)

    @property
    def schema(self) -> Dict[str, Any]:
        return getattr(self._agent, "_db_schema", {})

    @property
    def plugin(self) -> Any:
        return getattr(self._agent, "_db_plugin", None)

    @property
    def mode(self) -> Optional[str]:
        return getattr(self._agent, "_db_mode", None)

    @property
    def drift(self) -> Any:
        return getattr(self._agent, "_db_schema_drift", None)

    @property
    def memory(self) -> Any:
        return getattr(self._agent, "_db_memory", None)

    @property
    def memory_semantics(self) -> Any:
        return getattr(self._agent, "_db_memory_semantics", None)

    @property
    def lineage(self) -> Any:
        return getattr(self._agent, "_db_lineage", None)

    @property
    def history(self) -> Any:
        return getattr(self._agent, "_db_history", None)

    @property
    def quality(self) -> Any:
        return getattr(self._agent, "_db_quality", None)

    @property
    def summary(self) -> Dict[str, Any]:
        return getattr(self._agent, "_db_summary", {})

    @property
    def suggested_questions(self) -> List[str]:
        summary = self.summary
        return list(summary.get("suggested_questions", []))

    def register_monitors(
        self,
        monitors: Iterable[Dict[str, Any]],
        *,
        handler: Optional[Callable[["WatchEvent", Dict[str, Any]], Any]] = None,
        name_prefix: str = "db",
    ) -> List[Dict[str, Any]]:
        """Register structured DB monitor definitions as local watches."""
        from .monitors import register_local_monitors

        return register_local_monitors(
            self._agent,
            monitors,
            handler=handler,
            name_prefix=name_prefix,
        )

    @property
    def monitor_events(self) -> List[Dict[str, Any]]:
        return list(getattr(self._agent, "_db_monitor_events", []))
