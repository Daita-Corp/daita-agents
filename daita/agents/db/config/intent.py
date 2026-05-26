"""Typed request intent model for ``Agent.from_db()`` routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DbEvidenceMode(str, Enum):
    """Evidence source required before a DB run can answer safely."""

    NONE = "none"
    CATALOG = "catalog"
    QUERY = "query"
    MEMORY = "memory"


class DbIntentKind(str, Enum):
    """Structured DB request intent kinds.

    The values intentionally match the current diagnostic intent names so
    existing run summaries remain stable while the implementation moves away
    from stringly typed routing.
    """

    CONVERSATIONAL = "conversational"
    SCHEMA_ONLY = "schema_only"
    SCHEMA_QUESTION = "schema_question"
    SCHEMA_EXPLAIN = "schema_explain"
    DATA_QUERY_SIMPLE = "data_query_simple"
    DATA_QUERY_CATALOG_ASSISTED = "data_query_catalog_assisted"
    MANUAL_SQL = "manual_sql"
    ADMIN_OR_WRITE = "admin_or_write"
    MEMORY_ONLY = "memory_only"


CATALOG_INTENT_KINDS = frozenset(
    {
        DbIntentKind.SCHEMA_ONLY,
        DbIntentKind.SCHEMA_QUESTION,
        DbIntentKind.SCHEMA_EXPLAIN,
    }
)
QUERY_INTENT_KINDS = frozenset(
    {
        DbIntentKind.DATA_QUERY_SIMPLE,
        DbIntentKind.DATA_QUERY_CATALOG_ASSISTED,
        DbIntentKind.MANUAL_SQL,
        DbIntentKind.ADMIN_OR_WRITE,
    }
)


@dataclass(frozen=True)
class DbIntent:
    """Typed intent and routing facts for one DB request."""

    kind: DbIntentKind
    evidence_mode: DbEvidenceMode
    needs_catalog_resolution: bool = False
    needs_sql_execution: bool = False
    is_write_or_admin: bool = False
    requested_capabilities: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> str:
        """Stable diagnostic value for existing run metadata."""
        return self.kind.value

    @classmethod
    def from_kind(
        cls,
        kind: DbIntentKind,
        *,
        requested_capabilities: tuple[str, ...] = (),
        diagnostics: dict[str, Any] | None = None,
    ) -> "DbIntent":
        evidence_mode = evidence_mode_for_intent(kind)
        return cls(
            kind=kind,
            evidence_mode=evidence_mode,
            needs_catalog_resolution=kind == DbIntentKind.DATA_QUERY_CATALOG_ASSISTED
            or kind in CATALOG_INTENT_KINDS,
            needs_sql_execution=kind in QUERY_INTENT_KINDS,
            is_write_or_admin=kind == DbIntentKind.ADMIN_OR_WRITE,
            requested_capabilities=requested_capabilities,
            diagnostics=diagnostics or {},
        )


def evidence_mode_for_intent(kind: DbIntentKind) -> DbEvidenceMode:
    """Return the required evidence mode for an intent kind."""
    if kind in CATALOG_INTENT_KINDS:
        return DbEvidenceMode.CATALOG
    if kind == DbIntentKind.MEMORY_ONLY:
        return DbEvidenceMode.MEMORY
    if kind == DbIntentKind.CONVERSATIONAL:
        return DbEvidenceMode.NONE
    return DbEvidenceMode.QUERY
