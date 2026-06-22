"""Typed control-plane records for explicit DB memory commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from daita.db.memory import DB_SEMANTIC_MEMORY_KINDS

DB_MEMORY_COMMAND_ACTIONS = frozenset(
    {"remember", "update", "forget", "inspect", "list"}
)
DB_MEMORY_MUTATION_ACTIONS = frozenset({"remember", "update"})
DB_MEMORY_COMMAND_KINDS = frozenset(DB_SEMANTIC_MEMORY_KINDS)


@dataclass(frozen=True)
class DbMemoryCommand:
    """User-authored DB memory command routed from a request."""

    action: str
    prompt: str
    mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "prompt": self.prompt,
            "mode": self.mode,
            "metadata": dict(self.metadata),
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class DbMemoryIntent:
    """Normalized intent extracted from an explicit DB memory command."""

    action: str
    kind: str | None
    key: str | None
    text: str | None
    schema_refs: tuple[dict[str, str], ...] = ()
    catalog_refs: tuple[str, ...] = ()
    source_identity: str | None = None
    workspace_scope: str = "source"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.7
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "kind": self.kind,
            "key": self.key,
            "text": self.text,
            "schema_refs": [dict(item) for item in self.schema_refs],
            "catalog_refs": list(self.catalog_refs),
            "source_identity": self.source_identity,
            "workspace_scope": self.workspace_scope,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "importance": self.importance,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class DbMemoryValidation:
    """Validation outcome for a DB memory proposal."""

    accepted: bool
    status: Literal["accepted", "blocked", "rejected"]
    reasons: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
        }
