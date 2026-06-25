"""Typed monitor command records and validation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from daita.runtime import Evidence

from ..monitors import DbMonitor

DbMonitorCommandKind = Literal[
    "create",
    "list",
    "inspect",
    "update",
    "pause",
    "resume",
    "delete",
    "explain_run",
    "approve_action",
]


@dataclass(frozen=True)
class DbMonitorCommand:
    """Typed prompt-level monitor command.

    The command is intentionally a control-plane route. It carries only the
    monitor CRUD target and planner diagnostics; it does not contain SQL,
    runtime tasks, evidence plans, or governance decisions.
    """

    kind: DbMonitorCommandKind
    monitor_id: str | None = None
    patch: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "patch", dict(self.patch))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class DbMonitorValidation:
    """Machine-readable validation result for a planned monitor."""

    accepted: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    unsupported_actions: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(
            self, "required_capabilities", tuple(self.required_capabilities)
        )
        object.__setattr__(
            self, "missing_capabilities", tuple(self.missing_capabilities)
        )
        object.__setattr__(self, "unsupported_actions", tuple(self.unsupported_actions))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "required_capabilities": list(self.required_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "unsupported_actions": list(self.unsupported_actions),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DbMonitorValidation":
        values = dict(data)
        return cls(
            accepted=bool(values.get("accepted")),
            warnings=tuple(values.get("warnings") or ()),
            errors=tuple(values.get("errors") or ()),
            required_capabilities=tuple(values.get("required_capabilities") or ()),
            missing_capabilities=tuple(values.get("missing_capabilities") or ()),
            unsupported_actions=tuple(values.get("unsupported_actions") or ()),
            diagnostics=dict(values.get("diagnostics") or {}),
        )


@dataclass(frozen=True)
class DbMonitorResolution:
    """Store-aware monitor reference resolution result."""

    monitor: DbMonitor | None
    monitor_ref: str | None
    matches: tuple[DbMonitor, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    definition_evidence: Evidence | None = None
    proposal_evidence: Evidence | None = None
    operation_id: str | None = None
    resolution_source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "matches", tuple(self.matches))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def accepted(self) -> bool:
        return (
            self.monitor is not None
            or self.definition_evidence is not None
            or self.proposal_evidence is not None
        ) and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_id": None if self.monitor is None else self.monitor.id,
            "monitor_ref": self.monitor_ref,
            "matches": [monitor.id for monitor in self.matches],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "definition_evidence_id": (
                None
                if self.definition_evidence is None
                else self.definition_evidence.id
            ),
            "proposal_evidence_id": (
                None if self.proposal_evidence is None else self.proposal_evidence.id
            ),
            "operation_id": self.operation_id,
            "resolution_source": self.resolution_source,
        }
