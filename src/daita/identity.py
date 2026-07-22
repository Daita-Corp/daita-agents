"""Canonical persistent-agent identity records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    id: str
    display_name: str
    created_at: datetime

    def __post_init__(self) -> None:
        _required_text(self.id, "agent id")
        _required_text(self.display_name, "agent display_name")
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("agent created_at must be timezone-aware")


class AgentIdentityStoreError(RuntimeError):
    """Base failure for authoritative identity persistence."""


class AgentIdentityConflictError(AgentIdentityStoreError):
    """Raised when a database already belongs to another agent."""
