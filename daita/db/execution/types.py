"""Execution-local public and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daita.runtime import Evidence, Task


@dataclass(frozen=True)
class DbExecutionOutcome:
    """Result of executing one DB operation plan."""

    evidence: tuple[Evidence, ...]
    tasks: tuple[Task, ...]
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...] = ()
