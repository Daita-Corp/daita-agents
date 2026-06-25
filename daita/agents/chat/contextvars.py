"""Context-local access to the active Agent run state."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from .state import RunState

active_run_state: ContextVar[Optional[RunState]] = ContextVar(
    "daita_active_agent_run_state", default=None
)


def get_active_run_state() -> Optional[RunState]:
    """Return the current task-local run state, if an Agent run is active."""
    return active_run_state.get()
