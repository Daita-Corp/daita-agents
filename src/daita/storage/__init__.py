"""Concrete persistence for embedded agent state."""

from .sqlite import SQLiteStateStore

__all__ = ["SQLiteStateStore"]
