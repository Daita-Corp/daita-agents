"""Shared live data fixtures; importing them performs no external I/O."""

from __future__ import annotations

from tests.support.postgresql_live import database

__all__ = ["database"]
