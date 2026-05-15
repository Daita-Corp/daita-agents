"""Small shared helpers for ``from_db`` internals."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, TypeVar

T = TypeVar("T", bound=Hashable)

ANALYST_TOOL_PREFIXES = (
    "pivot_",
    "correlate",
    "detect_",
    "compare_",
    "find_",
    "forecast_",
)


def unique_preserving_order(
    values: Iterable[T], *, skip_empty: bool = False
) -> list[T]:
    """Return values once, preserving the first-seen order."""
    seen: set[T] = set()
    out: list[T] = []
    for value in values:
        if skip_empty and not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def plugin_database_name(plugin: Any) -> Any:
    if plugin is None:
        return None
    return (
        getattr(plugin, "database_name", None)
        or getattr(plugin, "database", None)
        or getattr(plugin, "db", None)
        or getattr(plugin, "path", None)
    )
