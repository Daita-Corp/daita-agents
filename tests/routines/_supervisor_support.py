"""Shared helpers extracted from ``test_supervisor.py``."""

from __future__ import annotations

from collections.abc import Callable


def _ids() -> Callable[[str], str]:
    counters: dict[str, int] = {}

    def create(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        if prefix == "run":
            return f"run-{counters[prefix]:032x}"
        return f"{prefix}-{counters[prefix]}"

    return create
