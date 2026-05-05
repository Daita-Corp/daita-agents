"""Shared helpers for deterministic eval assertions."""

from __future__ import annotations

from typing import Iterable

from ..models import AssertionResult


def has_errors(results: Iterable[AssertionResult]) -> bool:
    return any(r.status == "failed" and r.severity == "error" for r in results)


def matches(expected: str, observed: str) -> bool:
    return expected == observed or expected in observed


def matches_any(expected: str, observed_values: Iterable[str]) -> bool:
    return any(matches(expected, observed) for observed in observed_values)


def tool_row_count(result) -> int | None:
    if isinstance(result, dict):
        for key in ("row_count", "total_rows", "count"):
            value = result.get(key)
            if isinstance(value, int):
                return value
        rows = result.get("rows")
        if isinstance(rows, list):
            return len(rows)
    if isinstance(result, list):
        return len(result)
    return None


def fail(
    assertion_id: str,
    code: str,
    message: str,
    assertion_path: str,
    *,
    observed=None,
    expected=None,
    fix_hints: list[str] | None = None,
    related_tool_calls: list[int] | None = None,
) -> AssertionResult:
    return AssertionResult(
        id=assertion_id,
        status="failed",
        code=code,
        message=message,
        assertion_path=assertion_path,
        observed=observed,
        expected=expected,
        fix_hints=fix_hints or [],
        related_tool_calls=related_tool_calls or [],
    )
