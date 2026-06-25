"""Assertions over runtime query-result evidence."""

from __future__ import annotations

from numbers import Number
from typing import Any

from ..analysis import RunEvidence
from ..config import Expectations
from ..models import AssertionResult
from .common import fail


def result_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    if not _result_expected(exp):
        return []

    query_results = [item for item in evidence.evidence if item.kind == "query.result"]
    rows = [row for item in query_results for row in _rows(item.payload)]
    columns = _columns(query_results, rows)
    results: list[AssertionResult] = []

    if not query_results:
        results.append(
            fail(
                "result",
                "query_result_missing",
                "No query.result evidence was produced.",
                "expectations.result",
                observed=None,
                expected="query.result evidence",
            )
        )
        return results

    if exp.result.min_rows is not None and len(rows) < exp.result.min_rows:
        results.append(
            fail(
                "result.min_rows",
                "query_result_too_few_rows",
                f"Expected at least {exp.result.min_rows} query result rows, got {len(rows)}.",
                "expectations.result.min_rows",
                observed=len(rows),
                expected=exp.result.min_rows,
            )
        )
    if exp.result.max_rows is not None and len(rows) > exp.result.max_rows:
        results.append(
            fail(
                "result.max_rows",
                "query_result_too_many_rows",
                f"Expected at most {exp.result.max_rows} query result rows, got {len(rows)}.",
                "expectations.result.max_rows",
                observed=len(rows),
                expected=exp.result.max_rows,
            )
        )

    for index, column in enumerate(exp.result.required_columns):
        if column not in columns:
            results.append(
                fail(
                    f"result.required_columns[{index}]",
                    "query_result_required_column_missing",
                    f"Required query result column was missing: {column}.",
                    f"expectations.result.required_columns[{index}]",
                    observed=sorted(columns),
                    expected=column,
                )
            )

    for index, expected in enumerate(exp.result.required_rows):
        if not any(_row_matches(expected, row) for row in rows):
            results.append(
                fail(
                    f"result.required_rows[{index}]",
                    "query_result_required_row_missing",
                    "Required query result row was not observed.",
                    f"expectations.result.required_rows[{index}]",
                    observed=_preview_rows(rows),
                    expected=expected,
                )
            )

    for index, forbidden in enumerate(exp.result.forbidden_rows):
        if any(_row_matches(forbidden, row) for row in rows):
            results.append(
                fail(
                    f"result.forbidden_rows[{index}]",
                    "query_result_forbidden_row_observed",
                    "Forbidden query result row was observed.",
                    f"expectations.result.forbidden_rows[{index}]",
                    observed=forbidden,
                    expected=f"not {forbidden}",
                )
            )

    return results


def _result_expected(exp: Expectations) -> bool:
    result = exp.result
    return any(
        (
            bool(result.required_columns),
            bool(result.required_rows),
            bool(result.forbidden_rows),
            result.min_rows is not None,
            result.max_rows is not None,
        )
    )


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("rows")
    if not isinstance(raw, list):
        return []
    rows = []
    for row in raw:
        if isinstance(row, dict):
            rows.append(row)
        elif isinstance(row, (list, tuple)):
            rows.append({str(index): value for index, value in enumerate(row)})
        else:
            rows.append({"value": row})
    return rows


def _columns(query_results, rows: list[dict[str, Any]]) -> set[str]:
    columns: set[str] = set()
    for item in query_results:
        raw_columns = item.payload.get("columns")
        if isinstance(raw_columns, list):
            columns.update(str(column) for column in raw_columns)
    for row in rows:
        columns.update(str(column) for column in row)
    return columns


def _row_matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in observed:
            return False
        if not _value_matches(expected_value, observed[key]):
            return False
    return True


def _value_matches(expected: Any, observed: Any) -> bool:
    if isinstance(expected, Number) and isinstance(observed, Number):
        return abs(float(expected) - float(observed)) <= 1e-9
    return str(expected) == str(observed)


def _preview_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows[:10]
