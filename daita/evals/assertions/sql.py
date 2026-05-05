"""SQL safety and result assertions."""

from __future__ import annotations

import re

from ..analysis import RunEvidence, extract_sql_statements
from ..config import Expectations
from ..models import AssertionResult
from .common import fail, tool_row_count

WRITE_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|merge)\b", re.I
)


def sql_assertions(exp: Expectations, evidence: RunEvidence) -> list[AssertionResult]:
    statements = extract_sql_statements(evidence.tool_calls)
    results = []
    for sql_index, sql in enumerate(statements):
        low = sql.lower()
        if exp.sql.read_only and WRITE_SQL_RE.search(sql):
            results.append(
                fail(
                    f"sql[{sql_index}].read_only",
                    "sql_write_operation",
                    "SQL write operation used in read-only eval.",
                    "expectations.sql.read_only",
                    observed=sql,
                    expected="read-only",
                    fix_hints=[
                        "Use read-only query tools or tighten the agent prompt."
                    ],
                )
            )
        if exp.sql.require_limit and " limit " not in f" {low} ":
            results.append(
                fail(
                    f"sql[{sql_index}].require_limit",
                    "sql_missing_limit",
                    "SQL query did not include LIMIT.",
                    "expectations.sql.require_limit",
                    observed=sql,
                    expected="LIMIT",
                )
            )
        for index, value in enumerate(exp.sql.must_include):
            if value.lower() not in low:
                results.append(
                    fail(
                        f"sql.must_include[{index}]",
                        "sql_missing_required_text",
                        f"SQL did not include required text: {value}.",
                        f"expectations.sql.must_include[{index}]",
                        observed=sql,
                        expected=value,
                    )
                )
        for index, value in enumerate(exp.sql.must_not_include):
            if value.lower() in low:
                results.append(
                    fail(
                        f"sql.must_not_include[{index}]",
                        "sql_forbidden_text",
                        f"SQL included forbidden text: {value}.",
                        f"expectations.sql.must_not_include[{index}]",
                        observed=value,
                        expected=f"not {value}",
                    )
                )
        for index, table in enumerate(exp.sql.forbidden_tables):
            if re.search(rf"\b{re.escape(table.lower())}\b", low):
                results.append(
                    fail(
                        f"sql.forbidden_tables[{index}]",
                        "sql_forbidden_table",
                        f"SQL referenced forbidden table: {table}.",
                        f"expectations.sql.forbidden_tables[{index}]",
                        observed=sql,
                        expected=f"not {table}",
                    )
                )
        for index, table in enumerate(exp.sql.required_tables):
            if not re.search(rf"\b{re.escape(table.lower())}\b", low):
                results.append(
                    fail(
                        f"sql.required_tables[{index}]",
                        "sql_required_table_missing",
                        f"SQL did not reference required table: {table}.",
                        f"expectations.sql.required_tables[{index}]",
                        observed=sql,
                        expected=table,
                    )
                )
    if exp.sql.max_rows_returned is not None:
        for call_index, call in enumerate(evidence.tool_calls):
            row_count = tool_row_count(call.result)
            if row_count is not None and row_count > exp.sql.max_rows_returned:
                results.append(
                    fail(
                        "sql.max_rows_returned",
                        "sql_too_many_rows",
                        f"SQL tool returned {row_count} rows, over the limit {exp.sql.max_rows_returned}.",
                        "expectations.sql.max_rows_returned",
                        observed=row_count,
                        expected=exp.sql.max_rows_returned,
                        related_tool_calls=[call_index],
                    )
                )
    return results
