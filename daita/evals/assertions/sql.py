"""SQL assertions over runtime SQL validation and query-result evidence."""

from __future__ import annotations

import re

from ..analysis import RunEvidence, extract_sql_statements
from ..config import Expectations
from ..models import AssertionResult, RuntimeEvidenceRecord
from .common import fail, payload_row_count


def sql_assertions(exp: Expectations, evidence: RunEvidence) -> list[AssertionResult]:
    sql_records = _sql_records(evidence.evidence)
    results: list[AssertionResult] = []
    for sql_index, record in enumerate(sql_records):
        sql = str(record.get("sql") or "")
        low = sql.lower()
        facts = record["facts"]
        evidence_id = record.get("evidence_id")
        task_id = record.get("task_id")
        related_tasks = [task_id] if task_id else []
        related_evidence = [evidence_id] if evidence_id else []

        if exp.sql.read_only and _is_write_statement(facts, sql):
            results.append(
                fail(
                    f"sql[{sql_index}].read_only",
                    "sql_write_operation",
                    "SQL write operation used in read-only eval.",
                    "expectations.sql.read_only",
                    observed=sql,
                    expected="read-only",
                    fix_hints=[
                        "Use read-only query capabilities or tighten runtime policy."
                    ],
                    related_task_ids=related_tasks,
                    related_evidence_ids=related_evidence,
                )
            )
        if exp.sql.require_limit and not _has_limit(facts, sql):
            results.append(
                fail(
                    f"sql[{sql_index}].require_limit",
                    "sql_missing_limit",
                    "SQL query did not include LIMIT.",
                    "expectations.sql.require_limit",
                    observed=sql,
                    expected="LIMIT",
                    related_task_ids=related_tasks,
                    related_evidence_ids=related_evidence,
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
                        related_task_ids=related_tasks,
                        related_evidence_ids=related_evidence,
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
                        related_task_ids=related_tasks,
                        related_evidence_ids=related_evidence,
                    )
                )
        resources = _referenced_resources(record)
        for index, table in enumerate(exp.sql.forbidden_tables):
            if _resource_matches(table, resources) or re.search(
                rf"\b{re.escape(table.lower())}\b", low
            ):
                results.append(
                    fail(
                        f"sql.forbidden_tables[{index}]",
                        "sql_forbidden_table",
                        f"SQL referenced forbidden table: {table}.",
                        f"expectations.sql.forbidden_tables[{index}]",
                        observed=sql,
                        expected=f"not {table}",
                        related_task_ids=related_tasks,
                        related_evidence_ids=related_evidence,
                    )
                )
        for index, table in enumerate(exp.sql.required_tables):
            if not _resource_matches(table, resources) and not re.search(
                rf"\b{re.escape(table.lower())}\b", low
            ):
                results.append(
                    fail(
                        f"sql.required_tables[{index}]",
                        "sql_required_table_missing",
                        f"SQL did not reference required table: {table}.",
                        f"expectations.sql.required_tables[{index}]",
                        observed=sql,
                        expected=table,
                        related_task_ids=related_tasks,
                        related_evidence_ids=related_evidence,
                    )
                )
    if exp.sql.max_rows_returned is not None:
        for item in evidence.evidence:
            if item.kind != "query.result":
                continue
            row_count = payload_row_count(item.payload)
            if row_count is not None and row_count > exp.sql.max_rows_returned:
                results.append(
                    fail(
                        "sql.max_rows_returned",
                        "sql_too_many_rows",
                        f"SQL query returned {row_count} rows, over the limit {exp.sql.max_rows_returned}.",
                        "expectations.sql.max_rows_returned",
                        observed=row_count,
                        expected=exp.sql.max_rows_returned,
                        related_task_ids=[item.task_id] if item.task_id else [],
                        related_evidence_ids=[item.id] if item.id else [],
                    )
                )
    return results


def _sql_records(evidence: list[RuntimeEvidenceRecord]) -> list[dict]:
    records = []
    for item in evidence:
        if item.kind != "sql.validation":
            continue
        payload = item.payload
        facts = payload.get("statement_facts")
        if not isinstance(facts, dict):
            facts = {}
        records.append(
            {
                "sql": payload.get("sql") or payload.get("query") or "",
                "facts": facts,
                "referenced_tables": payload.get("referenced_tables") or [],
                "target_resources": facts.get("target_resources") or [],
                "task_id": item.task_id,
                "evidence_id": item.id,
            }
        )
    if records:
        return records
    return [{"sql": sql, "facts": {}} for sql in extract_sql_statements(evidence)]


def _is_write_statement(facts: dict, sql: str) -> bool:
    if facts.get("is_read") is False:
        return True
    has_mutating_facts = bool(
        facts.get("mutating_statement_classes")
        or facts.get("destructive_statement_classes")
        or facts.get("admin_statement_classes")
    )
    if facts:
        return has_mutating_facts
    return bool(
        re.search(
            r"\b(insert|update|delete|drop|truncate|alter|create|merge)\b", sql, re.I
        )
    )


def _has_limit(facts: dict, sql: str) -> bool:
    if "has_limit" in facts:
        return bool(facts["has_limit"])
    return " limit " in f" {sql.lower()} "


def _referenced_resources(record: dict) -> list[str]:
    values = [
        *list(record.get("referenced_tables") or []),
        *list(record.get("target_resources") or []),
    ]
    return [str(value).lower() for value in values]


def _resource_matches(expected: str, observed: list[str]) -> bool:
    expected_low = expected.lower()
    return any(
        expected_low == resource
        or expected_low in resource
        or resource.endswith(f".{expected_low}")
        for resource in observed
    )
