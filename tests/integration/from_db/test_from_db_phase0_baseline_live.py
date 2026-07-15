"""Phase 0 neutral live-SQLite baseline artifact capture.

Run:
    DAITA_RUN_LIVE_LLM=1 DAITA_PHASE0_OUTPUT_DIR=<path> pytest \
        tests/integration/from_db/test_from_db_phase0_baseline_live.py \
        -m "requires_llm and integration" -v -s
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
import sqlite3

import pytest

from daita.runtime import OperationStatus
from tests.integration.from_db.live_production_helpers import (
    all_sql_strings,
    create_live_sqlite_from_db_agent,
    diagnostic_text,
    query_rows,
    row_values,
    seed_rich_sqlite_schema,
    sql_from_result,
    task_capabilities,
)
from tests.performance.from_db.scale_runner import (
    NEUTRAL_ARTIFACT_SCHEMA_NAME,
    NEUTRAL_ARTIFACT_SCHEMA_VERSION,
    ScaleBenchmarkParameters,
    default_environment_metadata,
    measure_agent_operation,
    run_scale_benchmark,
    write_artifact,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]

BASELINE_SHA = "b87df31873d33fffbf50498f5dc4d8892115e8f8"
FIXTURE_REVISION = f"rich-sqlite-production-contract@{BASELINE_SHA}"


async def test_phase0_live_sqlite_comparable_baseline_artifacts(tmp_path):
    output_dir = Path(os.environ.get("DAITA_PHASE0_OUTPUT_DIR", tmp_path)) / "sqlite"
    db_path = await seed_rich_sqlite_schema(tmp_path / "phase0-rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "phase0-runtime.sqlite",
        name="Phase0BaselineSQLite",
    )
    artifacts: list[str] = []
    try:
        artifacts.append(
            await _capture(
                agent,
                output_dir,
                scenario="simple-customer-count",
                prompt="How many customers are there?",
                state="cold",
                evaluator=_simple_count_correctness,
            )
        )
        artifacts.append(
            await _capture(
                agent,
                output_dir,
                scenario="relationship-join",
                prompt=(
                    "Join orders to customers using their relationship and return records"
                ),
                state="warm",
                evaluator=_relationship_correctness,
            )
        )
        artifacts.append(
            await _capture(
                agent,
                output_dir,
                scenario="literal-grounding-completed-vs-complete",
                prompt=(
                    "Search catalog column values for orders.status, then show "
                    "completed orders by status."
                ),
                state="warm",
                evaluator=_literal_correctness,
            )
        )
        artifacts.append(
            await _capture(
                agent,
                output_dir,
                scenario="zero-row-behavior",
                prompt=(
                    "Return order_id and status for orders whose status is pending "
                    "and total is greater than 1000."
                ),
                state="warm",
                evaluator=_zero_row_correctness,
            )
        )
        artifacts.append(
            await _capture(
                agent,
                output_dir,
                scenario="repair-grounding-warmup",
                prompt=(
                    "Search catalog column values for support_tickets.status and "
                    "support_tickets.severity."
                ),
                state="warm",
                evaluator=_safe_terminal_correctness,
            )
        )
        artifacts.append(
            await _capture(
                agent,
                output_dir,
                scenario="invalid-literal-or-schema-repair",
                prompt=(
                    "Using observed catalog column values, answer: how many support "
                    "tickets have status='unresolved' and severity='urgent'?"
                ),
                state="warm",
                evaluator=_repair_correctness,
            )
        )
    finally:
        await agent.stop()

    stateful_db = await seed_rich_sqlite_schema(tmp_path / "phase0-stateful.sqlite")
    stateful = await create_live_sqlite_from_db_agent(
        stateful_db,
        runtime_path=tmp_path / "phase0-stateful-runtime.sqlite",
        name="Phase0BaselineStatefulSQLite",
        stateful=True,
    )
    try:
        artifacts.append(
            await _capture(
                stateful,
                output_dir,
                scenario="stateful-primer",
                prompt="Show enterprise customers in NA.",
                state="cold",
                evaluator=_stateful_primer_correctness,
            )
        )
        artifacts.append(
            await _capture(
                stateful,
                output_dir,
                scenario="stateful-followup",
                prompt="What are their completed order totals?",
                state="warm",
                evaluator=_stateful_followup_correctness,
            )
        )
    finally:
        await stateful.stop()

    manifest = {
        "schema": {
            "name": NEUTRAL_ARTIFACT_SCHEMA_NAME,
            "version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
        },
        "source_git_sha": BASELINE_SHA,
        "artifacts": artifacts,
        "coverage": {
            "simple_customer_count": "captured",
            "relationship_join": "captured",
            "literal_grounding": "captured",
            "stateful_followup": "captured",
            "invalid_sql_or_schema_repair": "captured_with_actual_repair_count",
            "zero_row_behavior": "captured",
            "multi_query_analysis": "not covered by the existing comparable SQLite suite",
        },
    }
    manifest_path = write_artifact(manifest, output_dir / "manifest.json")
    assert len(artifacts) == 8
    assert manifest_path.exists()


async def _capture(
    agent,
    output_dir: Path,
    *,
    scenario: str,
    prompt: str,
    state: str,
    evaluator,
) -> str:
    environment = default_environment_metadata(
        model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        provider="openai",
        model_parameters={"temperature": 0},
        dataset="rich_sqlite_production_contract",
        database_type="sqlite",
        database_version=sqlite3.sqlite_version,
        fixture_revision=FIXTURE_REVISION,
        source_git_sha=BASELINE_SHA,
        control_label="baseline",
    )

    async def operation(_index: int):
        return await measure_agent_operation(
            agent,
            prompt,
            measurement={
                "scenario": scenario,
                "run_id": "run-001",
                "control_label": "baseline",
                "provider": "openai",
                "model": environment["model"],
                "model_parameters": {"temperature": 0},
                "database": {
                    "type": "sqlite",
                    "version": sqlite3.sqlite_version,
                },
                "fixture_revision": FIXTURE_REVISION,
                "state": state,
                "concurrency": 1,
            },
            correctness_evaluator=evaluator,
        )

    artifact = await run_scale_benchmark(
        suite="from-db-slim-phase0-sqlite-baseline",
        parameters=ScaleBenchmarkParameters(
            concurrency=1,
            operations=1,
            scenario=scenario,
            extra={"state": state},
        ),
        operation_factory=operation,
        output_dir=output_dir,
        environment=environment,
        artifact_name=f"{scenario}.json",
    )
    operation_record = artifact["operations"][0]
    answer_passed = operation_record["correctness"]["answer"].get("passed")
    sql_passed = operation_record["correctness"]["sql"].get("passed")
    assert answer_passed is True, operation_record["correctness"]
    assert sql_passed is True, operation_record["correctness"]
    return str(output_dir / f"{scenario}.json")


def _simple_count_correctness(result, snapshot) -> dict:
    sql = sql_from_result(snapshot)
    rows = row_values(snapshot)
    return _correctness(
        answer=result.status is OperationStatus.SUCCEEDED
        and 4 in rows
        and bool(re.search(r"\b4\b", result.answer or "")),
        sql=_safe_read_sql(sql)
        and "customers" in sql.lower()
        and bool(re.search(r"(?i)\bcount\s*\(", sql)),
        facts={"expected_count": 4, "observed_values": sorted(map(str, rows))},
    )


def _relationship_correctness(result, snapshot) -> dict:
    sql = sql_from_result(snapshot)
    rows = query_rows(snapshot)
    values = {str(value) for row in rows for value in row.values()}
    return _correctness(
        answer=result.status is OperationStatus.SUCCEEDED
        and len(rows) >= 5
        and "Ada Lovelace" in values,
        sql=_safe_read_sql(sql)
        and all(token in sql.lower() for token in ("orders", "customers", "join")),
        facts={
            "row_count": len(rows),
            "catalog_relationship_task": _has_capability(
                snapshot, "catalog.relationship_paths.find"
            ),
        },
    )


def _literal_correctness(result, snapshot) -> dict:
    sql = sql_from_result(snapshot)
    rows = query_rows(snapshot)
    statuses = {row.get("status") for row in rows if "status" in row}
    return _correctness(
        answer=result.status is OperationStatus.SUCCEEDED and bool(rows),
        sql=_safe_read_sql(sql)
        and "orders" in sql.lower()
        and bool(re.search(r"(?i)['\"]complete['\"]", sql))
        and not bool(re.search(r"(?i)['\"]completed['\"]", sql)),
        facts={"observed_statuses": sorted(map(str, statuses))},
    )


def _zero_row_correctness(result, snapshot) -> dict:
    sql = sql_from_result(snapshot)
    answer = (result.answer or "").lower()
    truthful = any(
        phrase in answer
        for phrase in ("no rows", "no matching", "no orders", "0 rows", "zero rows")
    )
    return _correctness(
        answer=result.status is OperationStatus.SUCCEEDED
        and query_rows(snapshot) == []
        and truthful,
        sql=_safe_read_sql(sql)
        and all(token in sql.lower() for token in ("orders", "status", "total")),
        facts={
            "row_count": len(query_rows(snapshot)),
            "truthful_zero_row_answer": truthful,
        },
    )


def _safe_terminal_correctness(result, snapshot) -> dict:
    sql_values = all_sql_strings(snapshot)
    return _correctness(
        answer=result.status
        in {OperationStatus.SUCCEEDED, OperationStatus.BLOCKED, OperationStatus.FAILED},
        sql=all(_safe_read_sql(sql) for sql in sql_values),
        facts={"sql_statement_count": len(sql_values)},
    )


def _repair_correctness(result, snapshot) -> dict:
    sql_values = all_sql_strings(snapshot)
    failing_sql = [
        item.payload.get("sql")
        for item in snapshot.evidence
        if item.kind in {"sql.validation", "query.plan.validation"}
        and item.accepted is False
        and isinstance(item.payload.get("sql"), str)
    ]
    repeated = [sql for sql, count in Counter(failing_sql).items() if count > 2]
    if result.status is OperationStatus.SUCCEEDED:
        final_sql = sql_from_result(snapshot)
        answer_correct = bool(final_sql) and not re.search(
            r"(?i)['\"](?:unresolved|urgent)['\"]", final_sql
        )
    else:
        text = diagnostic_text(snapshot).lower()
        answer_correct = any(
            token in text
            for token in ("unobserved_filter_literal", "repair", "validation")
        )
    return _correctness(
        answer=answer_correct,
        sql=all(_safe_read_sql(sql) for sql in sql_values) and not repeated,
        facts={
            "terminal_status": result.status.value,
            "repair_task_count": task_capabilities(snapshot).count("db.query.repair"),
            "repeated_failing_sql": repeated,
        },
    )


def _stateful_primer_correctness(result, snapshot) -> dict:
    sql = sql_from_result(snapshot)
    values = row_values(snapshot)
    return _correctness(
        answer=result.status is OperationStatus.SUCCEEDED
        and "Ada Lovelace" in values
        and "Grace Hopper" in values,
        sql=_safe_read_sql(sql)
        and "enterprise" in sql.lower()
        and bool(re.search(r"(?i)\bNA\b|north america", sql)),
        facts={"session_id_present": bool(result.request.session_id)},
    )


def _stateful_followup_correctness(result, snapshot) -> dict:
    sql = sql_from_result(snapshot)
    values = row_values(snapshot)
    numeric = _numeric_values(query_rows(snapshot))
    return _correctness(
        answer=result.status is OperationStatus.SUCCEEDED
        and "Linus Torvalds" not in values
        and "Katherine Johnson" not in values
        and bool({120.0, 175.0, 295.0} & numeric),
        sql=_safe_read_sql(sql)
        and all(token in sql.lower() for token in ("orders", "complete", "enterprise"))
        and bool(re.search(r"(?i)\bNA\b|north america", sql)),
        facts={
            "numeric_values": sorted(numeric),
            "session_id_present": bool(result.request.session_id),
        },
    )


def _correctness(*, answer: bool, sql: bool, facts: dict) -> dict:
    return {
        "answer": {"passed": bool(answer), "facts": facts},
        "sql": {"passed": bool(sql), "safety": "read_only" if sql else "failed"},
    }


def _safe_read_sql(sql: str) -> bool:
    normalized = sql.strip().lower()
    return (
        bool(normalized)
        and normalized.startswith(("select", "with"))
        and not any(
            re.search(rf"(?i)\b{verb}\b", sql)
            for verb in ("delete", "drop", "update", "insert", "alter", "truncate")
        )
    )


def _has_capability(snapshot, capability_id: str) -> bool:
    return capability_id in task_capabilities(snapshot)


def _numeric_values(rows: list[dict[str, object]]) -> set[float]:
    values = set()
    for row in rows:
        for value in row.values():
            try:
                values.add(float(str(value).replace(",", "")))
            except (TypeError, ValueError):
                continue
    return values
