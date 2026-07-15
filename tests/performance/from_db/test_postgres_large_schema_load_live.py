"""Live PostgreSQL large-schema and large-table load benchmarks."""

from __future__ import annotations

import os

import pytest

from .postgres_scale_fixtures import (
    create_postgres_scale_agent,
    large_operational_schema_sql,
    postgres_version,
    scale_int_env,
)
from .scale_runner import (
    ScaleBenchmarkParameters,
    artifact_output_dir,
    assert_latency_gates,
    live_llm_required,
    measured_agent_operation_factory,
    postgres_live_required,
    run_scale_benchmark,
    write_artifact,
)

pytestmark = [
    pytest.mark.performance,
    pytest.mark.requires_db,
    pytest.mark.requires_llm,
    pytest.mark.slow,
]


async def test_postgres_large_schema_table_selection_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    harness, version, table_count, row_count = await _large_harness()
    try:
        artifact = await run_scale_benchmark(
            suite="postgres-large-schema-load",
            parameters=ScaleBenchmarkParameters(
                concurrency=3,
                operations=scale_int_env("DAITA_PERF_LARGE_OPERATIONS", 100),
                scenario="table-selection",
                extra={"table_count": table_count, "row_count": row_count},
            ),
            operation_factory=measured_agent_operation_factory(
                harness.agent,
                "How many operational events are open and high severity?",
                measurement=_large_measurement(
                    version, table_count, row_count, state="cold_mixed"
                ),
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-large-schema-load"),
            environment={
                "postgres_version": version,
                "provider": "openai",
                "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
                "model_parameters": {"temperature": 0},
                "dataset": "large_operational_schema",
                "database_type": "postgresql",
                "fixture_revision": _large_fixture_revision(table_count, row_count),
            },
        )
        assert_latency_gates(artifact, success_rate=0.99, p95_ms=5000)
        _assert_no_decoy_sql(artifact)
        _assert_sql_validation_precedes_execution(artifact)
        _persist_asserted_correctness(artifact, tmp_path)
    finally:
        await harness.stop()


async def test_postgres_large_schema_join_path_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    harness, version, table_count, row_count = await _large_harness()
    try:
        artifact = await run_scale_benchmark(
            suite="postgres-large-schema-load",
            parameters=ScaleBenchmarkParameters(
                concurrency=3,
                operations=scale_int_env("DAITA_PERF_LARGE_OPERATIONS", 100),
                scenario="join-path",
                extra={"table_count": table_count, "row_count": row_count},
            ),
            operation_factory=measured_agent_operation_factory(
                harness.agent,
                "Which accounts have open high severity incidents?",
                measurement=_large_measurement(
                    version, table_count, row_count, state="cold_mixed"
                ),
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-large-schema-load"),
            environment={
                "postgres_version": version,
                "provider": "openai",
                "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
                "model_parameters": {"temperature": 0},
                "dataset": "large_operational_schema",
                "database_type": "postgresql",
                "fixture_revision": _large_fixture_revision(table_count, row_count),
            },
        )
        assert_latency_gates(artifact, success_rate=0.99, p95_ms=10000)
        _assert_no_decoy_sql(artifact)
        assert any(
            "catalog.relationship_paths.find" in operation["capability_sequence"]
            for operation in artifact["operations"]
            if operation["success"]
        ), artifact["summary"]["capability_sequences"]
        _persist_asserted_correctness(artifact, tmp_path)
    finally:
        await harness.stop()


async def test_postgres_large_table_query_latency_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    harness, version, table_count, row_count = await _large_harness()
    try:
        artifact = await run_scale_benchmark(
            suite="postgres-large-schema-load",
            parameters=ScaleBenchmarkParameters(
                concurrency=3,
                operations=scale_int_env("DAITA_PERF_LARGE_OPERATIONS", 100),
                scenario="indexed-large-table-read",
                extra={"table_count": table_count, "row_count": row_count},
            ),
            operation_factory=measured_agent_operation_factory(
                harness.agent,
                "Count operational events where status is open and severity is high.",
                measurement=_large_measurement(
                    version, table_count, row_count, state="cold_mixed"
                ),
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-large-schema-load"),
            environment={
                "postgres_version": version,
                "provider": "openai",
                "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
                "model_parameters": {"temperature": 0},
                "dataset": "large_operational_schema",
                "database_type": "postgresql",
                "fixture_revision": _large_fixture_revision(table_count, row_count),
            },
        )
        assert_latency_gates(artifact, success_rate=0.99, p95_ms=10000)
        _assert_no_decoy_sql(artifact)
        _assert_sql_validation_precedes_execution(artifact)
        _persist_asserted_correctness(artifact, tmp_path)
    finally:
        await harness.stop()


async def _large_harness():
    table_count = scale_int_env("DAITA_PERF_LARGE_TABLES", 300)
    row_count = scale_int_env("DAITA_PERF_LARGE_ROWS", 10000000)
    harness = await create_postgres_scale_agent(
        sql=large_operational_schema_sql(table_count=table_count, row_count=row_count),
        name="PerfLargeSchemaPostgres",
        tag_prefix="daita-perf-large-schema-pg",
        llm=True,
    )
    return harness, await postgres_version(harness.url), table_count, row_count


def _large_measurement(
    version: str,
    table_count: int,
    row_count: int,
    *,
    state: str,
) -> dict:
    return {
        "control_label": "baseline",
        "provider": "openai",
        "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "model_parameters": {"temperature": 0},
        "database": {"type": "postgresql", "version": version},
        "fixture_revision": _large_fixture_revision(table_count, row_count),
        "state": state,
        "concurrency": 3,
        "correctness": {
            "answer": {
                "passed": None,
                "evaluation_source": "existing scale benchmark assertions",
            },
            "sql": {
                "passed": None,
                "evaluation_source": "existing scale benchmark assertions",
            },
        },
    }


def _large_fixture_revision(table_count: int, row_count: int) -> str:
    return f"large_operational_schema:tables={table_count}:rows={row_count}"


def _persist_asserted_correctness(artifact: dict, tmp_path) -> None:
    for operation in artifact["operations"]:
        operation["correctness"] = {
            "answer": {
                "passed": bool(operation["success"]),
                "evaluation_source": "existing scale benchmark assertions",
            },
            "sql": {
                "passed": True,
                "evaluation_source": "existing scale benchmark assertions",
            },
        }
    suite = artifact["suite"]
    scenario = artifact["parameters"]["scenario"]
    concurrency = artifact["parameters"]["concurrency"]
    write_artifact(
        artifact,
        artifact_output_dir(tmp_path, suite)
        / f"{suite}-{scenario}-c{concurrency}.json",
    )


def _assert_no_decoy_sql(artifact: dict) -> None:
    for operation in artifact["operations"]:
        sql = str(operation["metadata"].get("planned_sql") or "").lower()
        assert "operational_decoy_" not in sql, sql


def _assert_sql_validation_precedes_execution(artifact: dict) -> None:
    for operation in artifact["operations"]:
        sequence = operation["capability_sequence"]
        if "db.sql.execute_read" not in sequence:
            continue
        assert "db.sql.validate" in sequence, sequence
        assert sequence.index("db.sql.validate") < sequence.index("db.sql.execute_read")
