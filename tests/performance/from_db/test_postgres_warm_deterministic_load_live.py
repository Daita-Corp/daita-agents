"""Live PostgreSQL deterministic warm-path load benchmarks."""

from __future__ import annotations

import pytest

from .postgres_scale_fixtures import (
    create_postgres_scale_agent,
    medium_wide_schema_sql,
    postgres_version,
    profile_catalog_values,
    rich_schema_sql,
    scale_concurrency_env,
    scale_int_env,
)
from .scale_runner import (
    ScaleBenchmarkParameters,
    artifact_output_dir,
    assert_latency_gates,
    postgres_live_required,
    run_scale_benchmark,
)

pytestmark = [
    pytest.mark.performance,
    pytest.mark.requires_db,
]


async def test_postgres_warm_simple_count_load(tmp_path):
    postgres_live_required()
    operations = scale_int_env("DAITA_PERF_WARM_OPERATIONS", 1000)
    harness = await create_postgres_scale_agent(
        sql=rich_schema_sql(),
        name="PerfWarmCountPostgres",
        tag_prefix="daita-perf-warm-count-pg",
        llm=False,
    )
    try:
        await harness.agent.run_detailed("How many customers are there?")
        version = await postgres_version(harness.url)
        for concurrency in scale_concurrency_env(
            "DAITA_PERF_WARM_CONCURRENCY", "1,5,10,25"
        ):
            artifact = await run_scale_benchmark(
                suite="postgres-warm-deterministic-load",
                parameters=ScaleBenchmarkParameters(
                    concurrency=concurrency,
                    operations=operations,
                    scenario="simple-count",
                ),
                operation_factory=lambda _index: harness.agent.run_detailed(
                    "How many customers are there?"
                ),
                output_dir=artifact_output_dir(
                    tmp_path, "postgres-warm-deterministic-load"
                ),
                environment={
                    "postgres_version": version,
                    "model": None,
                    "dataset": "small_rich_schema",
                },
            )
            assert_latency_gates(
                artifact,
                success_rate=0.995,
                p50_ms=1500,
                p95_ms=4000,
                p99_ms=7000,
                llm_calls_per_operation=0,
            )
    finally:
        await harness.stop()


async def test_postgres_warm_literal_filter_load(tmp_path):
    postgres_live_required()
    operations = scale_int_env("DAITA_PERF_WARM_OPERATIONS", 1000)
    harness = await create_postgres_scale_agent(
        sql=rich_schema_sql(),
        name="PerfWarmLiteralPostgres",
        tag_prefix="daita-perf-warm-literal-pg",
        llm=False,
    )
    try:
        await harness.agent.run_detailed(
            "How many open high severity support tickets are there?"
        )
        await profile_catalog_values(
            harness.agent,
            [("support_tickets", "status"), ("support_tickets", "severity")],
        )
        version = await postgres_version(harness.url)
        artifact = await run_scale_benchmark(
            suite="postgres-warm-deterministic-load",
            parameters=ScaleBenchmarkParameters(
                concurrency=10,
                operations=operations,
                scenario="literal-filter",
            ),
            operation_factory=lambda index: harness.agent.run_detailed(
                _literal_prompt(index)
            ),
            output_dir=artifact_output_dir(
                tmp_path, "postgres-warm-deterministic-load"
            ),
            environment={
                "postgres_version": version,
                "model": None,
                "dataset": "small_rich_schema",
            },
        )
        assert_latency_gates(
            artifact,
            success_rate=0.995,
            p50_ms=1500,
            p95_ms=4000,
            p99_ms=7000,
            llm_calls_per_operation=0,
        )
        _assert_literal_predicates(artifact)
    finally:
        await harness.stop()


async def test_postgres_warm_wide_schema_disambiguation_load(tmp_path):
    postgres_live_required()
    operations = scale_int_env("DAITA_PERF_WIDE_OPERATIONS", 1000)
    table_count = scale_int_env("DAITA_PERF_WIDE_TABLES", 100)
    rows_per_table = scale_int_env("DAITA_PERF_WIDE_ROWS_PER_TABLE", 10000)
    harness = await create_postgres_scale_agent(
        sql=medium_wide_schema_sql(
            table_count=table_count, rows_per_table=rows_per_table
        ),
        name="PerfWarmWidePostgres",
        tag_prefix="daita-perf-warm-wide-pg",
        llm=False,
    )
    try:
        await harness.agent.run_detailed(
            "How many open high severity support tickets are there?"
        )
        await profile_catalog_values(
            harness.agent,
            [("support_tickets", "status"), ("support_tickets", "severity")],
        )
        version = await postgres_version(harness.url)
        artifact = await run_scale_benchmark(
            suite="postgres-warm-deterministic-load",
            parameters=ScaleBenchmarkParameters(
                concurrency=10,
                operations=operations,
                scenario="wide-schema-disambiguation",
                extra={"table_count": table_count, "rows_per_table": rows_per_table},
            ),
            operation_factory=lambda _index: harness.agent.run_detailed(
                "How many open high severity support tickets are there?"
            ),
            output_dir=artifact_output_dir(
                tmp_path, "postgres-warm-deterministic-load"
            ),
            environment={
                "postgres_version": version,
                "model": None,
                "dataset": "medium_wide_schema",
            },
        )
        assert_latency_gates(
            artifact,
            success_rate=0.995,
            p50_ms=1500,
            p95_ms=4000,
            p99_ms=7000,
            llm_calls_per_operation=0,
        )
        sequences = {
            tuple(operation["capability_sequence"])
            for operation in artifact["operations"]
            if operation["success"]
        }
        assert len(sequences) == 1, artifact["summary"]["capability_sequences"]
        assert all(
            "customer_activity_decoy"
            not in str(operation["metadata"].get("planned_sql", "")).lower()
            for operation in artifact["operations"]
        )
    finally:
        await harness.stop()


def _literal_prompt(index: int) -> str:
    if index % 2 == 0:
        return "How many support tickets have status='open' and severity='high'?"
    return "Count open high severity support tickets."


def _assert_literal_predicates(artifact: dict) -> None:
    for operation in artifact["operations"]:
        if not operation["success"]:
            continue
        sql = str(operation["metadata"].get("planned_sql") or "").lower()
        assert "support_tickets" in sql, sql
        assert "status" in sql and "open" in sql, sql
        assert "severity" in sql and "high" in sql, sql
