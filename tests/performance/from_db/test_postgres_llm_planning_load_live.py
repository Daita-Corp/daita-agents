"""Live PostgreSQL model-backed planning load benchmarks."""

from __future__ import annotations

import pytest

from .postgres_scale_fixtures import (
    create_postgres_scale_agent,
    medium_wide_schema_sql,
    postgres_version,
    rich_schema_sql,
    scale_concurrency_env,
    scale_int_env,
)
from .scale_runner import (
    ScaleBenchmarkParameters,
    artifact_output_dir,
    assert_latency_gates,
    live_llm_required,
    postgres_live_required,
    run_scale_benchmark,
)

pytestmark = [
    pytest.mark.performance,
    pytest.mark.requires_db,
    pytest.mark.requires_llm,
]


async def test_postgres_llm_join_planning_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    operations = scale_int_env("DAITA_PERF_LLM_OPERATIONS", 100)
    harness = await create_postgres_scale_agent(
        sql=rich_schema_sql(),
        name="PerfLlmJoinPostgres",
        tag_prefix="daita-perf-llm-join-pg",
        llm=True,
    )
    try:
        version = await postgres_version(harness.url)
        for concurrency in scale_concurrency_env("DAITA_PERF_LLM_CONCURRENCY", "1,3,5"):
            artifact = await run_scale_benchmark(
                suite="postgres-llm-planning-load",
                parameters=ScaleBenchmarkParameters(
                    concurrency=concurrency,
                    operations=operations,
                    scenario="join-planning",
                ),
                operation_factory=lambda _index: harness.agent.run_detailed(
                    "Which customers have open high severity support tickets? Join customers to support tickets."
                ),
                output_dir=artifact_output_dir(tmp_path, "postgres-llm-planning-load"),
                environment={
                    "postgres_version": version,
                    "model": "openai",
                    "dataset": "small_rich_schema",
                },
            )
            assert_latency_gates(
                artifact,
                success_rate=0.95,
                p95_ms=30000,
                p99_ms=45000,
            )
            _assert_no_unsafe_sql(artifact)
    finally:
        await harness.stop()


async def test_postgres_llm_ambiguous_query_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    operations = scale_int_env("DAITA_PERF_LLM_OPERATIONS", 100)
    table_count = scale_int_env("DAITA_PERF_LLM_WIDE_TABLES", 100)
    harness = await create_postgres_scale_agent(
        sql=medium_wide_schema_sql(table_count=table_count, rows_per_table=100),
        name="PerfLlmAmbiguousPostgres",
        tag_prefix="daita-perf-llm-ambiguous-pg",
        llm=True,
    )
    try:
        version = await postgres_version(harness.url)
        artifact = await run_scale_benchmark(
            suite="postgres-llm-planning-load",
            parameters=ScaleBenchmarkParameters(
                concurrency=3,
                operations=operations,
                scenario="ambiguous-query",
                extra={"table_count": table_count},
            ),
            operation_factory=lambda _index: harness.agent.run_detailed(
                "Show open high severity customer activity."
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-llm-planning-load"),
            environment={
                "postgres_version": version,
                "model": "openai",
                "dataset": "medium_wide_schema",
            },
        )
        assert_latency_gates(artifact, success_rate=0.95, p95_ms=30000, p99_ms=45000)
        _assert_no_unsafe_sql(artifact)
    finally:
        await harness.stop()


async def test_postgres_llm_repair_loop_bound_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    operations = scale_int_env("DAITA_PERF_LLM_REPAIR_OPERATIONS", 100)
    harness = await create_postgres_scale_agent(
        sql=rich_schema_sql(),
        name="PerfLlmRepairPostgres",
        tag_prefix="daita-perf-llm-repair-pg",
        llm=True,
    )
    try:
        version = await postgres_version(harness.url)
        artifact = await run_scale_benchmark(
            suite="postgres-llm-planning-load",
            parameters=ScaleBenchmarkParameters(
                concurrency=3,
                operations=operations,
                scenario="repair-loop-bound",
            ),
            operation_factory=lambda _index: harness.agent.run_detailed(
                "How many support tickets have status='unresolved' and severity='urgent'?"
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-llm-planning-load"),
            environment={
                "postgres_version": version,
                "model": "openai",
                "dataset": "small_rich_schema",
            },
        )
        assert_latency_gates(artifact, success_rate=0.95, p95_ms=30000, p99_ms=45000)
        _assert_no_unsafe_sql(artifact)
        for operation in artifact["operations"]:
            repairs = operation["capability_sequence"].count("db.query.repair")
            assert repairs <= 2, operation
    finally:
        await harness.stop()


def _assert_no_unsafe_sql(artifact: dict) -> None:
    for operation in artifact["operations"]:
        sql = str(operation["metadata"].get("planned_sql") or "").lower()
        assert "drop " not in sql and "delete " not in sql and "update " not in sql, sql
