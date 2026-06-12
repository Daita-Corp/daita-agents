"""Live PostgreSQL catalog value profile scale benchmarks."""

from __future__ import annotations

import pytest

from .postgres_scale_fixtures import (
    create_postgres_scale_agent,
    postgres_version,
    profile_catalog_values,
    rich_schema_sql,
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


async def test_postgres_fresh_value_profiles_reused_under_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    operations = scale_int_env("DAITA_PERF_PROFILE_OPERATIONS", 100)
    harness = await create_postgres_scale_agent(
        sql=rich_schema_sql(),
        name="PerfCatalogFreshProfilesPostgres",
        tag_prefix="daita-perf-catalog-fresh-pg",
        cache_ttl=3600,
        llm=True,
    )
    try:
        await harness.agent.run_detailed(
            "How many support tickets are open and high severity?"
        )
        await profile_catalog_values(
            harness.agent,
            [("support_tickets", "status"), ("support_tickets", "severity")],
        )
        version = await postgres_version(harness.url)
        artifact = await run_scale_benchmark(
            suite="postgres-catalog-profile-scale",
            parameters=ScaleBenchmarkParameters(
                concurrency=5,
                operations=operations,
                scenario="fresh-profile-reuse",
            ),
            operation_factory=lambda _index: harness.agent.run_detailed(
                "How many support tickets are open and high severity?"
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-catalog-profile-scale"),
            environment={
                "postgres_version": version,
                "model": "openai",
                "dataset": "small_rich_schema",
            },
        )
        assert_latency_gates(artifact, success_rate=0.99, p95_ms=5000)
        assert _max_capability_count(artifact, "db.column_values.profile") == 0
    finally:
        await harness.stop()


async def test_postgres_stale_value_profiles_refresh_under_load(tmp_path):
    postgres_live_required()
    live_llm_required()
    operations = scale_int_env("DAITA_PERF_PROFILE_OPERATIONS", 100)
    harness = await create_postgres_scale_agent(
        sql=rich_schema_sql(),
        name="PerfCatalogStaleProfilesPostgres",
        tag_prefix="daita-perf-catalog-stale-pg",
        cache_ttl=0,
        llm=True,
    )
    try:
        await harness.agent.run_detailed(
            "How many support tickets are open and high severity?"
        )
        await profile_catalog_values(
            harness.agent,
            [("support_tickets", "status"), ("support_tickets", "severity")],
        )
        version = await postgres_version(harness.url)
        artifact = await run_scale_benchmark(
            suite="postgres-catalog-profile-scale",
            parameters=ScaleBenchmarkParameters(
                concurrency=5,
                operations=operations,
                scenario="stale-profile-refresh",
            ),
            operation_factory=lambda _index: harness.agent.run_detailed(
                "How many support tickets are open and high severity?"
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-catalog-profile-scale"),
            environment={
                "postgres_version": version,
                "model": "openai",
                "dataset": "small_rich_schema",
            },
        )
        assert_latency_gates(artifact, success_rate=0.99, p95_ms=12000)
        assert _max_capability_count(artifact, "db.column_values.profile") <= 2
        _assert_no_unrelated_profile_scans(artifact)
    finally:
        await harness.stop()


async def test_postgres_missing_value_profiles_are_bounded(tmp_path):
    postgres_live_required()
    live_llm_required()
    operations = scale_int_env("DAITA_PERF_PROFILE_OPERATIONS", 100)
    harness = await create_postgres_scale_agent(
        sql=rich_schema_sql(),
        name="PerfCatalogMissingProfilesPostgres",
        tag_prefix="daita-perf-catalog-missing-pg",
        cache_ttl=3600,
        llm=True,
    )
    try:
        await harness.agent.run_detailed("How many support tickets are there?")
        version = await postgres_version(harness.url)
        artifact = await run_scale_benchmark(
            suite="postgres-catalog-profile-scale",
            parameters=ScaleBenchmarkParameters(
                concurrency=5,
                operations=operations,
                scenario="missing-profile-bounded",
            ),
            operation_factory=lambda _index: harness.agent.run_detailed(
                "How many support tickets have status='open' and severity='high'?"
            ),
            output_dir=artifact_output_dir(tmp_path, "postgres-catalog-profile-scale"),
            environment={
                "postgres_version": version,
                "model": "openai",
                "dataset": "small_rich_schema",
            },
        )
        assert_latency_gates(artifact, success_rate=0.99, p95_ms=12000)
        assert _max_capability_count(artifact, "db.column_values.profile") <= 2
    finally:
        await harness.stop()


def _max_capability_count(artifact: dict, capability_id: str) -> int:
    return max(
        (
            operation["capability_sequence"].count(capability_id)
            for operation in artifact["operations"]
        ),
        default=0,
    )


def _assert_no_unrelated_profile_scans(artifact: dict) -> None:
    for operation in artifact["operations"]:
        sql = str(operation["metadata"].get("planned_sql") or "").lower()
        if "support_tickets" in sql:
            assert "orders" not in sql and "customers" not in sql, sql
