"""Live PostgreSQL value-profile freshness benchmark."""

from __future__ import annotations

import pytest

from daita.evals import EvalSuite, EvalSuiteConfig
from daita.evals.reporters import render_pretty

from .from_db_postgres_live_helpers import (
    output_dir,
    postgres_rich_agent_config,
    require_live_postgres_openai,
    show_report,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
]


async def test_eval_live_from_db_postgres_profile_freshness_benchmark(tmp_path):
    require_live_postgres_openai()
    config = EvalSuiteConfig(
        name="live-from-db-postgres-profile-freshness",
        agent=postgres_rich_agent_config(cache_ttl=3600),
        defaults={"timeout_seconds": 45, "max_iterations": 14},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "pg-stale-needed-profile-full-refresh",
                "prompt": "Show support_tickets where severity = 'high'",
                "expectations": _profile_refresh_expectations(
                    table="support_tickets",
                    literal="high",
                    max_profile=1,
                    max_register=1,
                    rows=3,
                ),
            },
            {
                "id": "pg-fresh-profile-reused",
                "prompt": "Show support_tickets where severity = 'high'",
                "expectations": _profile_refresh_expectations(
                    table="support_tickets",
                    literal="high",
                    max_profile=0,
                    max_register=0,
                    rows=3,
                ),
            },
            {
                "id": "pg-stale-unrelated-profile-ignored",
                "prompt": "Show customers where lifecycle_stage = 'active'",
                "expectations": _profile_refresh_expectations(
                    table="customers",
                    literal="active",
                    max_profile=1,
                    max_register=1,
                    rows=3,
                ),
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=output_dir(tmp_path, config.name))
    show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 3


async def test_eval_live_from_db_postgres_cache_ttl_zero_operation_local_freshness(
    tmp_path,
):
    require_live_postgres_openai()
    config = EvalSuiteConfig(
        name="live-from-db-postgres-profile-freshness-ttl-zero",
        agent=postgres_rich_agent_config(cache_ttl=0),
        defaults={"timeout_seconds": 45, "max_iterations": 14},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "pg-cache-ttl-zero-operation-local-freshness",
                "prompt": "Show orders where status = 'complete'",
                "expectations": _profile_refresh_expectations(
                    table="orders",
                    literal="complete",
                    max_profile=1,
                    max_register=1,
                    rows=4,
                ),
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=output_dir(tmp_path, config.name))
    show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 1


def _profile_refresh_expectations(
    *,
    table: str,
    literal: str,
    max_profile: int,
    max_register: int,
    rows: int,
) -> dict:
    return {
        "capabilities": {
            "required": [
                "catalog.column_values.search",
                "db.sql.validate",
                "db.sql.execute_read",
            ],
            "required_owners": ["catalog", "postgresql"],
            "max_per_capability": {
                "db.column_values.profile": max_profile,
                "catalog.column_values.register": max_register,
                "db.planning.context.build": 2,
            },
            "max_calls": 14,
        },
        "tasks": {"max_errors": 0},
        "evidence": {
            "required_kinds": ["sql.validation", "query.result"],
            "max_per_kind": {"column_values.profile": max_profile},
        },
        "sql": {
            "read_only": True,
            "required_tables": [table],
            "must_include": ["WHERE", literal],
            "must_not_include": ["DELETE", "DROP"],
            "max_rows_returned": rows,
        },
        "result": {"min_rows": rows, "max_rows": rows},
        "budgets": {"max_latency_ms": 18000, "max_iterations": 14},
    }
