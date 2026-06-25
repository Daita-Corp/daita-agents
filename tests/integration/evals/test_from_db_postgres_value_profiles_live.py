"""Live PostgreSQL predicate-value grounding benchmark."""

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


VALUE_PROFILE_CASES = [
    (
        "pg-literal-orders-complete",
        "Show orders where status = 'complete'",
        ("orders", "status"),
        {
            "required_tables": ["orders"],
            "must_include": ["WHERE", "status", "complete"],
        },
        {"required_rows": [{"status": "complete"}], "min_rows": 4, "max_rows": 4},
    ),
    (
        "pg-literal-support-tickets-high",
        "Show support_tickets where severity = 'high'",
        ("support_tickets", "severity"),
        {
            "required_tables": ["support_tickets"],
            "must_include": ["WHERE", "severity", "high"],
        },
        {"required_rows": [{"severity": "high"}], "min_rows": 3, "max_rows": 3},
    ),
    (
        "pg-literal-customers-enterprise",
        "Show customers where segment = 'enterprise'",
        ("customers", "segment"),
        {
            "required_tables": ["customers"],
            "must_include": ["WHERE", "segment", "enterprise"],
        },
        {"required_rows": [{"segment": "enterprise"}], "min_rows": 3, "max_rows": 3},
    ),
    (
        "pg-literal-region-na",
        "Show regions where name = 'NA'",
        ("regions", "name"),
        {"required_tables": ["regions"], "must_include": ["WHERE", "name", "NA"]},
        {"required_rows": [{"name": "NA"}], "min_rows": 1, "max_rows": 1},
    ),
]


async def test_eval_live_from_db_postgres_value_profile_benchmark(tmp_path):
    require_live_postgres_openai()
    config = EvalSuiteConfig(
        name="live-from-db-postgres-value-profiles",
        agent=postgres_rich_agent_config(cache_ttl=3600),
        defaults={"timeout_seconds": 45, "max_iterations": 14},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": case_id,
                "prompt": prompt,
                "expectations": {
                    "capabilities": {
                        "required": [
                            "catalog.column_values.search",
                            "db.planning.context.build",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "required_owners": ["catalog", "postgresql", "db_runtime"],
                        "max_per_capability": {
                            "db.column_values.profile": 1,
                            "catalog.column_values.register": 1,
                            "db.planning.context.build": 2,
                        },
                        "max_calls": 14,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": [
                            "planning.context",
                            "sql.validation",
                            "query.result",
                        ],
                        "max_per_kind": {"column_values.profile": 1},
                    },
                    "sql": {
                        "read_only": True,
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": result["max_rows"],
                        **sql_expectations,
                    },
                    "result": result,
                    "budgets": {"max_latency_ms": 18000, "max_iterations": 14},
                },
            }
            for case_id, prompt, _, sql_expectations, result in VALUE_PROFILE_CASES
        ],
    )

    report = await EvalSuite(config).run(output_dir=output_dir(tmp_path, config.name))
    show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == len(VALUE_PROFILE_CASES)
    for case_result, (_, _, expected_profile, _, _) in zip(
        report.cases, VALUE_PROFILE_CASES, strict=True
    ):
        _assert_only_expected_profiled(case_result, expected_profile)


def _assert_only_expected_profiled(case_result, expected: tuple[str, str]) -> None:
    expected_table, expected_column = expected
    for run in case_result.runs:
        profile_tasks = [
            task
            for task in run.tasks
            if task.capability_id == "db.column_values.profile"
        ]
        register_tasks = [
            task
            for task in run.tasks
            if task.capability_id == "catalog.column_values.register"
        ]
        assert len(profile_tasks) <= 1
        assert len(register_tasks) == len(profile_tasks)
        if not profile_tasks:
            continue
        task_input = profile_tasks[0].input
        assert str(task_input.get("table") or "").endswith(expected_table)
        assert task_input.get("column") == expected_column
