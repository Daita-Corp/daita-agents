"""Live PostgreSQL wide and ambiguous schema benchmark."""

from __future__ import annotations

import pytest

from daita.evals import EvalSuite, EvalSuiteConfig
from daita.evals.reporters import render_pretty

from .from_db_postgres_live_helpers import (
    output_dir,
    postgres_wide_agent_config,
    require_live_postgres_openai,
    show_report,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
]


async def test_eval_live_from_db_postgres_wide_schema_benchmark(tmp_path):
    require_live_postgres_openai()
    config = EvalSuiteConfig(
        name="live-from-db-postgres-wide-schema",
        agent=postgres_wide_agent_config(cache_ttl=3600, table_count=48),
        defaults={"timeout_seconds": 60, "max_iterations": 20},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "pg-wide-obvious-table-count",
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 4."},
                    "capabilities": {
                        "required": ["db.sql.validate", "db.sql.execute_read"],
                        "required_owners": ["postgresql"],
                        "max_calls": 10,
                    },
                    "tasks": {"max_errors": 0},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "forbidden_tables": ["customer_activity_decoy_01"],
                        "must_include": ["COUNT"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 1,
                    },
                    "result": {
                        "required_rows": [{"count": 4}],
                        "min_rows": 1,
                        "max_rows": 1,
                    },
                    "budgets": {"max_latency_ms": 15000, "max_iterations": 10},
                },
            },
            {
                "id": "pg-wide-ambiguous-show-customers",
                "runs": 3,
                "prompt": "show customers",
                "expectations": {
                    "answer": {"equals": "Returned 4 rows."},
                    "capabilities": {
                        "required": ["db.sql.validate", "db.sql.execute_read"],
                        "required_owners": ["postgresql"],
                        "max_calls": 10,
                    },
                    "tasks": {"max_errors": 0},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "forbidden_tables": ["customer_activity_decoy_01"],
                        "must_include": ["SELECT"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 4,
                    },
                    "result": {"min_rows": 4, "max_rows": 4},
                    "budgets": {"max_latency_ms": 15000, "max_iterations": 10},
                    "stability": {
                        "require_same_capabilities": True,
                        "max_answer_variants": 1,
                        "max_latency_p95_ms": 15000,
                    },
                },
            },
            {
                "id": "pg-wide-multiple-join-paths",
                "prompt": (
                    "Using catalog relationships, join orders to refunds and return "
                    "the refunded order ids and refund amounts."
                ),
                "expectations": {
                    "answer": {"equals": "Returned 2 rows."},
                    "capabilities": {
                        "required": [
                            "catalog.relationship_paths.find",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "required_owners": ["catalog", "postgresql"],
                        "max_calls": 18,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {"required_kinds": ["schema.relationship_path"]},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["orders", "refunds"],
                        "forbidden_tables": ["customer_activity_decoy_01"],
                        "must_include": ["JOIN"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 2,
                    },
                    "result": {"min_rows": 2, "max_rows": 2},
                    "budgets": {"max_latency_ms": 24000, "max_iterations": 18},
                },
            },
            {
                "id": "pg-wide-literal-column-name-in-many-tables",
                "prompt": (
                    "Show support_tickets where status = 'open' and severity = 'high'"
                ),
                "expectations": {
                    "answer": {"equals": "Returned 2 rows."},
                    "capabilities": {
                        "required": [
                            "catalog.column_values.search",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "required_owners": ["catalog", "postgresql"],
                        "max_calls": 18,
                    },
                    "tasks": {"max_errors": 0},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["support_tickets"],
                        "forbidden_tables": ["customer_activity_decoy_01"],
                        "must_include": ["WHERE", "status", "open", "severity", "high"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 2,
                    },
                    "result": {
                        "required_rows": [
                            {"customer_id": 1, "severity": "high", "status": "open"},
                            {"customer_id": 3, "severity": "high", "status": "open"},
                        ],
                        "min_rows": 2,
                        "max_rows": 2,
                    },
                    "budgets": {"max_latency_ms": 24000, "max_iterations": 18},
                },
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=output_dir(tmp_path, config.name))
    show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 4
