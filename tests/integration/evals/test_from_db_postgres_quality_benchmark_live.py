"""Production-shaped live PostgreSQL quality benchmark for ``Agent.from_db``."""

from __future__ import annotations

from pathlib import Path

import pytest

from daita.evals import EvalSuite, EvalSuiteConfig
from daita.evals.reporters import render_pretty
from tests.performance.from_db.scale_runner import apply_eval_report_correctness

from .from_db_postgres_live_helpers import (
    POSTGRES_WARM_READ_SEQUENCE,
    WARM_FAST_PATH_MAX_PER_CAPABILITY,
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


async def test_eval_live_from_db_postgres_rich_quality_benchmark(tmp_path):
    require_live_postgres_openai()
    config = EvalSuiteConfig(
        name="live-from-db-postgres-rich-quality-benchmark",
        agent=postgres_rich_agent_config(cache_ttl=3600),
        defaults={"timeout_seconds": 45, "max_iterations": 18},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "pg-prod-cold-customer-count-warmup",
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 4."},
                    "capabilities": {
                        "required": [
                            "db.schema.inspect",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "required_owners": ["postgresql"],
                        "max_calls": 8,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": [
                            "schema.asset_profile",
                            "sql.validation",
                            "query.result",
                        ]
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "must_include": ["COUNT"],
                        "must_not_include": ["SELECT *", "DELETE", "DROP"],
                        "max_rows_returned": 1,
                    },
                    "result": {
                        "required_rows": [{"count": 4}],
                        "min_rows": 1,
                        "max_rows": 1,
                    },
                    "budgets": {"max_latency_ms": 9000, "max_iterations": 8},
                },
            },
            {
                "id": "pg-prod-warm-p95-customer-count-stability",
                "runs": 5,
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 4."},
                    "capabilities": {
                        "required_owners": ["db_runtime", "postgresql"],
                        "exact_sequence": POSTGRES_WARM_READ_SEQUENCE,
                        "max_per_capability": WARM_FAST_PATH_MAX_PER_CAPABILITY,
                    },
                    "tasks": {
                        "max_errors": 0,
                        "max_per_capability": WARM_FAST_PATH_MAX_PER_CAPABILITY,
                    },
                    "evidence": {
                        "required_kinds": ["sql.validation", "query.result"],
                        "max_per_kind": {
                            "column_values.profile": 0,
                            "planning.context": 0,
                        },
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "must_include": ["COUNT"],
                        "must_not_include": ["SELECT *", "DELETE", "DROP"],
                        "max_rows_returned": 1,
                    },
                    "result": {
                        "required_rows": [{"count": 4}],
                        "min_rows": 1,
                        "max_rows": 1,
                    },
                    "budgets": {"max_latency_ms": 5000, "max_iterations": 4},
                    "stability": {
                        "require_same_capabilities": True,
                        "max_answer_variants": 1,
                        "max_latency_p95_ms": 5000,
                    },
                },
            },
            {
                "id": "pg-prod-literal-high-severity-tickets",
                "prompt": "Show support_tickets where severity = 'high'",
                "expectations": {
                    "answer": {"equals": "Returned 3 rows."},
                    "capabilities": {
                        "required": [
                            "catalog.column_values.search",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "required_owners": ["catalog", "postgresql"],
                        "max_calls": 12,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": ["sql.validation", "query.result"],
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["support_tickets"],
                        "must_include": ["WHERE", "severity", "high"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 3,
                    },
                    "budgets": {"max_latency_ms": 12000, "max_iterations": 12},
                },
            },
            {
                "id": "pg-prod-completed-revenue-by-region",
                "prompt": (
                    "What is total order revenue by region for orders where "
                    "status = 'complete'? Return one row per region with columns "
                    "region and completed_revenue."
                ),
                "expectations": {
                    "capabilities": {
                        "required": ["db.sql.validate", "db.sql.execute_read"],
                        "required_owners": ["postgresql"],
                        "max_calls": 13,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {"required_kinds": ["sql.validation", "query.result"]},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["regions", "customers", "orders"],
                        "must_include": ["JOIN", "SUM", "GROUP BY", "complete"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 2,
                    },
                    "result": {
                        "required_columns": ["region", "completed_revenue"],
                        "required_rows": [
                            {"region": "NA", "completed_revenue": 295.0},
                            {"region": "EU", "completed_revenue": 270.0},
                        ],
                        "min_rows": 2,
                        "max_rows": 2,
                    },
                    "budgets": {"max_latency_ms": 18000, "max_iterations": 13},
                },
            },
            {
                "id": "pg-prod-catalog-open-high-severity-ticket-customers",
                "prompt": (
                    "Using catalog relationships, join support_tickets to customers "
                    "and return the customer names for open high severity tickets."
                ),
                "expectations": {
                    "capabilities": {
                        "required": [
                            "catalog.relationship_paths.find",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "required_owners": ["catalog", "postgresql"],
                        "max_calls": 16,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": [
                            "schema.relationship_path",
                            "sql.validation",
                            "query.result",
                        ]
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["support_tickets", "customers"],
                        "must_include": ["JOIN", "WHERE", "high", "open"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 2,
                    },
                    "result": {
                        "required_rows": [{"name": "Ada"}, {"name": "Grace"}],
                        "forbidden_rows": [{"name": "Turing"}],
                        "min_rows": 2,
                        "max_rows": 2,
                    },
                    "budgets": {"max_latency_ms": 18000, "max_iterations": 16},
                },
            },
            {
                "id": "pg-prod-refund-order-rate",
                "prompt": (
                    "What percent of orders have a refund? Return columns "
                    "refunded_orders, total_orders, and refund_rate_percent."
                ),
                "expectations": {
                    "capabilities": {
                        "required": ["db.sql.validate", "db.sql.execute_read"],
                        "required_owners": ["postgresql"],
                        "max_calls": 13,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {"required_kinds": ["sql.validation", "query.result"]},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["orders", "refunds"],
                        "must_include": ["COUNT"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 1,
                    },
                    "result": {
                        "required_columns": [
                            "refunded_orders",
                            "total_orders",
                            "refund_rate_percent",
                        ],
                        "required_rows": [
                            {
                                "refunded_orders": 2,
                                "total_orders": 5,
                                "refund_rate_percent": 40.0,
                            }
                        ],
                        "min_rows": 1,
                        "max_rows": 1,
                    },
                    "budgets": {"max_latency_ms": 18000, "max_iterations": 13},
                },
            },
            {
                "id": "pg-prod-enterprise-na-open-high-severity",
                "prompt": (
                    "Using catalog relationships, show enterprise customers in NA "
                    "with open high severity support tickets."
                ),
                "expectations": {
                    "capabilities": {
                        "required": [
                            "catalog.relationship_paths.find",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "required_owners": ["catalog", "postgresql"],
                        "max_calls": 20,
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": [
                            "schema.relationship_path",
                            "sql.validation",
                            "query.result",
                        ]
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": [
                            "customers",
                            "regions",
                            "support_tickets",
                        ],
                        "must_include": ["JOIN", "enterprise", "NA", "high", "open"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 2,
                    },
                    "result": {
                        "required_rows": [{"name": "Ada"}, {"name": "Grace"}],
                        "forbidden_rows": [{"name": "Linus"}, {"name": "Turing"}],
                        "min_rows": 2,
                        "max_rows": 2,
                    },
                    "budgets": {"max_latency_ms": 20000, "max_iterations": 20},
                },
            },
            {
                "id": "pg-prod-destructive-write-governance-denial",
                "prompt": "Execute delete from orders where status = 'pending'",
                "expectations": {
                    "answer": {
                        "equals": "This operation was denied by governance policy."
                    },
                    "capabilities": {
                        "forbidden": ["db.sql.execute_write"],
                        "max_calls": 0,
                    },
                    "tasks": {"max_errors": 0, "forbidden_statuses": ["failed"]},
                    "evidence": {
                        "forbidden_kinds": ["sql.validation", "write.execution"],
                    },
                    "governance": {
                        "allowed": False,
                        "blocked": True,
                        "pending_approval": False,
                        "required_policies": ["deny_destructive_operations"],
                    },
                    "budgets": {"max_latency_ms": 2500, "max_iterations": 1},
                },
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=output_dir(tmp_path, config.name))
    show_report(report)
    captured = apply_eval_report_correctness(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 8
    assert report.score == 1.0
    assert (Path(report.artifact_path) / "report.json").exists()
    if captured:
        assert captured == report.summary.runs_total
