"""Live PostgreSQL warm-runtime latency distribution benchmark."""

from __future__ import annotations

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


async def test_eval_live_from_db_postgres_warm_performance_benchmark(tmp_path):
    require_live_postgres_openai()
    config = EvalSuiteConfig(
        name="live-from-db-postgres-warm-performance",
        agent=postgres_rich_agent_config(cache_ttl=3600),
        defaults={"timeout_seconds": 30, "max_iterations": 8},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "pg-warm-simple-count-cold-warmup",
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 4."},
                    "capabilities": {
                        "required": ["db.schema.inspect", "db.sql.execute_read"],
                        "required_owners": ["postgresql"],
                        "max_calls": 8,
                    },
                    "tasks": {"max_errors": 0},
                    "result": {
                        "required_rows": [{"count": 4}],
                        "min_rows": 1,
                        "max_rows": 1,
                    },
                    "budgets": {"max_latency_ms": 9000, "max_iterations": 8},
                },
            },
            {
                "id": "pg-warm-simple-count-50-runs",
                "runs": 50,
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 4."},
                    "capabilities": {
                        "exact_sequence": POSTGRES_WARM_READ_SEQUENCE,
                        "required_owners": ["postgresql"],
                        "max_per_capability": WARM_FAST_PATH_MAX_PER_CAPABILITY,
                    },
                    "tasks": {
                        "max_errors": 0,
                        "max_per_capability": WARM_FAST_PATH_MAX_PER_CAPABILITY,
                    },
                    "evidence": {
                        "required_kinds": ["sql.validation", "query.result"],
                        "max_per_kind": {
                            "schema.search_result": 0,
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
                        "max_latency_p50_ms": 2500,
                        "max_latency_p95_ms": 4000,
                        "max_latency_p99_ms": 5000,
                    },
                },
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=output_dir(tmp_path, config.name))
    show_report(report)
    captured = apply_eval_report_correctness(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 2
    assert report.summary.runs_total == 51
    if captured:
        assert captured == report.summary.runs_total
