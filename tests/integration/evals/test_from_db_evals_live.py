"""Live eval suites for ``Agent.from_db`` quality reporting.

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/evals/test_from_db_evals_live.py \
        -m "requires_llm and integration" -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from daita.evals import EvalSuite, EvalSuiteConfig
from daita.evals.reporters import render_pretty

load_dotenv(Path.cwd() / ".env")

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


def _require_live_openai() -> None:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db eval tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def _show_report(report) -> None:
    print()
    print(render_pretty(report))
    print()


def _output_dir(tmp_path: Path, suite_name: str) -> Path:
    return Path(os.environ.get("DAITA_EVAL_OUTPUT_DIR", tmp_path)) / suite_name


async def test_eval_live_from_db_sqlite_quality_suite(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-from-db-sqlite-evals",
        agent={
            "factory": (
                "tests.integration.evals.eval_from_db_factories:"
                "create_sqlite_from_db_eval_agent"
            ),
            "kwargs": {
                "db_path": str(tmp_path / "from-db-eval.sqlite"),
                "cache_ttl": 3600,
            },
        },
        defaults={"timeout_seconds": 30, "max_iterations": 10},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "sqlite-cold-count-customers",
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 3."},
                    "capabilities": {
                        "required": [
                            "db.schema.inspect",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "max_calls": 8,
                        "required_owners": ["sqlite"],
                        "forbidden_owners": ["data_quality", "lineage", "memory"],
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": ["schema.asset_profile", "query.result"]
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "must_include": ["COUNT"],
                        "must_not_include": ["SELECT *", "DELETE", "DROP"],
                        "max_rows_returned": 1,
                    },
                    "budgets": {"max_latency_ms": 8000, "max_iterations": 8},
                },
            },
            {
                "id": "sqlite-warm-relationship-join",
                "prompt": "Join orders to customers using their relationship",
                "expectations": {
                    "answer": {"equals": "Returned 4 rows."},
                    "capabilities": {
                        "required": [
                            "catalog.schema.search",
                            "catalog.relationship_paths.find",
                            "db.sql.validate",
                            "db.sql.execute_read",
                        ],
                        "max_calls": 10,
                        "required_owners": ["catalog", "sqlite"],
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": [
                            "schema.search_result",
                            "schema.relationship_path",
                            "sql.validation",
                            "query.result",
                        ]
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers", "orders"],
                        "must_include": ["JOIN"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 4,
                    },
                    "budgets": {"max_latency_ms": 12000, "max_iterations": 10},
                },
            },
            {
                "id": "sqlite-warm-ambiguous-prompt",
                "runs": 2,
                "prompt": "show me customers",
                "expectations": {
                    "answer": {"equals": "Returned 3 rows."},
                    "capabilities": {
                        "required": ["db.sql.execute_read"],
                        "max_calls": 7,
                        "required_owners": ["sqlite"],
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {"required_kinds": ["query.result"]},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "must_include": ["SELECT"],
                        "must_not_include": ["DELETE", "DROP"],
                        "max_rows_returned": 3,
                    },
                    "budgets": {"max_latency_ms": 10000, "max_iterations": 7},
                    "stability": {
                        "require_same_capabilities": True,
                        "max_answer_variants": 1,
                        "max_latency_delta_pct": 500,
                    },
                },
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=_output_dir(tmp_path, config.name))
    _show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.score == 1.0
    assert (Path(report.artifact_path) / "report.json").exists()
    assert "db.schema.inspect" in render_pretty(report)


async def test_eval_live_from_db_sqlite_data_team_capabilities(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-from-db-data-team-evals",
        agent={
            "factory": (
                "tests.integration.evals.eval_from_db_factories:"
                "create_sqlite_data_team_from_db_eval_agent"
            ),
            "kwargs": {
                "db_path": str(tmp_path / "from-db-data-team.sqlite"),
                "memory_dir": str(tmp_path / "memory"),
            },
        },
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "quality-profile",
                "prompt": "Profile the customers table",
                "expectations": {
                    "answer": {"contains": ["verified evidence"]},
                    "capabilities": {
                        "required": ["quality.profile"],
                        "max_calls": 2,
                        "required_owners": ["sqlite", "data_quality"],
                    },
                    "tasks": {"max_errors": 0},
                    "budgets": {"max_latency_ms": 1000, "max_iterations": 2},
                },
            },
            {
                "id": "lineage-trace",
                "prompt": "Trace lineage for orders",
                "expectations": {
                    "answer": {"contains": ["verified evidence"]},
                    "capabilities": {
                        "required": ["lineage.trace"],
                        "max_calls": 2,
                        "required_owners": ["sqlite", "lineage"],
                    },
                    "tasks": {"max_errors": 0},
                    "budgets": {"max_latency_ms": 1000, "max_iterations": 2},
                },
            },
            {
                "id": "memory-update",
                "prompt": "Remember that revenue excludes tax",
                "expectations": {
                    "answer": {"contains": ["verified evidence"]},
                    "capabilities": {
                        "required": ["memory.semantic.write"],
                        "max_calls": 2,
                        "required_owners": ["sqlite", "memory"],
                    },
                    "tasks": {"max_errors": 0},
                    "budgets": {"max_latency_ms": 1000, "max_iterations": 2},
                },
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=_output_dir(tmp_path, config.name))
    _show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 3
    assert "owners: sqlite" in render_pretty(report)


@pytest.mark.requires_db
async def test_eval_live_from_db_postgres_quality_suite(tmp_path):
    _require_live_openai()
    if os.environ.get("DAITA_EVAL_POSTGRES") != "1":
        pytest.skip("Set DAITA_EVAL_POSTGRES=1 to run Docker Postgres from_db evals")

    config = EvalSuiteConfig(
        name="live-from-db-postgres-evals",
        agent={
            "factory": (
                "tests.integration.evals.eval_from_db_factories:"
                "create_postgres_from_db_eval_agent"
            ),
            "kwargs": {"cache_ttl": 3600},
        },
        defaults={"timeout_seconds": 30, "max_iterations": 10},
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "postgres-cold-count-customers",
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 3."},
                    "capabilities": {
                        "required": ["db.schema.inspect", "db.sql.execute_read"],
                        "max_calls": 8,
                        "required_owners": ["postgresql"],
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {"required_kinds": ["query.result"]},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "must_include": ["COUNT"],
                        "max_rows_returned": 1,
                    },
                    "budgets": {"max_latency_ms": 5000, "max_iterations": 8},
                },
            },
            {
                "id": "postgres-warm-count-customers",
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 3."},
                    "capabilities": {
                        "required": ["db.sql.execute_read"],
                        "max_calls": 7,
                        "required_owners": ["postgresql"],
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {"required_kinds": ["query.result"]},
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "must_include": ["COUNT"],
                        "max_rows_returned": 1,
                    },
                    "budgets": {"max_latency_ms": 4000, "max_iterations": 7},
                },
            },
            {
                "id": "postgres-warm-relationship-join",
                "prompt": "Join orders to customers using their relationship",
                "expectations": {
                    "answer": {"equals": "Returned 4 rows."},
                    "capabilities": {
                        "required": [
                            "catalog.relationship_paths.find",
                            "db.sql.execute_read",
                        ],
                        "max_calls": 10,
                        "required_owners": ["catalog", "postgresql"],
                    },
                    "tasks": {"max_errors": 0},
                    "evidence": {
                        "required_kinds": ["schema.relationship_path", "query.result"]
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers", "orders"],
                        "must_include": ["JOIN"],
                        "max_rows_returned": 4,
                    },
                    "budgets": {"max_latency_ms": 12000, "max_iterations": 10},
                },
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=_output_dir(tmp_path, config.name))
    _show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 3


async def test_eval_live_from_db_sqlite_openai_judge(tmp_path):
    _require_live_openai()
    if os.environ.get("DAITA_EVAL_OPENAI_JUDGE") != "1":
        pytest.skip("Set DAITA_EVAL_OPENAI_JUDGE=1 to run OpenAI judge scoring")

    judge_model = os.environ.get("OPENAI_JUDGE_TEST_MODEL") or os.environ.get(
        "OPENAI_TEST_MODEL", "gpt-5.4-mini"
    )
    config = EvalSuiteConfig(
        name="live-from-db-openai-judge-evals",
        agent={
            "factory": (
                "tests.integration.evals.eval_from_db_factories:"
                "create_sqlite_from_db_eval_agent"
            ),
            "kwargs": {"db_path": str(tmp_path / "from-db-judge.sqlite")},
        },
        artifacts={"include_evidence_payloads": True},
        cases=[
            {
                "id": "judge-count-customers",
                "prompt": "How many customers are there?",
                "expectations": {
                    "answer": {"equals": "The count is 3."},
                    "capabilities": {
                        "required": ["db.sql.execute_read"],
                        "max_calls": 3,
                    },
                    "sql": {
                        "read_only": True,
                        "required_tables": ["customers"],
                        "must_include": ["COUNT"],
                        "max_rows_returned": 1,
                    },
                    "judge": {
                        "provider": "openai",
                        "model": judge_model,
                        "api_key": os.environ["OPENAI_API_KEY"],
                        "pass_score": 0.8,
                        "require_all_criteria_pass": True,
                        "include_evidence_payloads": True,
                        "criteria": [
                            {
                                "id": "answer_correctness",
                                "description": (
                                    "The final answer correctly states that the "
                                    "customers table contains exactly 3 customers."
                                ),
                                "required": True,
                                "weight": 0.4,
                            },
                            {
                                "id": "runtime_grounding",
                                "description": (
                                    "The runtime evidence includes read-only SQL "
                                    "against the customers table and a query result "
                                    "supporting the answer."
                                ),
                                "required": True,
                                "weight": 0.4,
                            },
                            {
                                "id": "no_unsupported_claims",
                                "description": (
                                    "The answer does not add facts that are not "
                                    "supported by the DB evidence."
                                ),
                                "required": True,
                                "weight": 0.2,
                            },
                        ],
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=_output_dir(tmp_path, config.name))
    _show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.judge_calls == 1
    assert report.cases[0].runs[0].judges[0].score >= 0.8
