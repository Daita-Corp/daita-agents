"""Shared helpers for live PostgreSQL ``from_db`` eval suites."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from daita.evals.reporters import render_pretty

load_dotenv(Path.cwd() / ".env")

POSTGRES_WARM_READ_SEQUENCE = [
    "db.sql.validate",
    "db.sql.execute_read",
]

WARM_FAST_PATH_MAX_PER_CAPABILITY = {
    "db.schema.inspect": 0,
    "catalog.source.register": 0,
    "catalog.column_values.search": 0,
    "db.column_values.profile": 0,
    "db.planning.context.build": 0,
    "db.query.plan": 0,
    "db.query.plan.validate": 0,
    "db.answer.synthesize": 0,
}


def require_live_postgres_openai() -> None:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db benchmarks")
    if os.environ.get("DAITA_EVAL_POSTGRES") != "1":
        pytest.skip("Set DAITA_EVAL_POSTGRES=1 to run Docker Postgres benchmarks")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def output_dir(tmp_path: Path, suite_name: str) -> Path:
    return Path(os.environ.get("DAITA_EVAL_OUTPUT_DIR", tmp_path)) / suite_name


def show_report(report) -> None:
    print()
    print(render_pretty(report))
    print()


def postgres_rich_agent_config(*, cache_ttl: int | None = 3600) -> dict:
    return {
        "factory": (
            "tests.integration.evals.eval_from_db_factories:"
            "create_postgres_rich_from_db_benchmark_agent"
        ),
        "kwargs": {"cache_ttl": cache_ttl},
    }


def postgres_wide_agent_config(
    *, cache_ttl: int | None = 3600, table_count: int = 48
) -> dict:
    return {
        "factory": (
            "tests.integration.evals.eval_from_db_factories:"
            "create_postgres_wide_from_db_benchmark_agent"
        ),
        "kwargs": {"cache_ttl": cache_ttl, "table_count": table_count},
    }
