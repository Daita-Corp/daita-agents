"""Live-gated integration tests for ``Agent.from_db`` over PostgreSQL.

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/from_db/test_from_db_live_postgres.py \
        -m "requires_llm and requires_db and integration" -v -s

Requires docker and ``asyncpg``. The module starts a throwaway Postgres
container, seeds a small relational schema, and exercises ``DbRuntime`` through
``Agent.from_db``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

asyncpg = pytest.importorskip(
    "asyncpg",
    reason="asyncpg required: pip install 'daita-agents[postgresql]'",
)

from daita.agents.agent import Agent
from daita.db import DbIntentKind
from daita.runtime import OperationStatus

from tests.integration._harness import start_container

load_dotenv(Path.cwd() / ".env")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
]

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_from_db_test"

SEED_SQL = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    total NUMERIC(10, 2) NOT NULL,
    status TEXT NOT NULL
);

INSERT INTO customers (customer_id, name, region) VALUES
    (1, 'Ada', 'NA'),
    (2, 'Linus', 'EU'),
    (3, 'Grace', 'NA');

INSERT INTO orders (order_id, customer_id, total, status) VALUES
    (1, 1, 120.00, 'complete'),
    (2, 1, 80.00, 'pending'),
    (3, 2, 50.00, 'complete'),
    (4, 3, 175.00, 'complete');
"""


@pytest.fixture(scope="module")
def live_openai_kwargs() -> dict[str, Any]:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db Postgres tests")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return {
        "llm_provider": "openai",
        "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "api_key": api_key,
        "temperature": 0,
    }


@pytest.fixture(scope="module")
def postgres_container(live_openai_kwargs):
    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix="daita-from-db-pg",
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def seeded_postgres_url(postgres_container) -> str:
    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{postgres_container.host}:{postgres_container.host_port}/{POSTGRES_DB}"
    )
    asyncio.run(_seed_postgres(url))
    return url


async def test_from_db_live_postgres_query_uses_runtime_tasks_and_evidence(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbPostgresQuery",
        cache_ttl=0,
        **live_openai_kwargs,
    )

    try:
        result = await agent.run_detailed("How many customers are there?")
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.DATA_QUERY
    assert result.answer == "The count is 3."
    assert {
        "schema.asset_profile",
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
    } <= _evidence_kinds(result)
    assert {"db.schema.inspect", "db.sql.validate", "db.sql.execute_read"} <= set(
        _task_capabilities(result)
    )
    assert "postgresql" in inspection.plugin_ids
    assert "postgresql:db.sql.execute_read" in inspection.capability_ids
    assert result.diagnostics["verification"]["passed"] is True


async def test_from_db_live_postgres_catalog_assisted_join_records_relationship_path(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbPostgresJoin",
        cache_ttl=0,
        **live_openai_kwargs,
    )

    try:
        result = await agent.run_detailed(
            "Join orders to customers using their relationship and return records"
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY
    assert result.answer == "Returned 4 rows."
    assert {
        "catalog.source_registered",
        "schema.search_result",
        "schema.relationship_path",
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
    } <= _evidence_kinds(result)
    assert "catalog.relationship_paths.find" in _task_capabilities(result)
    assert result.diagnostics["verification"]["passed"] is True


async def test_from_db_live_postgres_grounds_completed_orders_to_observed_status(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbPostgresValueGrounding",
        cache_ttl=0,
        **live_openai_kwargs,
    )

    try:
        result = await agent.run_detailed("Show completed orders by status")
    finally:
        await agent.stop()

    query_result = next(item for item in result.evidence if item.kind == "query.result")
    planning_context = next(
        item for item in result.evidence if item.kind == "planning.context"
    )
    statuses = {row.get("status") for row in query_result.payload.get("rows", [])}

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.DATA_QUERY
    assert statuses == {"complete"}
    assert len(query_result.payload["rows"]) == 3
    assert "orders.status: complete" in planning_context.payload["rendered_context"]
    assert result.diagnostics["verification"]["passed"] is True


async def test_from_db_live_postgres_resolves_non_descriptive_prompt_without_looping(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbPostgresAmbiguous",
        cache_ttl=0,
        **live_openai_kwargs,
    )

    try:
        resolved = await agent.run_detailed("show me customers")
        bounded_fallback = await agent.run_detailed("customers")
    finally:
        await agent.stop()

    assert resolved.status is OperationStatus.SUCCEEDED
    assert resolved.intent.kind is DbIntentKind.DATA_QUERY
    assert resolved.answer == "Returned 3 rows."
    assert {
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
    } <= _evidence_kinds(resolved)
    assert resolved.diagnostics["execution"]["task_count"] == 3

    assert bounded_fallback.status is OperationStatus.SUCCEEDED
    assert bounded_fallback.intent.kind is DbIntentKind.CONVERSATIONAL
    assert _evidence_kinds(bounded_fallback) == {"schema.asset_profile"}
    assert bounded_fallback.diagnostics["execution"]["task_count"] == 1


async def _seed_postgres(url: str) -> None:
    deadline = time.time() + 30
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            connection = await asyncpg.connect(url, ssl=False)
            await connection.execute(SEED_SQL)
            await connection.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Could not seed Postgres test database: {last_error}")


def _evidence_kinds(result) -> set[str]:
    return {item.kind for item in result.evidence}


def _task_capabilities(result) -> list[str]:
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]
