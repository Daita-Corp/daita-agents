"""Live-gated performance benchmarks for ``Agent.from_db``.

Run:
    DAITA_BENCHMARK_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/performance/from_db/test_live_from_db_benchmarks.py \
        -m "performance and requires_llm" -v -s

Optional:
    DAITA_BENCHMARK_OUTPUT=.daita/benchmarks/live-from-db-benchmarks.jsonl

The current ``from_db`` path is deterministic, but these benchmarks pass live
LLM configuration through construction and record token/cost fields when the
runtime starts producing them.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.db import DbOperationResult
from daita.db.capabilities import (
    QUERY_PLAN_PROPOSAL_EVIDENCE,
    QUERY_PLAN_VALIDATION_EVIDENCE,
    QUERY_RESULT_EVIDENCE,
    SCHEMA_RELATIONSHIP_PATH_EVIDENCE,
    SQL_VALIDATION_EVIDENCE,
)
from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus

load_dotenv(Path.cwd() / ".env")

try:
    import asyncpg
except ImportError:  # pragma: no cover - exercised by optional dependency matrix
    asyncpg = None

from tests.integration._harness import start_container

pytestmark = [
    pytest.mark.performance,
    pytest.mark.requires_llm,
    pytest.mark.slow,
]

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_from_db_benchmark"
QUERY_PLAN_EVIDENCE = {
    QUERY_PLAN_PROPOSAL_EVIDENCE,
    QUERY_PLAN_VALIDATION_EVIDENCE,
}


@pytest.fixture(scope="session")
def benchmark_agent_kwargs() -> dict[str, Any]:
    if os.environ.get("DAITA_BENCHMARK_LIVE_LLM") != "1":
        pytest.fail(
            "Live from_db benchmarks require DAITA_BENCHMARK_LIVE_LLM=1. "
            "These benchmarks do not run against mock LLM configuration."
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail("Missing OPENAI_API_KEY for live from_db benchmarks.")
    return {
        "llm_provider": "openai",
        "model": os.environ.get("DAITA_BENCHMARK_MODEL")
        or os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "api_key": api_key,
        "temperature": 0,
    }


@pytest.fixture
async def benchmark_db_path(tmp_path) -> Path:
    db_path = tmp_path / "from-db-benchmark.sqlite"
    await _seed_benchmark_db(db_path)
    return db_path


@pytest.fixture(scope="session")
def benchmark_postgres_url(benchmark_agent_kwargs) -> str:
    if os.environ.get("DAITA_BENCHMARK_POSTGRES") != "1":
        pytest.skip("Set DAITA_BENCHMARK_POSTGRES=1 to run Postgres benchmarks")
    if asyncpg is None:
        pytest.skip("asyncpg required: pip install 'daita-agents[postgresql]'")
    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix="daita-from-db-bench-pg",
    )
    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{container.host}:{container.host_port}/{POSTGRES_DB}"
    )
    try:
        asyncio.run(_seed_postgres_benchmark_db(url))
        yield url
    finally:
        container.remove()


async def test_live_from_db_schema_lookup_latency_cost_and_output(
    benchmark_db_path,
    benchmark_agent_kwargs,
):
    agent = await Agent.from_db(
        str(benchmark_db_path),
        name="BenchFromDbSchema",
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            "What columns are in the customers table?",
            mode="schema.query",
        )
    finally:
        await agent.stop()

    _assert_result(result, elapsed_ms, expected_answer=["customers", "region"])
    _assert_evidence(result, {"schema.asset_profile", "schema.search_result"})
    _record_benchmark("from_db_schema_lookup", result, elapsed_ms, database="sqlite")


async def test_live_from_db_simple_query_latency_cost_and_output(
    benchmark_db_path,
    benchmark_agent_kwargs,
):
    agent = await Agent.from_db(
        str(benchmark_db_path),
        name="BenchFromDbQuery",
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            "How many customers are there?",
        )
    finally:
        await agent.stop()

    _assert_result(result, elapsed_ms, expected_answer=["3"])
    _assert_evidence(
        result,
        {*QUERY_PLAN_EVIDENCE, SQL_VALIDATION_EVIDENCE, QUERY_RESULT_EVIDENCE},
    )
    _record_benchmark("from_db_simple_query", result, elapsed_ms, database="sqlite")


async def test_live_from_db_relationships_latency_cost_and_output(
    benchmark_db_path,
    benchmark_agent_kwargs,
):
    agent = await Agent.from_db(
        str(benchmark_db_path),
        name="BenchFromDbRelationship",
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            "Join orders to customers using their relationship and return records",
        )
    finally:
        await agent.stop()

    _assert_result(result, elapsed_ms, expected_answer=["Returned 4 rows"])
    _assert_evidence(
        result,
        {
            SCHEMA_RELATIONSHIP_PATH_EVIDENCE,
            *QUERY_PLAN_EVIDENCE,
            SQL_VALIDATION_EVIDENCE,
            QUERY_RESULT_EVIDENCE,
        },
    )
    _record_benchmark(
        "from_db_relationships",
        result,
        elapsed_ms,
        database="sqlite",
    )


async def test_live_from_db_ambiguous_prompt_latency_cost_and_output(
    benchmark_db_path,
    benchmark_agent_kwargs,
):
    agent = await Agent.from_db(
        str(benchmark_db_path),
        name="BenchFromDbAmbiguous",
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(agent, "show me customers")
    finally:
        await agent.stop()

    _assert_result(result, elapsed_ms, expected_answer=["Returned 3 rows"])
    _assert_evidence(
        result,
        {*QUERY_PLAN_EVIDENCE, SQL_VALIDATION_EVIDENCE, QUERY_RESULT_EVIDENCE},
    )
    _record_benchmark(
        "from_db_ambiguous_prompt",
        result,
        elapsed_ms,
        database="sqlite",
    )


async def test_live_from_db_data_team_plugin_latency_cost_and_output(
    benchmark_db_path,
    benchmark_agent_kwargs,
    tmp_path,
):
    agent = await Agent.from_db(
        str(benchmark_db_path),
        name="BenchFromDbDataTeam",
        mode="data_team",
        quality=True,
        lineage=True,
        memory=_memory_plugin(tmp_path),
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        quality, quality_ms = await _timed_run(
            agent,
            "Profile the orders table",
            mode="quality.check",
        )
        lineage, lineage_ms = await _timed_run(
            agent,
            "Trace lineage for orders",
            mode="lineage.trace",
        )
        memory, memory_ms = await _timed_run(
            agent,
            "Remember that revenue excludes tax",
            mode="memory.update",
            metadata={"category": "db_semantics"},
        )
    finally:
        await agent.stop()

    _assert_result(quality, quality_ms, expected_answer=["completed"])
    _assert_result(lineage, lineage_ms, expected_answer=["completed"])
    _assert_result(memory, memory_ms, expected_answer=["completed"])
    _assert_evidence(quality, {"quality.profile"})
    _assert_evidence(lineage, {"lineage.trace"})
    _assert_evidence(memory, {"memory.semantic.write"})
    _record_benchmark("from_db_plugin_quality", quality, quality_ms, database="sqlite")
    _record_benchmark("from_db_plugin_lineage", lineage, lineage_ms, database="sqlite")
    _record_benchmark("from_db_plugin_memory", memory, memory_ms, database="sqlite")


async def test_live_from_db_postgres_simple_query_latency_cost_and_output(
    benchmark_postgres_url,
    benchmark_agent_kwargs,
):
    agent = await Agent.from_db(
        benchmark_postgres_url,
        name="BenchFromDbPostgresQuery",
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            "How many customers are there?",
        )
    finally:
        await agent.stop()

    _assert_result(result, elapsed_ms, expected_answer=["3"])
    _assert_evidence(
        result,
        {*QUERY_PLAN_EVIDENCE, SQL_VALIDATION_EVIDENCE, QUERY_RESULT_EVIDENCE},
    )
    _record_benchmark(
        "from_db_postgres_simple_query",
        result,
        elapsed_ms,
        database="postgresql",
    )


async def test_live_from_db_postgres_relationships_latency_cost_and_output(
    benchmark_postgres_url,
    benchmark_agent_kwargs,
):
    agent = await Agent.from_db(
        benchmark_postgres_url,
        name="BenchFromDbPostgresRelationship",
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            "Join orders to customers using their relationship and return records",
        )
    finally:
        await agent.stop()

    _assert_result(result, elapsed_ms, expected_answer=["Returned 4 rows"])
    _assert_evidence(
        result,
        {
            SCHEMA_RELATIONSHIP_PATH_EVIDENCE,
            *QUERY_PLAN_EVIDENCE,
            SQL_VALIDATION_EVIDENCE,
            QUERY_RESULT_EVIDENCE,
        },
    )
    _record_benchmark(
        "from_db_postgres_relationships",
        result,
        elapsed_ms,
        database="postgresql",
    )


async def test_live_from_db_postgres_ambiguous_prompt_latency_cost_and_output(
    benchmark_postgres_url,
    benchmark_agent_kwargs,
):
    agent = await Agent.from_db(
        benchmark_postgres_url,
        name="BenchFromDbPostgresAmbiguous",
        cache_ttl=0,
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(agent, "show me customers")
    finally:
        await agent.stop()

    _assert_result(result, elapsed_ms, expected_answer=["Returned 3 rows"])
    _assert_evidence(
        result,
        {*QUERY_PLAN_EVIDENCE, SQL_VALIDATION_EVIDENCE, QUERY_RESULT_EVIDENCE},
    )
    _record_benchmark(
        "from_db_postgres_ambiguous_prompt",
        result,
        elapsed_ms,
        database="postgresql",
    )


async def _seed_benchmark_db(path: Path) -> None:
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            total REAL NOT NULL,
            status TEXT NOT NULL
        );
        INSERT INTO customers (id, name, region) VALUES
            (1, 'Ada', 'NA'),
            (2, 'Linus', 'EU'),
            (3, 'Grace', 'NA');
        INSERT INTO orders (customer_id, total, status) VALUES
            (1, 120.00, 'complete'),
            (1, 80.00, 'pending'),
            (2, 50.00, 'complete'),
            (3, 175.00, 'complete');
        """
    )
    await plugin.disconnect()


async def _seed_postgres_benchmark_db(url: str) -> None:
    assert asyncpg is not None
    deadline = time.time() + 30
    last_error: Exception | None = None
    seed_sql = """
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
    while time.time() < deadline:
        try:
            connection = await asyncpg.connect(url, ssl=False)
            await connection.execute(seed_sql)
            await connection.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Could not seed Postgres benchmark database: {last_error}")


def _memory_plugin(tmp_path: Path) -> MemoryPlugin:
    embedder = MockEmbeddingProvider(dim=8)
    plugin = MemoryPlugin(workspace="from-db-benchmark-memory", embedder=embedder)
    plugin.backend = LocalMemoryBackend(
        workspace="from-db-benchmark-memory",
        agent_id="from-db-benchmark-memory",
        scope="project",
        base_dir=tmp_path,
        embedder=embedder,
    )
    plugin.environment = "local"
    return plugin


async def _timed_run(agent, prompt: str, **kwargs) -> tuple[DbOperationResult, float]:
    start = time.perf_counter()
    result = await agent.run_detailed(prompt, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def _assert_result(
    result: DbOperationResult,
    elapsed_ms: float,
    *,
    expected_answer: list[str],
) -> None:
    assert elapsed_ms > 0
    assert elapsed_ms <= float(
        os.environ.get("DAITA_BENCHMARK_MAX_LATENCY_MS", "90000")
    )
    assert result.status is OperationStatus.SUCCEEDED
    assert result.diagnostics["verification"]["passed"] is True
    answer = result.answer or ""
    for expected in expected_answer:
        assert expected.lower() in answer.lower()


def _assert_evidence(result: DbOperationResult, expected: set[str]) -> None:
    kinds = {item.kind for item in result.evidence}
    assert expected <= kinds


def _record_benchmark(
    case: str,
    result: DbOperationResult,
    elapsed_ms: float,
    *,
    database: str,
) -> None:
    output_path = Path(
        os.environ.get(
            "DAITA_BENCHMARK_OUTPUT",
            ".daita/benchmarks/live-from-db-benchmarks.jsonl",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    llm = result.diagnostics.get("llm") or {}
    payload = {
        "case": case,
        "database": database,
        "elapsed_ms": round(elapsed_ms, 2),
        "operation_id": result.operation_id,
        "operation_type": result.contract.operation_type,
        "status": result.status.value,
        "answer": result.answer,
        "warnings": list(result.warnings),
        "evidence_kinds": [item.kind for item in result.evidence],
        "task_count": result.diagnostics.get("execution", {}).get("task_count"),
        "task_capabilities": [
            task.get("capability_id")
            for task in result.diagnostics.get("execution", {}).get("tasks", [])
        ],
        "tokens": llm.get("tokens", {}),
        "cost": llm.get("cost"),
        "diagnostics": result.diagnostics,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    print("[from_db_benchmark]", json.dumps(payload, sort_keys=True, default=str))
