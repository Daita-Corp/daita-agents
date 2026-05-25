"""
Large database performance benchmarks for ``Agent.from_db()``.

These benchmarks measure the framework itself over synthetic warehouse-scale
SQLite and optional PostgreSQL databases. Most cases use a deterministic mock
LLM so build, catalog, runtime-context, schema navigation, and query compaction
performance can be tracked without provider variance.

Run deterministic large-db benchmarks:
    pytest tests/performance/test_large_from_db_benchmarks.py -v -s

Optional live LLM large-schema benchmark:
    DAITA_BENCHMARK_LIVE_LLM=1 OPENAI_API_KEY=... pytest \
        tests/performance/test_large_from_db_benchmarks.py \
        -m "performance and requires_llm" -v -s

Optional deterministic PostgreSQL large-db benchmark:
    DAITA_LARGE_BENCHMARK_POSTGRES_URL=postgresql://... pytest \
        tests/performance/test_large_from_db_benchmarks.py \
        -m "performance and requires_db" -v -s

Optional:
    DAITA_LARGE_BENCHMARK_OUTPUT=.daita/benchmarks/large-from-db-benchmarks.jsonl
    DAITA_LARGE_BENCHMARK_TABLES=1000
    DAITA_LARGE_BENCHMARK_WIDE_COLUMNS=250
    DAITA_LARGE_BENCHMARK_BULKY_ROWS=1000
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from daita.agents.agent import Agent
from daita.core.streaming import LLMChunk
from daita.llm.mock import MockLLMProvider

pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_COUNT = 1000
DEFAULT_WIDE_COLUMN_COUNT = 250
DEFAULT_BULKY_ROW_COUNT = 1000


@pytest.fixture(scope="session")
def large_sqlite_db_path(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("large_from_db") / "warehouse_scale.sqlite"
    _seed_large_sqlite(
        db_path,
        table_count=_env_int("DAITA_LARGE_BENCHMARK_TABLES", DEFAULT_TABLE_COUNT),
        wide_column_count=_env_int(
            "DAITA_LARGE_BENCHMARK_WIDE_COLUMNS", DEFAULT_WIDE_COLUMN_COUNT
        ),
        bulky_row_count=_env_int(
            "DAITA_LARGE_BENCHMARK_BULKY_ROWS", DEFAULT_BULKY_ROW_COUNT
        ),
    )
    return db_path


@pytest.fixture(scope="session")
def large_postgres_url() -> str:
    postgres_url = os.environ.get("DAITA_LARGE_BENCHMARK_POSTGRES_URL")
    if not postgres_url:
        pytest.skip("Set DAITA_LARGE_BENCHMARK_POSTGRES_URL to run Postgres benchmark")
    asyncpg = pytest.importorskip(
        "asyncpg",
        reason="asyncpg required: pip install 'daita-agents[postgresql]'",
    )
    asyncio.run(
        _seed_large_postgres(
            asyncpg,
            postgres_url,
            table_count=_env_int("DAITA_LARGE_BENCHMARK_TABLES", DEFAULT_TABLE_COUNT),
            wide_column_count=_env_int(
                "DAITA_LARGE_BENCHMARK_WIDE_COLUMNS", DEFAULT_WIDE_COLUMN_COUNT
            ),
            bulky_row_count=_env_int(
                "DAITA_LARGE_BENCHMARK_BULKY_ROWS", DEFAULT_BULKY_ROW_COUNT
            ),
        )
    )
    return postgres_url


async def test_large_from_db_build_prompt_and_runtime_context_benchmark(
    large_sqlite_db_path: Path, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    llm = MockLLMProvider(delay=0)

    build_start = time.perf_counter()
    agent = await _large_agent(large_sqlite_db_path, llm_provider=llm)
    build_ms = _elapsed_ms(build_start)
    try:
        description = agent.describe()
        prompt = agent.prompt

        run_start = time.perf_counter()
        result = await agent.run(
            "What can you tell me about this schema?",
            detailed=True,
            max_iterations=1,
        )
        run_ms = _elapsed_ms(run_start)
        user_message = llm.call_history[-1]["messages"][-1]["content"]
        runtime_context = user_message.split("User question:", 1)[0]

        metrics = {
            "build_ms": round(build_ms, 2),
            "run_ms": round(run_ms, 2),
            "table_count": description["db"]["table_count"],
            "column_count": description["db"]["column_count"],
            "prompt_chars": len(prompt),
            "runtime_context_chars": len(runtime_context),
            "selected_tool_count": (
                getattr(agent, "_db_last_context_metadata", {}) or {}
            ).get("selected_tool_count"),
            "iterations": result["iterations"],
            "tool_count": len(result.get("tool_calls") or []),
        }
        _record_large_benchmark("large_build_prompt_runtime_context", metrics)

        assert description["db"]["table_count"] == _expected_table_count()
        assert description["db"]["column_count"] == _expected_column_count()
        assert "## Database Schema" in prompt
        assert "feature_249" not in prompt
        assert metrics["prompt_chars"] < _env_int(
            "DAITA_LARGE_BENCHMARK_MAX_PROMPT_CHARS", 20000
        )
        assert metrics["runtime_context_chars"] <= _env_int(
            "DAITA_LARGE_BENCHMARK_MAX_CONTEXT_CHARS", 1800
        )
        assert build_ms <= _env_float("DAITA_LARGE_BENCHMARK_MAX_BUILD_MS", 20000)
        assert run_ms <= _env_float("DAITA_LARGE_BENCHMARK_MAX_CONTEXT_RUN_MS", 10000)
    finally:
        await agent.stop()


async def test_large_catalog_search_inspection_and_relationship_benchmark(
    large_sqlite_db_path: Path, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    agent = await _large_agent(large_sqlite_db_path)
    try:
        store_id = agent._db_catalog_store_id

        search, search_ms = await _timed_tool(
            agent,
            "catalog_search_schema",
            {"store_id": store_id, "query": "feature 249 event value", "limit": 5},
        )
        first_page, first_page_ms = await _timed_tool(
            agent,
            "catalog_inspect_table",
            {
                "store_id": store_id,
                "table_name": "wide_fact_events",
                "limit": 100,
                "offset": 0,
            },
        )
        late_page, late_page_ms = await _timed_tool(
            agent,
            "catalog_inspect_table",
            {
                "store_id": store_id,
                "table_name": "wide_fact_events",
                "limit": 100,
                "offset": 200,
            },
        )
        pattern_page, pattern_ms = await _timed_tool(
            agent,
            "catalog_inspect_table",
            {
                "store_id": store_id,
                "table_name": "wide_fact_events",
                "column_pattern": "feature_24",
                "limit": 30,
            },
        )
        relationships, relationships_ms = await _timed_tool(
            agent,
            "catalog_find_join_paths",
            {
                "store_id": store_id,
                "from_tables": ["wide_fact_events"],
                "to_tables": ["accounts"],
                "max_hops": 2,
                "max_paths": 2,
            },
        )

        metrics = {
            "search_ms": round(search_ms, 2),
            "first_page_ms": round(first_page_ms, 2),
            "late_page_ms": round(late_page_ms, 2),
            "pattern_ms": round(pattern_ms, 2),
            "relationships_ms": round(relationships_ms, 2),
            "search_total_matches": search.get("total_matches"),
            "search_result_count": len(search.get("tables") or []),
            "wide_column_count": first_page.get("column_count"),
            "first_page_columns": len(first_page.get("columns") or []),
            "late_page_columns": len(late_page.get("columns") or []),
            "pattern_matched_columns": pattern_page.get("matched_column_count"),
            "relationship_reachable": relationships.get("reachable"),
            "relationship_path_count": relationships.get("path_count"),
        }
        _record_large_benchmark(
            "large_catalog_search_inspection_relationships", metrics
        )

        assert search["tables"][0]["name"] == "wide_fact_events"
        assert first_page["column_count"] == _expected_wide_column_count()
        assert first_page["truncated"] is True
        assert any(column["name"] == "feature_249" for column in late_page["columns"])
        assert pattern_page["matched_column_count"] >= 10
        assert relationships["reachable"] is True
        assert search_ms <= _env_float("DAITA_LARGE_BENCHMARK_MAX_SEARCH_MS", 3000)
        assert late_page_ms <= _env_float("DAITA_LARGE_BENCHMARK_MAX_INSPECT_MS", 3000)
        assert relationships_ms <= _env_float(
            "DAITA_LARGE_BENCHMARK_MAX_RELATIONSHIP_MS", 3000
        )
    finally:
        await agent.stop()


async def test_large_result_compaction_and_query_benchmark(
    large_sqlite_db_path: Path, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    agent = await _large_agent(
        large_sqlite_db_path,
        query_default_limit=1000,
        query_max_rows=25,
        query_max_chars=2400,
    )
    try:
        result, elapsed_ms = await _timed_tool(
            agent,
            "db_query",
            {
                "sql": (
                    "SELECT event_id, account_id, payload, created_at "
                    "FROM bulky_events ORDER BY event_id LIMIT 1000"
                )
            },
        )
        serialized_rows = json.dumps(result.get("rows") or [], default=str)
        metrics = {
            "query_ms": round(elapsed_ms, 2),
            "total_rows": result.get("total_rows"),
            "returned_rows": len(result.get("rows") or []),
            "truncated": result.get("truncated"),
            "serialized_rows_chars": len(serialized_rows),
            "sql_chars": len(result.get("sql") or ""),
        }
        _record_large_benchmark("large_result_compaction_query", metrics)

        assert result["total_rows"] == _env_int(
            "DAITA_LARGE_BENCHMARK_BULKY_ROWS", DEFAULT_BULKY_ROW_COUNT
        )
        assert result["truncated"] is True
        assert len(result["rows"]) <= 25
        assert len(serialized_rows) <= 2400
        assert "payload-0001-" in serialized_rows
        assert "payload-1000-" not in serialized_rows
        assert elapsed_ms <= _env_float("DAITA_LARGE_BENCHMARK_MAX_QUERY_MS", 5000)
    finally:
        await agent.stop()


async def test_large_scripted_agent_navigation_benchmark(
    large_sqlite_db_path: Path, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    llm = ScriptedToolLLMProvider(
        [
            _tool_turn(
                "call_search",
                "catalog_search_schema",
                {"query": "wide_fact_events feature_249 event_value", "limit": 5},
            ),
            _tool_turn(
                "call_inspect",
                "catalog_inspect_table",
                {
                    "table_name": "wide_fact_events",
                    "offset": 200,
                    "limit": 80,
                },
            ),
            _tool_turn(
                "call_query",
                "db_query",
                {
                    "sql": (
                        "SELECT event_id, account_id, feature_000, feature_249, "
                        "event_value FROM wide_fact_events WHERE account_id = 42 "
                        "ORDER BY event_id LIMIT 1"
                    )
                },
            ),
            {
                "content": (
                    "For account_id 42, the first wide_fact_events row has "
                    "feature_000 segment-a and feature_249 channel-web."
                )
            },
        ]
    )
    agent = await _large_agent(large_sqlite_db_path, llm_provider=llm)
    try:
        start = time.perf_counter()
        result = await agent.run(
            (
                "Find the wide event feature table, inspect late columns, then "
                "query account_id 42 for feature_000 and feature_249."
            ),
            tools=["catalog_search_schema", "catalog_inspect_table", "db_query"],
            detailed=True,
            max_iterations=8,
        )
        elapsed_ms = _elapsed_ms(start)
        tool_sequence = [call.get("tool") for call in result.get("tool_calls") or []]
        query_call = next(
            call for call in result["tool_calls"] if call.get("tool") == "db_query"
        )
        rows = query_call.get("result", {}).get("rows") or []
        metrics = {
            "elapsed_ms": round(elapsed_ms, 2),
            "iterations": result["iterations"],
            "tool_count": len(result.get("tool_calls") or []),
            "tool_sequence": tool_sequence,
            "tool_durations_ms": [
                call.get("duration_ms") for call in result.get("tool_calls") or []
            ],
            "tokens": result.get("tokens"),
            "context": getattr(agent, "_db_last_context_metadata", {}) or {},
            "returned_rows": len(rows),
            "result_chars": len(str(result.get("result") or "")),
        }
        _record_large_benchmark("large_scripted_agent_navigation", metrics)

        assert tool_sequence == [
            "catalog_search_schema",
            "catalog_inspect_table",
            "db_query",
        ]
        assert rows
        assert rows[0]["feature_000"] == "segment-a"
        assert rows[0]["feature_249"] == "channel-web"
        assert "channel-web" in result["result"]
        assert elapsed_ms <= _env_float(
            "DAITA_LARGE_BENCHMARK_MAX_SCRIPTED_AGENT_MS", 20000
        )
    finally:
        await agent.stop()


@pytest.mark.requires_db
async def test_large_postgres_build_catalog_query_benchmark(
    large_postgres_url: str, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    build_start = time.perf_counter()
    agent = await _large_agent(
        large_postgres_url,
        name="BenchLargeFromDbPostgres",
        query_default_limit=1000,
        query_max_rows=25,
        query_max_chars=2400,
    )
    build_ms = _elapsed_ms(build_start)
    try:
        description = agent.describe()
        store_id = agent._db_catalog_store_id

        search, search_ms = await _timed_tool(
            agent,
            "catalog_search_schema",
            {
                "store_id": store_id,
                "query": "wide_fact_events feature_249 event_value",
                "limit": 5,
            },
        )
        inspect, inspect_ms = await _timed_tool(
            agent,
            "catalog_inspect_table",
            {
                "store_id": store_id,
                "table_name": "wide_fact_events",
                "column_pattern": "feature_24",
                "limit": 30,
            },
        )
        query, query_ms = await _timed_tool(
            agent,
            "db_query",
            {
                "sql": (
                    "SELECT event_id, account_id, feature_000, feature_249, "
                    "event_value FROM wide_fact_events WHERE account_id = 42 "
                    "ORDER BY event_id LIMIT 1"
                )
            },
        )

        rows = query.get("rows") or []
        metrics = {
            "database": "postgresql",
            "build_ms": round(build_ms, 2),
            "search_ms": round(search_ms, 2),
            "inspect_ms": round(inspect_ms, 2),
            "query_ms": round(query_ms, 2),
            "table_count": description["db"]["table_count"],
            "column_count": description["db"]["column_count"],
            "search_total_matches": search.get("total_matches"),
            "pattern_matched_columns": inspect.get("matched_column_count"),
            "returned_rows": len(rows),
            "truncated": query.get("truncated"),
        }
        _record_large_benchmark("large_postgres_build_catalog_query", metrics)

        assert description["db"]["database_type"] == "postgresql"
        assert description["db"]["table_count"] == _expected_table_count()
        assert description["db"]["column_count"] == _expected_column_count()
        assert search["tables"][0]["name"] == "wide_fact_events"
        assert inspect["matched_column_count"] >= 10
        assert rows
        assert rows[0]["feature_000"] == "segment-a"
        assert rows[0]["feature_249"] == "channel-web"
        assert build_ms <= _env_float(
            "DAITA_LARGE_BENCHMARK_MAX_POSTGRES_BUILD_MS", 60000
        )
        assert search_ms <= _env_float(
            "DAITA_LARGE_BENCHMARK_MAX_POSTGRES_SEARCH_MS", 5000
        )
        assert query_ms <= _env_float(
            "DAITA_LARGE_BENCHMARK_MAX_POSTGRES_QUERY_MS", 10000
        )
    finally:
        await agent.stop()


@pytest.mark.requires_llm
async def test_live_large_from_db_omitted_wide_table_navigation_benchmark(
    large_sqlite_db_path: Path, monkeypatch, tmp_path
):
    if os.environ.get("DAITA_BENCHMARK_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_BENCHMARK_LIVE_LLM=1 to run live large-db benchmark")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for the live large-db benchmark")

    monkeypatch.chdir(tmp_path)
    build_start = time.perf_counter()
    agent = await _large_agent(
        large_sqlite_db_path,
        name="BenchLargeFromDbLive",
        llm_provider="openai",
        model=os.environ.get("DAITA_BENCHMARK_MODEL")
        or os.environ.get("OPENAI_TEST_MODEL")
        or "gpt-4o-mini",
        api_key=api_key,
        temperature=0,
        max_tokens=_env_int("DAITA_BENCHMARK_MAX_TOKENS", 220),
    )
    build_ms = _elapsed_ms(build_start)
    try:
        run_start = time.perf_counter()
        result = await agent.run(
            (
                "This database has many omitted tables. Find the table that stores "
                "wide event feature columns, inspect schema metadata until you can "
                "identify feature_249, then query account_id 42 for the first event "
                "by event_id. What are feature_000 and feature_249?"
            ),
            detailed=True,
            max_iterations=12,
        )
        elapsed_ms = _elapsed_ms(run_start)
        tool_sequence = [call.get("tool") for call in result.get("tool_calls") or []]
        metrics = {
            "agent_build_ms": round(build_ms, 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "iterations": result["iterations"],
            "tool_count": len(result.get("tool_calls") or []),
            "tool_sequence": tool_sequence,
            "tool_durations_ms": [
                call.get("duration_ms") for call in result.get("tool_calls") or []
            ],
            "tokens": result.get("tokens"),
            "cost": result.get("cost"),
            "diagnostics": result.get("diagnostics"),
            "context": getattr(agent, "_db_last_context_metadata", {}) or {},
        }
        _record_large_benchmark("live_large_omitted_wide_table_navigation", metrics)

        assert any(
            tool in tool_sequence
            for tool in ("catalog_search_schema", "search_catalog")
        ), tool_sequence
        assert "catalog_inspect_table" in tool_sequence
        assert "db_query" in tool_sequence
        assert "segment-a" in str(result.get("result", "")).lower()
        assert "channel-web" in str(result.get("result", "")).lower()
        assert result["iterations"] <= 12
        assert elapsed_ms <= _env_float(
            "DAITA_LARGE_BENCHMARK_MAX_LIVE_AGENT_MS", 120000
        )
    finally:
        await agent.stop()


class ScriptedToolLLMProvider(MockLLMProvider):
    def __init__(self, turns: list[dict[str, Any]]):
        super().__init__(delay=0)
        self.turns = list(turns)

    async def _generate_impl(self, messages, tools=None, **kwargs):
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "params": kwargs,
                "timestamp": time.perf_counter(),
            }
        )
        if self.turns:
            return self.turns.pop(0)
        return {"content": "done", "tool_calls": None}

    async def _stream_impl(self, messages, tools=None, **kwargs):
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "params": kwargs,
                "timestamp": time.perf_counter(),
            }
        )
        turn = self.turns.pop(0) if self.turns else {"content": "done"}
        for tool_call in turn.get("tool_calls") or []:
            yield LLMChunk(
                type="tool_call_complete",
                tool_name=tool_call["name"],
                tool_args=tool_call.get("arguments") or {},
                tool_call_id=tool_call.get("id", "call_1"),
            )
        content = turn.get("content") or ""
        if content:
            yield LLMChunk(type="text", content=content)


async def _large_agent(source: Any, **overrides: Any) -> Agent:
    kwargs = {
        "name": "BenchLargeFromDb",
        "llm_provider": MockLLMProvider(delay=0),
        "mode": "analyst",
        "cache_ttl": None,
        "include_sample_values": False,
        "query_default_limit": 5,
        "query_max_rows": 10,
        "query_max_chars": 1400,
    }
    kwargs.update(overrides)
    return await Agent.from_db(str(source), **kwargs)


async def _timed_tool(
    agent: Agent, tool_name: str, args: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    result = await agent.tool_registry.execute(tool_name, args)
    return result, _elapsed_ms(start)


def _tool_turn(call_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "name": tool_name,
                "arguments": args,
            }
        ],
    }


def _seed_large_sqlite(
    path: Path,
    *,
    table_count: int,
    wide_column_count: int,
    bulky_row_count: int,
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        statements = ["""
            CREATE TABLE accounts (
                account_id INTEGER PRIMARY KEY,
                account_name TEXT NOT NULL
            );
            """]
        for index in range(table_count):
            statements.append(f"""
                CREATE TABLE warehouse_table_{index:04d} (
                    id INTEGER PRIMARY KEY,
                    metric_value REAL,
                    status TEXT,
                    created_at TEXT
                );
                """)
        wide_columns = ",\n".join(
            f"                    feature_{index:03d} TEXT"
            for index in range(wide_column_count)
        )
        statements.append(f"""
            CREATE TABLE wide_fact_events (
                event_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL REFERENCES accounts(account_id),
{wide_columns},
                event_value REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
        statements.append("""
            CREATE TABLE bulky_events (
                event_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
        conn.executescript("\n".join(statements))
        conn.executemany(
            "INSERT INTO accounts VALUES (?, ?)",
            [(42, "Acme Analytics"), (7, "Globex")],
        )
        wide_insert_columns = [
            "event_id",
            "account_id",
            "feature_000",
            f"feature_{wide_column_count - 1:03d}",
            "event_value",
            "created_at",
        ]
        conn.executemany(
            (
                f"INSERT INTO wide_fact_events "
                f"({', '.join(wide_insert_columns)}) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            [
                (
                    1,
                    42,
                    "segment-a",
                    "channel-web",
                    123.45,
                    "2026-04-01T00:00:00",
                ),
                (
                    2,
                    42,
                    "segment-b",
                    "channel-app",
                    54.0,
                    "2026-04-02T00:00:00",
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO bulky_events VALUES (?, ?, ?, ?)",
            [
                (
                    index,
                    index % 17,
                    f"payload-{index:04d}-" + ("x" * 160),
                    f"2026-05-{(index % 28) + 1:02d}T00:00:00",
                )
                for index in range(1, bulky_row_count + 1)
            ],
        )
        conn.commit()
    finally:
        conn.close()


async def _seed_large_postgres(
    asyncpg: Any,
    url: str,
    *,
    table_count: int,
    wide_column_count: int,
    bulky_row_count: int,
) -> None:
    conn = await asyncpg.connect(url)
    try:
        drop_names = ["wide_fact_events", "bulky_events", "accounts"] + [
            f"warehouse_table_{index:04d}" for index in range(table_count)
        ]
        await conn.execute(f"DROP TABLE IF EXISTS {', '.join(drop_names)} CASCADE")

        statements = ["""
            CREATE TABLE accounts (
                account_id INTEGER PRIMARY KEY,
                account_name TEXT NOT NULL
            );
            """]
        for index in range(table_count):
            statements.append(f"""
                CREATE TABLE warehouse_table_{index:04d} (
                    id INTEGER PRIMARY KEY,
                    metric_value DOUBLE PRECISION,
                    status TEXT,
                    created_at TEXT
                );
                """)
        wide_columns = ",\n".join(
            f"                    feature_{index:03d} TEXT"
            for index in range(wide_column_count)
        )
        statements.append(f"""
            CREATE TABLE wide_fact_events (
                event_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL REFERENCES accounts(account_id),
{wide_columns},
                event_value DOUBLE PRECISION NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
        statements.append("""
            CREATE TABLE bulky_events (
                event_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
        await conn.execute("\n".join(statements))
        await conn.executemany(
            "INSERT INTO accounts VALUES ($1, $2)",
            [(42, "Acme Analytics"), (7, "Globex")],
        )
        wide_insert_columns = [
            "event_id",
            "account_id",
            "feature_000",
            f"feature_{wide_column_count - 1:03d}",
            "event_value",
            "created_at",
        ]
        await conn.executemany(
            (
                f"INSERT INTO wide_fact_events "
                f"({', '.join(wide_insert_columns)}) "
                "VALUES ($1, $2, $3, $4, $5, $6)"
            ),
            [
                (
                    1,
                    42,
                    "segment-a",
                    "channel-web",
                    123.45,
                    "2026-04-01T00:00:00",
                ),
                (
                    2,
                    42,
                    "segment-b",
                    "channel-app",
                    54.0,
                    "2026-04-02T00:00:00",
                ),
            ],
        )
        await conn.executemany(
            "INSERT INTO bulky_events VALUES ($1, $2, $3, $4)",
            [
                (
                    index,
                    index % 17,
                    f"payload-{index:04d}-" + ("x" * 160),
                    f"2026-05-{(index % 28) + 1:02d}T00:00:00",
                )
                for index in range(1, bulky_row_count + 1)
            ],
        )
    finally:
        await conn.close()


def _record_large_benchmark(case: str, metrics: dict[str, Any]) -> None:
    output_path = Path(
        os.environ.get(
            "DAITA_LARGE_BENCHMARK_OUTPUT",
            str(REPO_ROOT / ".daita/benchmarks/large-from-db-benchmarks.jsonl"),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case,
        "benchmark_kind": "large_from_db",
        "database": "sqlite",
        "table_count": _expected_table_count(),
        "wide_column_count": _expected_wide_column_count(),
        "bulky_row_count": _env_int(
            "DAITA_LARGE_BENCHMARK_BULKY_ROWS", DEFAULT_BULKY_ROW_COUNT
        ),
        **metrics,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    print("[large-benchmark]", json.dumps(payload, sort_keys=True, default=str))


def _expected_table_count() -> int:
    return _env_int("DAITA_LARGE_BENCHMARK_TABLES", DEFAULT_TABLE_COUNT) + 3


def _expected_wide_column_count() -> int:
    return _env_int("DAITA_LARGE_BENCHMARK_WIDE_COLUMNS", DEFAULT_WIDE_COLUMN_COUNT) + 4


def _expected_column_count() -> int:
    table_columns = _env_int("DAITA_LARGE_BENCHMARK_TABLES", DEFAULT_TABLE_COUNT) * 4
    return table_columns + 2 + _expected_wide_column_count() + 4


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default
