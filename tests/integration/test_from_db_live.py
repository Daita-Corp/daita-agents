"""
Live integration tests for Agent.from_db().

These tests exercise real local databases, not mocked schema discovery:

* SQLite uses a temporary file-backed database fixture.
* PostgreSQL uses a throwaway Docker container seeded with a known schema.

Most tests use MockLLMProvider and execute registered tools directly so they
validate from_db runtime wiring cheaply. The requires_llm tests use a real
OpenAI connection and verify that the model actually calls DB tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.agents.agent import Agent
from daita.agents.db.memory import DBMemoryRecord
from daita.core.streaming import LLMChunk
from daita.embeddings.base import BaseEmbeddingProvider
from daita.core.watch import WatchEvent
from daita.llm.mock import MockLLMProvider
from daita.plugins.memory import MemoryPlugin
from daita.plugins.sqlite import SQLitePlugin

from ._harness import assert_answer_mentions, assert_tool_called, start_container

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_from_db_test"

CORE_BUILD_MODES = {
    "simple": {
        "analyst_tools": False,
        "quality_tools": False,
        "lineage_tools": False,
        "history": False,
    },
    "analyst": {
        "analyst_tools": True,
        "quality_tools": False,
        "lineage_tools": False,
        "history": False,
    },
    "governed": {
        "analyst_tools": True,
        "quality_tools": False,
        "lineage_tools": True,
        "history": True,
    },
    "data_team": {
        "analyst_tools": True,
        "quality_tools": True,
        "lineage_tools": True,
        "history": True,
    },
}

ANALYST_TOOL_NAMES = {
    "pivot_table",
    "correlate",
    "detect_anomalies",
    "compare_entities",
    "find_similar",
    "forecast_trend",
}
QUALITY_TOOL_NAMES = {
    "dq_profile",
    "dq_detect_anomaly",
    "dq_check_freshness",
    "dq_report",
}
LINEAGE_TOOL_NAMES = {
    "trace_lineage",
    "analyze_impact",
    "find_lineage_paths",
    "register_flow",
    "export_lineage",
}
DISCOVERY_TOOL_NAMES = {
    "postgres_list_tables",
    "postgres_get_schema",
    "postgres_inspect",
    "sqlite_list_tables",
    "sqlite_get_schema",
    "sqlite_inspect",
}

SEED_SQL = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS daily_metrics;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT UNIQUE,
    signup_date  DATE NOT NULL
);

CREATE TABLE orders (
    order_id     SERIAL PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    total_amount NUMERIC(10, 2) NOT NULL,
    status       TEXT NOT NULL,
    created_at   TIMESTAMP NOT NULL
);

CREATE TABLE daily_metrics (
    metric_date  DATE PRIMARY KEY,
    revenue      NUMERIC(10, 2) NOT NULL,
    orders_count INTEGER NOT NULL,
    sessions     INTEGER NOT NULL
);

INSERT INTO customers (name, email, signup_date) VALUES
    ('Alice', 'alice@example.com', '2026-01-01'),
    ('Bob', 'bob@example.com', '2026-01-02'),
    ('Cara', 'cara@example.com', '2026-01-03');

INSERT INTO orders (customer_id, total_amount, status, created_at) VALUES
    (1, 50.00,  'shipped', '2026-02-01T10:00:00'),
    (1, 75.00,  'shipped', '2026-02-02T10:00:00'),
    (2, 150.00, 'pending', '2026-02-03T10:00:00');

INSERT INTO daily_metrics (metric_date, revenue, orders_count, sessions) VALUES
    ('2026-02-01', 100.00, 10, 100),
    ('2026-02-02', 120.00, 12, 120),
    ('2026-02-03', 140.00, 14, 140),
    ('2026-02-04', 160.00, 16, 160),
    ('2026-02-05', 180.00, 18, 180),
    ('2026-02-06', 1000.00, 20, 200);
"""

MULTI_SCHEMA_SQL = """
DROP SCHEMA IF EXISTS analytics CASCADE;
DROP TABLE IF EXISTS public.orders;
DROP TABLE IF EXISTS public.customers;

CREATE TABLE public.customers (
    customer_id  SERIAL PRIMARY KEY,
    name         TEXT NOT NULL
);

CREATE TABLE public.orders (
    order_id     SERIAL PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES public.customers(customer_id),
    total_amount NUMERIC(10, 2) NOT NULL,
    status       TEXT NOT NULL
);

CREATE SCHEMA analytics;

CREATE TABLE analytics.orders (
    analytic_order_id SERIAL PRIMARY KEY,
    customer_id       INTEGER NOT NULL,
    recognized_revenue NUMERIC(10, 2) NOT NULL,
    channel           TEXT NOT NULL
);

INSERT INTO public.customers (name) VALUES ('Alice'), ('Bob');
INSERT INTO public.orders (customer_id, total_amount, status) VALUES
    (1, 50.00, 'shipped'),
    (1, 75.00, 'shipped'),
    (2, 150.00, 'pending');
INSERT INTO analytics.orders (customer_id, recognized_revenue, channel) VALUES
    (1, 110.00, 'web'),
    (2, 90.00, 'app');
"""


async def _seed_sqlite(path: str) -> None:
    db = SQLitePlugin(path=path, wal_mode=False)
    await db.connect()
    try:
        await db.execute_script("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                signup_date TEXT NOT NULL
            );

            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                total_amount REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE daily_metrics (
                metric_date TEXT PRIMARY KEY,
                revenue REAL NOT NULL,
                orders_count INTEGER NOT NULL,
                sessions INTEGER NOT NULL
            );
            """)
        await db.insert_many(
            "customers",
            [
                {
                    "customer_id": 1,
                    "name": "Alice",
                    "email": "alice@example.com",
                    "signup_date": "2026-01-01",
                },
                {
                    "customer_id": 2,
                    "name": "Bob",
                    "email": "bob@example.com",
                    "signup_date": "2026-01-02",
                },
                {
                    "customer_id": 3,
                    "name": "Cara",
                    "email": "cara@example.com",
                    "signup_date": "2026-01-03",
                },
            ],
        )
        await db.insert_many(
            "orders",
            [
                {
                    "order_id": 1,
                    "customer_id": 1,
                    "total_amount": 50.0,
                    "status": "shipped",
                    "created_at": "2026-02-01T10:00:00",
                },
                {
                    "order_id": 2,
                    "customer_id": 1,
                    "total_amount": 75.0,
                    "status": "shipped",
                    "created_at": "2026-02-02T10:00:00",
                },
                {
                    "order_id": 3,
                    "customer_id": 2,
                    "total_amount": 150.0,
                    "status": "pending",
                    "created_at": "2026-02-03T10:00:00",
                },
            ],
        )
        await db.insert_many(
            "daily_metrics",
            [
                {
                    "metric_date": "2026-02-01",
                    "revenue": 100.0,
                    "orders_count": 10,
                    "sessions": 100,
                },
                {
                    "metric_date": "2026-02-02",
                    "revenue": 120.0,
                    "orders_count": 12,
                    "sessions": 120,
                },
                {
                    "metric_date": "2026-02-03",
                    "revenue": 140.0,
                    "orders_count": 14,
                    "sessions": 140,
                },
                {
                    "metric_date": "2026-02-04",
                    "revenue": 160.0,
                    "orders_count": 16,
                    "sessions": 160,
                },
                {
                    "metric_date": "2026-02-05",
                    "revenue": 180.0,
                    "orders_count": 18,
                    "sessions": 180,
                },
                {
                    "metric_date": "2026-02-06",
                    "revenue": 1000.0,
                    "orders_count": 20,
                    "sessions": 200,
                },
            ],
        )
    finally:
        await db.disconnect()


async def _seed_large_guardrail_sqlite(path: str) -> None:
    db = SQLitePlugin(path=path, wal_mode=False)
    await db.connect()
    try:
        statements = []
        for i in range(90):
            statements.append(f"""
                CREATE TABLE table_{i:02d} (
                    id INTEGER PRIMARY KEY,
                    metric_value REAL NOT NULL,
                    long_dimension_name_{i:02d} TEXT,
                    created_at TEXT
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
        await db.execute_script("\n".join(statements))
        await db.insert_many(
            "bulky_events",
            [
                {
                    "event_id": i,
                    "account_id": i % 7,
                    "payload": f"payload-{i:03d}-" + ("x" * 120),
                    "created_at": f"2026-03-{(i % 28) + 1:02d}T00:00:00",
                }
                for i in range(1, 121)
            ],
        )
    finally:
        await db.disconnect()


async def _seed_warehouse_scale_sqlite(path: str) -> None:
    db = SQLitePlugin(path=path, wal_mode=False)
    await db.connect()
    try:
        statements = []
        for i in range(1000):
            statements.append(f"""
                CREATE TABLE warehouse_table_{i:04d} (
                    id INTEGER PRIMARY KEY,
                    metric_value REAL,
                    status TEXT,
                    created_at TEXT
                );
                """)

        wide_columns = ",\n".join(
            f"                    feature_{i:03d} TEXT" for i in range(250)
        )
        statements.append(f"""
            CREATE TABLE wide_fact_events (
                event_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
{wide_columns},
                event_value REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
        await db.execute_script("\n".join(statements))
        await db.insert_many(
            "wide_fact_events",
            [
                {
                    "event_id": 1,
                    "account_id": 42,
                    "feature_000": "segment-a",
                    "feature_249": "channel-web",
                    "event_value": 123.45,
                    "created_at": "2026-04-01T00:00:00",
                },
                {
                    "event_id": 2,
                    "account_id": 42,
                    "feature_000": "segment-b",
                    "feature_249": "channel-app",
                    "event_value": 54.0,
                    "created_at": "2026-04-02T00:00:00",
                },
            ],
        )
    finally:
        await db.disconnect()


@pytest.fixture
async def sqlite_db_path(tmp_path):
    db_path = tmp_path / "from_db.sqlite"
    await _seed_sqlite(str(db_path))
    return db_path


@pytest.fixture
async def large_guardrail_sqlite_db_path(tmp_path):
    db_path = tmp_path / "from_db_large_guardrails.sqlite"
    await _seed_large_guardrail_sqlite(str(db_path))
    return db_path


@pytest.fixture
async def warehouse_scale_sqlite_db_path(tmp_path):
    db_path = tmp_path / "from_db_warehouse_scale.sqlite"
    await _seed_warehouse_scale_sqlite(str(db_path))
    return db_path


@pytest.fixture(scope="module")
def postgres_container():
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
def postgres_url(postgres_container) -> str:
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{postgres_container.host}:{postgres_container.host_port}/{POSTGRES_DB}"
        "?sslmode=disable"
    )


@pytest.fixture(scope="module")
def seeded_postgres(postgres_url) -> str:
    asyncpg = pytest.importorskip(
        "asyncpg",
        reason="asyncpg required for PostgreSQL from_db integration tests",
    )

    async def _seed() -> None:
        deadline = time.time() + 30
        last_err = None
        while time.time() < deadline:
            try:
                conn = await asyncpg.connect(postgres_url)
                await conn.execute(SEED_SQL)
                await conn.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Could not seed Postgres from_db test DB: {last_err}")

    asyncio.run(_seed())
    return postgres_url


@pytest.fixture(scope="module")
def seeded_multischema_postgres(postgres_url) -> str:
    asyncpg = pytest.importorskip(
        "asyncpg",
        reason="asyncpg required for PostgreSQL from_db integration tests",
    )

    async def _seed() -> None:
        deadline = time.time() + 30
        last_err = None
        while time.time() < deadline:
            try:
                conn = await asyncpg.connect(postgres_url)
                await conn.execute(MULTI_SCHEMA_SQL)
                await conn.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Could not seed multi-schema Postgres DB: {last_err}")

    asyncio.run(_seed())
    return postgres_url


async def _close_agent_db(agent) -> None:
    plugin = getattr(getattr(agent, "db", None), "plugin", None)
    if plugin is not None and hasattr(plugin, "disconnect"):
        await plugin.disconnect()


def _query_tool_name(database_type: str) -> str:
    return "db_query"


def _count_tool_name(database_type: str) -> str:
    return "db_count"


def _sample_tool_name(database_type: str) -> str:
    return "db_sample"


async def _core_build_matrix_result(
    agent,
    *,
    source_kind: str,
    mode: str,
) -> dict:
    description = agent.describe()
    database_type = description["db"]["database_type"]
    query_tool = _query_tool_name(database_type)
    query_result = await agent.tool_registry.execute(
        query_tool,
        {"sql": "SELECT COUNT(*) AS order_count FROM orders"},
    )

    return {
        "source_kind": source_kind,
        "mode": mode,
        "agent_name": description["name"],
        "database_type": database_type,
        "capabilities": description["capabilities"],
        "tools": description["tools"],
        "tool_count": description["tool_count"],
        "schema": {
            "table_count": description["db"]["table_count"],
            "column_count": description["db"]["column_count"],
            "relationship_count": description["db"]["relationship_count"],
            "tables": sorted(t["name"] for t in agent.db.schema["tables"]),
            "foreign_keys": agent.db.schema.get("foreign_keys", []),
        },
        "db_context": {
            "mode": agent.db.mode,
            "has_plugin": agent.db.plugin is not None,
            "has_schema": bool(agent.db.schema),
            "has_summary": bool(agent.db.summary),
            "suggested_question_count": len(agent.db.suggested_questions),
            "audit_entry_count": len(agent.db.audit.entries),
            "monitor_event_count": len(agent.db.monitor_events),
            "finding_count": len(agent.db.findings.all),
            "quality_attached": agent.db.quality is not None,
            "lineage_attached": agent.db.lineage is not None,
            "history_attached": agent.db.history is not None,
        },
        "query_sample": query_result,
        "query_tool_parameters": agent.tool_registry.get(query_tool).parameters,
        "describe_shape": {
            "top_level_keys": sorted(description.keys()),
            "db_keys": sorted(description["db"].keys()),
            "query_policy_keys": sorted(description["db"]["query_policy"].keys()),
        },
    }


def _write_core_build_matrix_result(tmp_path, result: dict) -> None:
    output_dir = os.environ.get("DAITA_FROM_DB_MATRIX_OUTPUT")
    if not output_dir:
        return
    path = Path(output_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{result['source_kind']}_{result['mode']}.json"
    (path / filename).write_text(json.dumps(result, indent=2, default=str))


def _assert_core_build_matrix_result(result: dict) -> None:
    expected = CORE_BUILD_MODES[result["mode"]]
    tools = set(result["tools"])
    capabilities = set(result["capabilities"])
    db_context = result["db_context"]
    database_type = result["database_type"]

    def fail(message: str) -> str:
        return f"{message}\nCore build result:\n{json.dumps(result, indent=2, default=str)}"

    assert result["schema"]["table_count"] == 3, fail("schema table count mismatch")
    assert result["schema"]["column_count"] == 13, fail("schema column count mismatch")
    assert result["schema"]["tables"] == [
        "customers",
        "daily_metrics",
        "orders",
    ], fail("schema table names mismatch")

    if database_type == "postgresql":
        assert result["schema"]["relationship_count"] == 1, fail(
            "postgres relationship count mismatch"
        )
        assert result["schema"]["foreign_keys"] == [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "customer_id",
            }
        ], fail("postgres FK discovery mismatch")
    else:
        assert result["schema"]["relationship_count"] == 0, fail(
            "sqlite relationship count mismatch"
        )

    assert _query_tool_name(database_type) in tools, fail("query tool missing")
    assert _count_tool_name(database_type) in tools, fail("count tool missing")
    assert _sample_tool_name(database_type) in tools, fail("sample tool missing")
    assert tools.isdisjoint(DISCOVERY_TOOL_NAMES), fail("discovery tools leaked")
    assert "focus" not in result["query_tool_parameters"]["properties"], fail(
        "Focus DSL leaked into from_db query tool schema"
    )

    if expected["analyst_tools"]:
        assert ANALYST_TOOL_NAMES.issubset(tools), fail("analyst tools missing")
        assert "analyst_tools" in capabilities, fail("analyst capability missing")
    else:
        assert tools.isdisjoint(ANALYST_TOOL_NAMES), fail("analyst tools leaked")
        assert "analyst_tools" not in capabilities, fail("analyst capability leaked")

    if expected["quality_tools"]:
        assert QUALITY_TOOL_NAMES.issubset(tools), fail("quality tools missing")
        assert "data_quality" in capabilities, fail("quality capability missing")
        assert db_context["quality_attached"] is True, fail("quality context missing")
    else:
        assert tools.isdisjoint(QUALITY_TOOL_NAMES), fail("quality tools leaked")
        assert "data_quality" not in capabilities, fail("quality capability leaked")
        assert db_context["quality_attached"] is False, fail("quality context leaked")

    if expected["lineage_tools"]:
        assert LINEAGE_TOOL_NAMES.issubset(tools), fail("lineage tools missing")
        assert "lineage" in capabilities, fail("lineage capability missing")
        assert db_context["lineage_attached"] is True, fail("lineage context missing")
    else:
        assert tools.isdisjoint(LINEAGE_TOOL_NAMES), fail("lineage tools leaked")
        assert "lineage" not in capabilities, fail("lineage capability leaked")
        assert db_context["lineage_attached"] is False, fail("lineage context leaked")

    assert db_context["history_attached"] is expected["history"], fail(
        "history context mismatch"
    )
    assert db_context["mode"] == result["mode"], fail("agent.db mode mismatch")
    assert db_context["has_plugin"] is True, fail("agent.db plugin missing")
    assert db_context["has_schema"] is True, fail("agent.db schema missing")
    assert db_context["has_summary"] is True, fail("agent.db summary missing")
    assert db_context["suggested_question_count"] > 0, fail(
        "suggested questions missing"
    )

    assert result["query_sample"]["total_rows"] == 1, fail("query sample row mismatch")
    assert result["query_sample"]["rows"][0]["order_count"] == 3, fail(
        "query sample value mismatch"
    )
    assert result["query_sample"]["sql"].endswith("LIMIT 20"), fail(
        "query sample did not inject LIMIT"
    )

    assert result["describe_shape"]["top_level_keys"] == [
        "agent_id",
        "capabilities",
        "db",
        "kind",
        "llm",
        "name",
        "tool_count",
        "tools",
    ], fail("describe top-level shape changed")
    for key in (
        "database_type",
        "mode",
        "query_policy",
        "summary",
        "tools",
        "table_count",
        "column_count",
    ):
        assert key in result["describe_shape"]["db_keys"], fail(
            f"describe db shape missing {key}"
        )


def _openai_kwargs() -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping live from_db LLM test")
    return {
        "llm_provider": "openai",
        "api_key": api_key,
        "model": os.environ.get("DAITA_LIVE_LLM_MODEL", "gpt-4o-mini"),
    }


class ScriptedToolLLMProvider(MockLLMProvider):
    """Mock provider that emits deterministic tool calls before a final answer."""

    def __init__(self, turns: list[dict], **kwargs):
        super().__init__(delay=0, **kwargs)
        self.turns = list(turns)

    async def _generate_impl(self, messages, tools=None, **kwargs):
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "params": kwargs,
                "timestamp": asyncio.get_event_loop().time(),
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
                "timestamp": asyncio.get_event_loop().time(),
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


class KeywordEmbeddingProvider(BaseEmbeddingProvider):
    """Small deterministic embedder for local memory integration tests."""

    TERMS = (
        "revenue",
        "refund",
        "refunded",
        "order",
        "orders",
        "metric",
        "semantics",
        "calculate",
        "calculated",
        "customer",
        "email",
        "row",
        "lookup",
    )

    def __init__(self):
        super().__init__(model="keyword-test-embedder")
        self.embedded_texts: list[str] = []

    @property
    def dimensions(self) -> int:
        return len(self.TERMS)

    async def _embed_text_impl(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        normalized = text.lower()
        return [float(normalized.count(term)) for term in self.TERMS]

    async def _embed_texts_impl(self, texts: list[str]) -> list[list[float]]:
        return [await self._embed_text_impl(text) for text in texts]


def _from_db_metrics(result: dict, tool_name: str, expected_value: str) -> dict:
    tool_calls = assert_tool_called(result, tool_name)
    answer = result.get("result") or ""
    accurate = expected_value in answer
    return {
        "tool": tool_name,
        "accurate": accurate,
        "expected_value": expected_value,
        "answer": answer,
        "query_duration_ms": tool_calls[0].get("duration_ms"),
        "iterations": result.get("iterations"),
        "tokens": result.get("tokens"),
        "cost": result.get("cost"),
    }


def _write_live_accuracy_result(tmp_path, case_result: dict) -> None:
    output_dir = os.environ.get("DAITA_FROM_DB_ACCURACY_OUTPUT")
    if not output_dir:
        return
    path = Path(output_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{case_result['case_id']}.json"
    (path / filename).write_text(json.dumps(case_result, indent=2, default=str))


def _assert_any_tool_called(result: dict, tool_names: set[str]) -> dict:
    calls = result.get("tool_calls") or []
    for call in calls:
        if call.get("tool") in tool_names:
            return call
    raise AssertionError(
        f"Expected one of {sorted(tool_names)}. "
        f"Tools actually called: {[call.get('tool') for call in calls]}"
    )


def _assert_answer_tokens(
    result: dict,
    *,
    all_of: list[str] | None = None,
    any_of: list[str] | None = None,
    none_of: list[str] | None = None,
) -> None:
    answer = (result.get("result") or "").lower()
    normalized_answer = answer.replace(",", "")
    missing = [
        token
        for token in all_of or []
        if token.lower().replace(",", "") not in normalized_answer
    ]
    if missing:
        raise AssertionError(f"Answer missing tokens {missing}: {answer[:500]!r}")
    if any_of and not any(
        token.lower().replace(",", "") in normalized_answer for token in any_of
    ):
        raise AssertionError(f"Answer contains none of {any_of}: {answer[:500]!r}")
    leaked = [token for token in none_of or [] if token.lower() in answer]
    if leaked:
        raise AssertionError(
            f"Answer leaked forbidden tokens {leaked}: {answer[:500]!r}"
        )


async def _assert_single_seeded_fk_lineage(agent) -> dict:
    from daita.core.graph.models import (
        AgentGraphEdge,
        AgentGraphNode,
        EdgeType,
        NodeType,
    )
    from daita.plugins.catalog.persistence import _derive_store

    store = _derive_store(agent.db.schema)
    expected_source = AgentGraphNode.make_id(NodeType.TABLE, f"{store}.orders")
    expected_target = AgentGraphNode.make_id(NodeType.TABLE, f"{store}.customers")
    expected_edge_id = AgentGraphEdge.make_id(
        expected_source,
        EdgeType.REFERENCES,
        expected_target,
    )

    backend = agent.db.lineage._graph_backend
    edges = await backend.get_edges(edge_types=[EdgeType.REFERENCES])
    matching = [
        edge
        for edge in edges
        if edge.from_node_id == expected_source and edge.to_node_id == expected_target
    ]

    assert len(matching) == 1
    assert matching[0].edge_id == expected_edge_id
    assert matching[0].properties["flow_type"] == "references"
    assert matching[0].properties["transformation"] == "customer_id → customer_id"
    assert "postgresql:" in matching[0].from_node_id
    assert "public.orders" in matching[0].from_node_id
    assert "public.customers" in matching[0].to_node_id

    lineage = await agent.db.lineage.trace_lineage(
        expected_source,
        direction="downstream",
        edge_types=[EdgeType.REFERENCES],
        max_depth=2,
    )
    assert lineage["downstream_count"] == 1
    assert lineage["lineage"]["downstream"][0]["node_id"] == expected_target

    return {
        "store": store,
        "source_id": expected_source,
        "target_id": expected_target,
        "edge_id": expected_edge_id,
        "edge_count": len(edges),
    }


def _live_accuracy_metrics(result: dict, case: dict, elapsed_ms: float) -> dict:
    tool_call = _assert_any_tool_called(result, set(case["expected_tools"]))
    _assert_live_tool_result(case, tool_call)
    _assert_answer_tokens(
        result,
        all_of=case.get("answer_all_of"),
        any_of=case.get("answer_any_of"),
        none_of=case.get("answer_none_of"),
    )
    return {
        "case_id": case["id"],
        "tool": tool_call.get("tool"),
        "tool_arguments": tool_call.get("arguments"),
        "tool_duration_ms": tool_call.get("duration_ms"),
        "final_answer": result.get("result"),
        "accurate": True,
        "query_duration_ms": tool_call.get("duration_ms"),
        "run_duration_ms": round(elapsed_ms, 1),
        "tokens": result.get("tokens"),
        "cost": result.get("cost"),
        "iterations": result.get("iterations"),
        "audit_entry_count": case.get("audit_entry_count"),
    }


def _assert_live_tool_result(case: dict, tool_call: dict) -> None:
    tool_result = tool_call.get("result") or {}
    case_id = case["id"]

    if case_id == "simple_aggregate":
        rows = tool_result.get("rows") or []
        assert rows, f"Aggregate tool returned no rows: {tool_result}"
        assert float(rows[0]["total_revenue"]) == 125.0
    elif case_id == "relationship_customers_with_orders":
        rows = tool_result.get("rows") or []
        names = {str(row.get("name")).lower() for row in rows}
        assert {"alice", "bob"}.issubset(names), tool_result
        assert "cara" not in names, tool_result
    elif case_id == "freshness_orders_created_at":
        assert tool_result.get("success") is True, tool_result
        assert tool_result.get("is_fresh") is False, tool_result
    elif case_id == "anomaly_daily_revenue":
        assert tool_result.get("success") is True, tool_result
        assert tool_result.get("anomaly_count", 0) >= 1, tool_result
        assert "1000" in json.dumps(tool_result, default=str), tool_result
    elif case_id == "blocked_email_column":
        error = str(tool_result.get("error") or "").lower()
        assert "blocked column" in error, tool_result


def _fresh_cache_file(source) -> Path:
    from daita.agents.db.schema.cache import cache_key

    return Path(".daita") / "schema_cache" / f"{cache_key(source)}.json"


def _expire_and_mutate_cache(source, mutate_schema) -> None:
    cache_file = _fresh_cache_file(source)
    payload = json.loads(cache_file.read_text())
    payload["schema"] = mutate_schema(payload["schema"])
    payload["cached_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    cache_file.write_text(json.dumps(payload, indent=2, default=str))


async def _assert_data_quality_tools(agent) -> None:
    profile = await agent.tool_registry.execute(
        "dq_profile",
        {"table": "orders", "columns": ["total_amount", "status"]},
    )
    assert profile["success"] is True
    assert profile["profile"]["total_amount"]["total_rows"] == 3
    assert profile["profile"]["total_amount"]["null_rate"] == 0
    assert profile["profile"]["status"]["distinct_count"] == 2

    report = await agent.tool_registry.execute("dq_report", {"table": "customers"})
    assert report["success"] is True
    assert report["completeness_score"] == 1.0
    assert report["profile"]["email"]["null_rate"] == 0

    anomaly = await agent.tool_registry.execute(
        "dq_detect_anomaly",
        {"table": "daily_metrics", "column": "revenue", "method": "iqr"},
    )
    assert anomaly["success"] is True
    assert anomaly["anomaly_count"] >= 1
    assert 1000.0 in anomaly["anomaly_values"]

    fresh = await agent.tool_registry.execute(
        "dq_check_freshness",
        {
            "table": "orders",
            "timestamp_column": "created_at",
            "expected_interval_hours": 3000,
        },
    )
    assert fresh["success"] is True
    assert fresh["is_fresh"] is True

    stale = await agent.tool_registry.execute(
        "dq_check_freshness",
        {
            "table": "orders",
            "timestamp_column": "created_at",
            "expected_interval_hours": 24,
        },
    )
    assert stale["success"] is True
    assert stale["is_fresh"] is False


async def _assert_analyst_tools(agent) -> None:
    pytest.importorskip("pandas", reason="pandas required for analyst integration")
    pytest.importorskip("numpy", reason="numpy required for analyst integration")

    pivot = await agent.tool_registry.execute(
        "pivot_table",
        {
            "sql": (
                "SELECT customer_id, status, total_amount FROM orders "
                "ORDER BY customer_id, status"
            ),
            "rows": "customer_id",
            "columns": "status",
            "values": "total_amount",
            "aggfunc": "sum",
        },
    )
    assert pivot["success"] is True
    assert pivot["dimensions"]["values"] == "total_amount"
    assert pivot["row_count"] == 2

    correlations = await agent.tool_registry.execute(
        "correlate",
        {
            "sql": "SELECT orders_count, sessions FROM daily_metrics",
            "columns": ["orders_count", "sessions"],
            "min_correlation": 0.99,
        },
    )
    assert correlations["success"] is True
    assert correlations["sample_size"] == 6
    assert correlations["correlations"][0]["correlation"] == 1.0

    anomalies = await agent.tool_registry.execute(
        "detect_anomalies",
        {
            "sql": "SELECT metric_date, revenue FROM daily_metrics",
            "column": "revenue",
            "method": "iqr",
        },
    )
    assert anomalies["success"] is True
    assert anomalies["anomaly_count"] >= 1
    assert anomalies["anomalies"][0]["revenue"] == 1000.0

    forecast = await agent.tool_registry.execute(
        "forecast_trend",
        {
            "sql": (
                "SELECT metric_date, orders_count FROM daily_metrics "
                "ORDER BY metric_date"
            ),
            "date_column": "metric_date",
            "metric_column": "orders_count",
            "periods": 2,
        },
    )
    assert forecast["success"] is True
    assert forecast["trend"]["direction"] == "up"
    assert forecast["trend"]["frequency"] == "daily"
    assert len(forecast["forecast"]) == 2

    comparison = await agent.tool_registry.execute(
        "compare_entities",
        {
            "entity_table": "customers",
            "entity_ids": [1, 2],
            "id_column": "customer_id",
            "dimensions": [
                {
                    "expression": "SUM(c.total_amount)",
                    "alias": "orders_total_amount_sum",
                    "child_table": "orders",
                    "fk_col": "customer_id",
                },
                {
                    "expression": "COUNT(c.order_id)",
                    "alias": "orders_count",
                    "child_table": "orders",
                    "fk_col": "customer_id",
                },
            ],
        },
    )
    assert comparison["success"] is True
    assert comparison["entities"] == [1, 2]
    assert comparison["biggest_differences"]

    similar = await agent.tool_registry.execute(
        "find_similar",
        {
            "entity_table": "customers",
            "entity_id": 1,
            "id_column": "customer_id",
            "dimensions": [
                {
                    "expression": "SUM(c.total_amount)",
                    "alias": "orders_total_amount_sum",
                    "child_table": "orders",
                    "fk_col": "customer_id",
                }
            ],
            "top_k": 2,
        },
    )
    assert similar["success"] is True
    assert similar["reference"]["id"] == 1
    assert similar["similar"]

    bad = await agent.tool_registry.execute(
        "forecast_trend",
        {
            "sql": "SELECT metric_date, revenue FROM daily_metrics",
            "date_column": "metric_date",
            "metric_column": "missing_revenue",
        },
    )
    assert bad["success"] is False
    assert "missing_revenue" in bad["error"]


async def _assert_sql_safety(agent, *, tool_name: str, param_sql: str) -> None:
    limited = await agent.tool_registry.execute(
        tool_name, {"sql": "SELECT order_id FROM orders ORDER BY order_id"}
    )
    assert limited["sql"].endswith("LIMIT 2")
    assert limited["total_rows"] == 2

    max_rows = await agent.tool_registry.execute(
        tool_name,
        {
            "sql": (
                "SELECT order_id, total_amount FROM orders "
                "ORDER BY order_id LIMIT 10"
            )
        },
    )
    assert max_rows["total_rows"] == 3
    assert len(max_rows["rows"]) == 2
    assert max_rows["truncated"] is True

    max_chars = await agent.tool_registry.execute(
        tool_name,
        {
            "sql": (
                "SELECT status || ':' || total_amount || ':' || created_at AS payload "
                "FROM orders ORDER BY order_id"
            )
        },
    )
    assert max_chars["total_rows"] == 2
    assert max_chars["truncated"] is True
    assert len(json.dumps(max_chars["rows"], default=str)) <= 80

    parameterized = await agent.tool_registry.execute(
        tool_name, {"sql": param_sql, "params": [1]}
    )
    assert (
        parameterized["rows"][0]["revenue"] == 125
        or float(parameterized["rows"][0]["revenue"]) == 125.0
    )

    for sql, message in [
        ("SELECT * FROM customers", "blocked table"),
        ("SELECT email FROM orders", "blocked column"),
        ("SELECT * FROM daily_metrics", "outside allowlist"),
        ("DELETE FROM orders WHERE order_id = 1", "non-read query|read-only mode"),
        (
            "WITH deleted AS (DELETE FROM orders WHERE order_id = 1 RETURNING *) "
            "SELECT * FROM deleted",
            "read-only mode",
        ),
        ("EXPLAIN ANALYZE DELETE FROM orders WHERE order_id = 1", "read-only mode"),
    ]:
        with pytest.raises(Exception, match=message):
            await agent.tool_registry.execute(tool_name, {"sql": sql})


async def _assert_local_monitor_findings(agent) -> None:
    monitor = {
        "name": "orders non-empty",
        "type": "row_count",
        "severity": "info",
        "entity": {"table": "orders"},
        "sql": "SELECT COUNT(*) AS row_count FROM orders",
        "threshold": {"min_rows": 4},
        "interval": "1h",
    }
    registered = agent.db.register_monitors([monitor])
    assert registered[0]["watch_name"] == "db:orders_non_empty"

    handler_calls = []

    async def custom_handler(event, monitor_dict):
        handler_calls.append({"event": event, "monitor": monitor_dict})

    duplicate_registered = agent.db.register_monitors(
        [monitor, monitor], handler=custom_handler
    )
    assert duplicate_registered[0]["watch_name"] == "db:orders_non_empty_2"
    assert duplicate_registered[1]["watch_name"] == "db:orders_non_empty_3"

    opened = WatchEvent(
        value=3,
        previous_value=5,
        triggered_at=datetime.now(timezone.utc),
        source_type="polling",
    )
    resolved = WatchEvent(
        value=5,
        previous_value=3,
        triggered_at=datetime.now(timezone.utc),
        source_type="polling",
        resolved=True,
    )

    await agent._watches[-1].handler(opened)
    opened_id = agent.db.findings.open[0]["id"]
    await agent._watches[-1].handler(resolved)

    assert agent.db.findings.open == []
    assert agent.db.findings.resolved[0]["id"] == opened_id
    assert agent.db.findings.resolved[0]["observed"]["resolved"] is True
    assert len(agent.db.monitor_events) == 2
    assert len(handler_calls) == 2
    json.dumps(agent.db.findings.resolved[0], default=str)
    assert agent.db.monitor_events[0]["finding_id"] == opened_id


@pytest.mark.integration
class TestFromDbCoreBuildMatrix:
    @pytest.mark.parametrize("mode", list(CORE_BUILD_MODES))
    async def test_sqlite_path_core_build_matrix(
        self, sqlite_db_path, mode, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            name=f"matrix-sqlite-path-{mode}",
            llm_provider=MockLLMProvider(delay=0),
            mode=mode,
            cache_ttl=None,
            query_default_limit=20,
        )
        try:
            result = await _core_build_matrix_result(
                agent,
                source_kind="sqlite_path",
                mode=mode,
            )
            _write_core_build_matrix_result(tmp_path, result)
            _assert_core_build_matrix_result(result)
        finally:
            await _close_agent_db(agent)

    @pytest.mark.parametrize("mode", list(CORE_BUILD_MODES))
    async def test_sqlite_plugin_instance_core_build_matrix(
        self, sqlite_db_path, mode, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        plugin = SQLitePlugin(path=str(sqlite_db_path), wal_mode=False)
        agent = await Agent.from_db(
            plugin,
            name=f"matrix-sqlite-plugin-{mode}",
            llm_provider=MockLLMProvider(delay=0),
            mode=mode,
            cache_ttl=None,
            query_default_limit=20,
        )
        try:
            result = await _core_build_matrix_result(
                agent,
                source_kind="sqlite_plugin",
                mode=mode,
            )
            _write_core_build_matrix_result(tmp_path, result)
            _assert_core_build_matrix_result(result)
        finally:
            await _close_agent_db(agent)

    @pytest.mark.requires_db
    @pytest.mark.parametrize("mode", list(CORE_BUILD_MODES))
    async def test_postgres_connection_string_core_build_matrix(
        self, seeded_postgres, mode, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_postgres,
            name=f"matrix-postgres-connection-{mode}",
            llm_provider=MockLLMProvider(delay=0),
            mode=mode,
            cache_ttl=None,
            query_default_limit=20,
        )
        try:
            result = await _core_build_matrix_result(
                agent,
                source_kind="postgres_connection",
                mode=mode,
            )
            _write_core_build_matrix_result(tmp_path, result)
            _assert_core_build_matrix_result(result)
        finally:
            await _close_agent_db(agent)


@pytest.mark.integration
class TestFromDbSQLiteLive:
    async def test_from_db_builds_sqlite_agent_and_executes_real_queries(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            name="sqlite-from-db-live",
            llm_provider=MockLLMProvider(delay=0),
            mode="data_team",
            cache_ttl=3600,
            query_default_limit=10,
        )
        try:
            description = agent.describe()
            assert description["kind"] == "database"
            assert description["db"]["database_type"] == "sqlite"
            assert description["db"]["table_count"] == 3
            assert description["db"]["mode"] == "data_team"
            assert description["db"]["quality_enabled"] is True
            assert "data_quality" in description["capabilities"]
            assert "db_query" in agent.tool_registry.tool_names
            assert "db_count" in agent.tool_registry.tool_names
            assert "sqlite_list_tables" not in agent.tool_registry.tool_names
            assert "sqlite_inspect" not in agent.tool_registry.tool_names
            assert agent.db.summary["candidate_metrics"]
            assert agent.db.suggested_questions

            query_result = await agent.tool_registry.execute(
                "db_query",
                {
                    "sql": (
                        "SELECT customer_id, SUM(total_amount) AS revenue "
                        "FROM orders GROUP BY customer_id ORDER BY revenue DESC"
                    )
                },
            )
            assert query_result["sql"].endswith("LIMIT 10")
            assert query_result["rows"][0]["customer_id"] == 2
            assert query_result["rows"][0]["revenue"] == 150.0

            freshness = await agent.tool_registry.execute(
                "dq_check_freshness",
                {
                    "table": "orders",
                    "timestamp_column": "created_at",
                    "expected_interval_hours": 24,
                },
            )
            assert freshness["success"] is True
            assert freshness["is_fresh"] is False

            with pytest.raises(Exception, match="read-only mode|non-read query"):
                await agent.tool_registry.execute(
                    "db_query", {"sql": "DELETE FROM orders WHERE order_id = 1"}
                )
        finally:
            await _close_agent_db(agent)

    async def test_from_db_sqlite_quality_analyst_and_monitor_systems(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            name="sqlite-from-db-systems",
            llm_provider=MockLLMProvider(delay=0),
            mode="data_team",
            cache_ttl=3600,
            query_default_limit=20,
        )
        try:
            await _assert_data_quality_tools(agent)
            await _assert_analyst_tools(agent)
            await _assert_local_monitor_findings(agent)
        finally:
            await _close_agent_db(agent)

    async def test_from_db_sqlite_analyst_tools_use_db_query_policy(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            name="sqlite-from-db-policy",
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            blocked_columns=["email"],
            cache_ttl=3600,
        )
        try:
            blocked = await agent.tool_registry.execute(
                "pivot_table",
                {
                    "sql": "SELECT customer_id, email, 1 AS value FROM customers",
                    "rows": "customer_id",
                    "columns": "email",
                    "values": "value",
                },
            )
            assert blocked["success"] is False
            assert "blocked column" in blocked["error"]

            mutating = await agent.tool_registry.execute(
                "correlate",
                {
                    "sql": (
                        "WITH deleted AS (DELETE FROM orders WHERE order_id = 1 "
                        "RETURNING total_amount) SELECT total_amount FROM deleted"
                    )
                },
            )
            assert mutating["success"] is False
            assert "read-only mode" in mutating["error"]
        finally:
            await _close_agent_db(agent)

    async def test_from_db_sqlite_query_tool_ignores_focus_argument(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            name="sqlite-from-db-no-focus",
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=None,
            focus="name",
        )
        try:
            assert agent.default_focus is None
            tool = agent.tool_registry.get("db_query")
            assert "focus" not in tool.parameters["properties"]

            result = await agent.tool_registry.execute(
                "db_query",
                {
                    "sql": (
                        "SELECT DISTINCT c.name AS name FROM customers c "
                        "JOIN orders o ON c.customer_id = o.customer_id "
                        "ORDER BY c.name"
                    ),
                    "focus": "name",
                },
            )
            assert result["rows"] == [{"name": "Alice"}, {"name": "Bob"}]
        finally:
            await _close_agent_db(agent)

    async def test_from_db_sqlite_schema_cache_skips_rediscovery(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=3600,
        )
        await _close_agent_db(agent)

        import daita.agents.db.builder as builder

        async def fail_discovery(*args, **kwargs):
            raise AssertionError("schema discovery should not run on fresh cache")

        monkeypatch.setattr(builder, "discover_schema", fail_discovery)
        cached_agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=3600,
        )
        try:
            assert cached_agent.describe()["db"]["table_count"] == 3
        finally:
            await _close_agent_db(cached_agent)


@pytest.mark.integration
class TestFromDbCacheAndDrift:
    async def test_expired_sqlite_cache_rediscover_reports_table_and_column_drift(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=3600,
        )
        await _close_agent_db(agent)

        def mutate_cached_baseline(schema):
            schema = json.loads(json.dumps(schema))
            schema["tables"] = [
                table for table in schema["tables"] if table["name"] != "daily_metrics"
            ]
            for table in schema["tables"]:
                if table["name"] == "orders":
                    table["columns"] = [
                        col for col in table["columns"] if col["name"] != "status"
                    ]
                    table["columns"].append(
                        {
                            "name": "legacy_note",
                            "type": "TEXT",
                            "nullable": True,
                            "is_primary_key": False,
                        }
                    )
            schema["tables"].append(
                {
                    "name": "legacy_table",
                    "type": "table",
                    "columns": [
                        {
                            "name": "legacy_id",
                            "type": "INTEGER",
                            "nullable": False,
                            "is_primary_key": True,
                        }
                    ],
                    "row_count": 0,
                }
            )
            return schema

        _expire_and_mutate_cache(str(sqlite_db_path), mutate_cached_baseline)

        drifted_agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=1,
        )
        try:
            assert drifted_agent.db.drift["added_tables"] == ["daily_metrics"]
            assert drifted_agent.db.drift["removed_tables"] == ["legacy_table"]
            orders_change = next(
                item
                for item in drifted_agent.db.drift["column_changes"]
                if item["table"] == "orders"
            )
            assert orders_change["added_columns"] == ["status"]
            assert orders_change["removed_columns"] == ["legacy_note"]

            result = await drifted_agent.tool_registry.execute(
                "db_query",
                {"sql": "SELECT COUNT(*) AS order_count FROM orders"},
            )
            assert result["rows"][0]["order_count"] == 3
        finally:
            await _close_agent_db(drifted_agent)

    async def test_expired_sqlite_cache_used_when_discovery_fails(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        source = str(sqlite_db_path)
        agent = await Agent.from_db(
            source,
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=3600,
        )
        await _close_agent_db(agent)
        _expire_and_mutate_cache(source, lambda schema: schema)

        import daita.agents.db.builder as builder

        async def fail_discovery(*args, **kwargs):
            raise RuntimeError("discovery unavailable")

        monkeypatch.setattr(builder, "discover_schema", fail_discovery)
        cached_agent = await Agent.from_db(
            source,
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=1,
        )
        try:
            assert cached_agent.describe()["db"]["table_count"] == 3
            assert cached_agent.db.drift is None
            result = await cached_agent.tool_registry.execute(
                "db_query",
                {"sql": "SELECT SUM(total_amount) AS revenue FROM orders"},
            )
            assert result["rows"][0]["revenue"] == 275.0
        finally:
            await _close_agent_db(cached_agent)


@pytest.mark.integration
class TestFromDbSQLSafety:
    async def test_sqlite_query_guardrails_on_seeded_database(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=None,
            query_default_limit=2,
            query_max_rows=2,
            query_max_chars=80,
            allowed_tables=["orders"],
            blocked_tables=["customers"],
            blocked_columns=["email"],
        )
        try:
            await _assert_sql_safety(
                agent,
                tool_name="db_query",
                param_sql=(
                    "SELECT SUM(total_amount) AS revenue FROM orders "
                    "WHERE customer_id = ?"
                ),
            )
        finally:
            await _close_agent_db(agent)

    @pytest.mark.requires_db
    async def test_postgres_query_guardrails_on_seeded_database(
        self, seeded_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_postgres,
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=None,
            query_default_limit=2,
            query_max_rows=2,
            query_max_chars=80,
            allowed_tables=["orders"],
            blocked_tables=["customers"],
            blocked_columns=["email"],
        )
        try:
            await _assert_sql_safety(
                agent,
                tool_name="db_query",
                param_sql=(
                    "SELECT SUM(total_amount) AS revenue FROM orders "
                    "WHERE customer_id = $1"
                ),
            )
        finally:
            await _close_agent_db(agent)


@pytest.mark.integration
class TestFromDbLargeGuardrails:
    async def test_large_sqlite_schema_uses_summary_prompt_and_bounded_runtime_context(
        self, large_guardrail_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        llm = MockLLMProvider(delay=0)
        agent = await Agent.from_db(
            str(large_guardrail_sqlite_db_path),
            llm_provider=llm,
            mode="analyst",
            cache_ttl=None,
            include_sample_values=True,
            query_default_limit=5,
        )
        try:
            prompt = agent.prompt
            assert "## Database Schema (91 tables)" in prompt
            assert "| Column | Type | PK | Nullable |" not in prompt
            assert "Columns:" not in prompt
            assert "long_dimension_name_00" not in prompt
            assert "- table_00" in prompt
            assert "- bulky_events" in prompt

            await agent.run("What tables are available?", max_iterations=1)
            user_message = llm.call_history[-1]["messages"][-1]["content"]
            runtime_context = user_message.split("User question:", 1)[0]
            assert len(runtime_context) <= 1800
            assert "tables=91" in runtime_context
            assert user_message.count("<db_runtime_context>") == 1
            assert "CREATE TABLE" not in user_message

            assert "db_search_schema" in agent.tool_registry.tool_names
            assert "db_inspect_table" in agent.tool_registry.tool_names

            query_result = await agent.tool_registry.execute(
                "db_query",
                {"sql": "SELECT COUNT(*) AS event_count FROM bulky_events"},
            )
            assert query_result["rows"][0]["event_count"] == 120
            assert query_result["sql"].endswith("LIMIT 5")
        finally:
            await _close_agent_db(agent)

    async def test_large_sqlite_result_set_is_row_and_character_bounded(
        self, large_guardrail_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(large_guardrail_sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=None,
            query_default_limit=100,
            query_max_rows=12,
            query_max_chars=900,
        )
        try:
            result = await agent.tool_registry.execute(
                "db_query",
                {
                    "sql": (
                        "SELECT event_id, account_id, payload, created_at "
                        "FROM bulky_events ORDER BY event_id LIMIT 120"
                    )
                },
            )
            serialized_rows = json.dumps(result["rows"], default=str)

            assert result["total_rows"] == 120
            assert result["truncated"] is True
            assert len(result["rows"]) <= 12
            assert len(serialized_rows) <= 900
            assert "payload-001-" in serialized_rows
            assert "payload-120-" not in serialized_rows
        finally:
            await _close_agent_db(agent)

    async def test_thousand_table_wide_schema_prompt_is_capped_and_queryable(
        self, warehouse_scale_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        llm = MockLLMProvider(delay=0)
        agent = await Agent.from_db(
            str(warehouse_scale_sqlite_db_path),
            llm_provider=llm,
            mode="analyst",
            cache_ttl=None,
            include_sample_values=False,
            query_default_limit=5,
            query_max_rows=5,
            query_max_chars=1200,
        )
        try:
            description = agent.describe()
            prompt = agent.prompt

            assert description["db"]["table_count"] == 1001
            assert description["db"]["column_count"] == 4254
            assert "## Database Schema (1001 tables)" in prompt
            assert "| Column | Type | PK | Nullable |" not in prompt
            assert "Columns:" not in prompt
            assert prompt.count("- warehouse_table_") == 30
            assert "- ... 971 additional tables omitted from prompt summary" in prompt
            assert "wide_fact_events" not in prompt
            assert "feature_249" not in prompt
            assert len(prompt) < 20000

            await agent.run("What can you tell me about this schema?", max_iterations=1)
            user_message = llm.call_history[-1]["messages"][-1]["content"]
            runtime_context = user_message.split("User question:", 1)[0]
            assert len(runtime_context) <= 1800
            assert "tables=1001" in runtime_context
            assert "columns=4254" in runtime_context
            assert "feature_249" not in user_message

            result = await agent.tool_registry.execute(
                "db_query",
                {
                    "sql": (
                        "SELECT event_id, feature_000, feature_249, event_value "
                        "FROM wide_fact_events ORDER BY event_id"
                    )
                },
            )
            assert result["sql"].endswith("LIMIT 5")
            assert result["total_rows"] == 2
            assert result["truncated"] is False
            assert result["rows"][0]["feature_249"] == "channel-web"
        finally:
            await _close_agent_db(agent)

    async def test_schema_navigation_finds_omitted_and_wide_tables_without_rows(
        self, warehouse_scale_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(warehouse_scale_sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=None,
            blocked_columns=["feature_249"],
        )
        try:
            assert "db_list_tables" in agent.tool_registry.tool_names
            assert "db_search_schema" in agent.tool_registry.tool_names
            assert "db_inspect_table" in agent.tool_registry.tool_names
            assert "db_describe_relationships" in agent.tool_registry.tool_names

            listed = await agent.tool_registry.execute(
                "db_list_tables",
                {"pattern": "wide", "limit": 10},
            )
            assert listed["total_matches"] == 1
            assert listed["tables"][0] == {
                "name": "wide_fact_events",
                "row_count": None,
                "column_count": 254,
            }

            searched = await agent.tool_registry.execute(
                "db_search_schema",
                {"query": "feature 249 event value", "limit": 5},
            )
            assert searched["tables"][0]["name"] == "wide_fact_events"
            matched_columns = {
                column["name"] for column in searched["tables"][0]["matched_columns"]
            }
            assert {"feature_249", "event_value"}.issubset(matched_columns)

            inspected = await agent.tool_registry.execute(
                "db_inspect_table",
                {
                    "table_name": "wide_fact_events",
                    "column_pattern": "feature_24",
                    "limit": 20,
                },
            )
            assert inspected["success"] is True
            assert inspected["column_count"] == 254
            assert inspected["matched_column_count"] >= 10
            assert inspected["truncated"] is False
            assert not any("_samples" in column for column in inspected["columns"])
            assert any(
                column["name"] == "feature_249" and column["blocked_by_policy"] is True
                for column in inspected["columns"]
            )
            assert "segment-a" not in json.dumps(inspected, default=str)
            assert "channel-web" not in json.dumps(inspected, default=str)

            missing = await agent.tool_registry.execute(
                "db_inspect_table",
                {"table_name": "wide_fact"},
            )
            assert missing["success"] is False
            assert missing["candidates"][0]["name"] == "wide_fact_events"
        finally:
            await _close_agent_db(agent)

    async def test_wide_table_inspection_paginates_columns(
        self, warehouse_scale_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(warehouse_scale_sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=None,
        )
        try:
            first_page = await agent.tool_registry.execute(
                "db_inspect_table",
                {"table_name": "wide_fact_events", "limit": 100, "offset": 0},
            )
            second_page = await agent.tool_registry.execute(
                "db_inspect_table",
                {"table_name": "wide_fact_events", "limit": 100, "offset": 100},
            )
            third_page = await agent.tool_registry.execute(
                "db_inspect_table",
                {"table_name": "wide_fact_events", "limit": 100, "offset": 200},
            )

            assert first_page["success"] is True
            assert first_page["column_count"] == 254
            assert len(first_page["columns"]) == 100
            assert first_page["truncated"] is True
            assert first_page["columns"][0]["name"] == "event_id"
            assert first_page["columns"][-1]["name"] == "feature_097"

            assert len(second_page["columns"]) == 100
            assert second_page["truncated"] is True
            assert second_page["columns"][0]["name"] == "feature_098"
            assert second_page["columns"][-1]["name"] == "feature_197"

            assert len(third_page["columns"]) == 54
            assert third_page["truncated"] is False
            assert third_page["columns"][0]["name"] == "feature_198"
            assert third_page["columns"][-1]["name"] == "created_at"

            all_seen = {
                column["name"]
                for page in (first_page, second_page, third_page)
                for column in page["columns"]
            }
            assert len(all_seen) == 254
            assert "feature_249" in all_seen
            assert "event_value" in all_seen
        finally:
            await _close_agent_db(agent)

    async def test_schema_navigation_recovers_from_bad_table_and_column_names(
        self, warehouse_scale_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(warehouse_scale_sqlite_db_path),
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=None,
        )
        try:
            missing_table = await agent.tool_registry.execute(
                "db_inspect_table",
                {"table_name": "wide_eventz"},
            )
            assert missing_table["success"] is False
            assert missing_table["candidates"][0]["name"] == "wide_fact_events"

            inspected = await agent.tool_registry.execute(
                "db_inspect_table",
                {
                    "table_name": "wide_fact_events",
                    "column_pattern": "feat_249",
                    "limit": 20,
                },
            )
            assert inspected["success"] is True
            assert inspected["matched_column_count"] == 1
            assert inspected["columns"][0]["name"] == "feature_249"

            recovered = await agent.tool_registry.execute(
                "db_search_schema",
                {"query": "feature 249 wide events", "limit": 5},
            )
            assert recovered["tables"][0]["name"] == "wide_fact_events"
            assert any(
                column["name"] == "feature_249"
                for column in recovered["tables"][0]["matched_columns"]
            )
        finally:
            await _close_agent_db(agent)


@pytest.mark.integration
class TestFromDbAuditAndPrivacy:
    async def test_run_audit_records_original_prompt_and_sanitized_tool_call(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        prompt = "Look up the safe display name for customer 1."
        llm = ScriptedToolLLMProvider(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "db_query",
                            "arguments": {
                                "sql": (
                                    "SELECT name FROM customers "
                                    "WHERE customer_id = ? AND email = ?"
                                ),
                                "params": [1, "alice@example.com"],
                            },
                        }
                    ],
                },
                {"content": "Customer 1 is Alice.", "tool_calls": None},
            ]
        )
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=llm,
            cache_ttl=None,
            query_default_limit=10,
        )
        try:
            result = await agent.run(prompt, detailed=True, max_iterations=3)
            assert result["result"] == "Customer 1 is Alice."

            audit = agent.db.audit.last()
            assert audit["prompt"] == prompt
            assert "<db_runtime_context>" not in audit["prompt"]
            call = audit["tool_calls"][0]
            assert call["tool"] == "db_query"
            assert call["arguments"]["sql"].startswith("SELECT name FROM customers")
            assert call["arguments"]["param_count"] == 2
            assert "params" not in call["arguments"]
            assert call["result"]["total_rows"] == 1
            assert call["result"]["row_count"] == 1
            assert "rows" not in call["result"]

            audit_json = agent.db.audit.export_json()
            assert "alice@example.com" not in audit_json
            assert '"name": "Alice"' not in audit_json
        finally:
            await _close_agent_db(agent)

    async def test_failed_tool_call_audit_records_sanitized_error(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        llm = ScriptedToolLLMProvider(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "db_query",
                            "arguments": {
                                "sql": "SELECT email FROM customers WHERE email = ?",
                                "params": ["alice@example.com"],
                            },
                        }
                    ],
                },
                {"content": "The request is blocked.", "tool_calls": None},
            ]
        )
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=llm,
            cache_ttl=None,
            blocked_columns=["email"],
        )
        try:
            await agent.run("show customer emails", detailed=True, max_iterations=3)
            audit = agent.db.audit.last()
            call = audit["tool_calls"][0]
            assert call["result"]["error"].startswith("Tool 'db_query' failed")
            assert "blocked column" in call["result"]["error"]
            assert call["arguments"]["param_count"] == 1
            assert "alice@example.com" not in agent.db.audit.export_json()
        finally:
            await _close_agent_db(agent)

    async def test_stream_audit_matches_run_audit_privacy_shape(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        llm = ScriptedToolLLMProvider(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "db_query",
                            "arguments": {
                                "sql": "SELECT name FROM customers WHERE customer_id = ?",
                                "params": [1],
                            },
                        }
                    ],
                },
                {"content": "Customer 1 is Alice.", "tool_calls": None},
            ]
        )
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=llm,
            cache_ttl=None,
        )
        try:
            events = [
                event
                async for event in agent.stream(
                    "stream the safe customer name", max_iterations=3
                )
            ]
            assert any(event.tool_name == "db_query" for event in events)
            audit = agent.db.audit.last()
            assert audit["prompt"] == "stream the safe customer name"
            assert audit["tool_calls"][0]["result"]["row_count"] == 1
            assert "rows" not in audit["tool_calls"][0]["result"]
        finally:
            await _close_agent_db(agent)


@pytest.mark.integration
class TestFromDbRuntimeContextAndHistory:
    async def test_compact_runtime_context_is_added_and_bounded(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        llm = MockLLMProvider(delay=0)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=llm,
            cache_ttl=None,
            query_default_limit=20,
        )
        try:
            await agent.run("What tables are available?", max_iterations=1)
            user_message = llm.call_history[-1]["messages"][-1]["content"]
            runtime_context = user_message.split("User question:", 1)[0]

            assert "<db_runtime_context>" in runtime_context
            assert "tables=3" in runtime_context
            assert "Candidate metrics:" in runtime_context
            assert len(runtime_context) <= 1800
            assert user_message.count("<db_runtime_context>") == 1

            await agent.run("What candidate metrics exist?", max_iterations=1)
            second_user_message = llm.call_history[-1]["messages"][-1]["content"]
            assert second_user_message.count("<db_runtime_context>") == 1
            assert "CREATE TABLE" not in second_user_message
        finally:
            await _close_agent_db(agent)

    async def test_history_auto_injection_reuses_object_and_allows_override(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        llm = MockLLMProvider(delay=0)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=llm,
            cache_ttl=None,
            history=True,
        )
        try:
            history = agent.db.history
            assert history is not None
            await agent.run("First DB question", max_iterations=1)
            assert agent.db.history is history
            assert history.turn_count == 1

            await agent.run("Follow up using the prior answer", max_iterations=1)
            assert agent.db.history is history
            assert history.turn_count == 2

            await agent.run("Do not store this turn", history=None, max_iterations=1)
            assert history.turn_count == 2
            assert agent.db.audit.last()["prompt"] == "Do not store this turn"
        finally:
            await _close_agent_db(agent)

    async def test_generic_agent_run_is_not_db_augmented(self):
        llm = MockLLMProvider(delay=0)
        agent = Agent(name="plain-agent", llm_provider=llm)

        await agent.run("Hello", max_iterations=1)

        assert not hasattr(agent, "db")
        user_message = llm.call_history[-1]["messages"][-1]["content"]
        assert "<db_runtime_context>" not in user_message

    async def test_local_memory_semantics_persist_and_skip_row_level_lookup(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        workspace = "from_db_memory_semantics"
        snippet = "Revenue excludes refunded orders."

        writer_llm = MockLLMProvider(delay=0)
        writer_embedder = KeywordEmbeddingProvider()
        writer_memory = MemoryPlugin(
            workspace=workspace,
            auto_curate="manual",
            embedder=writer_embedder,
        )
        writer_agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=writer_llm,
            cache_ttl=None,
            memory=writer_memory,
        )
        try:
            await writer_agent.db.memory_semantics.remember(
                DBMemoryRecord(
                    kind="business_rule",
                    key="business_rule:revenue_refunds",
                    text=snippet,
                    metadata={"metric": "revenue"},
                    importance=0.8,
                )
            )
        finally:
            await _close_agent_db(writer_agent)

        reader_llm = MockLLMProvider(delay=0)
        reader_embedder = KeywordEmbeddingProvider()
        reader_memory = MemoryPlugin(
            workspace=workspace,
            auto_curate="manual",
            embedder=reader_embedder,
        )
        reader_agent = await Agent.from_db(
            str(sqlite_db_path),
            llm_provider=reader_llm,
            cache_ttl=None,
            memory=reader_memory,
        )
        try:
            reader_llm.call_history.clear()
            reader_embedder.embedded_texts.clear()

            await reader_agent.run(
                "Does revenue exclude refunded orders when calculated?",
                max_iterations=1,
            )
            user_message = reader_llm.call_history[-1]["messages"][-1]["content"]
            runtime_context = user_message.split("User question:", 1)[0]

            assert "<db_runtime_context>" in runtime_context
            assert snippet in runtime_context

            reader_llm.call_history.clear()
            reader_embedder.embedded_texts.clear()

            await reader_agent.run(
                "Show me the email for customer_id 1", max_iterations=1
            )
            row_level_messages = reader_llm.call_history[-1]["messages"]
            row_level_prompt = "\n".join(
                str(message.get("content", "")) for message in row_level_messages
            )
            row_level_user_message = row_level_messages[-1]["content"]
            row_level_runtime_context = row_level_user_message.split(
                "User question:", 1
            )[0]

            assert "<db_runtime_context>" in row_level_runtime_context
            assert snippet not in row_level_runtime_context
            assert snippet not in row_level_prompt
            assert reader_embedder.embedded_texts == []
        finally:
            await _close_agent_db(reader_agent)


@pytest.mark.integration
@pytest.mark.requires_llm
class TestFromDbSQLiteOpenAILive:
    LIVE_ACCURACY_CASES = [
        {
            "id": "simple_aggregate",
            "prompt": (
                "Use db_query to answer from the database: what is the total "
                "revenue for customer_id 1? Return the numeric total and mention "
                "customer_id 1."
            ),
            "expected_tools": ["db_query"],
            "answer_all_of": ["125", "1"],
        },
        {
            "id": "relationship_customers_with_orders",
            "prompt": (
                "Use db_query exactly once with this SQL: "
                "SELECT DISTINCT c.name AS name FROM customers c "
                "JOIN orders o ON c.customer_id = o.customer_id "
                "ORDER BY c.name. Which customer names have at least one order? "
                "Return only names that have orders."
            ),
            "expected_tools": ["db_query"],
            "answer_all_of": ["alice", "bob"],
            "answer_none_of": ["cara"],
        },
        {
            "id": "freshness_orders_created_at",
            "prompt": (
                "Use dq_check_freshness on table orders with timestamp_column "
                "created_at and expected_interval_hours 24. Is the table fresh? "
                "Answer with fresh or stale and a short reason."
            ),
            "expected_tools": ["dq_check_freshness"],
            "answer_any_of": ["stale", "not fresh", "false"],
        },
        {
            "id": "anomaly_daily_revenue",
            "prompt": (
                "Use dq_detect_anomaly on table daily_metrics, column revenue, "
                "method iqr. What seeded revenue value is the outlier? "
                "Answer by naming the anomalous value."
            ),
            "expected_tools": ["dq_detect_anomaly"],
            "answer_all_of": ["1000"],
            "answer_any_of": ["outlier", "anomal", "unusual"],
            "answer_none_of": ["no outlier", "no anomal"],
        },
        {
            "id": "blocked_email_column",
            "prompt": (
                "Use db_query to try to show customer emails with "
                "SELECT email FROM customers. If the database guardrail blocks it, "
                "say it was blocked and do not reveal any email address."
            ),
            "expected_tools": ["db_query"],
            "answer_any_of": ["blocked", "guardrail", "not allowed"],
            "answer_none_of": [
                "alice@example.com",
                "bob@example.com",
                "cara@example.com",
            ],
        },
    ]

    async def test_openai_agent_uses_sqlite_query_against_seeded_db(
        self, sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            name="sqlite-from-db-openai-live",
            mode="analyst",
            cache_ttl=3600,
            query_default_limit=10,
            **_openai_kwargs(),
        )
        try:
            result = await agent.run(
                "Use db_query to answer this exactly from the database: "
                "what is the total revenue for customer_id 1? "
                "Return the numeric total and mention customer_id 1.",
                detailed=True,
                max_iterations=4,
            )

            metrics = _from_db_metrics(result, "db_query", "125")
            result["from_db_metrics"] = metrics
            assert metrics["accurate"] is True
            assert metrics["query_duration_ms"] is not None
            assert metrics["query_duration_ms"] >= 0
            assert metrics["tokens"]["total_tokens"] > 0
            assert metrics["cost"] >= 0
            assert_answer_mentions(result, ["125"], any_of=True)
            assert any(
                call.get("tool") == "db_query"
                for call in result.get("tool_calls") or []
            )

            audit_json = agent.db.audit.export_json()
            assert "db_query" in audit_json
            assert "alice@example.com" not in audit_json
        finally:
            await _close_agent_db(agent)

    @pytest.mark.parametrize("case", LIVE_ACCURACY_CASES, ids=lambda case: case["id"])
    async def test_openai_sqlite_compact_accuracy_suite(
        self, sqlite_db_path, case, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(sqlite_db_path),
            name=f"sqlite-from-db-accuracy-{case['id']}",
            mode="data_team",
            cache_ttl=3600,
            query_default_limit=10,
            blocked_columns=["email"],
            **_openai_kwargs(),
        )
        try:
            started = time.perf_counter()
            result = await agent.run(
                case["prompt"],
                detailed=True,
                max_iterations=5,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            case["audit_entry_count"] = len(agent.db.audit.entries)
            metrics = _live_accuracy_metrics(result, case, elapsed_ms)
            result["from_db_accuracy"] = metrics

            assert metrics["tool_duration_ms"] is not None
            assert metrics["tool_duration_ms"] >= 0
            assert metrics["tokens"]["total_tokens"] > 0
            assert metrics["cost"] >= 0
            assert metrics["iterations"] <= 5
            assert metrics["audit_entry_count"] == 1
            assert "focus" not in (metrics["tool_arguments"] or {})

            audit_json = agent.db.audit.export_json()
            assert "alice@example.com" not in audit_json
            assert "bob@example.com" not in audit_json
            assert "cara@example.com" not in audit_json

            _write_live_accuracy_result(tmp_path, metrics)
        finally:
            await _close_agent_db(agent)

    async def test_openai_finds_omitted_table_before_querying_large_schema(
        self, warehouse_scale_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(warehouse_scale_sqlite_db_path),
            name="sqlite-from-db-large-schema-openai-live",
            mode="analyst",
            cache_ttl=None,
            query_default_limit=5,
            query_max_rows=5,
            query_max_chars=1200,
            **_openai_kwargs(),
        )
        try:
            result = await agent.run(
                "This database has many omitted tables. Find the table that stores "
                "wide event feature columns, inspect its schema metadata, then query "
                "the database for account_id 42. What are the feature_000 and "
                "feature_249 values on the first event by event_id? Use schema "
                "navigation before writing SQL.",
                detailed=True,
                max_iterations=6,
            )

            tool_names = [call.get("tool") for call in result.get("tool_calls") or []]
            assert any(
                name in tool_names for name in ("db_search_schema", "db_list_tables")
            ), tool_names
            assert "db_inspect_table" in tool_names, tool_names
            assert "db_query" in tool_names, tool_names

            query_call = next(
                call for call in result["tool_calls"] if call.get("tool") == "db_query"
            )
            rows = query_call.get("result", {}).get("rows") or []
            assert rows
            assert rows[0]["feature_000"] == "segment-a"
            assert rows[0]["feature_249"] == "channel-web"
            assert result["tokens"]["total_tokens"] > 0
            assert result["cost"] >= 0
            _assert_answer_tokens(
                result,
                all_of=["segment-a", "channel-web"],
            )
        finally:
            await _close_agent_db(agent)

    async def test_openai_paginates_wide_table_before_querying_late_column(
        self, warehouse_scale_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(warehouse_scale_sqlite_db_path),
            name="sqlite-from-db-wide-pagination-openai-live",
            mode="analyst",
            cache_ttl=None,
            query_default_limit=5,
            query_max_rows=5,
            query_max_chars=1200,
            **_openai_kwargs(),
        )
        try:
            result = await agent.run(
                "Find the wide event feature table, inspect its columns in pages "
                "until you locate the late feature column numbered 249, then query "
                "account_id 42 for the first event by event_id. What is that late "
                "feature value? Use db_inspect_table pagination rather than guessing.",
                detailed=True,
                max_iterations=12,
            )

            inspect_calls = [
                call
                for call in result.get("tool_calls") or []
                if call.get("tool") == "db_inspect_table"
            ]
            assert inspect_calls, result.get("tool_calls")
            assert any(
                int(call.get("arguments", {}).get("offset") or 0) >= 100
                for call in inspect_calls
            ), inspect_calls
            assert any(
                "feature_249" in json.dumps(call.get("result") or {}, default=str)
                for call in inspect_calls
            ), inspect_calls
            assert "db_query" in [
                call.get("tool") for call in result.get("tool_calls") or []
            ]

            query_call = next(
                call for call in result["tool_calls"] if call.get("tool") == "db_query"
            )
            rows = query_call.get("result", {}).get("rows") or []
            assert rows
            assert rows[0]["feature_249"] == "channel-web"
            _assert_answer_tokens(result, all_of=["channel-web"])
        finally:
            await _close_agent_db(agent)

    async def test_openai_recovers_from_bad_schema_navigation_before_sql(
        self, warehouse_scale_sqlite_db_path, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            str(warehouse_scale_sqlite_db_path),
            name="sqlite-from-db-bad-navigation-openai-live",
            mode="analyst",
            cache_ttl=None,
            query_default_limit=5,
            query_max_rows=5,
            query_max_chars=1200,
            **_openai_kwargs(),
        )
        try:
            result = await agent.run(
                "Use schema navigation to recover from these bad names before SQL. "
                "First try db_inspect_table with table_name='wide_eventz'. Use the "
                "returned candidate table. Then inspect that table with "
                "column_pattern='feat_249' to recover the exact column. Only after "
                "the tool result shows the corrected column should you query "
                "account_id 42 for the first event by event_id and return that "
                "corrected column value.",
                detailed=True,
                max_iterations=10,
            )

            tool_calls = result.get("tool_calls") or []
            inspect_calls = [
                call for call in tool_calls if call.get("tool") == "db_inspect_table"
            ]
            assert inspect_calls, tool_calls
            first_sql_index = next(
                index
                for index, call in enumerate(tool_calls)
                if call.get("tool") == "db_query"
            )
            assert any(
                call.get("tool")
                in ("db_search_schema", "db_list_tables", "db_inspect_table")
                for call in tool_calls[:first_sql_index]
            ), tool_calls
            assert "db_query" in [call.get("tool") for call in tool_calls]

            query_call = next(
                call for call in tool_calls if call.get("tool") == "db_query"
            )
            sql = query_call.get("arguments", {}).get("sql", "")
            assert "wide_fact_events" in sql
            assert "feature_249" in sql
            assert "wide_eventz" not in sql
            assert "feat_249" not in sql

            rows = query_call.get("result", {}).get("rows") or []
            assert rows
            assert rows[0]["feature_249"] == "channel-web"
            _assert_answer_tokens(result, all_of=["channel-web"])
        finally:
            await _close_agent_db(agent)


@pytest.mark.integration
@pytest.mark.requires_db
class TestFromDbPostgresLive:
    async def test_from_db_discovers_postgres_via_catalog_modules_and_queries_seed(
        self, seeded_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_postgres,
            name="postgres-from-db-live",
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            cache_ttl=3600,
            query_default_limit=10,
            blocked_columns=["email"],
        )
        try:
            description = agent.describe()
            assert description["db"]["database_type"] == "postgresql"
            assert description["db"]["database_name"] == "public"
            assert description["db"]["table_count"] == 3
            assert description["db"]["relationship_count"] == 1
            assert description["db"]["query_policy"]["has_column_blocklist"] is True
            assert "db_query" in agent.tool_registry.tool_names
            assert "db_count" in agent.tool_registry.tool_names
            assert "postgres_query" not in agent.tool_registry.tool_names
            assert "postgres_count" not in agent.tool_registry.tool_names
            assert "postgres_list_tables" not in agent.tool_registry.tool_names
            assert "postgres_inspect" not in agent.tool_registry.tool_names

            tables = {table["name"] for table in agent.db.schema["tables"]}
            assert tables == {"customers", "orders", "daily_metrics"}
            assert agent.db.schema["foreign_keys"] == [
                {
                    "source_table": "orders",
                    "source_column": "customer_id",
                    "target_table": "customers",
                    "target_column": "customer_id",
                }
            ]

            query_result = await agent.tool_registry.execute(
                "db_query",
                {
                    "sql": (
                        "SELECT customer_id, SUM(total_amount) AS revenue "
                        "FROM orders GROUP BY customer_id ORDER BY revenue DESC"
                    )
                },
            )
            assert query_result["sql"].endswith("LIMIT 10")
            assert query_result["rows"][0]["customer_id"] == 2
            assert float(query_result["rows"][0]["revenue"]) == 150.0

            with pytest.raises(Exception, match="blocked column"):
                await agent.tool_registry.execute(
                    "db_query", {"sql": "SELECT email FROM customers"}
                )
            with pytest.raises(Exception, match="read-only mode"):
                await agent.tool_registry.execute(
                    "db_query",
                    {
                        "sql": (
                            "WITH deleted AS (DELETE FROM orders WHERE order_id = 1 "
                            "RETURNING *) SELECT * FROM deleted"
                        )
                    },
                )
        finally:
            await _close_agent_db(agent)

    async def test_from_db_postgres_quality_analyst_and_monitor_systems(
        self, seeded_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_postgres,
            name="postgres-from-db-systems",
            llm_provider=MockLLMProvider(delay=0),
            mode="data_team",
            cache_ttl=3600,
            query_default_limit=20,
        )
        try:
            await _assert_data_quality_tools(agent)
            await _assert_analyst_tools(agent)
            await _assert_local_monitor_findings(agent)
        finally:
            await _close_agent_db(agent)

    async def test_from_db_postgres_cache_reuses_live_discovered_schema(
        self, seeded_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_postgres,
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=3600,
        )
        await _close_agent_db(agent)

        import daita.agents.db.builder as builder

        async def fail_discovery(*args, **kwargs):
            raise AssertionError("schema discovery should not run on fresh cache")

        monkeypatch.setattr(builder, "discover_schema", fail_discovery)
        cached_agent = await Agent.from_db(
            seeded_postgres,
            llm_provider=MockLLMProvider(delay=0),
            cache_ttl=3600,
        )
        try:
            assert cached_agent.describe()["db"]["relationship_count"] == 1
            assert json.loads(cached_agent.db.audit.export_json()) == []
        finally:
            await _close_agent_db(cached_agent)

    async def test_from_db_postgres_multischema_navigation_disambiguates_tables(
        self, seeded_multischema_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_multischema_postgres,
            name="postgres-from-db-multischema",
            llm_provider=MockLLMProvider(delay=0),
            mode="analyst",
            db_schema="*",
            cache_ttl=None,
            query_default_limit=10,
        )
        try:
            tables = {table["name"] for table in agent.db.schema["tables"]}
            assert {"public.orders", "analytics.orders", "public.customers"}.issubset(
                tables
            )
            assert "db_search_schema" in agent.tool_registry.tool_names
            assert "db_inspect_table" in agent.tool_registry.tool_names

            searched = await agent.tool_registry.execute(
                "db_search_schema",
                {"query": "orders recognized revenue channel", "limit": 10},
            )
            names = [table["name"] for table in searched["tables"]]
            assert names[0] == "analytics.orders"
            assert "public.orders" in names

            public_inspect = await agent.tool_registry.execute(
                "db_inspect_table",
                {"table_name": "public.orders"},
            )
            analytics_inspect = await agent.tool_registry.execute(
                "db_inspect_table",
                {"table_name": "analytics.orders"},
            )
            assert public_inspect["success"] is True
            assert analytics_inspect["success"] is True
            assert any(
                column["name"] == "total_amount" for column in public_inspect["columns"]
            )
            assert any(
                column["name"] == "recognized_revenue"
                for column in analytics_inspect["columns"]
            )

            relationships = await agent.tool_registry.execute(
                "db_describe_relationships",
                {"table_name": "public.orders"},
            )
            assert relationships["relationship_count"] == 1
            assert relationships["relationships"][0] == {
                "source_table": "public.orders",
                "source_column": "customer_id",
                "target_table": "public.customers",
                "target_column": "customer_id",
            }

            public_result = await agent.tool_registry.execute(
                "db_query",
                {
                    "sql": (
                        "SELECT SUM(total_amount) AS revenue "
                        "FROM public.orders WHERE customer_id = 1"
                    )
                },
            )
            analytics_result = await agent.tool_registry.execute(
                "db_query",
                {
                    "sql": (
                        "SELECT SUM(recognized_revenue) AS revenue "
                        "FROM analytics.orders WHERE customer_id = 1"
                    )
                },
            )
            assert float(public_result["rows"][0]["revenue"]) == 125.0
            assert float(analytics_result["rows"][0]["revenue"]) == 110.0
        finally:
            await _close_agent_db(agent)

    async def test_from_db_postgres_lineage_fk_persistence_is_idempotent(
        self, seeded_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        first_agent = await Agent.from_db(
            seeded_postgres,
            name="postgres-from-db-lineage-idempotent",
            llm_provider=MockLLMProvider(delay=0),
            mode="governed",
            cache_ttl=None,
        )
        try:
            first = await _assert_single_seeded_fk_lineage(first_agent)
        finally:
            await _close_agent_db(first_agent)

        second_agent = await Agent.from_db(
            seeded_postgres,
            name="postgres-from-db-lineage-idempotent",
            llm_provider=MockLLMProvider(delay=0),
            mode="governed",
            cache_ttl=None,
        )
        try:
            second = await _assert_single_seeded_fk_lineage(second_agent)
        finally:
            await _close_agent_db(second_agent)

        assert second == first


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.requires_llm
class TestFromDbPostgresOpenAILive:
    async def test_openai_agent_uses_postgres_query_against_seeded_container(
        self, seeded_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_postgres,
            name="postgres-from-db-openai-live",
            mode="analyst",
            cache_ttl=3600,
            query_default_limit=10,
            **_openai_kwargs(),
        )
        try:
            result = await agent.run(
                "Use db_query to answer this exactly from the database: "
                "what is the total revenue for customer_id 1? "
                "Return the numeric total and mention customer_id 1.",
                detailed=True,
                max_iterations=4,
            )

            metrics = _from_db_metrics(result, "db_query", "125")
            result["from_db_metrics"] = metrics
            assert metrics["accurate"] is True
            assert metrics["query_duration_ms"] is not None
            assert metrics["query_duration_ms"] >= 0
            assert metrics["tokens"]["total_tokens"] > 0
            assert metrics["cost"] >= 0
            assert_answer_mentions(result, ["125"], any_of=True)
            assert any(
                call.get("tool") == "db_query"
                for call in result.get("tool_calls") or []
            )

            audit_json = agent.db.audit.export_json()
            assert "db_query" in audit_json
            assert "alice@example.com" not in audit_json
        finally:
            await _close_agent_db(agent)

    async def test_openai_agent_uses_persisted_fk_lineage_from_seeded_postgres(
        self, seeded_postgres, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_db(
            seeded_postgres,
            name="postgres-from-db-lineage-openai-live",
            mode="governed",
            cache_ttl=None,
            **_openai_kwargs(),
        )
        try:
            lineage_snapshot = await _assert_single_seeded_fk_lineage(agent)
            started = time.perf_counter()
            result = await agent.run(
                "Use trace_lineage exactly once to inspect this seeded Postgres "
                f"foreign-key lineage: entity_id={lineage_snapshot['source_id']}, "
                "direction=downstream, max_depth=2, edge_types=['references']. "
                "Then answer which table the orders table references. Mention "
                "customers and customer_id.",
                detailed=True,
                max_iterations=4,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            tool_call = _assert_any_tool_called(result, {"trace_lineage"})
            tool_result = tool_call.get("result") or {}
            assert tool_call.get("duration_ms") is not None
            assert tool_call.get("duration_ms") >= 0
            assert (
                tool_call.get("arguments", {}).get("entity_id")
                == lineage_snapshot["source_id"]
            )
            assert tool_result.get("downstream_count") == 1
            assert lineage_snapshot["target_id"] in json.dumps(tool_result, default=str)
            assert result["tokens"]["total_tokens"] > 0
            assert result["cost"] >= 0
            assert elapsed_ms >= 0
            _assert_answer_tokens(
                result,
                all_of=["customers"],
                any_of=["customer_id", "foreign", "references"],
            )
        finally:
            await _close_agent_db(agent)
