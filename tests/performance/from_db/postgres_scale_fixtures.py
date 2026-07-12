"""Synthetic PostgreSQL scale data and live harnesses for from_db performance tests."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import os
import time
from typing import Any, Iterable

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbSourceOptions
from daita.runtime import Task

from tests.integration._harness import start_container
from tests.integration.evals.eval_from_db_factories import (
    POSTGRES_DB,
    POSTGRES_IMAGE,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
    RICH_BENCHMARK_POSTGRES_SQL,
    wide_benchmark_postgres_sql,
)


@dataclass
class PostgresScaleHarness:
    url: str
    container: Any
    agent: Any | None = None

    async def stop(self) -> None:
        if self.agent is not None:
            await self.agent.stop()
        self.container.remove()


async def create_postgres_scale_agent(
    *,
    sql: str,
    name: str,
    tag_prefix: str,
    cache_ttl: int | None = 3600,
    llm: bool = False,
    extra_agent_kwargs: dict[str, Any] | None = None,
) -> PostgresScaleHarness:
    harness = await start_seeded_postgres(sql, tag_prefix=tag_prefix)
    kwargs = dict(extra_agent_kwargs or {})
    if llm:
        kwargs.update(live_llm_kwargs())
    try:
        harness.agent = await Agent.from_db(
            harness.url,
            name=name,
            source_options=DbSourceOptions(cache_ttl=cache_ttl),
            **kwargs,
        )
    except Exception:
        await harness.stop()
        raise
    return harness


async def start_seeded_postgres(sql: str, *, tag_prefix: str) -> PostgresScaleHarness:
    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix=tag_prefix,
    )
    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{container.host}:{container.host_port}/{POSTGRES_DB}"
    )
    try:
        await execute_postgres_script(url, sql)
    except Exception:
        container.remove()
        raise
    return PostgresScaleHarness(url=url, container=container)


async def execute_postgres_script(url: str, sql: str) -> None:
    asyncpg = _asyncpg()
    deadline = time.time() + 60
    last_error: Exception | None = None
    while time.time() < deadline:
        connection = None
        try:
            connection = await asyncpg.connect(url, ssl=False)
            await connection.execute(sql)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
        finally:
            if connection is not None:
                with suppress(Exception):
                    await connection.close()
    raise RuntimeError(f"Could not seed Postgres scale database: {last_error}")


async def postgres_version(url: str) -> str:
    asyncpg = _asyncpg()
    connection = await asyncpg.connect(url, ssl=False)
    try:
        return str(await connection.fetchval("SHOW server_version"))
    finally:
        await connection.close()


async def seed_small_rich_schema(connection) -> None:
    """Seed the rich benchmark schema used by existing correctness evals."""

    await connection.execute(RICH_BENCHMARK_POSTGRES_SQL)


async def seed_medium_wide_schema(
    connection,
    *,
    table_count: int = 100,
    rows_per_table: int = 10000,
) -> None:
    """Seed a wide schema with repeated column names and decoy tables."""

    await connection.execute(wide_benchmark_postgres_sql(table_count))
    for index in range(1, table_count + 1):
        table = f"customer_activity_decoy_{index:02d}"
        await connection.execute(
            f"""
            INSERT INTO {table}
                (id, customer_id, customer_name, status, region, severity, total)
            SELECT
                generate_series + 1,
                generate_series,
                'Decoy ' || generate_series::text,
                CASE WHEN generate_series % 2 = 0 THEN 'open' ELSE 'closed' END,
                CASE WHEN generate_series % 3 = 0 THEN 'NA' ELSE 'EU' END,
                CASE WHEN generate_series % 5 = 0 THEN 'high' ELSE 'low' END,
                (generate_series % 100)::numeric
            FROM generate_series(1, $1)
            ON CONFLICT (id) DO NOTHING;
            """,
            max(0, rows_per_table - 1),
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_status_idx ON {table} (status)"
        )


async def seed_large_operational_schema(
    connection,
    *,
    table_count: int = 300,
    row_count: int = 10000000,
) -> None:
    """Seed a large operational schema with business tables and many decoys."""

    drops = [
        "DROP TABLE IF EXISTS operational_events",
        "DROP TABLE IF EXISTS incident_events",
        "DROP TABLE IF EXISTS incidents",
        "DROP TABLE IF EXISTS customer_accounts",
        "DROP TABLE IF EXISTS accounts",
    ]
    drops.extend(
        f"DROP TABLE IF EXISTS operational_decoy_{index:03d}"
        for index in range(1, table_count + 1)
    )
    await connection.execute(";\n".join(drops) + ";")
    await connection.execute("""
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY,
            account_name TEXT NOT NULL,
            region TEXT NOT NULL,
            tier TEXT NOT NULL
        );
        CREATE TABLE customer_accounts (
            id INTEGER PRIMARY KEY,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            customer_name TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE incidents (
            id INTEGER PRIMARY KEY,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            opened_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE incident_events (
            id INTEGER PRIMARY KEY,
            incident_id INTEGER NOT NULL REFERENCES incidents(id),
            event_type TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE operational_events (
            id BIGINT PRIMARY KEY,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            event_type TEXT NOT NULL,
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            value NUMERIC(10, 2) NOT NULL
        );
        INSERT INTO accounts (id, account_name, region, tier)
        SELECT generate_series, 'Account ' || generate_series::text,
               CASE WHEN generate_series % 2 = 0 THEN 'NA' ELSE 'EU' END,
               CASE WHEN generate_series % 3 = 0 THEN 'enterprise' ELSE 'startup' END
        FROM generate_series(1, 100);
        INSERT INTO customer_accounts (id, account_id, customer_name, status)
        SELECT generate_series, ((generate_series - 1) % 100) + 1,
               'Customer ' || generate_series::text,
               CASE WHEN generate_series % 4 = 0 THEN 'inactive' ELSE 'active' END
        FROM generate_series(1, 1000);
        INSERT INTO incidents (id, account_id, severity, status, opened_at)
        SELECT generate_series, ((generate_series - 1) % 100) + 1,
               CASE WHEN generate_series % 5 = 0 THEN 'high' ELSE 'low' END,
               CASE WHEN generate_series % 7 = 0 THEN 'closed' ELSE 'open' END,
               now() - (generate_series || ' hours')::interval
        FROM generate_series(1, 5000);
        INSERT INTO incident_events (id, incident_id, event_type, created_at)
        SELECT generate_series, ((generate_series - 1) % 5000) + 1,
               CASE WHEN generate_series % 2 = 0 THEN 'comment' ELSE 'status_change' END,
               now() - (generate_series || ' minutes')::interval
        FROM generate_series(1, 20000);
    """)
    await connection.execute(
        """
        INSERT INTO operational_events
            (id, account_id, event_type, status, severity, created_at, value)
        SELECT generate_series::bigint,
               ((generate_series - 1) % 100) + 1,
               CASE WHEN generate_series % 2 = 0 THEN 'ticket' ELSE 'order' END,
               CASE WHEN generate_series % 7 = 0 THEN 'closed' ELSE 'open' END,
               CASE WHEN generate_series % 11 = 0 THEN 'high' ELSE 'low' END,
               now() - (generate_series || ' seconds')::interval,
               (generate_series % 1000)::numeric
        FROM generate_series(1, $1);
        """,
        row_count,
    )
    await connection.execute("""
        CREATE INDEX operational_events_status_severity_idx
            ON operational_events (status, severity);
        CREATE INDEX operational_events_account_created_idx
            ON operational_events (account_id, created_at DESC);
        CREATE INDEX incidents_status_severity_idx ON incidents (status, severity);
    """)
    for index in range(1, table_count + 1):
        table = f"operational_decoy_{index:03d}"
        await connection.execute(f"""
            CREATE TABLE {table} (
                id INTEGER PRIMARY KEY,
                account_id INTEGER,
                customer_name TEXT,
                status TEXT,
                severity TEXT,
                event_type TEXT,
                created_at TIMESTAMPTZ
            );
            INSERT INTO {table} (id, account_id, customer_name, status, severity, event_type, created_at)
            VALUES (1, {index}, 'Decoy {index}', 'inactive', 'low', 'decoy', now());
            """)


async def seed_worker_task_backlog(kernel, *, count: int) -> tuple[Task, ...]:
    """Persist worker-capable runtime tasks for worker scale tests."""

    tasks: list[Task] = []
    for index in range(count):
        operation = await kernel.create_operation(
            operation_type="worker.query",
            request={"kind": "worker-backlog", "index": index},
            required_evidence={"live_runtime.worker.result"},
            evaluate_governance=False,
        )
        tasks.append(
            await kernel.plan_task(
                operation_id=operation.id,
                capability_id="live_runtime.worker.query",
                owner="live_runtime",
                input={
                    "sql": (
                        "SELECT c.name "
                        "FROM customers c "
                        "JOIN support_tickets st ON st.customer_id = c.id "
                        "WHERE st.status = 'open' AND st.severity = 'high' "
                        "ORDER BY c.name"
                    ),
                    "index": index,
                },
                metadata={"owner": "live_runtime", "queue": "from_db_perf"},
            )
        )
    return tuple(tasks)


async def profile_catalog_values(
    agent_or_runtime,
    columns: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Pre-warm catalog-owned value profiles through runtime capabilities."""

    runtime = getattr(agent_or_runtime, "runtime", agent_or_runtime)
    store_id = _catalog_store_id(runtime)
    results: list[dict[str, Any]] = []
    for table, column in columns:
        raw = await runtime.execute_capability(
            "db.column_values.profile",
            owner="postgresql",
            operation_type="source.profile",
            input={"table": table, "column": column},
        )
        payloads = [
            item.payload for item in raw if item.kind == "column_values.profile"
        ]
        if not payloads:
            continue
        registered = await runtime.execute_capability(
            "catalog.column_values.register",
            owner="catalog",
            operation_type="source.profile",
            input={
                "store_id": store_id,
                "profiles": payloads,
                "persist": False,
            },
        )
        results.extend(item.payload for item in registered)
    return results


def rich_schema_sql() -> str:
    return RICH_BENCHMARK_POSTGRES_SQL


def medium_wide_schema_sql(
    *, table_count: int = 100, rows_per_table: int = 10000
) -> str:
    inserts = []
    for index in range(1, table_count + 1):
        table = f"customer_activity_decoy_{index:02d}"
        rows = max(0, rows_per_table - 1)
        inserts.append(f"""
        INSERT INTO {table} (id, customer_id, customer_name, status, region, severity, total)
        SELECT
            generate_series + 1,
            generate_series,
            'Decoy ' || generate_series::text,
            CASE WHEN generate_series % 2 = 0 THEN 'open' ELSE 'closed' END,
            CASE WHEN generate_series % 3 = 0 THEN 'NA' ELSE 'EU' END,
            CASE WHEN generate_series % 5 = 0 THEN 'high' ELSE 'low' END,
            (generate_series % 100)::numeric
        FROM generate_series(1, {rows})
        ON CONFLICT (id) DO NOTHING;
        CREATE INDEX IF NOT EXISTS {table}_status_idx ON {table} (status);
        """)
    return "\n".join([wide_benchmark_postgres_sql(table_count), *inserts])


def large_operational_schema_sql(
    *,
    table_count: int = 300,
    row_count: int = 10000000,
) -> str:
    # Keep the SQL-producing helper simple by using the async seeder when direct
    # connection access is available; tests use this bounded SQL variant.
    rows = max(1, row_count)
    decoys = []
    for index in range(1, table_count + 1):
        decoys.append(f"""
        DROP TABLE IF EXISTS operational_decoy_{index:03d};
        CREATE TABLE operational_decoy_{index:03d} (
            id INTEGER PRIMARY KEY,
            account_id INTEGER,
            customer_name TEXT,
            status TEXT,
            severity TEXT,
            event_type TEXT,
            created_at TIMESTAMPTZ
        );
        INSERT INTO operational_decoy_{index:03d}
        VALUES (1, {index}, 'Decoy {index}', 'inactive', 'low', 'decoy', now());
        """)
    return f"""
    DROP TABLE IF EXISTS operational_events;
    DROP TABLE IF EXISTS incident_events;
    DROP TABLE IF EXISTS incidents;
    DROP TABLE IF EXISTS customer_accounts;
    DROP TABLE IF EXISTS accounts;
    CREATE TABLE accounts (
        id INTEGER PRIMARY KEY,
        account_name TEXT NOT NULL,
        region TEXT NOT NULL,
        tier TEXT NOT NULL
    );
    CREATE TABLE customer_accounts (
        id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES accounts(id),
        customer_name TEXT NOT NULL,
        status TEXT NOT NULL
    );
    CREATE TABLE incidents (
        id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES accounts(id),
        severity TEXT NOT NULL,
        status TEXT NOT NULL,
        opened_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE incident_events (
        id INTEGER PRIMARY KEY,
        incident_id INTEGER NOT NULL REFERENCES incidents(id),
        event_type TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE operational_events (
        id BIGINT PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES accounts(id),
        event_type TEXT NOT NULL,
        status TEXT NOT NULL,
        severity TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        value NUMERIC(10, 2) NOT NULL
    );
    INSERT INTO accounts
    SELECT generate_series, 'Account ' || generate_series::text,
           CASE WHEN generate_series % 2 = 0 THEN 'NA' ELSE 'EU' END,
           CASE WHEN generate_series % 3 = 0 THEN 'enterprise' ELSE 'startup' END
    FROM generate_series(1, 100);
    INSERT INTO customer_accounts
    SELECT generate_series, ((generate_series - 1) % 100) + 1,
           'Customer ' || generate_series::text,
           CASE WHEN generate_series % 4 = 0 THEN 'inactive' ELSE 'active' END
    FROM generate_series(1, 1000);
    INSERT INTO incidents
    SELECT generate_series, ((generate_series - 1) % 100) + 1,
           CASE WHEN generate_series % 5 = 0 THEN 'high' ELSE 'low' END,
           CASE WHEN generate_series % 7 = 0 THEN 'closed' ELSE 'open' END,
           now() - (generate_series || ' hours')::interval
    FROM generate_series(1, 5000);
    INSERT INTO incident_events
    SELECT generate_series, ((generate_series - 1) % 5000) + 1,
           CASE WHEN generate_series % 2 = 0 THEN 'comment' ELSE 'status_change' END,
           now() - (generate_series || ' minutes')::interval
    FROM generate_series(1, 20000);
    INSERT INTO operational_events
    SELECT generate_series::bigint,
           ((generate_series - 1) % 100) + 1,
           CASE WHEN generate_series % 2 = 0 THEN 'ticket' ELSE 'order' END,
           CASE WHEN generate_series % 7 = 0 THEN 'closed' ELSE 'open' END,
           CASE WHEN generate_series % 11 = 0 THEN 'high' ELSE 'low' END,
           now() - (generate_series || ' seconds')::interval,
           (generate_series % 1000)::numeric
    FROM generate_series(1, {rows});
    CREATE INDEX operational_events_status_severity_idx
        ON operational_events (status, severity);
    CREATE INDEX operational_events_account_created_idx
        ON operational_events (account_id, created_at DESC);
    CREATE INDEX incidents_status_severity_idx ON incidents (status, severity);
    {"".join(decoys)}
    """


def live_llm_kwargs() -> dict[str, Any]:
    return {
        "llm": DbLLMConfig(
            provider="openai",
            model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    }


def scale_int_env(name: str, default: int) -> int:
    return max(1, int(os.environ.get(name, str(default))))


def scale_concurrency_env(name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(name, default)
    return tuple(max(1, int(item.strip())) for item in raw.split(",") if item.strip())


def _catalog_store_id(runtime) -> str:
    options = runtime.config.metadata.get("from_db_options") or {}
    return str(options.get("catalog_store_id") or options.get("catalog_keys", [""])[0])


def _asyncpg():
    try:
        import asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        ) from exc
    return asyncpg
