"""Factory targets for evaluating ``Agent.from_db`` with ``daita.evals``."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbMemoryConfig, DbSourceOptions
from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.sqlite import SQLitePlugin

from tests.integration._harness import start_container

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_from_db_eval"

SALES_SQLITE_SQL = """
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

SALES_POSTGRES_SQL = """
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

RICH_BENCHMARK_SQLITE_SQL = """
CREATE TABLE regions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    quota REAL NOT NULL
);
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region_id INTEGER NOT NULL REFERENCES regions(id),
    segment TEXT NOT NULL,
    lifecycle_stage TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    total REAL NOT NULL
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE support_tickets (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    amount REAL NOT NULL,
    reason TEXT NOT NULL
);

INSERT INTO regions (id, name, quota) VALUES
    (1, 'NA', 500.00),
    (2, 'EU', 300.00);

INSERT INTO customers (id, name, region_id, segment, lifecycle_stage) VALUES
    (1, 'Ada', 1, 'enterprise', 'active'),
    (2, 'Linus', 2, 'startup', 'active'),
    (3, 'Grace', 1, 'enterprise', 'expansion'),
    (4, 'Turing', 2, 'enterprise', 'active');

INSERT INTO products (id, sku, name, category) VALUES
    (1, 'HW-A', 'Widget A', 'hardware'),
    (2, 'HW-B', 'Widget B', 'hardware'),
    (3, 'SVC-PLAN', 'Service Plan', 'services');

INSERT INTO orders (id, customer_id, order_date, status, total) VALUES
    (1, 1, '2026-01-05', 'complete', 120.00),
    (2, 1, '2026-01-20', 'pending', 80.00),
    (3, 2, '2026-02-12', 'complete', 50.00),
    (4, 3, '2026-02-18', 'complete', 175.00),
    (5, 4, '2026-03-03', 'complete', 220.00);

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 2, 40.00),
    (2, 1, 3, 1, 40.00),
    (3, 2, 2, 1, 80.00),
    (4, 3, 3, 1, 50.00),
    (5, 4, 1, 1, 75.00),
    (6, 4, 2, 1, 100.00),
    (7, 5, 1, 3, 60.00),
    (8, 5, 3, 1, 40.00);

INSERT INTO support_tickets (id, customer_id, severity, status, opened_at) VALUES
    (1, 1, 'high', 'open', '2026-01-07'),
    (2, 2, 'medium', 'closed', '2026-01-15'),
    (3, 3, 'high', 'open', '2026-02-21'),
    (4, 4, 'low', 'open', '2026-03-05'),
    (5, 4, 'high', 'closed', '2026-03-08');

INSERT INTO refunds (id, order_id, amount, reason) VALUES
    (1, 2, 20.00, 'partial return'),
    (2, 4, 35.00, 'service credit');
"""

RICH_BENCHMARK_POSTGRES_SQL = """
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS regions;

CREATE TABLE regions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    quota NUMERIC(10, 2) NOT NULL
);
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region_id INTEGER NOT NULL REFERENCES regions(id),
    segment TEXT NOT NULL,
    lifecycle_stage TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date DATE NOT NULL,
    status TEXT NOT NULL,
    total NUMERIC(10, 2) NOT NULL
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL
);
CREATE TABLE support_tickets (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at DATE NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    amount NUMERIC(10, 2) NOT NULL,
    reason TEXT NOT NULL
);

INSERT INTO regions (id, name, quota) VALUES
    (1, 'NA', 500.00),
    (2, 'EU', 300.00);

INSERT INTO customers (id, name, region_id, segment, lifecycle_stage) VALUES
    (1, 'Ada', 1, 'enterprise', 'active'),
    (2, 'Linus', 2, 'startup', 'active'),
    (3, 'Grace', 1, 'enterprise', 'expansion'),
    (4, 'Turing', 2, 'enterprise', 'active');

INSERT INTO products (id, sku, name, category) VALUES
    (1, 'HW-A', 'Widget A', 'hardware'),
    (2, 'HW-B', 'Widget B', 'hardware'),
    (3, 'SVC-PLAN', 'Service Plan', 'services');

INSERT INTO orders (id, customer_id, order_date, status, total) VALUES
    (1, 1, '2026-01-05', 'complete', 120.00),
    (2, 1, '2026-01-20', 'pending', 80.00),
    (3, 2, '2026-02-12', 'complete', 50.00),
    (4, 3, '2026-02-18', 'complete', 175.00),
    (5, 4, '2026-03-03', 'complete', 220.00);

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 2, 40.00),
    (2, 1, 3, 1, 40.00),
    (3, 2, 2, 1, 80.00),
    (4, 3, 3, 1, 50.00),
    (5, 4, 1, 1, 75.00),
    (6, 4, 2, 1, 100.00),
    (7, 5, 1, 3, 60.00),
    (8, 5, 3, 1, 40.00);

INSERT INTO support_tickets (id, customer_id, severity, status, opened_at) VALUES
    (1, 1, 'high', 'open', '2026-01-07'),
    (2, 2, 'medium', 'closed', '2026-01-15'),
    (3, 3, 'high', 'open', '2026-02-21'),
    (4, 4, 'low', 'open', '2026-03-05'),
    (5, 4, 'high', 'closed', '2026-03-08');

INSERT INTO refunds (id, order_id, amount, reason) VALUES
    (1, 2, 20.00, 'partial return'),
    (2, 4, 35.00, 'service credit');
"""


def wide_benchmark_postgres_sql(table_count: int = 48) -> str:
    """Generate a wide, ambiguous Postgres schema around the rich benchmark data."""

    decoys = []
    for index in range(1, table_count + 1):
        table = f"customer_activity_decoy_{index:02d}"
        decoys.append(f"DROP TABLE IF EXISTS {table};")
    statements = ["\n".join(decoys), RICH_BENCHMARK_POSTGRES_SQL]
    for index in range(1, table_count + 1):
        table = f"customer_activity_decoy_{index:02d}"
        statements.append(
            f"""
CREATE TABLE {table} (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    customer_name TEXT,
    status TEXT,
    region TEXT,
    severity TEXT,
    total NUMERIC(10, 2)
);
INSERT INTO {table} (id, customer_id, customer_name, status, region, severity, total)
VALUES
    (1, {index}, 'Decoy {index}', 'inactive', 'ZZ', 'low', 0.00);
"""
        )
    return "\n".join(statements)


class FromDbEvalTarget:
    """Runnable eval adapter for DB runtime results."""

    def __init__(
        self,
        agent,
        *,
        name: str,
        cleanup=None,
        database_type: str | None = None,
        fixture_revision: str = "from-db-eval-fixture@b87df318",
    ) -> None:
        self.agent = agent
        self.name = name
        self.agent_id = name
        self.llm = getattr(agent, "llm", None)
        self._cleanup = cleanup
        self._database_type = database_type or (
            "postgresql" if "postgres" in name.lower() else "sqlite"
        )
        self._fixture_revision = fixture_revision
        self._phase0_prompt_runs: dict[str, int] = {}

    async def run(
        self,
        prompt: str,
        *,
        detailed: bool = False,
        max_iterations: int | None = None,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        del max_iterations, timeout_seconds
        result = await self.run_detailed(prompt, **kwargs)
        return result if detailed else result.answer or ""

    async def run_detailed(self, prompt: str, **kwargs):
        if os.environ.get("DAITA_PHASE0_OUTPUT_DIR"):
            return await self._run_detailed_with_phase0_capture(prompt, **kwargs)
        result = await self.agent.run_detailed(prompt, **kwargs)
        snapshot = await self.agent.runtime.inspect_operation(result.operation_id)
        return _eval_runtime_payload(result, snapshot)

    async def _run_detailed_with_phase0_capture(self, prompt: str, **kwargs):
        from tests.performance.from_db.scale_runner import (
            NEUTRAL_ARTIFACT_SCHEMA_NAME,
            NEUTRAL_ARTIFACT_SCHEMA_VERSION,
            default_environment_metadata,
            measure_agent_operation,
            operation_record,
            summarize_operations,
            write_artifact,
        )

        prompt_hash = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(prompt, sort_keys=True, default=str).encode()
            ).hexdigest()
        )
        run_number = self._phase0_prompt_runs.get(prompt_hash, 0) + 1
        self._phase0_prompt_runs[prompt_hash] = run_number
        pytest_node_id = os.environ.get("PYTEST_CURRENT_TEST", "unknown-test").split(
            " (", 1
        )[0]
        scenario = f"prompt-{prompt_hash.removeprefix('sha256:')[:12]}"
        environment = default_environment_metadata(
            model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
            provider="openai",
            model_parameters={"temperature": 0},
            dataset=self._fixture_revision,
            database_type=self._database_type,
            fixture_revision=self._fixture_revision,
            control_label="baseline",
        )
        started_at = datetime.now(timezone.utc).isoformat()
        envelope = await measure_agent_operation(
            self.agent,
            prompt,
            measurement={
                "scenario": scenario,
                "run_id": f"run-{run_number:03d}",
                "control_label": "baseline",
                "provider": "openai",
                "model": environment["model"],
                "model_parameters": {"temperature": 0},
                "database": {"type": self._database_type, "version": None},
                "fixture_revision": self._fixture_revision,
                "state": (
                    "cold" if sum(self._phase0_prompt_runs.values()) == 1 else "warm"
                ),
                "concurrency": 1,
                "prompt_hash": prompt_hash,
                "pytest_node_id": pytest_node_id,
                "correctness": {
                    "answer": {
                        "passed": None,
                        "evaluation_source": "existing eval report",
                    },
                    "sql": {
                        "passed": None,
                        "evaluation_source": "existing eval report",
                    },
                },
            },
            run_kwargs=kwargs,
        )
        record = operation_record(
            index=run_number - 1,
            latency_ms=float(envelope["operation_latency_ms"]),
            started_at=started_at,
            result=envelope["runtime_result"],
            snapshot=envelope["operation_snapshot"],
            model_traces=envelope["model_traces"],
            provider_delta=envelope["provider_delta"],
            measurement={
                **envelope["measurement"],
                "environment": environment,
            },
            metadata={
                "prompt_hash": prompt_hash,
                "pytest_node_id": pytest_node_id,
            },
        )
        artifact = {
            "schema": {
                "name": NEUTRAL_ARTIFACT_SCHEMA_NAME,
                "version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
            },
            "suite": "from-db-slim-phase0-eval-capture",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "environment": environment,
            "parameters": {
                "scenario": scenario,
                "operations": 1,
                "concurrency": 1,
                "pytest_node_id": pytest_node_id,
            },
            "summary": summarize_operations(
                [record], max(float(envelope["operation_latency_ms"]) / 1000, 0.000001)
            ),
            "operations": [record],
        }
        safe_node_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", pytest_node_id).strip("-")
        output = (
            Path(os.environ["DAITA_PHASE0_OUTPUT_DIR"])
            / "neutral-eval-operations"
            / safe_node_id
            / self.name
            / scenario
            / f"run-{run_number:03d}.json"
        )
        write_artifact(artifact, output)
        return _eval_runtime_payload(
            envelope["runtime_result"], envelope["operation_snapshot"]
        )

    async def stop(self) -> None:
        await self.agent.stop()
        if self._cleanup is not None:
            self._cleanup()


def _eval_runtime_payload(result, snapshot):
    """Combine public answer diagnostics with persisted tasks and evidence."""

    if snapshot is None:
        return result
    if is_dataclass(result) and not isinstance(result, type):
        public_payload = asdict(result)
    else:
        public_payload = result.to_dict()
    snapshot_payload = snapshot.to_dict()
    public_payload.update(snapshot_payload)
    public_payload["answer"] = result.answer

    diagnostics = dict(result.diagnostics or {})
    execution = dict(diagnostics.get("execution") or {})
    execution["tasks"] = list(snapshot_payload.get("tasks") or [])
    execution["task_count"] = len(execution["tasks"])
    diagnostics["execution"] = execution
    public_payload["diagnostics"] = diagnostics
    return public_payload


async def create_sqlite_from_db_eval_agent(
    db_path: str,
    *,
    cache_ttl: int | None = 0,
) -> FromDbEvalTarget:
    path = Path(db_path)
    await _seed_sqlite(path)
    agent = await Agent.from_db(
        str(path),
        name="EvalFromDbSQLite",
        source_options=DbSourceOptions(cache_ttl=cache_ttl),
        **_openai_kwargs(),
    )
    return FromDbEvalTarget(agent, name="EvalFromDbSQLite")


async def create_sqlite_data_team_from_db_eval_agent(
    db_path: str,
    memory_dir: str,
    *,
    cache_ttl: int | None = 0,
) -> FromDbEvalTarget:
    path = Path(db_path)
    await _seed_sqlite(path)
    agent = await Agent.from_db(
        str(path),
        name="EvalFromDbSQLiteDataTeam",
        mode="data_team",
        quality=True,
        lineage=True,
        memory=_memory_config(Path(memory_dir)),
        source_options=DbSourceOptions(cache_ttl=cache_ttl),
        **_openai_kwargs(),
    )
    return FromDbEvalTarget(agent, name="EvalFromDbSQLiteDataTeam")


async def create_sqlite_rich_from_db_benchmark_agent(
    db_path: str,
    *,
    cache_ttl: int | None = 0,
) -> FromDbEvalTarget:
    path = Path(db_path)
    await _seed_sqlite_script(path, RICH_BENCHMARK_SQLITE_SQL)
    agent = await Agent.from_db(
        str(path),
        name="BenchmarkFromDbSQLiteRich",
        source_options=DbSourceOptions(cache_ttl=cache_ttl),
        **_openai_kwargs(),
    )
    return FromDbEvalTarget(agent, name="BenchmarkFromDbSQLiteRich")


async def create_postgres_from_db_eval_agent(
    *,
    cache_ttl: int | None = 0,
) -> FromDbEvalTarget:
    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix="daita-from-db-eval-pg",
    )
    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{container.host}:{container.host_port}/{POSTGRES_DB}"
    )
    try:
        await _seed_postgres(url)
        agent = await Agent.from_db(
            url,
            name="EvalFromDbPostgres",
            source_options=DbSourceOptions(cache_ttl=cache_ttl),
            **_openai_kwargs(),
        )
    except Exception:
        container.remove()
        raise
    return FromDbEvalTarget(
        agent,
        name="EvalFromDbPostgres",
        cleanup=container.remove,
    )


async def create_postgres_rich_from_db_benchmark_agent(
    *,
    cache_ttl: int | None = 0,
) -> FromDbEvalTarget:
    return await _postgres_agent_from_sql(
        RICH_BENCHMARK_POSTGRES_SQL,
        name="BenchmarkFromDbPostgresRich",
        cache_ttl=cache_ttl,
        tag_prefix="daita-from-db-rich-eval-pg",
    )


async def create_postgres_wide_from_db_benchmark_agent(
    *,
    cache_ttl: int | None = 0,
    table_count: int = 48,
) -> FromDbEvalTarget:
    return await _postgres_agent_from_sql(
        wide_benchmark_postgres_sql(table_count),
        name="BenchmarkFromDbPostgresWide",
        cache_ttl=cache_ttl,
        tag_prefix="daita-from-db-wide-eval-pg",
    )


async def _postgres_agent_from_sql(
    sql: str,
    *,
    name: str,
    cache_ttl: int | None,
    tag_prefix: str,
) -> FromDbEvalTarget:
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
        await _seed_postgres_script(url, sql)
        agent = await Agent.from_db(
            url,
            name=name,
            source_options=DbSourceOptions(cache_ttl=cache_ttl),
            **_openai_kwargs(),
        )
    except Exception:
        container.remove()
        raise
    return FromDbEvalTarget(agent, name=name, cleanup=container.remove)


async def _seed_sqlite(path: Path) -> None:
    await _seed_sqlite_script(path, SALES_SQLITE_SQL)


async def _seed_sqlite_script(path: Path, sql: str) -> None:
    if path.exists():
        path.unlink()
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script(sql)
    await plugin.disconnect()


async def _seed_postgres(url: str) -> None:
    await _seed_postgres_script(url, SALES_POSTGRES_SQL)


async def _seed_postgres_script(url: str, sql: str) -> None:
    try:
        import asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        ) from exc

    deadline = time.time() + 30
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            connection = await asyncpg.connect(url, ssl=False)
            await connection.execute(sql)
            await connection.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Could not seed Postgres eval database: {last_error}")


def _openai_kwargs() -> dict[str, Any]:
    return {
        "llm": DbLLMConfig(
            provider="openai",
            model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    }


def _memory_config(base_dir: Path) -> DbMemoryConfig:
    embedder = MockEmbeddingProvider(dim=8)
    return DbMemoryConfig(
        backend=LocalMemoryBackend(
            workspace="from-db-eval-memory",
            agent_id="from-db-eval-memory",
            scope="project",
            base_dir=base_dir,
            embedder=embedder,
        ),
        embedder=embedder,
    )
