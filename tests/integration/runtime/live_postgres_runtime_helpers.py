"""Shared live PostgreSQL runtime helpers for monitor and worker tests."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from daita.plugins import ExtensionRegistry, PluginKind, PluginManifest
from daita.plugins.postgresql import PostgreSQLPlugin
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    InMemoryRuntimeStore,
    RiskLevel,
    RuntimeKernel,
    Task,
    Worker,
)

from tests.integration._harness import start_container
from tests.integration.evals.eval_from_db_factories import (
    POSTGRES_DB,
    POSTGRES_IMAGE,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
    RICH_BENCHMARK_POSTGRES_SQL,
)

load_dotenv(Path.cwd() / ".env")


def require_live_postgres_runtime() -> None:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live runtime benchmarks")
    if os.environ.get("DAITA_EVAL_POSTGRES") != "1":
        pytest.skip("Set DAITA_EVAL_POSTGRES=1 to run Docker Postgres benchmarks")


def start_seeded_postgres(tag_prefix: str):
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
        asyncio.run(seed_postgres(url))
    except Exception:
        container.remove()
        raise
    return container, url


async def seed_postgres(url: str) -> None:
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
            await connection.execute(RICH_BENCHMARK_POSTGRES_SQL)
            await connection.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Could not seed Postgres runtime database: {last_error}")


@dataclass
class LiveKernelHarness:
    kernel: RuntimeKernel
    store: InMemoryRuntimeStore
    plugin: "LiveRuntimeDbPlugin"
    postgres: PostgreSQLPlugin

    async def stop(self) -> None:
        await self.postgres.disconnect()


async def build_live_kernel(url: str) -> LiveKernelHarness:
    postgres = PostgreSQLPlugin(connection_string=url)
    await postgres.connect()
    plugin = LiveRuntimeDbPlugin()
    registry = ExtensionRegistry()
    registry.register(postgres)
    registry.register(plugin)
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="live-postgres-runtime",
        runtime_kind="integration",
        extension_registry=registry,
        runtime_store=store,
    )
    plugin.kernel = kernel
    return LiveKernelHarness(
        kernel=kernel, store=store, plugin=plugin, postgres=postgres
    )


class LiveRuntimeDbExecutor:
    def __init__(self, plugin: "LiveRuntimeDbPlugin") -> None:
        self.id = "live_runtime.db.executor"
        self.capability_ids = frozenset(
            {
                "live_runtime.monitor.source",
                "live_runtime.monitor.action",
                "live_runtime.worker.query",
                "live_runtime.worker.side_effect",
            }
        )
        self.plugin = plugin

    async def execute(self, task: Task, operation, context):
        if self.plugin.kernel is None:
            raise RuntimeError("test plugin is not bound to a RuntimeKernel")
        sql = _sql_for_task(task)
        rows = await _validate_and_read(
            self.plugin.kernel,
            operation.id,
            sql,
            context=context,
        )
        payload: dict[str, Any] = {
            "sql": sql,
            "rows": rows,
            "row_count": len(rows),
            "worker_id": context.get("worker_id"),
            "monitor_id": context.get("monitor_id"),
        }
        if task.capability_id == "live_runtime.monitor.source":
            payload["open_high_count"] = int(rows[0]["open_high_count"])
            kind = "live_runtime.monitor.source"
        elif task.capability_id == "live_runtime.monitor.action":
            kind = "live_runtime.monitor.action_result"
        else:
            kind = "live_runtime.worker.result"
        return [
            Evidence(
                kind=kind,
                owner="live_runtime",
                payload=payload,
            )
        ]


class LiveRuntimeDbPlugin:
    manifest = PluginManifest(
        id="live_runtime",
        display_name="Live Runtime DB",
        version="1.0.0",
        kind=PluginKind.WORKER_PROVIDER,
        domains=frozenset({"db", "runtime"}),
    )

    def __init__(self) -> None:
        self.kernel: RuntimeKernel | None = None
        self.executor = LiveRuntimeDbExecutor(self)

    def declare_capabilities(self):
        common = {
            "owner": "live_runtime",
            "domains": frozenset({"db", "runtime"}),
            "access": AccessMode.READ,
            "risk": RiskLevel.LOW,
            "input_schema": {"type": "object"},
            "executor": self.executor.id,
            "runtime_only": True,
            "replay_safe": True,
            "side_effecting": False,
        }
        return (
            Capability(
                id="live_runtime.monitor.source",
                description="Read live monitor source data from Postgres.",
                operation_types=frozenset({"monitor.source"}),
                output_evidence=frozenset({"live_runtime.monitor.source"}),
                **common,
            ),
            Capability(
                id="live_runtime.monitor.action",
                description="Run a monitor-triggered live Postgres read.",
                operation_types=frozenset({"monitor.triggered"}),
                output_evidence=frozenset({"live_runtime.monitor.action_result"}),
                **common,
            ),
            Capability(
                id="live_runtime.worker.query",
                description="Run live Postgres worker query.",
                operation_types=frozenset({"worker.query"}),
                output_evidence=frozenset({"live_runtime.worker.result"}),
                specialist_only=True,
                **common,
            ),
            Capability(
                id="live_runtime.worker.side_effect",
                owner="live_runtime",
                description="Side-effecting worker task used for lease recovery tests.",
                domains=frozenset({"db", "runtime"}),
                operation_types=frozenset({"worker.query"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.HIGH,
                input_schema={"type": "object"},
                output_evidence=frozenset({"live_runtime.worker.result"}),
                executor=self.executor.id,
                runtime_only=True,
                specialist_only=True,
                replay_safe=False,
                side_effecting=True,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def declare_evidence_schemas(self):
        return (
            EvidenceSchema(
                kind="live_runtime.monitor.source",
                owner="live_runtime",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="live_runtime.monitor.action_result",
                owner="live_runtime",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="live_runtime.worker.result",
                owner="live_runtime",
                json_schema={"type": "object"},
            ),
        )

    def get_workers(self):
        return (
            Worker(
                id="live_runtime.db_worker",
                owner="live_runtime",
                role="db_worker",
                capability_ids=frozenset(
                    {
                        "live_runtime.worker.query",
                        "live_runtime.worker.side_effect",
                    }
                ),
                input_schema={"type": "object"},
                output_evidence=frozenset({"live_runtime.worker.result"}),
                max_concurrency=5,
            ),
        )


async def _validate_and_read(
    kernel: RuntimeKernel,
    operation_id: str,
    sql: str,
    *,
    context,
) -> list[dict[str, Any]]:
    await kernel.execute_capability(
        "db.sql.validate",
        owner="postgresql",
        operation_id=operation_id,
        input={"sql": sql, "operation": "query"},
        task_metadata={"owner": "postgresql", "reason": "live_runtime_sql_validate"},
        context=context,
    )
    read = await kernel.execute_capability(
        "db.sql.execute_read",
        owner="postgresql",
        operation_id=operation_id,
        input={"sql": sql},
        task_metadata={"owner": "postgresql", "reason": "live_runtime_sql_read"},
        context=context,
    )
    query_result = next(item for item in read.evidence if item.kind == "query.result")
    return list(query_result.payload.get("rows") or [])


def _sql_for_task(task: Task) -> str:
    if task.capability_id == "live_runtime.monitor.source":
        return (
            "SELECT COUNT(*)::int AS open_high_count "
            "FROM support_tickets "
            "WHERE status = 'open' AND severity = 'high'"
        )
    return str(
        task.input.get("sql")
        or (
            "SELECT c.name "
            "FROM customers c "
            "JOIN support_tickets st ON st.customer_id = c.id "
            "WHERE st.status = 'open' AND st.severity = 'high' "
            "ORDER BY c.name"
        )
    )


def event_types(events) -> set:
    return {event.type for event in events}
