"""Durable local order monitor with Agent.from_db().

Run:
    python examples/07_monitor_orders.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from daita.agents.agent import Agent
from daita.db import DbRuntimeOptions

from local_sqlite_fixtures import seed_sales_sqlite

NOW = "2026-06-25T12:00:00+00:00"


def llm_options(use_live_llm: bool) -> dict[str, Any]:
    """Use OpenAI only when the caller explicitly asks for live synthesis."""
    if not use_live_llm:
        print("Using deterministic DB runtime output. Pass --live-llm to use OpenAI.\n")
        return {}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set; using deterministic DB runtime output.\n")
        return {}
    return {
        "llm_provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        "api_key": api_key,
        "temperature": 0,
    }


def pending_orders_observation() -> dict[str, Any]:
    return {
        "kind": "metric_sql",
        "metric": "pending_count",
        "sql": "select count(*) as pending_count from orders where status = 'pending'",
        "value_path": "rows.0.pending_count",
        "source_scope": ["orders"],
        "capability_owner": "sqlite",
    }


def evidence_kinds(snapshot) -> list[str]:
    return [item.kind for item in snapshot.evidence]


def task_capability_sequence(snapshot) -> list[str]:
    return [task.capability_id for task in snapshot.tasks]


def print_monitor(monitor) -> None:
    print("Created monitor")
    print(f"  id: {monitor.id}")
    print(f"  status: {monitor.status}")
    print(f"  schedule: {monitor.schedule}")
    print(f"  trigger: {monitor.trigger}")
    print(f"  observation kind: {monitor.observation_plan.get('kind')}")
    print()


def print_inspection(label: str, inspection) -> None:
    print(label)
    print(f"  monitor id: {inspection.monitor.id}")
    print(f"  status: {inspection.monitor.status}")
    print(f"  run count: {len(inspection.runs)}")
    if inspection.state is not None:
        print(f"  last tick operation id: {inspection.state.last_tick_operation_id}")
        print(
            "  last triggered operation id: "
            f"{inspection.state.last_triggered_operation_id}"
        )
        print(f"  last value summary: {inspection.state.last_value_summary}")
    print()


def print_run(run) -> None:
    print("Monitor run")
    print(f"  run id: {run.id}")
    print(f"  operation id: {run.operation_id}")
    print(f"  status: {run.status}")
    print(f"  triggered: {run.triggered}")
    print(f"  reason: {run.summary.get('reason')}")
    print(f"  triggered operation id: {run.summary.get('triggered_operation_id')}")
    print()


def print_operation_snapshot(label: str, snapshot) -> None:
    print(label)
    print(f"  operation id: {snapshot.operation.id}")
    print(f"  status: {snapshot.operation.status.value}")
    print("  task capability sequence:")
    for capability_id in task_capability_sequence(snapshot):
        print(f"    - {capability_id}")
    print("  evidence kinds:")
    for kind in evidence_kinds(snapshot):
        print(f"    - {kind}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Seed SQLite and initialize Agent.from_db() without running prompts.",
    )
    parser.add_argument(
        "--live-llm", action="store_true", help="Use OpenAI if configured."
    )
    args = parser.parse_args()

    with TemporaryDirectory(prefix="daita_examples_") as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = await seed_sales_sqlite(tmp_path / "daita_sales.sqlite")
        store_path = tmp_path / "runtime_store.sqlite"
        agent = await Agent.from_db(
            str(db_path),
            name="MonitorOrders",
            cache_ttl=0,
            memory=False,
            runtime=DbRuntimeOptions(store="sqlite", store_path=store_path),
            **llm_options(args.live_llm),
        )
        try:
            inspection = await agent.describe()
            print(f"SQLite fixture: {db_path}")
            print(f"Runtime store: {store_path}")
            print(f"Runtime capabilities: {inspection.capability_count}")
            print()

            monitor = await agent.monitor(
                monitor_id="pending_orders_local",
                name="Pending Orders Local",
                schedule={"interval_seconds": 0},
                watch="Count pending orders in the local sales fixture.",
                observation_plan=pending_orders_observation(),
                trigger={"path": "pending_count", "gt": 10},
                source_scope=("orders",),
                metadata={"example": "07_monitor_orders"},
            )
            print_monitor(monitor)

            listed = await agent.list_monitors()
            print(f"List monitors: {', '.join(item.id for item in listed)}")
            print()

            before = await agent.inspect_monitor(monitor.id)
            print_inspection("Before tick", before)

            if args.setup_only:
                return

            run = (await agent.runtime.tick_monitors(now=NOW))[0]
            print_run(run)

            tick_snapshot = await agent.runtime.inspect_operation(run.operation_id)
            print_operation_snapshot("Tick operation", tick_snapshot)

            child_id = run.summary.get("triggered_operation_id")
            if child_id:
                child_snapshot = await agent.runtime.inspect_operation(child_id)
                print_operation_snapshot("Triggered operation", child_snapshot)

            after = await agent.inspect_monitor(monitor.id)
            print_inspection("After tick", after)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
