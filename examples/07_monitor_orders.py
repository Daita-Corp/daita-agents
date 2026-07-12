"""Durable local order monitor with Agent.from_db().

Run:
    python examples/07_monitor_orders.py
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from daita.agents.agent import Agent
from daita.db import (
    DbLLMConfig,
    DbMemoryConfig,
    DbRuntimeOptions,
    DbSourceOptions,
)
from daita.db.monitor_scheduler import DbMonitorScheduler

from local_sqlite_fixtures import seed_sales_sqlite

NOW = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
MAX_PASS_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 0.05
MAX_BACKOFF_SECONDS = 0.2


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
        "llm": DbLLMConfig(
            provider="openai",
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            api_key=api_key,
            temperature=0,
        )
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


def new_host_metrics() -> dict[str, int]:
    return {
        "due": 0,
        "claimed": 0,
        "lease_lost": 0,
        "succeeded": 0,
        "blocked": 0,
        "failed": 0,
        "triggered": 0,
        "pass_failed": 0,
    }


def record_host_metrics(metrics: dict[str, int], results) -> None:
    for result in results:
        reason = result.run.summary.get("reason")
        if result.claimed or reason == "lease_lost":
            metrics["due"] += 1
        if result.claimed:
            metrics["claimed"] += 1
        if reason == "lease_lost":
            metrics["lease_lost"] += 1
        if result.run.status in {"succeeded", "triggered"}:
            metrics["succeeded"] += 1
        elif result.run.status == "blocked":
            metrics["blocked"] += 1
        elif result.run.status == "failed":
            metrics["failed"] += 1
        if result.run.triggered:
            metrics["triggered"] += 1


async def run_hosted_monitor_pass(scheduler: DbMonitorScheduler, *, now: datetime):
    """Run one deterministic hosted pass with bounded application retry."""
    stop = asyncio.Event()  # A recurring host would set this from SIGINT/SIGTERM.
    metrics = new_host_metrics()
    backoff = INITIAL_BACKOFF_SECONDS
    active_pass = None
    try:
        for attempt in range(1, MAX_PASS_ATTEMPTS + 1):
            if stop.is_set():
                break
            active_pass = asyncio.create_task(scheduler.run_once(now=now))
            try:
                results = await active_pass
            except Exception:
                metrics["pass_failed"] += 1
                if attempt == MAX_PASS_ATTEMPTS:
                    raise
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            else:
                backoff = INITIAL_BACKOFF_SECONDS  # Reset after success.
                record_host_metrics(metrics, results)
                return results, metrics
    finally:
        stop.set()  # Stop new passes before teardown.
        if active_pass is not None and not active_pass.done():
            await active_pass
    return (), metrics


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
            source_options=DbSourceOptions(cache_ttl=0),
            memory=DbMemoryConfig(enabled=False),
            runtime=DbRuntimeOptions(store="sqlite", store_path=store_path),
            **llm_options(args.live_llm),
        )
        scheduler = DbMonitorScheduler(
            runtime=agent.runtime,
            scheduler_id=(
                os.getenv("DAITA_MONITOR_SCHEDULER_ID")
                or f"monitor-orders-example-{os.getpid()}"
            ),
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

            results, metrics = await run_hosted_monitor_pass(scheduler, now=NOW)
            run = results[0].run
            print(f"Scheduler: {scheduler.scheduler_id}")
            print(f"Host metrics: {metrics}")
            print()
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
