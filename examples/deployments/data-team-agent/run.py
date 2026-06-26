"""Run the data-team-agent deployment template locally."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.data_team_agent import (  # noqa: E402
    CATALOG_QUERY,
    DEMO_NOW,
    LINEAGE_REQUEST,
    MEMORY_RULE,
    QUALITY_REQUEST,
    create_data_team_agent,
    default_paths,
    ensure_pending_orders_monitor,
    evidence_kinds,
    seed_local_sqlite,
    task_capability_sequence,
)


def print_runtime_summary(inspection) -> None:
    print("Runtime")
    print(f"  profile: {inspection.profile}")
    print(f"  plugins: {', '.join(inspection.plugin_ids)}")
    print(f"  capabilities: {inspection.capability_count}")
    print()


def print_operation(title: str, result) -> None:
    print(title)
    print(f"  operation id: {result.operation_id}")
    print(f"  status: {result.status.value}")
    print(f"  intent kind: {result.intent.kind.value}")
    print("  task capability sequence:")
    for capability_id in task_capability_sequence(result):
        print(f"    - {capability_id}")
    print("  evidence kinds:")
    for kind in evidence_kinds(result):
        print(f"    - {kind}")
    print(f"  answer: {result.answer}")
    print()


def print_monitor_inspection(label: str, inspection) -> None:
    print(label)
    print(f"  monitor id: {inspection.monitor.id}")
    print(f"  status: {inspection.monitor.status}")
    print(f"  observation kind: {inspection.monitor.observation_plan.get('kind')}")
    print(f"  run count: {len(inspection.runs)}")
    if inspection.state is not None:
        print(f"  last operation id: {inspection.state.last_operation_id}")
        print(f"  last value summary: {inspection.state.last_value_summary}")
    print()


async def run_demo(args: argparse.Namespace) -> None:
    paths = default_paths(args.base_dir)
    await seed_local_sqlite(paths.db_path)

    print("Local files")
    print(f"  SQLite fixture: {paths.db_path}")
    print(f"  Runtime store: {paths.runtime_store_path}")
    print(f"  Memory dir: {paths.memory_dir}")
    if not args.live_llm:
        print("  LLM: deterministic runtime output")
    elif not args.openai_configured:
        print("  LLM: OPENAI_API_KEY not set; using deterministic runtime output")
    else:
        print("  LLM: OpenAI live synthesis enabled")
    print()

    agent = await create_data_team_agent(paths, use_live_llm=args.live_llm)
    try:
        inspection = await agent.describe()
        print_runtime_summary(inspection)

        monitor = await ensure_pending_orders_monitor(agent)
        monitor_inspection = await agent.inspect_monitor(monitor.id)
        print_monitor_inspection("Monitor", monitor_inspection)

        if args.setup_only:
            return

        memory = await agent.run_detailed(
            f"Remember that {MEMORY_RULE}",
            mode="memory.update",
            metadata={
                "kind": "business_rule",
                "key": "business_rule:completed_orders_status",
                "text": MEMORY_RULE,
                "importance": 0.9,
                "metadata": {"table": "orders", "column": "status"},
            },
        )
        print_operation("Memory update", memory)

        quality = await agent.run_detailed(QUALITY_REQUEST, mode="quality.check")
        print_operation(f"Quality request: {QUALITY_REQUEST}", quality)

        lineage = await agent.run_detailed(LINEAGE_REQUEST, mode="lineage.trace")
        print_operation(f"Lineage request: {LINEAGE_REQUEST}", lineage)

        query = await agent.run_detailed(CATALOG_QUERY)
        print_operation(f"Catalog-backed query: {CATALOG_QUERY}", query)

        if args.tick_monitor:
            run = (await agent.runtime.tick_monitors(now=DEMO_NOW))[0]
            print("Monitor tick")
            print(f"  run id: {run.id}")
            print(f"  operation id: {run.operation_id}")
            print(f"  status: {run.status}")
            print(f"  triggered: {run.triggered}")
            print()
            after = await agent.inspect_monitor(monitor.id)
            print_monitor_inspection("Monitor after tick", after)
    finally:
        await agent.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        help="Directory for local SQLite, runtime store, and memory files.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Seed SQLite, initialize DbRuntime, and create/inspect the monitor.",
    )
    parser.add_argument(
        "--tick-monitor",
        action="store_true",
        help="Ask DbRuntime to run due monitors once after the demo operations.",
    )
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Use OpenAI synthesis only when OPENAI_API_KEY is also set.",
    )
    args = parser.parse_args()
    import os

    args.openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    return args


if __name__ == "__main__":
    asyncio.run(run_demo(parse_args()))
