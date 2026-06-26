"""Inspectable operation: see what the DB runtime did for a question.

Run:
    python examples/01_inspectable_operation.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from daita.agents.agent import Agent

from local_sqlite_fixtures import temporary_sales_sqlite


def llm_options(use_live_llm: bool) -> dict[str, Any]:
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


def task_capability_sequence(result) -> list[str]:
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]


def evidence_kinds(result) -> list[str]:
    return [item.kind for item in result.evidence]


def print_operation(result) -> None:
    verification = result.diagnostics.get("verification", {})
    print(f"operation_id: {result.operation_id}")
    print(f"status: {result.status.value}")
    print(f"intent kind: {result.intent.kind.value}")
    print(f"required capabilities: {', '.join(result.contract.required_capabilities)}")
    print("task capability sequence:")
    for capability_id in task_capability_sequence(result):
        print(f"  - {capability_id}")
    print("evidence kinds:")
    for kind in evidence_kinds(result):
        print(f"  - {kind}")
    print(f"verification passed: {verification.get('passed')}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument(
        "--live-llm", action="store_true", help="Use OpenAI if configured."
    )
    args = parser.parse_args()

    async with temporary_sales_sqlite() as db_path:
        options = llm_options(args.live_llm)
        agent = await Agent.from_db(
            str(db_path),
            name="InspectableOperation",
            cache_ttl=0,
            memory=False,
            **options,
        )
        try:
            inspection = await agent.describe()
            print("Runtime")
            print(f"  plugins: {len(inspection.plugin_ids)}")
            print(f"  capabilities: {inspection.capability_count}")
            print(f"  evidence schemas: {inspection.evidence_schema_count}")
            print()

            if args.setup_only:
                return

            result = await agent.run_detailed("What are the top products by revenue?")
            print(f"Answer: {result.answer}\n")
            print_operation(result)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
