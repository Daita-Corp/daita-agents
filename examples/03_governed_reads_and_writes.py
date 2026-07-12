"""Governed reads and source-safe write blocking with Agent.from_db().

Run:
    python examples/03_governed_reads_and_writes.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbMemoryConfig, DbSourceOptions

from local_sqlite_fixtures import temporary_sales_sqlite

READ_QUESTION = "How many customers are there?"
WRITE_PROPOSAL = "UPDATE orders SET status = 'complete' WHERE id = 2"


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


def task_capability_sequence(snapshot) -> list[str]:
    return [task.capability_id for task in snapshot.tasks]


def evidence_kinds(snapshot) -> list[str]:
    return [item.kind for item in snapshot.evidence]


def skipped_task_ids(snapshot) -> list[str]:
    return [
        event.task_id
        for event in snapshot.events
        if event.type.value == "task.skipped" and event.task_id is not None
    ]


def first_approval_id(snapshot) -> str | None:
    if not snapshot.approval_requests:
        return None
    return snapshot.approval_requests[0].approval_id


def print_snapshot(label: str, snapshot) -> None:
    print(label)
    print(f"  operation id: {snapshot.operation.id}")
    print(f"  status: {snapshot.operation.status.value}")
    print(f"  approval id: {first_approval_id(snapshot) or 'none'}")
    print("  task capability sequence:")
    for capability_id in task_capability_sequence(snapshot):
        print(f"    - {capability_id}")
    print("  evidence kinds:")
    for kind in evidence_kinds(snapshot):
        print(f"    - {kind}")
    if snapshot.completed_task_ids:
        print("  completed task ids:")
        for task_id in snapshot.completed_task_ids:
            print(f"    - {task_id}")
    if snapshot.resumable_task_ids:
        print("  resumable task ids:")
        for task_id in snapshot.resumable_task_ids:
            print(f"    - {task_id}")
    skipped = skipped_task_ids(snapshot)
    if skipped:
        print("  skipped task ids:")
        for task_id in skipped:
            print(f"    - {task_id}")
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
            name="GovernedReadsAndWrites",
            lineage=True,
            source_options=DbSourceOptions(cache_ttl=0),
            memory=DbMemoryConfig(enabled=False),
            **options,
        )
        try:
            inspection = await agent.describe()
            print(f"SQLite fixture: {db_path}")
            print(f"Runtime mode: {inspection.profile}")
            print()

            if args.setup_only:
                return

            read_result = await agent.run_detailed(READ_QUESTION)
            read_snapshot = await agent.runtime.inspect_operation(
                read_result.operation_id
            )
            print(f"Safe read: {READ_QUESTION}")
            print(f"  answer: {read_result.answer}")
            print_snapshot("Read operation", read_snapshot)

            write_result = await agent.run_detailed(WRITE_PROPOSAL)
            write_snapshot = await agent.runtime.inspect_operation(
                write_result.operation_id
            )
            print(f"Write proposal: {WRITE_PROPOSAL}")
            print_snapshot("Read-only operation", write_snapshot)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
