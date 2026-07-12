"""Catalog-assisted joins over a local relational SQLite fixture.

Run:
    python examples/02_catalog_assisted_joins.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbMemoryConfig, DbSourceOptions

from local_sqlite_fixtures import temporary_sales_sqlite

JOIN_QUESTION = "Join orders to customers using their relationship and return records"


def llm_options(use_live_llm: bool) -> dict[str, Any]:
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


def task_capability_sequence(result) -> list[str]:
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]


def print_catalog_evidence(result) -> None:
    interesting = {
        "catalog.source_registered",
        "schema.search_result",
        "schema.relationship_path",
    }
    print("Catalog evidence:")
    for item in result.evidence:
        if item.kind not in interesting:
            continue
        payload = item.payload
        keys = ", ".join(sorted(str(key) for key in payload))
        print(f"  - {item.kind} ({keys})")


def print_query_rows(result) -> None:
    query_result = next(
        (item.payload for item in result.evidence if item.kind == "query.result"),
        {},
    )
    rows = query_result.get("rows") or []
    if not rows:
        return
    print("Rows:")
    for row in rows[:5]:
        print(f"  - {row}")


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
            name="CatalogAssistedJoins",
            source_options=DbSourceOptions(cache_ttl=0),
            memory=DbMemoryConfig(enabled=False),
            **options,
        )
        try:
            inspection = await agent.describe()
            print(f"SQLite fixture: {db_path}")
            print(f"Registered plugins: {', '.join(inspection.plugin_ids)}")
            print()

            if args.setup_only:
                return

            result = await agent.run_detailed(JOIN_QUESTION)
            print(f"Question: {JOIN_QUESTION}")
            print(f"Answer: {result.answer}")
            print(f"Intent: {result.intent.kind.value}")
            print("Task capability sequence:")
            for capability_id in task_capability_sequence(result):
                print(f"  - {capability_id}")
            print()
            print_catalog_evidence(result)
            print_query_rows(result)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
