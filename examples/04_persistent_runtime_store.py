"""Persistent runtime state across Agent.from_db() instances.

Run:
    python examples/04_persistent_runtime_store.py
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

QUESTION = "How many customers are there?"


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


async def create_agent(db_path: Path, store_path: Path, options: dict[str, Any]):
    return await Agent.from_db(
        str(db_path),
        name="PersistentRuntimeStore",
        cache_ttl=0,
        memory=False,
        runtime=DbRuntimeOptions(store="sqlite", store_path=store_path),
        **options,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument(
        "--live-llm", action="store_true", help="Use OpenAI if configured."
    )
    args = parser.parse_args()

    with TemporaryDirectory(prefix="daita_examples_") as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = await seed_sales_sqlite(tmp_path / "daita_sales.sqlite")
        store_path = tmp_path / "runtime_store.sqlite"
        options = llm_options(args.live_llm)

        first = await create_agent(db_path, store_path, options)
        try:
            inspection = await first.describe()
            print(f"SQLite fixture: {db_path}")
            print(f"Runtime store: {store_path}")
            print(f"Initial operation count: {inspection.operation_count}")
            print()

            if args.setup_only:
                return

            result = await first.run_detailed(QUESTION)
            print(f"Question: {QUESTION}")
            print(f"Answer: {result.answer}")
            print(f"Original operation id: {result.operation_id}")
        finally:
            await first.stop()

        second = await create_agent(db_path, store_path, options)
        try:
            reopened = await second.runtime.inspect_operation(result.operation_id)
            if reopened is None:
                print("Reopened runtime could not find the original operation.")
                return
            print()
            print("Reopened runtime")
            print(f"  inspected operation id: {reopened.operation.id}")
            print(f"  status: {reopened.operation.status.value}")
            print(f"  persisted task count: {len(reopened.tasks)}")
            print(f"  persisted evidence count: {len(reopened.evidence)}")
        finally:
            await second.stop()


if __name__ == "__main__":
    asyncio.run(main())
