"""Quickstart: ask questions of a local SQLite database with Agent.from_db().

Run:
    python examples/00_quickstart_sqlite_from_db.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from daita.agents.agent import Agent

from local_sqlite_fixtures import temporary_sales_sqlite

QUESTIONS = (
    "How many customers are there?",
    "What are the top products by revenue?",
    "Count orders where status = complete.",
)


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


def print_operation_metadata(result) -> None:
    telemetry = result.telemetry
    print("Operation")
    print(f"  id: {result.operation_id}")
    print(f"  status: {result.status.value}")
    print(f"  intent: {result.intent.kind.value}")
    print(f"  telemetry: {telemetry['provider']} / {telemetry['model']}")
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

    async with temporary_sales_sqlite() as db_path:
        print(f"SQLite fixture: {db_path}")
        options = llm_options(args.live_llm)
        agent = await Agent.from_db(
            str(db_path),
            name="SQLiteQuickstart",
            cache_ttl=0,
            memory=False,
            **options,
        )
        try:
            if args.setup_only:
                inspection = await agent.describe()
                print(
                    f"Ready: {inspection.source_type}, "
                    f"{inspection.capability_count} capabilities"
                )
                return

            for question in QUESTIONS:
                print(f"Question: {question}")
                result = await agent.run_detailed(question)
                print(f"Answer: {result.answer}")
                print_operation_metadata(result)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
