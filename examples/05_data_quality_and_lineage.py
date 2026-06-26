"""Data quality and lineage with Agent.from_db().

Run:
    python examples/05_data_quality_and_lineage.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from daita.agents.agent import Agent

from local_sqlite_fixtures import temporary_sales_sqlite

QUALITY_REQUEST = "Profile data quality for the orders table"
LINEAGE_REQUEST = "Trace lineage for the orders table"


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


def task_capability_sequence(result) -> list[str]:
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]


def evidence_kinds(result) -> list[str]:
    return [item.kind for item in result.evidence]


def print_runtime_summary(inspection) -> None:
    data_team_capabilities = [
        capability_id
        for capability_id in inspection.capability_ids
        if capability_id.startswith(("data_quality:", "lineage:"))
    ]
    print("Runtime")
    print(f"  plugins: {', '.join(inspection.plugin_ids)}")
    print("  data team capabilities:")
    for capability_id in data_team_capabilities:
        print(f"    - {capability_id}")
    print()


def print_operation(result, *, title: str) -> None:
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


def print_quality_evidence(result) -> None:
    quality = next(
        (item.payload for item in result.evidence if item.kind == "quality.profile"),
        None,
    )
    if not quality:
        print("Quality evidence: not available")
        print()
        return

    profile = quality.get("profile") or {}
    print("Quality evidence")
    print(f"  table: {quality.get('table')}")
    print(f"  columns profiled: {quality.get('columns_profiled')}")
    for column, stats in list(profile.items())[:4]:
        if not isinstance(stats, dict):
            continue
        print(
            "  "
            f"{column}: nulls={stats.get('null_count')}, "
            f"distinct={stats.get('distinct_count')}"
        )
    print()


def print_lineage_evidence(result) -> None:
    trace = next(
        (item.payload for item in result.evidence if item.kind == "lineage.trace"),
        None,
    )
    if not trace:
        print("Lineage evidence: not available")
        print()
        return

    lineage = trace.get("lineage") or {}
    print("Lineage evidence")
    print(f"  entity: {lineage.get('entity_id')}")
    print(f"  upstream entities: {trace.get('upstream_count')}")
    print(f"  downstream entities: {trace.get('downstream_count')}")
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
            name="DataQualityAndLineage",
            mode="data_team",
            quality=True,
            lineage=True,
            cache_ttl=0,
            memory=False,
            **options,
        )
        try:
            inspection = await agent.describe()
            print_runtime_summary(inspection)

            if args.setup_only:
                return

            quality = await agent.run_detailed(
                QUALITY_REQUEST,
                mode="quality.check",
            )
            print_operation(quality, title=f"Quality request: {QUALITY_REQUEST}")
            print_quality_evidence(quality)

            lineage = await agent.run_detailed(
                LINEAGE_REQUEST,
                mode="lineage.trace",
            )
            print_operation(lineage, title=f"Lineage request: {LINEAGE_REQUEST}")
            print_lineage_evidence(lineage)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
