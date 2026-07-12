"""Business semantics memory for Agent.from_db().

Run:
    python examples/06_memory_for_business_semantics.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbMemoryConfig, DbSourceOptions

from local_sqlite_fixtures import seed_sales_sqlite

BUSINESS_RULE = "Completed orders use status value 'complete', not 'completed'."
FOLLOW_UP = "Which status value means completed orders in this database?"


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


def memory_options(base_dir: Path) -> DbMemoryConfig:
    """Keep example memory local to this temporary run."""
    from daita.plugins.memory import LocalMemoryBackend

    backend = LocalMemoryBackend(
        workspace="example_06_db_memory",
        scope="project",
        base_dir=base_dir,
    )
    return DbMemoryConfig(
        backend=backend,
        recall="auto",
        learning="off",
        retrieval_mode="structured",
        limit=3,
        char_budget=800,
    )


def evidence_kinds(result) -> list[str]:
    return [item.kind for item in result.evidence]


def task_capability_sequence(result) -> list[str]:
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]


def print_operation(label: str, result) -> None:
    print(label)
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


def print_memory_write(result) -> None:
    evidence = next(
        (
            item.payload
            for item in result.evidence
            if item.kind == "memory.semantic.write"
        ),
        None,
    )
    if not evidence:
        print("Memory write evidence: not available")
        print()
        return
    print("Memory write evidence")
    print(f"  status: {evidence.get('status')}")
    print(f"  kind: {evidence.get('kind')}")
    print(f"  category: {evidence.get('category')}")
    print(f"  structured: {(evidence.get('stored') or {}).get('structured')}")
    print()


def print_memory_recall(result) -> None:
    evidence = next(
        (
            item.payload
            for item in result.evidence
            if item.kind == "memory.semantic.recall"
        ),
        None,
    )
    if not evidence:
        print("Memory recall evidence: not available")
        print()
        return

    diagnostics = evidence.get("diagnostics") or {}
    results = evidence.get("results") or []
    print("Memory recall evidence")
    print(f"  retrieval mode: {diagnostics.get('retrieval_mode')}")
    print(f"  recalled records: {len(results)}")
    for item in results[:3]:
        text = memory_result_text(item)
        print(f"  - {text}")
    print()


def memory_result_text(item: dict[str, Any]) -> str:
    text = item.get("text") or item.get("content") or item.get("value")
    if isinstance(text, str) and text.startswith("DB memory record:"):
        _, _, raw = text.partition("\n")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return text
        return str(record.get("text") or text)
    return str(text)


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
        agent = await Agent.from_db(
            str(db_path),
            name="MemoryForBusinessSemantics",
            source_options=DbSourceOptions(cache_ttl=0),
            memory=memory_options(tmp_path / "memory"),
            **llm_options(args.live_llm),
        )
        try:
            inspection = await agent.describe()
            print(f"SQLite fixture: {db_path}")
            print(f"Memory plugin registered: {'memory' in inspection.plugin_ids}")
            print(f"Memory workspace: {tmp_path / 'memory'}")
            print()

            if args.setup_only:
                return

            write = await agent.run_detailed(
                f"Remember that {BUSINESS_RULE}",
                mode="memory.update",
                metadata={
                    "kind": "business_rule",
                    "key": "business_rule:completed_orders_status",
                    "text": BUSINESS_RULE,
                    "importance": 0.9,
                    "metadata": {"table": "orders", "column": "status"},
                },
            )
            print_operation("Memory update", write)
            print_memory_write(write)

            follow_up = await agent.run_detailed(FOLLOW_UP, mode="schema.query")
            print_operation(f"Follow-up: {FOLLOW_UP}", follow_up)
            print_memory_recall(follow_up)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
