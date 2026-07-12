"""Promote a CSV file into SQLite, then use Agent.from_db().

Run:
    python examples/10_csv_to_sqlite_data_app.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbMemoryConfig, DbSourceOptions
from daita.plugins.sqlite import SQLitePlugin

QUESTION = "Show rows from csv_orders."


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


def write_orders_csv(path: Path) -> None:
    rows = [
        {
            "order_id": "1001",
            "customer": "Ada Lovelace",
            "region": "North America",
            "status": "complete",
            "product": "Analytics Notebook",
            "quantity": "2",
            "unit_price": "120.00",
        },
        {
            "order_id": "1002",
            "customer": "Grace Hopper",
            "region": "North America",
            "status": "complete",
            "product": "Pipeline Support Plan",
            "quantity": "1",
            "unit_price": "450.00",
        },
        {
            "order_id": "1003",
            "customer": "Katherine Johnson",
            "region": "Europe",
            "status": "pending",
            "product": "Data Quality Audit",
            "quantity": "1",
            "unit_price": "300.00",
        },
        {
            "order_id": "1004",
            "customer": "Mary Jackson",
            "region": "Europe",
            "status": "complete",
            "product": "Data Quality Audit",
            "quantity": "2",
            "unit_price": "300.00",
        },
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def promote_csv_to_sqlite(csv_path: Path, db_path: Path) -> Path:
    plugin = SQLitePlugin(path=str(db_path))
    try:
        await plugin.execute_script("""
            DROP TABLE IF EXISTS csv_orders;
            CREATE TABLE csv_orders (
                order_id INTEGER NOT NULL,
                customer TEXT NOT NULL,
                region TEXT NOT NULL,
                status TEXT NOT NULL,
                product TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL
            );
            """)
        with csv_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            await plugin.execute(
                """
                INSERT INTO csv_orders (
                    order_id, customer, region, status, product, quantity, unit_price
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["order_id"]),
                    row["customer"],
                    row["region"],
                    row["status"],
                    row["product"],
                    int(row["quantity"]),
                    float(row["unit_price"]),
                ),
            )
    except ImportError as exc:
        raise ImportError(
            "SQLite examples require aiosqlite. Install with: "
            "pip install 'daita-agents[sqlite]'"
        ) from exc
    finally:
        await plugin.disconnect()
    return db_path


def task_capability_sequence(result) -> list[str]:
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]


def evidence_kinds(result) -> list[str]:
    return [item.kind for item in result.evidence]


def print_operation(result) -> None:
    print("Operation")
    print(f"  id: {result.operation_id}")
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


def print_query_rows(result) -> None:
    query_result = next(
        (item.payload for item in result.evidence if item.kind == "query.result"),
        {},
    )
    rows = query_result.get("rows") or []
    if not rows:
        return
    print("Rows")
    for row in rows[:5]:
        print(f"  - {row}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Generate CSV, promote it to SQLite, and initialize Agent.from_db().",
    )
    parser.add_argument(
        "--live-llm", action="store_true", help="Use OpenAI if configured."
    )
    args = parser.parse_args()

    with TemporaryDirectory(prefix="daita_examples_") as tmpdir:
        tmp_path = Path(tmpdir)
        csv_path = tmp_path / "orders.csv"
        db_path = tmp_path / "csv_orders.sqlite"
        write_orders_csv(csv_path)
        await promote_csv_to_sqlite(csv_path, db_path)

        agent = await Agent.from_db(
            str(db_path),
            name="CsvToSQLiteDataApp",
            source_options=DbSourceOptions(cache_ttl=0),
            memory=DbMemoryConfig(enabled=False),
            **llm_options(args.live_llm),
        )
        try:
            inspection = await agent.describe()
            print(f"CSV fixture: {csv_path}")
            print(f"SQLite database: {db_path}")
            print(f"Registered plugins: {', '.join(inspection.plugin_ids)}")
            print(f"Runtime capabilities: {inspection.capability_count}")
            print()

            if args.setup_only:
                return

            result = await agent.run_detailed(QUESTION)
            print(f"Question: {QUESTION}")
            print_operation(result)
            print_query_rows(result)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
