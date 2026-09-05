"""Promote a local CSV with stdlib SQLite, then query it through Daita."""

from __future__ import annotations

import asyncio
import csv
import sqlite3
from pathlib import Path

from _shared import (
    ScriptedModel,
    create_offline_agent,
    example_root,
    final_response,
    parser,
    tool_response,
)
from daita import SQLiteSource

ROWS = (
    ("1001", "Ada Lovelace", "complete", "120.00"),
    ("1002", "Grace Hopper", "complete", "450.00"),
    ("1003", "Katherine Johnson", "pending", "300.00"),
)


def write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("order_id", "customer", "status", "amount"))
        writer.writerows(ROWS)


def promote(csv_path: Path, database: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TABLE csv_orders (
                order_id TEXT PRIMARY KEY,
                customer TEXT NOT NULL,
                status TEXT NOT NULL,
                amount REAL NOT NULL
            )
            """)
        connection.executemany(
            "INSERT INTO csv_orders VALUES (?, ?, ?, ?)",
            (
                (
                    row["order_id"],
                    row["customer"],
                    row["status"],
                    float(row["amount"]),
                )
                for row in rows
            ),
        )


async def run() -> None:
    arguments = parser(__doc__).parse_args()
    with example_root(arguments.root, "csv-app") as root:
        csv_path = root / "orders.csv"
        database = root / "orders.sqlite"
        write_csv(csv_path)
        promote(csv_path, database)
        model = ScriptedModel()
        agent = await create_offline_agent("csv-data-app", root, model)
        try:
            source = await agent.attach(SQLiteSource(database, name="CSV orders"))
            resources = await agent.list_catalog_resources(source_id=source.id)
            model.extend(
                tool_response(
                    "summarize-csv-orders",
                    "data_query",
                    {
                        "source_id": source.id,
                        "resource_ids": (resources[0].id,),
                        "sql": (
                            "SELECT status, COUNT(*) AS order_count, "
                            "SUM(amount) AS total_amount FROM csv_orders "
                            "GROUP BY status ORDER BY status"
                        ),
                        "parameters": [],
                    },
                ),
                final_response(
                    "The CSV contains 2 complete orders and 1 pending order."
                ),
            )
            result = await agent.run("Summarize CSV orders by status.")
            print(f"CSV: {csv_path}")
            print(f"SQLite: {database}")
            print(f"answer: {result.final_text}")
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
