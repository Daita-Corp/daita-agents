"""Inspect the durable tasks, evidence, and events behind one answer."""

from __future__ import annotations

import asyncio

from daita import SQLiteSource

from _shared import (
    ScriptedModel,
    create_offline_agent,
    example_root,
    final_response,
    parser,
    print_snapshot,
    seed_sales_database,
    tool_response,
)


async def run() -> None:
    arguments = parser(__doc__).parse_args()
    with example_root(arguments.root, "inspect") as root:
        database = seed_sales_database(root / "sales.sqlite")
        model = ScriptedModel()
        agent = await create_offline_agent("inspectable", root, model)
        try:
            source = await agent.attach(SQLiteSource(database, name="Sales"))
            model.extend(
                tool_response(
                    "regional-revenue",
                    "data_query_sqlite",
                    {
                        "source_id": source.id,
                        "sql": (
                            "SELECT c.region, SUM(o.amount) AS revenue "
                            "FROM orders AS o JOIN customers AS c "
                            "ON c.id = o.customer_id GROUP BY c.region "
                            "ORDER BY revenue DESC"
                        ),
                        "parameters": [],
                    },
                ),
                final_response(
                    "North America has 570 in revenue and Europe has 300.",
                    "evidence-1",
                ),
            )
            result = await agent.run("Summarize revenue by region.")
            snapshot = await agent.inspect(result.operation_id)
            print(f"answer: {result.final_text}")
            print_snapshot(snapshot)
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
