"""Use catalog search and relationship-aware SQL in one MVP run."""

from __future__ import annotations

import asyncio

from daita import SQLiteSource

from _shared import (
    ScriptedModel,
    create_offline_agent,
    example_root,
    final_response,
    parser,
    seed_sales_database,
    tool_response,
)


async def run() -> None:
    arguments = parser(__doc__).parse_args()
    with example_root(arguments.root, "joins") as root:
        database = seed_sales_database(root / "sales.sqlite")
        model = ScriptedModel()
        agent = await create_offline_agent("catalog-joins", root, model)
        try:
            source = await agent.attach(SQLiteSource(database, name="Sales"))
            model.extend(
                tool_response(
                    "find-orders",
                    "catalog_search",
                    {"query": "orders customers", "source_id": source.id, "limit": 5},
                ),
                tool_response(
                    "join-orders",
                    "data_query_sqlite",
                    {
                        "source_id": source.id,
                        "sql": (
                            "SELECT o.id, c.name, o.status, o.amount "
                            "FROM orders AS o JOIN customers AS c "
                            "ON c.id = o.customer_id ORDER BY o.id"
                        ),
                        "parameters": [],
                    },
                ),
                final_response(
                    "The joined result contains 3 orders with customer names."
                ),
            )
            result = await agent.run("Join orders to their customers.")
            transcript = await agent.transcript(result.run_id)
            print(f"answer: {result.final_text}")
            print(f"run: {result.run_id} ({result.kind.value})")
            print(f"persisted messages: {len(transcript.messages)}")
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
