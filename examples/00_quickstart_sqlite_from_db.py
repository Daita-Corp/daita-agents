"""Ask one grounded question of a cataloged SQLite source with the MVP Agent."""

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
    with example_root(arguments.root, "quickstart") as root:
        database = seed_sales_database(root / "sales.sqlite")
        model = ScriptedModel()
        agent = await create_offline_agent("quickstart", root, model)
        try:
            source = await agent.attach(SQLiteSource(database, name="Sales"))
            model.extend(
                tool_response(
                    "count-orders",
                    "data_query_sqlite",
                    {
                        "source_id": source.id,
                        "sql": "SELECT COUNT(*) AS order_count FROM orders",
                        "parameters": [],
                    },
                ),
                final_response("There are 3 orders."),
            )
            result = await agent.run("How many orders are there?")
            print(f"state root: {root}")
            print(f"source: {source.display_name} ({source.id})")
            print(f"answer: {result.final_text}")
            print(f"run: {result.run_id} ({result.kind.value})")
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
