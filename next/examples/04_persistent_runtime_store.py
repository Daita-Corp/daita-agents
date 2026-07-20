"""Create, close, reopen, and inspect one durable v2 operation."""

from __future__ import annotations

import asyncio

from daita import Agent, SQLiteSource

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
    with example_root(arguments.root, "persistence") as root:
        database = seed_sales_database(root / "sales.sqlite")
        model = ScriptedModel()
        first = await create_offline_agent("persistent", root, model)
        try:
            source = await first.attach(SQLiteSource(database, name="Sales"))
            model.extend(
                tool_response(
                    "count-customers",
                    "data_query_sqlite",
                    {
                        "source_id": source.id,
                        "sql": "SELECT COUNT(*) AS customer_count FROM customers",
                        "parameters": [],
                    },
                ),
                final_response("There are 3 customers.", "evidence-1"),
            )
            result = await first.run("How many customers are there?")
            operation_id = result.operation_id
            agent_id = first.id
            print(f"created agent: {agent_id}")
            print(f"operation before close: {operation_id}")
        finally:
            await first.close()

        reopened = await Agent.open("persistent", root=root)
        try:
            snapshot = await reopened.inspect(operation_id)
            print(f"reopened same agent: {reopened.id == agent_id}")
            print(f"persisted status: {snapshot.operation.status.value}")
            print(f"persisted tasks: {len(snapshot.tasks)}")
            print(f"persisted evidence: {len(snapshot.evidence)}")
        finally:
            await reopened.close()


if __name__ == "__main__":
    asyncio.run(run())
