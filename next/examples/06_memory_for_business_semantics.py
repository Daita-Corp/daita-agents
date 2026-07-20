"""Learn one explicit, resource-revision-bound business-term correction."""

from __future__ import annotations

import asyncio

from daita import SQLiteSource
from daita.catalog import ResourceKind, catalog_resource_id
from daita.memory import MemoryListRequest, MemoryScope, ResourceAliasCorrection

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
    with example_root(arguments.root, "memory") as root:
        database = seed_sales_database(root / "sales.sqlite")
        model = ScriptedModel()
        agent = await create_offline_agent("business-memory", root, model)
        try:
            source = await agent.attach(SQLiteSource(database, name="Sales"))
            resource_id = catalog_resource_id(
                source.id,
                ResourceKind.TABLE,
                "main.orders",
            )
            count_sql = "SELECT COUNT(*) AS order_count FROM orders WHERE status = ?"
            model.extend(
                tool_response(
                    "find-orders",
                    "catalog_search",
                    {"query": "orders", "source_id": source.id, "limit": 5},
                ),
                tool_response(
                    "count-completed-spelling",
                    "data_query_sqlite",
                    {
                        "source_id": source.id,
                        "sql": count_sql,
                        "parameters": ["completed"],
                    },
                ),
                final_response(
                    "No rows use the literal status completed.",
                    "evidence-2",
                ),
            )
            initial = await agent.run("How many completed orders are there?")
            initial_snapshot = await agent.inspect(initial.operation_id)
            hits = initial_snapshot.evidence[0].payload["hits"]
            revision = hits[0]["revision"]
            assert isinstance(revision, str)

            correction = ResourceAliasCorrection(
                source_id=source.id,
                resource_id=resource_id,
                resource_revision=revision,
                field="status",
                business_term="completed",
                stored_value="complete",
            )
            model.extend(
                tool_response(
                    "count-corrected-status",
                    "data_query_sqlite",
                    {
                        "source_id": source.id,
                        "sql": count_sql,
                        "parameters": ["complete"],
                    },
                ),
                final_response(
                    "The stored value complete matches 2 orders.",
                    "evidence-3",
                ),
            )
            learned_operation = await agent.run(correction.to_trigger_message())
            memories = await agent.list_memories(
                MemoryListRequest(
                    scope=MemoryScope(
                        agent_id=agent.id,
                        source_id=source.id,
                        resource_id=resource_id,
                    ),
                    current_resource_revision=revision,
                )
            )
            print(f"correction operation: {learned_operation.operation_id}")
            print(f"learning notices: {learned_operation.post_operation_notices}")
            print(f"current resource memories: {len(memories.items)}")
            if memories.items:
                version = memories.items[0].snapshot.version
                print(f"learned business term: {version.attributes['business_term']}")
                print(f"stored value: {version.attributes['stored_value']}")
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
