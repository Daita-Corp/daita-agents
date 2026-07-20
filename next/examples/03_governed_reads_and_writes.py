"""Preview one controlled SQLite update and stop for explicit human approval."""

from __future__ import annotations

import asyncio
import sqlite3

from daita import SQLiteSource
from daita.catalog import ResourceKind, catalog_resource_id

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


def order_status(database) -> str:
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT status FROM orders WHERE id = 'order-1'"
        ).fetchone()
    assert row is not None
    return str(row[0])


async def run() -> None:
    arguments = parser(__doc__).parse_args()
    with example_root(arguments.root, "governed-write") as root:
        database = seed_sales_database(root / "sales.sqlite")
        model = ScriptedModel()
        agent = await create_offline_agent("governed-write", root, model)
        try:
            source = await agent.attach(
                SQLiteSource(database, name="Orders", allow_writes=True)
            )
            resource_id = catalog_resource_id(
                source.id,
                ResourceKind.TABLE,
                "main.orders",
            )
            recipe: dict[str, object] = {
                "source_id": source.id,
                "resource_id": resource_id,
                "key_column": "id",
                "key_value": "order-1",
                "target_column": "status",
                "expected_value": "pending",
                "new_value": "complete",
            }
            model.extend(
                tool_response(
                    "preview-order-update",
                    "data_preview_sqlite_update",
                    recipe,
                ),
                tool_response(
                    "apply-order-update",
                    "data_update_sqlite",
                    {**recipe, "impact_evidence_id": "evidence-1"},
                ),
                final_response("order-1 is complete.", "evidence-2"),
            )
            result = await agent.run(
                "Preview changing order-1 from pending to complete, then wait "
                "for approval."
            )
            snapshot = await agent.inspect(result.operation_id)
            print_snapshot(snapshot)
            print(f"database status remains: {order_status(database)}")
            if snapshot.approvals:
                approval = snapshot.approvals[0]
                print(f"pending approval: {approval.id}")
                print(
                    "No decision was made. Inspect the impact evidence, then call "
                    "Agent.approve(...) or Agent.reject(...) explicitly."
                )
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
