"""Propose a durable orders monitor; activation requires explicit confirmation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from daita import Agent, SQLiteSource
from daita.monitors import IntervalSchedule, MonitorDefinition, MonitorScope

from _shared import example_root, parser, seed_sales_database


async def run() -> None:
    argument_parser = parser(__doc__)
    argument_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly confirm the exact proposed hash after printing it.",
    )
    arguments = argument_parser.parse_args()
    with example_root(arguments.root, "monitor") as root:
        database = seed_sales_database(root / "sales.sqlite")
        agent = await Agent.create("orders-monitor", root=root)
        try:
            source = await agent.attach(SQLiteSource(database, name="Sales"))
            definition = MonitorDefinition(
                name="Pending orders",
                objective="Report the current count of pending orders.",
                scope=MonitorScope(source_ids=(source.id,)),
                schedule=IntervalSchedule(
                    interval_seconds=300,
                    anchor_at=datetime.now(timezone.utc),
                ),
            )
            proposal = await agent.propose_monitor(
                "pending-orders",
                definition,
                idempotency_key="example-monitor-proposal-v1",
            )
            print(f"proposal: {proposal.id}")
            print(f"candidate hash: {proposal.candidate_hash}")
            if not arguments.confirm:
                print("Monitor remains inert. Re-run with --confirm after review.")
                return
            inspection = await agent.confirm_monitor(
                proposal.id,
                candidate_hash=proposal.candidate_hash,
                actor_id="example-operator",
                reason="The scope and five-minute schedule were reviewed.",
            )
            print(f"monitor status: {inspection.monitor.status.value}")
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
