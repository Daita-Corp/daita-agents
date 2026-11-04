"""Observe the catalog's background refreshes without reaching into a task."""

from __future__ import annotations

import asyncio
import sqlite3

from daita import Agent


async def settled(agent: Agent) -> None:
    """Wait until every background refresh the agent has started is over."""

    await asyncio.wait_for(
        asyncio.gather(*agent._embedded._source_refreshes.values()), timeout=10
    )


def sync_count(agent: Agent) -> int:
    """Discoveries so far: the attach, then one per refresh."""

    with sqlite3.connect(agent.home / "state.db") as connection:
        return int(connection.execute("SELECT count(*) FROM syncs").fetchone()[0])


__all__ = ["settled", "sync_count"]
