"""Drive the catalog keeper from a test without sleeping."""

from __future__ import annotations

import asyncio
import sqlite3

from daita import Agent


async def settled(agent: Agent) -> None:
    """Let the keeper finish everything that is due, then return."""

    async def finish() -> None:
        for _ in range(200):
            if not await agent._embedded.keeper.tick():
                return

    await asyncio.wait_for(finish(), timeout=10)


def stale_probes(agent: Agent) -> None:
    """Make every source look unprobed for long, so the next trigger probes."""

    for mem in agent._embedded.keeper._mems.values():
        mem.last_probe = None


def sync_count(agent: Agent) -> int:
    """Discoveries so far: the attach, then one per refresh."""

    with sqlite3.connect(agent.home / "state.db") as connection:
        return int(connection.execute("SELECT count(*) FROM syncs").fetchone()[0])


__all__ = ["settled", "stale_probes", "sync_count"]
