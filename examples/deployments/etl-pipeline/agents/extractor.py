"""
Extractor Agent

Reads unprocessed raw event logs from the source database and emits them
to the raw_data relay channel for the Transformer to consume.
"""

import os
from typing import Any, Dict

from daita import Agent
from daita.core.tools import tool
from daita.plugins import postgresql


@tool
async def get_unprocessed_events(batch_size: int = 1000) -> Dict[str, Any]:
    """
    Fetch a batch of unprocessed event records from the source database.

    Reads from the `raw_events` table. Returns records with `processed = false`,
    ordered by creation time (oldest first), limited to `batch_size` rows.

    Args:
        batch_size: Maximum number of records to fetch (default 1000, max 5000).

    Returns:
        Dict with `records` (list of event dicts) and `count`.
    """
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        )

    url = os.getenv("SOURCE_DATABASE_URL")
    if not url:
        return {"error": "SOURCE_DATABASE_URL environment variable is not set."}

    batch_size = min(max(1, batch_size), 5000)

    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(
            """
            SELECT id, event_type, user_id, session_id, properties, created_at
            FROM raw_events
            WHERE processed = false
            ORDER BY created_at ASC
            LIMIT $1
            """,
            batch_size,
        )
        records = [dict(r) for r in rows]

        # Serialise non-JSON-safe types
        for record in records:
            if record.get("created_at"):
                record["created_at"] = record["created_at"].isoformat()

        return {"records": records, "count": len(records)}
    finally:
        await conn.close()


def create_agent() -> Agent:
    """Create the Extractor agent."""
    src_db = postgresql(
        connection_string=os.getenv(
            "SOURCE_DATABASE_URL", "postgresql://localhost/source_db"
        )
    )

    return Agent(
        name="Extractor",
        model="gpt-4o-mini",
        prompt="""You extract raw event records from the source database and summarise what was found.

Your steps:
1. Call get_unprocessed_events with batch_size=1000 to fetch unprocessed records.
2. Report how many records were fetched, the date range, and a breakdown of event types.
3. If count is 0, report "No new records to process" and stop.
4. Pass the full records list downstream for transformation.

Source schema:
  raw_events (id, event_type, user_id, session_id, properties JSONB, created_at, processed BOOLEAN)

Always report anomalies: null user_ids, missing event types, or unexpected event type values.""",
        tools=[get_unprocessed_events],
        plugins=[src_db],
    )
