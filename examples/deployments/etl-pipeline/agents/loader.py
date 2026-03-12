"""
Loader Agent

Receives cleaned, normalised event records from the Transformer and inserts
them into the destination reporting database.
"""

import json
import os
from typing import Any, Dict, List

from daita import Agent
from daita.core.tools import tool
from daita.plugins import postgresql


@tool
async def load_to_destination(records_json: str) -> Dict[str, Any]:
    """
    Insert cleaned event records into the destination `fact_events` table.

    Uses INSERT ... ON CONFLICT DO NOTHING to make the operation idempotent —
    re-running the loader with the same records is always safe.

    Destination schema:
        fact_events (
            id          TEXT PRIMARY KEY,
            event_type  TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            session_id  TEXT,
            properties  JSONB,
            event_date  DATE,
            processed_at TIMESTAMPTZ
        )

    Args:
        records_json: JSON string of cleaned event records.

    Returns:
        Dict with `inserted`, `skipped`, and `total` counts.
    """
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        )

    url = os.getenv("DEST_DATABASE_URL")
    if not url:
        return {"error": "DEST_DATABASE_URL environment variable is not set."}

    try:
        records = json.loads(records_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid JSON input: {e}"}

    if not records:
        return {"inserted": 0, "skipped": 0, "total": 0}

    conn = await asyncpg.connect(url)
    inserted = 0
    skipped = 0
    errors: List[str] = []

    try:
        for record in records:
            try:
                result = await conn.execute(
                    """
                    INSERT INTO fact_events (id, event_type, user_id, session_id,
                                            properties, event_date, processed_at)
                    VALUES ($1, $2, $3, $4, $5::jsonb,
                            $6::date, $7::timestamptz)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    str(record.get("id", "")),
                    record.get("event_type", ""),
                    str(record.get("user_id", "")),
                    record.get("session_id"),
                    json.dumps(record.get("properties") or {}),
                    (
                        record.get("created_at", "").split("T")[0]
                        if record.get("created_at")
                        else None
                    ),
                    record.get("processed_at"),
                )
                # asyncpg returns "INSERT 0 N" — N=1 means inserted, N=0 means skipped
                n = int(result.split()[-1])
                if n == 1:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                errors.append(str(e))
                skipped += 1

        result_dict: Dict[str, Any] = {
            "inserted": inserted,
            "skipped": skipped,
            "total": len(records),
        }
        if errors:
            result_dict["errors"] = errors[:10]  # First 10 errors only

        return result_dict
    finally:
        await conn.close()


def create_agent() -> Agent:
    """Create the Loader agent."""
    dest_db = postgresql(
        connection_string=os.getenv(
            "DEST_DATABASE_URL", "postgresql://localhost/dest_db"
        )
    )

    return Agent(
        name="Loader",
        model="gpt-4o-mini",
        prompt="""You load cleaned event records into the destination reporting database.

Your steps:
1. Receive the cleaned records JSON from the relay channel.
2. Call load_to_destination with the records JSON.
3. Report: how many records were inserted vs skipped (duplicates), and total.
4. If there were errors, list them.
5. Summarise the ETL run: total extracted → cleaned → loaded.

Destination table: fact_events
  (id, event_type, user_id, session_id, properties JSONB, event_date, processed_at)

The insert is idempotent — re-running with the same records is safe (ON CONFLICT DO NOTHING).""",
        tools=[load_to_destination],
        plugins=[dest_db],
    )
