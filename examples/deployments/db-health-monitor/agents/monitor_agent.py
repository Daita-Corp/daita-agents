"""
Database Health Monitor Agent

Provides a PostgreSQL-focused agent with tools for investigating operational
anomalies. Runtime-native monitor scheduling should be declared through
``daita.runtime.monitors`` / ``daita.runtime.scheduler`` rather than an agent-owned
polling API.
"""

import os
from typing import Any, Dict, Optional

from daita import Agent
from daita.core.tools import tool
from daita.plugins import postgresql

# ---------------------------------------------------------------------------
# Tools — give the agent the ability to investigate runtime monitor actions
# ---------------------------------------------------------------------------


@tool
async def get_slow_queries(min_duration_seconds: int = 30) -> Dict[str, Any]:
    """
    List currently running queries that exceed the given duration.

    Args:
        min_duration_seconds: Minimum query runtime in seconds (default 30).

    Returns:
        Dict with slow query details (pid, duration, query text, user, database).
    """
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        )

    url = os.getenv("DATABASE_URL")
    if not url:
        return {"error": "DATABASE_URL environment variable is not set."}

    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(
            """
            SELECT pid,
                   now() - pg_stat_activity.query_start AS duration,
                   query,
                   usename,
                   datname,
                   state
            FROM pg_stat_activity
            WHERE (now() - pg_stat_activity.query_start) > $1 * interval '1 second'
              AND state != 'idle'
              AND pid != pg_backend_pid()
            ORDER BY duration DESC
            LIMIT 20
            """,
            min_duration_seconds,
        )
        return {
            "slow_queries": [
                {
                    "pid": r["pid"],
                    "duration": str(r["duration"]),
                    "query": r["query"][:200],
                    "user": r["usename"],
                    "database": r["datname"],
                    "state": r["state"],
                }
                for r in rows
            ],
            "count": len(rows),
        }
    finally:
        await conn.close()


@tool
async def get_table_bloat(min_dead_tuples: int = 10000) -> Dict[str, Any]:
    """
    List tables with significant dead tuple bloat that may need VACUUM.

    Args:
        min_dead_tuples: Minimum dead tuple count to include (default 10000).

    Returns:
        Dict with bloated tables and their dead tuple counts.
    """
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        )

    url = os.getenv("DATABASE_URL")
    if not url:
        return {"error": "DATABASE_URL environment variable is not set."}

    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(
            """
            SELECT schemaname, relname,
                   n_live_tup, n_dead_tup,
                   CASE WHEN n_live_tup > 0
                        THEN round(n_dead_tup::numeric / n_live_tup * 100, 1)
                        ELSE 0 END AS dead_pct,
                   last_autovacuum, last_autoanalyze
            FROM pg_stat_user_tables
            WHERE n_dead_tup >= $1
            ORDER BY n_dead_tup DESC
            LIMIT 20
            """,
            min_dead_tuples,
        )
        return {
            "bloated_tables": [
                {
                    "schema": r["schemaname"],
                    "table": r["relname"],
                    "live_tuples": r["n_live_tup"],
                    "dead_tuples": r["n_dead_tup"],
                    "dead_pct": float(r["dead_pct"]),
                    "last_autovacuum": (
                        str(r["last_autovacuum"]) if r["last_autovacuum"] else None
                    ),
                }
                for r in rows
            ],
            "count": len(rows),
        }
    finally:
        await conn.close()


@tool
async def get_connection_stats() -> Dict[str, Any]:
    """
    Get current connection pool statistics.

    Returns:
        Dict with active, idle, and total connections plus the configured max.
    """
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        )

    url = os.getenv("DATABASE_URL")
    if not url:
        return {"error": "DATABASE_URL environment variable is not set."}

    conn = await asyncpg.connect(url)
    try:
        row = await conn.fetchrow("""
            SELECT
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') AS active,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') AS idle,
                (SELECT count(*) FROM pg_stat_activity) AS total,
                (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_conn
            """)
        return {
            "active": row["active"],
            "idle": row["idle"],
            "total": row["total"],
            "max_connections": row["max_conn"],
            "utilization_pct": round(row["total"] / row["max_conn"] * 100, 1),
        }
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def create_agent() -> Agent:
    """Create the Database Health Monitor investigation agent."""
    db = postgresql(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/mydb")
    )

    agent = Agent(
        name="DB Health Monitor",
        model="gpt-4o-mini",
        prompt="""\
You are a PostgreSQL database health monitor. When operators ask about an
operational anomaly, your job is to:

1. INVESTIGATE: Use the available tools to gather more context about the issue.
2. DIAGNOSE: Identify the likely root cause.
3. RECOMMEND: Suggest specific corrective actions (e.g. "VACUUM ANALYZE orders",
   "Consider adding an index on users.email", "Kill PID 12345").

Keep responses concise — operators read these under pressure. Lead with severity
(INFO / WARNING / CRITICAL) and the key metric.""",
        tools=[get_slow_queries, get_table_bloat, get_connection_stats],
        plugins=[db],
    )

    return agent
