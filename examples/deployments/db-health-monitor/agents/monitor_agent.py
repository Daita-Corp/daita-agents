"""
Database Health Monitor Agent

Continuously monitors a PostgreSQL database for operational anomalies using
the @agent.watch() system. Watches poll SQL conditions on an interval and
fire when thresholds are crossed — the agent then diagnoses the issue and
recommends corrective action.

Three watches:
1. Slow queries     — fires when queries running longer than 30s exceed a count
2. Connection usage — fires when active connections approach the pool limit
3. Dead tuples      — fires when bloat in any table exceeds a threshold

Each watch also uses on_resolve=True to notify when conditions return to normal.
"""

import os
from datetime import timedelta
from typing import Any, Dict, Optional

from daita import Agent
from daita.core.tools import tool
from daita.core.watch import WatchEvent
from daita.plugins import postgresql


# ---------------------------------------------------------------------------
# Tools — give the agent the ability to investigate further when a watch fires
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
                    "last_autovacuum": str(r["last_autovacuum"])
                    if r["last_autovacuum"]
                    else None,
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
        row = await conn.fetchrow(
            """
            SELECT
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') AS active,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') AS idle,
                (SELECT count(*) FROM pg_stat_activity) AS total,
                (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_conn
            """
        )
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
# Agent creation with watches
# ---------------------------------------------------------------------------


def create_agent(
    slow_query_interval: str = "1m",
    connection_interval: str = "30s",
    bloat_interval: str = "5m",
) -> Agent:
    """Create the Database Health Monitor agent with three watches.

    Args:
        slow_query_interval: How often to check for slow queries (default "1m").
        connection_interval: How often to check connection usage (default "30s").
        bloat_interval: How often to check table bloat (default "5m").
    """
    db = postgresql(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/mydb")
    )

    agent = Agent(
        name="DB Health Monitor",
        model="gpt-4o-mini",
        prompt="""\
You are a PostgreSQL database health monitor. When a watch fires, you receive a
WatchEvent describing an operational anomaly. Your job is to:

1. INVESTIGATE: Use the available tools to gather more context about the issue.
2. DIAGNOSE: Identify the likely root cause.
3. RECOMMEND: Suggest specific corrective actions (e.g. "VACUUM ANALYZE orders",
   "Consider adding an index on users.email", "Kill PID 12345").

When a watch resolves (event.resolved=True), briefly acknowledge recovery.

Keep responses concise — operators read these under pressure. Lead with severity
(INFO / WARNING / CRITICAL) and the key metric.""",
        tools=[get_slow_queries, get_table_bloat, get_connection_stats],
        plugins=[db],
    )

    # -- Watch 1: Slow queries ------------------------------------------------
    @agent.watch(
        source=db,
        condition="SELECT COUNT(*) FROM pg_stat_activity "
        "WHERE state != 'idle' "
        "AND (now() - query_start) > interval '30 seconds' "
        "AND pid != pg_backend_pid()",
        threshold=lambda count: count >= 3,
        interval=slow_query_interval,
        on_resolve=True,
        cooldown=True,
    )
    async def on_slow_queries(event: WatchEvent):
        if event.resolved:
            result = await agent.run(
                f"Slow query alert RESOLVED. "
                f"Count dropped from {event.previous_value} to {event.value}."
            )
        else:
            result = await agent.run(
                f"ALERT: {event.value} queries running longer than 30s "
                f"(previous check: {event.previous_value}). "
                f"Investigate with get_slow_queries and recommend action."
            )
        print(f"\n{'='*65}\n[SLOW QUERIES] {result}\n{'='*65}")

    # -- Watch 2: Connection pool pressure ------------------------------------
    @agent.watch(
        source=db,
        condition="SELECT round(count(*)::numeric / "
        "(SELECT setting::int FROM pg_settings WHERE name = 'max_connections') "
        "* 100, 1) FROM pg_stat_activity",
        threshold=lambda pct: pct > 80,
        interval=connection_interval,
        on_resolve=True,
        cooldown=True,
    )
    async def on_connection_pressure(event: WatchEvent):
        if event.resolved:
            result = await agent.run(
                f"Connection pressure RESOLVED. "
                f"Utilization dropped from {event.previous_value}% to {event.value}%."
            )
        else:
            result = await agent.run(
                f"ALERT: Connection utilization at {event.value}% "
                f"(previous: {event.previous_value}%). "
                f"The database is under connection pressure. If get_connection_stats "
                f"fails with 'too many clients', that confirms the issue — diagnose "
                f"based on the utilization percentage alone and recommend action "
                f"(e.g. kill idle connections, increase max_connections, add pgbouncer)."
            )
        print(f"\n{'='*65}\n[CONNECTIONS] {result}\n{'='*65}")

    # -- Watch 3: Table bloat -------------------------------------------------
    @agent.watch(
        source=db,
        condition="SELECT COALESCE(MAX(n_dead_tup), 0) FROM pg_stat_user_tables",
        threshold=lambda dead: dead > 100_000,
        interval=bloat_interval,
        on_resolve=True,
        cooldown=True,
    )
    async def on_table_bloat(event: WatchEvent):
        if event.resolved:
            result = await agent.run(
                f"Table bloat RESOLVED. "
                f"Max dead tuples dropped from {event.previous_value} to {event.value}."
            )
        else:
            result = await agent.run(
                f"ALERT: Table bloat detected — max dead tuples: {event.value} "
                f"(previous: {event.previous_value}). "
                f"Investigate with get_table_bloat and recommend VACUUM targets."
            )
        print(f"\n{'='*65}\n[TABLE BLOAT] {result}\n{'='*65}")

    return agent
