"""
SQL Data Agent

Translates natural language questions into SQL, runs them against a PostgreSQL
database, and explains the results in plain English.

Use case: "Show me customers who haven't ordered in 90 days" → the agent
inspects the schema, generates the right SQL, executes it, and summarises output.
"""

import os
from typing import Any, Dict, List, Optional

from daita import Agent
from daita.core.tools import tool
from daita.plugins import postgresql


@tool
async def inspect_schema(table_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspect the database schema — list tables with their columns and data types.

    Always call this first so you know the exact table and column names before
    writing any SQL.

    Args:
        table_filter: Optional substring to filter table names (e.g. "order" to
                      show only tables whose name contains "order"). If omitted,
                      all tables in the public schema are returned.

    Returns:
        Dict with a list of tables, each containing column definitions.
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
        # Fetch all tables in the public schema
        tables_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        table_rows = await conn.fetch(tables_query)
        table_names = [r["table_name"] for r in table_rows]

        if table_filter:
            table_names = [t for t in table_names if table_filter.lower() in t.lower()]

        # Fetch columns for each table
        schema: List[Dict[str, Any]] = []
        for table_name in table_names:
            cols_query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = $1
                ORDER BY ordinal_position
            """
            col_rows = await conn.fetch(cols_query, table_name)
            schema.append(
                {
                    "table": table_name,
                    "columns": [
                        {
                            "name": r["column_name"],
                            "type": r["data_type"],
                            "nullable": r["is_nullable"] == "YES",
                            "default": r["column_default"],
                        }
                        for r in col_rows
                    ],
                }
            )

        return {"tables": schema, "table_count": len(schema)}
    finally:
        await conn.close()


@tool
async def run_query(sql: str, max_rows: int = 50) -> Dict[str, Any]:
    """
    Execute a SQL SELECT query and return the results.

    Only SELECT statements are allowed — any other statement type will be rejected.

    Args:
        sql: A valid SQL SELECT statement.
        max_rows: Maximum number of rows to return (default 50, max 500).

    Returns:
        Query results as a list of row dicts, plus row count and column names.
    """
    # Safety: only allow SELECT statements
    sql_stripped = sql.strip()
    if not sql_stripped.upper().startswith("SELECT"):
        return {
            "error": "Only SELECT statements are allowed. Received: "
            + sql_stripped[:60]
        }

    # Clamp max_rows
    max_rows = min(max(1, max_rows), 500)

    # Append LIMIT if not already present
    sql_upper = sql_stripped.upper()
    if "LIMIT" not in sql_upper:
        sql_with_limit = f"{sql_stripped}\nLIMIT {max_rows}"
    else:
        sql_with_limit = sql_stripped

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
        rows = await conn.fetch(sql_with_limit)
        records = [dict(r) for r in rows]

        # Convert non-serialisable types to strings
        for record in records:
            for key, value in record.items():
                if hasattr(value, "isoformat"):  # datetime, date
                    record[key] = value.isoformat()
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    record[key] = str(value)

        return {
            "sql": sql_stripped,
            "row_count": len(records),
            "columns": list(records[0].keys()) if records else [],
            "rows": records,
            "truncated": len(records) == max_rows,
        }
    except Exception as e:
        return {"error": str(e), "sql": sql_stripped}
    finally:
        await conn.close()


def create_agent() -> Agent:
    """Create the SQL Data Agent."""
    db = postgresql(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/mydb")
    )

    return Agent(
        name="SQL Data Agent",
        model="gpt-4o-mini",
        prompt="""You are a data analyst who answers natural language questions about a \
PostgreSQL database by writing and running SQL queries.

Your process for every question:
1. INSPECT: Call inspect_schema (optionally with a table_filter) to understand the
   available tables, column names, and data types. Skip this only if you have
   inspected the schema in the current session.
2. PLAN: Think about which tables and columns answer the question. Write the
   simplest correct SQL — avoid unnecessary subqueries or joins.
3. QUERY: Call run_query with your SELECT statement. Start with max_rows=50 unless
   the question requires more.
4. ANSWER: Explain the results in plain English. Include key numbers. Format tables
   as markdown when showing multiple rows.

Rules:
- Always use exact column names from inspect_schema — never guess.
- If a query returns an error, analyse the error, fix the SQL, and retry once.
- For aggregations, use GROUP BY and ORDER BY to produce ranked results.
- Round monetary values to 2 decimal places.
- If results are truncated (truncated=true), note that only the first N rows are shown.
- Never fabricate data — only report what the query actually returned.""",
        tools=[inspect_schema, run_query],
        plugins=[db],
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_agent()
        await agent.start()

        try:
            result = await agent.run(
                "Show me the top 10 customers by total order value in the last 90 days."
            )
            print(result)
        finally:
            await agent.stop()

    asyncio.run(main())
