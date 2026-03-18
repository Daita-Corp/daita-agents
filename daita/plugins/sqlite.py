"""
SQLite plugin for Daita Agents.

File-based (or in-memory) async SQLite access via aiosqlite.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_db import BaseDatabasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class SQLitePlugin(BaseDatabasePlugin):
    """
    SQLite plugin for agents.

    Supports both file-based and in-memory databases. Uses ? placeholders
    for parameterized queries (SQLite native style).

    Example:
        # File-based
        async with sqlite(path="./data.db") as db:
            await db.execute_script(\"\"\"
                CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT);
            \"\"\")
            await db.execute("INSERT INTO users (name) VALUES (?)", ["Alice"])
            rows = await db.query("SELECT * FROM users")

        # In-memory (default)
        async with sqlite() as db:
            rows = await db.query("SELECT 1 AS n")
    """

    sql_dialect = "sqlite"

    def __init__(
        self,
        path: str = ":memory:",
        wal_mode: bool = True,
        timeout: float = 5.0,
        **kwargs,
    ):
        """
        Initialize SQLite connection.

        Args:
            path: Path to the SQLite database file, or ":memory:" for an
                  in-process database that is discarded on close.
            wal_mode: Enable WAL journal mode for better concurrent read
                      performance. Defaults to True.
            timeout: Seconds to wait when the database is locked before
                     raising an error. Defaults to 5.0.
            **kwargs: Forwarded to BaseDatabasePlugin.
        """
        self.path = path
        self.wal_mode = wal_mode
        # BaseDatabasePlugin sets self.timeout, but we want float support
        kwargs["timeout"] = timeout

        super().__init__(path=path, wal_mode=wal_mode, **kwargs)

        # aiosqlite connection — overrides _connection from base
        self._db = None

        logger.debug(f"SQLitePlugin configured for path={path!r}")

    # ------------------------------------------------------------------
    # is_connected — override base since we use _db, not _connection/_pool
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._db is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the SQLite database connection."""
        if self._db is not None:
            return

        try:
            import aiosqlite
        except ImportError:
            self._handle_connection_error(
                ImportError(
                    "aiosqlite is required. Install with: pip install 'daita-agents[sqlite]'"
                ),
                "connection",
            )

        try:
            self._db = await aiosqlite.connect(
                self.path,
                timeout=float(self.timeout),
            )
            # Return rows as dict-like objects
            self._db.row_factory = aiosqlite.Row

            if self.wal_mode and self.path != ":memory:":
                await self._db.execute("PRAGMA journal_mode=WAL")

            logger.info(f"Connected to SQLite database: {self.path!r}")
        except Exception as e:
            self._db = None
            self._handle_connection_error(e, "connection")

    async def disconnect(self) -> None:
        """Close the SQLite database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info(f"Disconnected from SQLite database: {self.path!r}")

    # ------------------------------------------------------------------
    # Core query interface
    # ------------------------------------------------------------------

    async def query(
        self, sql: str, params: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a SELECT query and return results as a list of dicts.

        Uses ? placeholders:
            rows = await db.query("SELECT * FROM users WHERE age > ?", [25])

        Args:
            sql: SQL SELECT statement with ? placeholders.
            params: Optional list of parameter values.

        Returns:
            List of rows as dictionaries.
        """
        sql = self._normalize_sql(sql)
        if self._db is None:
            await self.connect()

        async with self._db.execute(sql, params or []) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def execute(self, sql: str, params: Optional[List] = None) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE statement.

        Uses ? placeholders:
            count = await db.execute("UPDATE users SET active=? WHERE id=?", [1, 42])

        Args:
            sql: SQL statement with ? placeholders.
            params: Optional list of parameter values.

        Returns:
            Number of rows affected.
        """
        sql = self._normalize_sql(sql)
        if self._db is None:
            await self.connect()

        async with self._db.execute(sql, params or []) as cursor:
            await self._db.commit()
            return cursor.rowcount if cursor.rowcount >= 0 else 0

    async def execute_script(self, sql: str) -> None:
        """
        Execute a multi-statement SQL script (no parameter binding).

        Ideal for schema setup, migrations, and seeding:
            await db.execute_script(\"\"\"
                CREATE TABLE IF NOT EXISTS users (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id),
                    total   REAL
                );
            \"\"\")

        Args:
            sql: One or more SQL statements separated by semicolons.
        """
        if self._db is None:
            await self.connect()

        await self._db.executescript(sql)
        # executescript issues a COMMIT internally, but commit again for safety
        await self._db.commit()

    async def insert_many(self, table: str, data: List[Dict[str, Any]]) -> int:
        """
        Bulk-insert rows into a table.

        Args:
            table: Target table name.
            data: List of dicts — all dicts must have the same keys.

        Returns:
            Number of rows inserted.
        """
        if not data:
            return 0

        if self._db is None:
            await self.connect()

        columns = list(data[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        rows = [[row[col] for col in columns] for row in data]

        await self._db.executemany(sql, rows)
        await self._db.commit()
        return len(data)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    async def tables(self) -> List[str]:
        """
        List all user-created tables in the database.

        SQLite internal tables (e.g. sqlite_sequence, sqlite_stat1) are
        excluded automatically.

        Returns:
            Sorted list of table names.
        """
        rows = await self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row["name"] for row in rows]

    async def describe(self, table: str) -> List[Dict[str, Any]]:
        """
        Return column info for a table.

        Normalized to the same shape used by the PostgreSQL/MySQL plugins:
        column_name, data_type, is_nullable.

        Args:
            table: Table name to inspect.

        Returns:
            List of column descriptor dicts.
        """
        rows = await self.query(f"PRAGMA table_info({table})")
        return [
            {
                "column_name": row["name"],
                "data_type": row["type"],
                "is_nullable": "NO" if row["notnull"] else "YES",
                "default_value": row["dflt_value"],
                "is_primary_key": bool(row["pk"]),
            }
            for row in rows
        ]

    async def pragma(self, key: str, value: Any = None) -> Any:
        """
        Get or set a SQLite PRAGMA value.

        Examples:
            page_size = await db.pragma("page_size")
            await db.pragma("cache_size", -64000)   # 64 MB cache

        Args:
            key: PRAGMA name (e.g. "journal_mode", "cache_size").
            value: Value to set. If None, the current value is returned.

        Returns:
            Current value when reading, None when setting.
        """
        if self._db is None:
            await self.connect()

        if value is None:
            rows = await self.query(f"PRAGMA {key}")
            if rows:
                return list(rows[0].values())[0]
            return None
        else:
            await self._db.execute(f"PRAGMA {key} = {value}")
            await self._db.commit()
            return None

    # ------------------------------------------------------------------
    # Agent tools
    # ------------------------------------------------------------------

    def get_tools(self) -> List["AgentTool"]:
        """Expose SQLite operations as agent tools."""
        from ..core.tools import AgentTool

        tools = [
            AgentTool(
                name="sqlite_query",
                description="Run a SELECT query on SQLite. Use ? placeholders for parameters. Use limit and columns to keep responses small.",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query with ? placeholders",
                        },
                        "params": {
                            "type": "array",
                            "description": "Optional parameter values for ? placeholders",
                            "items": {"type": "string"},
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max rows to return (default: 50)",
                        },
                        "columns": {
                            "type": "array",
                            "description": "Specific columns to return (returns all if omitted)",
                            "items": {"type": "string"},
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus DSL to filter/project results, e.g. \"status == 'active' | SELECT id, name | LIMIT 100\"",
                        },
                    },
                    "required": ["sql"],
                },
                handler=self._tool_query,
                category="database",
                source="plugin",
                plugin_name="SQLite",
                timeout_seconds=60,
            ),
            AgentTool(
                name="sqlite_list_tables",
                description="List all tables in the SQLite database.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=self._tool_list_tables,
                category="database",
                source="plugin",
                plugin_name="SQLite",
                timeout_seconds=15,
            ),
            AgentTool(
                name="sqlite_get_schema",
                description="Get column info (name, type, nullable, primary key) for a SQLite table.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Table name to inspect",
                        }
                    },
                    "required": ["table_name"],
                },
                handler=self._tool_get_schema,
                category="database",
                source="plugin",
                plugin_name="SQLite",
                timeout_seconds=15,
            ),
            AgentTool(
                name="sqlite_inspect",
                description="List all tables and their column schemas in one call. Prefer this over calling sqlite_list_tables then sqlite_get_schema for each table.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tables": {
                            "type": "array",
                            "description": "Filter to specific tables (returns all if omitted)",
                            "items": {"type": "string"},
                        }
                    },
                    "required": [],
                },
                handler=self._tool_inspect,
                category="database",
                source="plugin",
                plugin_name="SQLite",
                timeout_seconds=15,
            ),
        ]
        if not self.read_only:
            tools.append(
                AgentTool(
                    name="sqlite_execute",
                    description="Execute INSERT, UPDATE, or DELETE on SQLite. Use ? placeholders. Returns affected row count.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL statement (INSERT, UPDATE, or DELETE) with ? placeholders",
                            },
                            "params": {
                                "type": "array",
                                "description": "Optional parameter values for ? placeholders",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["sql"],
                    },
                    handler=self._tool_execute,
                    category="database",
                    source="plugin",
                    plugin_name="SQLite",
                    timeout_seconds=60,
                )
            )
        return tools

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sql = self._normalize_sql(args.get("sql"))
        params = args.get("params") or []
        focus_dsl = args.get("focus")

        if focus_dsl:
            results = await self._run_focus_query(sql, params, focus_dsl)
        else:
            limit = args.get("limit", 50)
            columns = args.get("columns")
            if columns:
                safe_cols = ", ".join(
                    f'"{c}"' for c in columns if re.match(r"^[A-Za-z0-9_]+$", c)
                )
                if safe_cols:
                    sql = f"SELECT {safe_cols} FROM ({sql})"
            if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
                sql = f"{sql} LIMIT {int(limit)}"
            results = await self.query(sql, params or None)

        return {"success": True, "rows": results, "row_count": len(results)}

    async def _tool_execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sql = args.get("sql")
        params = args.get("params")
        affected_rows = await self.execute(sql, params)
        return {"success": True, "affected_rows": affected_rows}

    async def _tool_list_tables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        tables = await self.tables()
        return {"success": True, "tables": tables, "count": len(tables)}

    async def _tool_get_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table_name = args.get("table_name")
        columns = await self.describe(table_name)
        return {
            "success": True,
            "table": table_name,
            "columns": columns,
            "column_count": len(columns),
        }

    async def _tool_inspect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        filter_tables = args.get("tables")
        all_tables = await self.tables()
        targets = (
            [t for t in all_tables if t in filter_tables]
            if filter_tables
            else all_tables
        )
        schemas = await asyncio.gather(*[self.describe(t) for t in targets])
        return {
            "success": True,
            "tables": [{"name": t, "columns": s} for t, s in zip(targets, schemas)],
            "count": len(targets),
        }


def sqlite(**kwargs) -> SQLitePlugin:
    """Create a SQLite plugin.

    Args:
        path: Path to the database file, or ":memory:" (default).
        wal_mode: Enable WAL journal mode (default: True).
        timeout: Lock wait timeout in seconds (default: 5.0).

    Example:
        async with sqlite(path="./app.db") as db:
            rows = await db.query("SELECT * FROM products WHERE price < ?", [50])
    """
    return SQLitePlugin(**kwargs)
