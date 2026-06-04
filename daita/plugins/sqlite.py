"""
SQLite plugin for Daita Agents.

File-based (or in-memory) async SQLite access via aiosqlite.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import PluginContext
from .base_db import BaseDatabasePlugin
from .sqlite_extensions import (
    SQLITE_MANIFEST,
    SQLiteExecutor,
    sqlite_capabilities,
    sqlite_evidence_schemas,
    sqlite_tool_views,
)

if TYPE_CHECKING:
    from ..core.tools import LocalTool

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
    manifest = SQLITE_MANIFEST

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

    async def setup(self, context: PluginContext) -> None:
        """Set up the SQLite connector for a runtime."""
        await self.connect()

    async def teardown(self) -> None:
        """Disconnect the SQLite connector from a runtime."""
        await self.disconnect()

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

    # ------------------------------------------------------------------
    # Runtime extension declarations
    # ------------------------------------------------------------------

    def declare_capabilities(self):
        return sqlite_capabilities()

    def get_executors(self):
        return (
            SQLiteExecutor(
                id="sqlite.schema.inspect",
                capability_ids=frozenset({"db.schema.inspect"}),
                evidence_kind="schema.asset_profile",
                handler=self._execute_schema_inspect,
            ),
            SQLiteExecutor(
                id="sqlite.sql.validate",
                capability_ids=frozenset({"db.sql.validate"}),
                evidence_kind="sql.validation",
                handler=self._execute_sql_validate,
            ),
            SQLiteExecutor(
                id="sqlite.sql.execute_read",
                capability_ids=frozenset({"db.sql.execute_read"}),
                evidence_kind="query.result",
                handler=self._execute_sql_read,
            ),
            SQLiteExecutor(
                id="sqlite.sql.execute_write",
                capability_ids=frozenset({"db.sql.execute_write"}),
                evidence_kind="write.execution",
                handler=self._execute_sql_write,
            ),
            SQLiteExecutor(
                id="sqlite.sql.explain",
                capability_ids=frozenset({"db.sql.explain"}),
                evidence_kind="query.plan",
                handler=self._execute_sql_explain,
            ),
        )

    def declare_evidence_schemas(self):
        return sqlite_evidence_schemas()

    def get_tool_views(self):
        return sqlite_tool_views()

    async def _execute_schema_inspect(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        requested = args.get("tables")
        all_tables = await self.tables()
        targets = (
            [table for table in all_tables if table in requested]
            if requested
            else all_tables
        )
        schemas = await asyncio.gather(*[self.describe(table) for table in targets])
        tables = []
        for table, columns in zip(targets, schemas):
            tables.append(
                {
                    "name": table,
                    "columns": [
                        {
                            "name": column["column_name"],
                            "data_type": column["data_type"],
                            "is_nullable": column["is_nullable"],
                            "default_value": column["default_value"],
                            "is_primary_key": column["is_primary_key"],
                        }
                        for column in columns
                    ],
                }
            )
        return {
            "database_type": "sqlite",
            "database_name": self.path,
            "table_count": len(tables),
            "tables": tables,
            "foreign_keys": await self.foreign_keys(),
        }

    async def _execute_sql_validate(self, payload: Any) -> Dict[str, Any]:
        from daita.db.query_sql_validation import sql_statement_facts

        args = dict(payload or {})
        sql = self._normalize_sql(str(args.get("sql") or ""))
        operation = str(args.get("operation") or "query")
        analysis = self._validate_sql_policy(sql, operation=operation)
        return {
            "valid": True,
            "sql": sql,
            "operation": operation,
            "statement_type": analysis.statement_type,
            "is_read": analysis.is_read,
            "has_limit": analysis.has_limit,
            "tables": [table.short_key for table in analysis.tables],
            "columns": sorted(analysis.referenced_column_names),
            "statement_facts": sql_statement_facts(sql, analysis),
        }

    async def _execute_sql_read(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return await self._run_guarded_tool_query(
            str(args.get("sql") or ""),
            list(args.get("params") or []),
            args.get("focus"),
        )

    async def _execute_sql_write(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        sql = self._prepare_tool_execute_sql(str(args.get("sql") or ""))
        affected_rows = await self.execute(sql, list(args.get("params") or []))
        return {"sql": sql, "affected_rows": affected_rows}

    async def _execute_sql_explain(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        sql = self._prepare_tool_query_sql(str(args.get("sql") or ""))
        rows = await self.query(
            f"EXPLAIN QUERY PLAN {sql}", list(args.get("params") or [])
        )
        return {"sql": sql, "plan": rows}

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

    async def foreign_keys(self) -> List[Dict[str, Any]]:
        """Return declared SQLite foreign key relationships."""
        keys: List[Dict[str, Any]] = []
        for table in await self.tables():
            quoted_table = '"' + table.replace('"', '""') + '"'
            rows = await self.query(f"PRAGMA foreign_key_list({quoted_table})")
            for row in rows:
                keys.append(
                    {
                        "source_table": table,
                        "source_column": row.get("from", ""),
                        "target_table": row.get("table", ""),
                        "target_column": row.get("to", ""),
                    }
                )
        return keys

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

    async def count_rows(self, table: str, filter: Optional[str] = None) -> int:
        """
        Count rows in a table.

        Args:
            table: Table name
            filter: Optional WHERE clause (without the WHERE keyword)

        Returns:
            Row count
        """
        sql = f'SELECT COUNT(*) as cnt FROM "{table}"'
        if filter:
            sql += f" WHERE {filter}"
        rows = await self.query(sql)
        return rows[0]["cnt"] if rows else 0

    async def sample_rows(self, table: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Return a random sample of rows from a table.

        Args:
            table: Table name
            n: Number of rows to return

        Returns:
            List of sampled rows
        """
        sql = f'SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT {int(n)}'
        return await self.query(sql)

    # ------------------------------------------------------------------
    # Agent tools
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _tool_list_tables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Kept for backward compatibility."""
        tables = await self.tables()
        return {"tables": tables}

    async def _tool_get_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Kept for backward compatibility."""
        table_name = args.get("table_name")
        columns = await self.describe(table_name)
        return {
            "table": table_name,
            "columns": [self._compact_column(c) for c in columns],
        }

    async def _tool_inspect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        filter_tables = args.get("tables")
        all_tables = await self.tables()
        targets = (
            [t for t in all_tables if t in filter_tables]
            if filter_tables
            else all_tables
        )

        # Cap at 50 tables to avoid token bloat
        total_tables = len(targets)
        truncated = total_tables > 50
        targets = targets[:50]

        schemas = await asyncio.gather(*[self.describe(t) for t in targets])
        return {
            "tables": [
                {"name": t, "columns": [self._compact_column(c) for c in s]}
                for t, s in zip(targets, schemas)
            ],
            "total_tables": total_tables,
            "truncated": truncated,
        }

    async def _tool_count(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table = args.get("table")
        filter_clause = args.get("filter")
        count = await self.count_rows(table, filter_clause)
        return {"table": table, "count": count}

    async def _tool_sample(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table = args.get("table")
        n = args.get("n", 5)
        rows = await self.sample_rows(table, n)
        return {"table": table, "rows": rows}


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
