"""
MySQL plugin for Daita Agents.

Simple MySQL connection and querying - no over-engineering.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, quote
from .base_db import BaseDatabasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class MySQLPlugin(BaseDatabasePlugin):
    """
    MySQL plugin for agents with standardized connection management.

    Inherits common database functionality from BaseDatabasePlugin.
    """

    sql_dialect = "mysql"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "",
        username: str = "",
        password: str = "",
        connection_string: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize MySQL connection.

        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Username
            password: Password
            connection_string: Full connection string (overrides individual params)
            **kwargs: Additional aiomysql parameters
        """
        if connection_string:
            self.connection_string = connection_string
            parsed = urlparse(connection_string)
            self.host = parsed.hostname or host
            self.port = parsed.port or port
            self.user = parsed.username or username
            self.password = parsed.password or password
            self.db = parsed.path.lstrip("/") if parsed.path else database
        else:
            self.connection_string = f"mysql://{quote(username, safe='')}:{quote(password, safe='')}@{host}:{port}/{database}"
            self.host = host
            self.port = port
            self.user = username
            self.password = password
            self.db = database

        self.pool_config = {
            "minsize": kwargs.get("min_size", 1),
            "maxsize": kwargs.get("max_size", 10),
            "charset": kwargs.get("charset", "utf8mb4"),
            "autocommit": kwargs.get("autocommit", True),
        }

        # Initialize base class with all config
        super().__init__(
            host=host,
            port=port,
            database=database,
            username=username,
            connection_string=connection_string,
            **kwargs,
        )

        logger.debug(f"MySQL plugin configured for {host}:{port}/{database}")

    async def connect(self):
        """Connect to MySQL database."""
        if self._pool is not None:
            return  # Already connected

        try:
            import aiomysql

            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
                **self.pool_config,
            )
            logger.info("Connected to MySQL")
        except ImportError:
            self._handle_connection_error(
                ImportError(
                    "aiomysql not installed. Install with: pip install 'daita-agents[mysql]'"
                ),
                "connection",
            )
        except Exception as e:
            self._handle_connection_error(e, "connection")

    async def disconnect(self):
        """Disconnect from the database."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            logger.info("Disconnected from MySQL")

    async def query(
        self, sql: str, params: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a SELECT query and return results.

        Args:
            sql: SQL query with %s placeholders
            params: List of parameters for the query

        Returns:
            List of rows as dictionaries

        Example:
            results = await db.query("SELECT * FROM users WHERE age > %s", [25])
        """
        sql = self._normalize_sql(sql)
        # Only auto-connect if pool is None - allows manual mocking
        if self._pool is None:
            await self.connect()

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if params:
                    await cursor.execute(sql, params)
                else:
                    await cursor.execute(sql)

                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                return [dict(zip(columns, row)) for row in rows]

    async def execute(self, sql: str, params: Optional[List] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE and return affected rows.

        Args:
            sql: SQL statement
            params: List of parameters

        Returns:
            Number of affected rows
        """
        sql = self._normalize_sql(sql)
        # Only auto-connect if pool is None - allows manual mocking
        if self._pool is None:
            await self.connect()

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if params:
                    await cursor.execute(sql, params)
                else:
                    await cursor.execute(sql)

                return cursor.rowcount

    async def insert_many(self, table: str, data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert data into a table.

        Args:
            table: Table name
            data: List of dictionaries to insert

        Returns:
            Number of rows inserted
        """
        if not data:
            return 0

        # Only auto-connect if pool is None - allows manual mocking
        if self._pool is None:
            await self.connect()

        # Get columns from first row
        columns = list(data[0].keys())
        placeholders = ", ".join(["%s"] * len(columns))

        sql = f"INSERT INTO {table} (`{'`, `'.join(columns)}`) VALUES ({placeholders})"

        # Convert to list of tuples for executemany
        rows = [tuple(row[col] for col in columns) for row in data]

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.executemany(sql, rows)
                return cursor.rowcount

    async def tables(self) -> List[str]:
        """List all tables in the database."""
        sql = """
        SELECT TABLE_NAME as table_name
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        AND TABLE_SCHEMA = DATABASE()
        ORDER BY TABLE_NAME
        """
        results = await self.query(sql)
        return [row["table_name"] for row in results]

    async def describe(self, table: str) -> List[Dict[str, Any]]:
        """Get table schema information."""
        sql = """
        SELECT
            COLUMN_NAME as column_name,
            DATA_TYPE as data_type,
            IS_NULLABLE as is_nullable,
            COLUMN_DEFAULT as column_default,
            COLUMN_TYPE as column_type
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = %s
        AND TABLE_SCHEMA = DATABASE()
        ORDER BY ORDINAL_POSITION
        """
        return await self.query(sql, [table])

    async def count_rows(
        self, table: str, filter: Optional[str] = None
    ) -> int:
        """
        Count rows in a table.

        Args:
            table: Table name
            filter: Optional WHERE clause (without the WHERE keyword)

        Returns:
            Row count
        """
        sql = f"SELECT COUNT(*) as cnt FROM `{table}`"
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
        sql = f"SELECT * FROM `{table}` ORDER BY RAND() LIMIT {int(n)}"
        return await self.query(sql)

    def get_tools(self) -> List["AgentTool"]:
        """
        Expose MySQL operations as agent tools.

        Returns:
            List of AgentTool instances for database operations
        """
        from ..core.tools import AgentTool

        tools = [
            AgentTool(
                name="mysql_query",
                description="Run a SELECT query on MySQL. Include LIMIT in your SQL to control result size.",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query with %s placeholders",
                        },
                        "params": {
                            "type": "array",
                            "description": "Optional parameter values",
                            "items": {},
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus DSL to filter/project at the database level, e.g. \"status == 'active' | SELECT id, name | LIMIT 100\"",
                        },
                    },
                    "required": ["sql"],
                },
                handler=self._tool_query,
                category="database",
                source="plugin",
                plugin_name="MySQL",
                timeout_seconds=60,
            ),
            AgentTool(
                name="mysql_inspect",
                description="List all tables and their column schemas in one call. Use tables param to filter specific tables.",
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
                plugin_name="MySQL",
                timeout_seconds=30,
            ),
            AgentTool(
                name="mysql_count",
                description="Count rows in a MySQL table, optionally filtered by a WHERE clause.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name",
                        },
                        "filter": {
                            "type": "string",
                            "description": "Optional WHERE clause (without WHERE keyword), e.g. 'status = \"active\"'",
                        },
                    },
                    "required": ["table"],
                },
                handler=self._tool_count,
                category="database",
                source="plugin",
                plugin_name="MySQL",
                timeout_seconds=30,
            ),
            AgentTool(
                name="mysql_sample",
                description="Return a random sample of rows from a MySQL table.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of rows to sample (default: 5)",
                        },
                    },
                    "required": ["table"],
                },
                handler=self._tool_sample,
                category="database",
                source="plugin",
                plugin_name="MySQL",
                timeout_seconds=30,
            ),
        ]
        if not self.read_only:
            tools.append(
                AgentTool(
                    name="mysql_execute",
                    description="Execute INSERT, UPDATE, or DELETE on MySQL. Returns affected row count.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL statement (INSERT, UPDATE, or DELETE)",
                            },
                            "params": {
                                "type": "array",
                                "description": "Optional parameter values",
                                "items": {},
                            },
                        },
                        "required": ["sql"],
                    },
                    handler=self._tool_execute,
                    category="database",
                    source="plugin",
                    plugin_name="MySQL",
                    timeout_seconds=60,
                )
            )
        return tools

    async def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mysql_query"""
        sql = self._normalize_sql(args.get("sql"))
        params = args.get("params") or []
        focus_dsl = args.get("focus")

        if focus_dsl:
            results = await self._run_focus_query(sql, params, focus_dsl)
        else:
            if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
                sql = f"{sql} LIMIT 50"
            results = await self.query(sql, params or None)

        truncated = self._truncate_result(results)
        return {
            "rows": truncated["rows"],
            "total_rows": truncated["total_rows"],
            "truncated": truncated["truncated"],
        }

    async def _tool_list_tables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mysql_list_tables (kept for backward compat)"""
        tables = await self.tables()
        return {"tables": tables}

    async def _tool_get_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mysql_get_schema (kept for backward compat)"""
        table_name = args.get("table_name")
        columns = await self.describe(table_name)
        return {
            "table": table_name,
            "columns": [self._compact_column(c) for c in columns],
        }

    async def _tool_execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mysql_execute"""
        sql = args.get("sql")
        params = args.get("params")

        affected_rows = await self.execute(sql, params)

        return {"affected_rows": affected_rows}

    async def _tool_inspect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mysql_inspect — fetch all table schemas in parallel."""
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
        """Tool handler for mysql_count"""
        table = args.get("table")
        filter_clause = args.get("filter")
        count = await self.count_rows(table, filter_clause)
        return {"table": table, "count": count}

    async def _tool_sample(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mysql_sample"""
        table = args.get("table")
        n = args.get("n", 5)
        rows = await self.sample_rows(table, n)
        return {"table": table, "rows": rows}


def mysql(**kwargs) -> MySQLPlugin:
    """Create MySQL plugin with simplified interface."""
    return MySQLPlugin(**kwargs)
