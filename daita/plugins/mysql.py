"""
MySQL plugin for Daita Agents.

Simple MySQL connection and querying - no over-engineering.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, quote

from .base import PluginContext
from .base_db import BaseDatabasePlugin
from .mysql_extensions import (
    MYSQL_MANIFEST,
    MySQLExecutor,
    mysql_capabilities,
    mysql_evidence_schemas,
    mysql_tool_views,
)

if TYPE_CHECKING:
    from ..core.tools import LocalTool

logger = logging.getLogger(__name__)


class MySQLPlugin(BaseDatabasePlugin):
    """
    MySQL plugin for agents with standardized connection management.

    Inherits common database functionality from BaseDatabasePlugin.
    """

    sql_dialect = "mysql"
    manifest = MYSQL_MANIFEST

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

    async def setup(self, context: PluginContext) -> None:
        """Set up the MySQL connector for a runtime."""
        await self.connect()

    async def teardown(self) -> None:
        """Disconnect the MySQL connector from a runtime."""
        await self.disconnect()

    # ------------------------------------------------------------------
    # Runtime extension declarations
    # ------------------------------------------------------------------

    def declare_capabilities(self):
        return mysql_capabilities()

    def get_executors(self):
        return (
            MySQLExecutor(
                id="mysql.schema.inspect",
                capability_ids=frozenset({"db.schema.inspect"}),
                evidence_kind="schema.asset_profile",
                handler=self._execute_schema_inspect,
            ),
            MySQLExecutor(
                id="mysql.sql.validate",
                capability_ids=frozenset({"db.sql.validate"}),
                evidence_kind="sql.validation",
                handler=self._execute_sql_validate,
            ),
            MySQLExecutor(
                id="mysql.sql.execute_read",
                capability_ids=frozenset({"db.sql.execute_read"}),
                evidence_kind="query.result",
                handler=self._execute_sql_read,
            ),
            MySQLExecutor(
                id="mysql.sql.execute_write",
                capability_ids=frozenset({"db.sql.execute_write"}),
                evidence_kind="write.execution",
                handler=self._execute_sql_write,
            ),
            MySQLExecutor(
                id="mysql.sql.explain",
                capability_ids=frozenset({"db.sql.explain"}),
                evidence_kind="query.plan",
                handler=self._execute_sql_explain,
            ),
        )

    def declare_evidence_schemas(self):
        return mysql_evidence_schemas()

    def get_tool_views(self):
        return mysql_tool_views()

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
                            "name": column.get("column_name") or column.get("name"),
                            "data_type": column.get("data_type") or column.get("type"),
                            "is_nullable": column.get("is_nullable"),
                            "default_value": column.get("column_default"),
                            "is_primary_key": bool(column.get("is_primary_key")),
                        }
                        for column in columns
                    ],
                }
            )
        return {
            "database_type": "mysql",
            "database_name": self.db,
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
        rows = await self.query(f"EXPLAIN {sql}", list(args.get("params") or []))
        return {"sql": sql, "plan": rows}

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
            COLUMN_TYPE as column_type,
            COLUMN_KEY = 'PRI' as is_primary_key
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = %s
        AND TABLE_SCHEMA = DATABASE()
        ORDER BY ORDINAL_POSITION
        """
        return await self.query(sql, [table])

    async def foreign_keys(self) -> List[Dict[str, Any]]:
        """List foreign-key relationships in the current MySQL database."""
        sql = """
        SELECT
            TABLE_NAME AS table_name,
            COLUMN_NAME AS column_name,
            REFERENCED_TABLE_NAME AS referenced_table_name,
            REFERENCED_COLUMN_NAME AS referenced_column_name,
            CONSTRAINT_NAME AS constraint_name
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE()
        AND REFERENCED_TABLE_NAME IS NOT NULL
        ORDER BY TABLE_NAME, COLUMN_NAME
        """
        return await self.query(sql)

    async def count_rows(self, table: str, filter: Optional[str] = None) -> int:
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
