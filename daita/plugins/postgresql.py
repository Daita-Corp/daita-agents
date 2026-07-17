"""
PostgreSQL plugin for Daita Agents.

Simple database connection and querying - no over-engineering.
"""

import asyncio
from datetime import date, datetime, time
from decimal import Decimal
import logging
import re
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING
from urllib.parse import quote
from uuid import UUID

from .base import PluginContext
from .base_db import BaseDatabasePlugin
from .postgresql_extensions import (
    POSTGRESQL_MANIFEST,
    PostgreSQLExecutor,
    postgresql_capabilities,
    postgresql_evidence_schemas,
    postgresql_tool_views,
)
from .sql_params import coerce_sql_params, param_specs_from_payload
from ..core.exceptions import PluginError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from types import TracebackType

    class _PostgreSQLConnection(Protocol):
        async def fetch(
            self, query: str, *args: object
        ) -> Sequence[Mapping[str, object]]: ...

        async def execute(self, command: str, *args: object) -> str: ...

        async def executemany(
            self, command: str, args: Iterable[Sequence[object]]
        ) -> None: ...

        async def fetchrow(
            self, query: str, *args: object
        ) -> Mapping[str, object] | None: ...

    class _PostgreSQLPoolAcquireContext(Protocol):
        async def __aenter__(self) -> _PostgreSQLConnection: ...

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None: ...

    class _PostgreSQLPool(Protocol):
        def acquire(
            self, *, timeout: float | None = None
        ) -> _PostgreSQLPoolAcquireContext: ...

        async def close(self) -> None: ...


logger = logging.getLogger(__name__)


class PostgreSQLPlugin(BaseDatabasePlugin):
    """
    PostgreSQL plugin for agents with standardized connection management.

    Inherits common database functionality from BaseDatabasePlugin.
    """

    sql_dialect = "postgresql"
    manifest = POSTGRESQL_MANIFEST

    @property
    def schema(self) -> str:
        return self._schema

    @schema.setter
    def schema(self, value: Any) -> None:
        self._schema = _validate_postgresql_identifier(str(value or "public"))

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "",
        username: str = "",
        user: Optional[str] = None,  # Add this
        password: str = "",
        connection_string: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize PostgreSQL connection.

        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Username
            user: Username (alias for username)
            password: Password
            connection_string: Full connection string (overrides individual params)
            **kwargs: Additional asyncpg parameters
        """
        # Use 'user' parameter as alias for 'username' if provided
        effective_username = user if user is not None else username

        # Build connection string
        if connection_string:
            self.connection_string = connection_string
        else:
            # Build connection string - handle empty password case
            if password:
                self.connection_string = f"postgresql://{quote(effective_username, safe='')}:{quote(password, safe='')}@{host}:{port}/{database}"
            else:
                self.connection_string = f"postgresql://{quote(effective_username, safe='')}@{host}:{port}/{database}"

        # PostgreSQL-specific pool configuration
        self.pool_config = {
            "min_size": kwargs.get("min_size", 1),
            "max_size": kwargs.get("max_size", 10),
            "command_timeout": kwargs.get("command_timeout", 60),
            "statement_cache_size": kwargs.get(
                "statement_cache_size", 0
            ),  # Set to 0 for pgbouncer compatibility
        }

        # Initialize base class with all config
        super().__init__(
            host=host,
            port=port,
            database=database,
            username=effective_username,
            connection_string=connection_string,
            **kwargs,
        )
        self._pool: Optional["_PostgreSQLPool"] = None
        self.schema = _validate_postgresql_identifier(str(self.schema or "public"))
        self.config["schema"] = self.schema

        logger.debug(f"PostgreSQL plugin configured for {host}:{port}/{database}")

    @property
    def pool(self) -> "_PostgreSQLPool":
        """Return the active connection pool owned by this plugin."""
        if self._pool is None:
            raise ValidationError(
                "PostgreSQLPlugin is not connected to database",
                field="connection_state",
            )
        return self._pool

    async def setup(self, context: PluginContext) -> None:
        """Set up the PostgreSQL connector for a runtime."""
        await self.connect()

    async def teardown(self) -> None:
        """Disconnect the PostgreSQL connector from a runtime."""
        await self.disconnect()

    def declare_capabilities(self):
        return postgresql_capabilities()

    def get_executors(self):
        return (
            PostgreSQLExecutor(
                id="postgresql.schema.inspect",
                capability_ids=frozenset({"db.schema.inspect"}),
                evidence_kind="schema.asset_profile",
                handler=self._execute_schema_inspect,
            ),
            PostgreSQLExecutor(
                id="postgresql.source.revision",
                capability_ids=frozenset({"db.source.revision"}),
                evidence_kind="source.revision",
                handler=self._execute_source_revision,
            ),
            PostgreSQLExecutor(
                id="postgresql.sql.validate",
                capability_ids=frozenset({"db.sql.validate"}),
                evidence_kind="sql.validation",
                handler=self._execute_sql_validate,
            ),
            PostgreSQLExecutor(
                id="postgresql.sql.execute_read",
                capability_ids=frozenset({"db.sql.execute_read"}),
                evidence_kind="query.result",
                handler=self._execute_sql_read,
            ),
            PostgreSQLExecutor(
                id="postgresql.sql.execute_write",
                capability_ids=frozenset({"db.sql.execute_write"}),
                evidence_kind="write.execution",
                handler=self._execute_sql_write,
            ),
            PostgreSQLExecutor(
                id="postgresql.sql.explain",
                capability_ids=frozenset({"db.sql.explain"}),
                evidence_kind="sql.explain.plan",
                handler=self._execute_sql_explain,
            ),
            PostgreSQLExecutor(
                id="postgresql.column_values.profile",
                capability_ids=frozenset({"db.column_values.profile"}),
                evidence_kind="column_values.profile",
                handler=self._execute_column_values_profile,
            ),
        )

    def declare_evidence_schemas(self):
        return postgresql_evidence_schemas()

    def get_tool_views(self):
        return postgresql_tool_views()

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
            "database_type": "postgresql",
            "database_name": self.config.get("database") or "",
            "schema": self.schema,
            "table_count": len(tables),
            "tables": tables,
            "foreign_keys": await self.foreign_keys(),
        }

    async def _execute_source_revision(self, _payload: Any) -> Dict[str, Any]:
        """Return the declared PostgreSQL structural revision."""

        rows = await self.query(
            """
            SELECT md5(COALESCE(string_agg(definition, '|' ORDER BY definition), ''))
                AS revision
            FROM (
                SELECT concat_ws(':', table_schema, table_name, column_name,
                                 data_type, is_nullable, COALESCE(column_default, ''))
                    AS definition
                FROM information_schema.columns
                WHERE table_schema = $1
                UNION ALL
                SELECT concat_ws(':', tc.table_schema, tc.table_name,
                                 tc.constraint_type, tc.constraint_name,
                                 COALESCE(kcu.column_name, ''),
                                 COALESCE(ccu.table_name, ''),
                                 COALESCE(ccu.column_name, '')) AS definition
                FROM information_schema.table_constraints tc
                LEFT JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_catalog = kcu.constraint_catalog
                 AND tc.constraint_schema = kcu.constraint_schema
                 AND tc.constraint_name = kcu.constraint_name
                LEFT JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_catalog = ccu.constraint_catalog
                 AND tc.constraint_schema = ccu.constraint_schema
                 AND tc.constraint_name = ccu.constraint_name
                WHERE tc.table_schema = $1
            ) structural_catalog
            """,
            [self.schema],
        )
        revision = rows[0].get("revision") if rows else None
        return {
            "revision": f"postgresql-schema:{revision}" if revision else None,
            "status": "authoritative" if revision else "unavailable",
            "reason": "postgresql_information_schema",
        }

    async def _execute_sql_validate(self, payload: Any) -> Dict[str, Any]:
        from daita.core.exceptions import ValidationError
        from daita.db.query_sql_validation import (
            sql_fingerprint,
            sql_statement_facts,
            validate_sql_against_schema,
        )

        args = dict(payload or {})
        sql = self._normalize_sql(str(args.get("sql") or ""))
        operation = str(args.get("operation") or "query")
        analysis = self._validate_sql_policy(sql, operation=operation)
        schema = args.get("schema")
        if isinstance(schema, dict):
            preflight = validate_sql_against_schema(
                sql,
                schema,
                dialect="postgresql",
                analysis=analysis,
                params=list(args.get("params") or ()),
                groundings=list(args.get("groundings") or ()),
                source_owner=str(args.get("source_owner") or "postgresql"),
            )
            if preflight.get("ok") is not True:
                if preflight.get("error_type") == "grounding_coverage_error":
                    return {
                        "valid": False,
                        "sql": sql,
                        "sql_fingerprint": sql_fingerprint(sql),
                        "operation": operation,
                        "statement_type": analysis.statement_type,
                        "is_read": analysis.is_read,
                        "has_limit": analysis.has_limit,
                        "tables": [table.short_key for table in analysis.tables],
                        "columns": sorted(analysis.referenced_column_names),
                        "statement_facts": sql_statement_facts(sql, analysis),
                        "grounding_coverage": dict(
                            preflight.get("grounding_coverage") or {}
                        ),
                    }
                safe_keys = {
                    "available_columns",
                    "available_tables",
                    "column_candidates",
                    "do_not_retry_same_sql",
                    "error_type",
                    "inspect_tables",
                    "missing_columns",
                    "repair_required",
                    "sql_fingerprint",
                    "table_candidates",
                    "unknown_tables",
                }
                raise ValidationError(
                    "SQL validation failed against the current catalog schema.",
                    field="sql",
                    context={
                        key: preflight[key] for key in safe_keys if key in preflight
                    },
                )
        return {
            "valid": True,
            "sql": sql,
            "sql_fingerprint": sql_fingerprint(sql),
            "operation": operation,
            "statement_type": analysis.statement_type,
            "is_read": analysis.is_read,
            "has_limit": analysis.has_limit,
            "tables": [table.short_key for table in analysis.tables],
            "columns": sorted(analysis.referenced_column_names),
            "statement_facts": sql_statement_facts(sql, analysis),
            "grounding_coverage": dict(
                (preflight if isinstance(schema, dict) else {}).get(
                    "grounding_coverage"
                )
                or {}
            ),
        }

    async def _execute_sql_read(self, payload: Any) -> Dict[str, Any]:
        from daita.db.query_sql_validation import sql_fingerprint

        args = dict(payload or {})
        params = coerce_sql_params(
            list(args.get("params") or []),
            param_specs_from_payload(args),
            dialect="postgresql",
            json_binding="text",
        )
        result = await self._run_guarded_tool_query(
            str(args.get("sql") or ""),
            params,
            args.get("focus"),
        )
        return {
            **result,
            "sql_fingerprint": str(
                args.get("sql_fingerprint")
                or sql_fingerprint(str(args.get("sql") or ""))
            ),
            "executed_sql_fingerprint": sql_fingerprint(str(result.get("sql") or "")),
        }

    async def _execute_sql_write(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        sql = self._prepare_tool_execute_sql(str(args.get("sql") or ""))
        params = coerce_sql_params(
            list(args.get("params") or []),
            param_specs_from_payload(args),
            dialect="postgresql",
            json_binding="text",
        )
        affected_rows = await self.execute(sql, params)
        return {"sql": sql, "affected_rows": affected_rows}

    async def _execute_sql_explain(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        sql = self._prepare_tool_query_sql(str(args.get("sql") or ""))
        params = coerce_sql_params(
            list(args.get("params") or []),
            param_specs_from_payload(args),
            dialect="postgresql",
            json_binding="text",
        )
        rows = await self.query(f"EXPLAIN {sql}", params)
        return {"sql": sql, "plan": rows}

    async def _execute_column_values_profile(self, payload: Any) -> Dict[str, Any]:
        from datetime import datetime, timezone

        args = dict(payload or {})
        schema_name, table = _postgresql_table_parts(
            str(args.get("table") or ""),
            schema=args.get("schema") or self.schema,
        )
        column = _validate_postgresql_identifier(str(args.get("column") or ""))
        max_values = max(1, min(int(args.get("max_values") or 25), 100))
        max_distinct = max(1, int(args.get("max_distinct_count") or 100))
        max_value_length = max(1, int(args.get("max_value_length") or 80))
        max_profile_rows = max(1, int(args.get("max_profile_rows") or 1_000_000))
        timeout_seconds = max(1, min(int(args.get("profile_timeout_seconds") or 5), 60))
        fingerprint_only = bool(args.get("fingerprint_only", False))
        include_source_revision = bool(
            args.get("include_source_revision") or fingerprint_only
        )

        table_ref = f"{schema_name}.{table}" if schema_name != "public" else table
        blocked_tables = {
            str(item).lower() for item in getattr(self, "blocked_tables", set())
        }
        blocked_columns = {
            str(item).lower() for item in getattr(self, "blocked_columns", set())
        }
        include_sample_values = bool(self.include_sample_values)
        redact_pii_columns = bool(self.redact_pii_columns)
        profile: Dict[str, Any] = {
            "table": table_ref,
            "schema": schema_name,
            "column": column,
            "profile_kind": "categorical_values",
            "profile_status": "profiled",
            "max_values": max_values,
            "sampled": False,
            "truncated": False,
            "redacted": False,
            "top_values": [],
            "policy": {
                "policy_owner": "postgresql",
                "bounded_aggregate": True,
                "eligibility_checks": [
                    "blocked_table",
                    "sensitive_or_blocked_column",
                    "max_profile_rows",
                    "max_distinct_count",
                    "max_value_length",
                    "profile_timeout",
                ],
                "max_distinct_count": max_distinct,
                "max_value_length": max_value_length,
                "max_profile_rows": max_profile_rows,
                "profile_timeout_seconds": timeout_seconds,
                "profile_only_readable_tables": True,
                "include_sample_values": include_sample_values,
                "redact_pii_columns": redact_pii_columns,
                "fingerprint_only_supported": True,
                "include_source_revision": include_source_revision,
            },
            "profiled_at": datetime.now(timezone.utc).isoformat(),
        }
        if table.lower() in blocked_tables or table_ref.lower() in blocked_tables:
            return {
                **profile,
                "profile_status": "skipped",
                "skipped_reason": "blocked_table",
            }
        column_refs = {
            column.lower(),
            f"{table}.{column}".lower(),
            f"{schema_name}.{table}.{column}".lower(),
        }
        if blocked_columns & column_refs or (
            redact_pii_columns and _looks_sensitive_column(column)
        ):
            return {
                **profile,
                "profile_status": "skipped",
                "redacted": True,
                "skipped_reason": "sensitive_or_blocked_column",
            }
        if not include_sample_values and not fingerprint_only:
            return {
                **profile,
                "profile_status": "skipped",
                "skipped_reason": "sample_values_disabled",
            }

        source_info = (
            await _postgresql_live_source_revision(
                self,
                schema_name,
                table,
                timeout_seconds=timeout_seconds,
            )
            if include_source_revision
            else {
                "revision": None,
                "status": "best_effort",
                "reason": "source_revision_not_requested",
            }
        )
        source_revision = source_info.get("revision")
        source_status = str(source_info.get("status") or "unavailable")
        profile["source_fingerprint_status"] = source_status
        if source_info.get("reason"):
            profile["source_fingerprint_reason"] = source_info["reason"]
        if source_revision is not None:
            profile["source_revision"] = source_revision
        if source_status != "unavailable":
            profile["source_fingerprint"] = _postgresql_source_fingerprint(
                schema_name,
                table,
                column,
                max_values=max_values,
                max_distinct=max_distinct,
                max_value_length=max_value_length,
                source_revision=source_revision,
            )
        if fingerprint_only:
            return {
                **profile,
                "profile_kind": "source_fingerprint",
                "profile_status": "fingerprint",
                "policy": {
                    **profile["policy"],
                    "fingerprint_only": True,
                    "include_source_revision": True,
                },
            }

        quoted_table = _quote_postgresql_table(schema_name, table)
        quoted_column = _quote_postgresql_identifier(column)
        stats_sql = (
            "SELECT COUNT(*)::bigint AS row_count, "
            f"SUM(CASE WHEN {quoted_column} IS NULL THEN 1 ELSE 0 END)::bigint "
            "AS null_count, "
            f"COUNT(DISTINCT {quoted_column})::bigint AS distinct_count, "
            f"MAX(LENGTH(CAST({quoted_column} AS TEXT)))::bigint "
            "AS max_value_length "
            f"FROM {quoted_table}"
        )
        try:
            stats_rows = await asyncio.wait_for(
                self.query(stats_sql),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                **profile,
                "profile_status": "skipped",
                "skipped_reason": "profile_timeout",
            }
        stats = stats_rows[0] if stats_rows else {}
        distinct_count = stats.get("distinct_count") or 0
        profile.update(
            {
                "row_count": stats.get("row_count") or 0,
                "null_count": stats.get("null_count") or 0,
                "distinct_count": distinct_count,
                "max_observed_value_length": stats.get("max_value_length") or 0,
            }
        )
        if (stats.get("row_count") or 0) > max_profile_rows:
            return {
                **profile,
                "profile_status": "skipped",
                "skipped_reason": "row_count_exceeds_profile_limit",
            }
        if distinct_count > max_distinct:
            return {
                **profile,
                "profile_status": "skipped",
                "skipped_reason": "high_distinct_count",
            }
        if (stats.get("max_value_length") or 0) > max_value_length:
            return {
                **profile,
                "profile_status": "skipped",
                "redacted": True,
                "skipped_reason": "value_too_long",
            }

        values_sql = (
            f"SELECT {quoted_column} AS value, COUNT(*)::bigint AS count "
            f"FROM {quoted_table} "
            f"WHERE {quoted_column} IS NOT NULL "
            f"GROUP BY {quoted_column} "
            f"ORDER BY COUNT(*) DESC, {quoted_column} ASC "
            f"LIMIT {max_values}"
        )
        try:
            rows = await asyncio.wait_for(
                self.query(values_sql),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                **profile,
                "profile_status": "skipped",
                "skipped_reason": "profile_timeout",
            }
        profile["top_values"] = [
            {"value": row.get("value"), "count": row.get("count")} for row in rows
        ]
        profile["truncated"] = distinct_count > len(rows)
        return profile

    async def connect(self):
        """Connect to PostgreSQL database."""
        if self._pool is not None:
            return  # Already connected

        try:
            from importlib import import_module

            asyncpg = import_module("asyncpg")

            logger.debug(
                f"Connecting to PostgreSQL with connection string (password masked)"
            )
            create_pool = getattr(asyncpg, "create_pool")
            self._pool = await create_pool(self.connection_string, **self.pool_config)
            logger.info("Connected to PostgreSQL")
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
            ) from exc
        except Exception as e:
            # Enhance error message with troubleshooting tips
            error_msg = str(e)
            troubleshooting = []

            if "role" in error_msg and "does not exist" in error_msg:
                troubleshooting.append(
                    "Database user may not exist. Check POSTGRES_USER setting."
                )
                troubleshooting.append(
                    "For Docker: ensure container started with correct POSTGRES_USER env var."
                )

            if "database" in error_msg and "does not exist" in error_msg:
                troubleshooting.append(
                    "Database does not exist. Check POSTGRES_DB setting."
                )
                troubleshooting.append(
                    f"For ankane/pgvector: default database is 'postgres', not custom names."
                )
                troubleshooting.append(
                    "Create database first or connect to existing database."
                )

            if "password authentication failed" in error_msg:
                troubleshooting.append(
                    "Password authentication failed. Check POSTGRES_PASSWORD setting."
                )
                troubleshooting.append(
                    "For Docker: ensure -e POSTGRES_PASSWORD=<pwd> was set when starting container."
                )

            if troubleshooting:
                enhanced_error = f"{error_msg}\n\nTroubleshooting:\n" + "\n".join(
                    f"  - {tip}" for tip in troubleshooting
                )
                logger.error(enhanced_error)

            self._handle_connection_error(e, "connection")

    async def disconnect(self):
        """Disconnect from PostgreSQL database."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from PostgreSQL")

    async def query(
        self, sql: str, params: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a SELECT query and return results.

        Args:
            sql: SQL query with $1, $2, etc. placeholders
            params: List of parameters for the query

        Returns:
            List of rows as dictionaries

        Example:
            results = await db.query("SELECT * FROM users WHERE age > $1", [25])
        """
        sql = self._normalize_sql(sql)
        # Only auto-connect if pool is None - allows manual mocking
        if self._pool is None:
            await self.connect()

        pool = self.pool
        async with pool.acquire() as conn:
            if params:
                rows = await conn.fetch(sql, *params)
            else:
                rows = await conn.fetch(sql)

            return [_json_safe_row(dict(row)) for row in rows]

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

        pool = self.pool
        async with pool.acquire() as conn:
            if params:
                result = await conn.execute(sql, *params)
            else:
                result = await conn.execute(sql)

            # Extract number from result like "INSERT 0 5" or "UPDATE 3"
            # Some commands like "CREATE EXTENSION" don't return a count
            if result:
                try:
                    return int(result.split()[-1])
                except ValueError:
                    # Command succeeded but doesn't return a count (e.g., CREATE EXTENSION)
                    return 0
            return 0

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
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])

        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

        # Convert to list of tuples for executemany
        rows = [[row[col] for col in columns] for row in data]

        pool = self.pool
        async with pool.acquire() as conn:
            await conn.executemany(sql, rows)

        return len(data)

    async def tables(self) -> List[str]:
        """List all tables in the database."""
        sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = $1
        ORDER BY table_name
        """
        results = await self.query(sql, [self.schema])
        return [row["table_name"] for row in results]

    async def describe(self, table: str) -> List[Dict[str, Any]]:
        """Get table column information."""
        sql = """
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                     AND tc.table_name = kcu.table_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_schema = c.table_schema
                      AND tc.table_name = c.table_name
                      AND kcu.column_name = c.column_name
                ) AS is_primary_key
            FROM information_schema.columns c
            WHERE c.table_schema = $1
              AND c.table_name = $2
            ORDER BY ordinal_position
        """
        return await self.query(sql, [self.schema, table])

    async def foreign_keys(self) -> List[Dict[str, Any]]:
        """Return declared PostgreSQL foreign key relationships."""
        sql = """
            SELECT
                kcu.table_name AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name AS target_table,
                ccu.column_name AS target_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
             AND tc.table_name = kcu.table_name
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = $1
            ORDER BY kcu.table_name, kcu.column_name
        """
        return await self.query(sql, [self.schema])

    async def count_rows(self, table: str, filter: Optional[str] = None) -> int:
        """Count rows in a table with an optional WHERE clause."""
        where_clause = f"WHERE {filter}" if filter else ""
        sql = f"SELECT COUNT(*) AS cnt FROM {table} {where_clause}"
        result = await self.query(sql)
        return result[0]["cnt"] if result else 0

    async def sample_rows(self, table: str, n: int = 5) -> List[Dict[str, Any]]:
        """Return a random sample of n rows from a table."""
        sql = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {int(n)}"
        return await self.query(sql)

    async def vector_search(
        self,
        table: str,
        vector_column: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[str] = None,
        select_columns: str = "*",
        distance_type: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using pgvector extension.

        Args:
            table: Table name containing vectors
            vector_column: Column name with vector data
            query_vector: Query vector as list of floats
            top_k: Number of results to return
            filter: Optional SQL WHERE clause (e.g., "category = 'tech'")
            select_columns: Columns to select (default "*")
            distance_type: Distance metric - "cosine", "l2", or "inner_product"

        Returns:
            List of rows with similarity scores
        """
        # Only auto-connect if pool is None
        if self._pool is None:
            await self.connect()

        # Validate filter param for injection prevention
        if filter:
            if ";" in filter:
                raise ValidationError(
                    "filter contains invalid character: ';'", field="filter"
                )
            if "--" in filter or "/*" in filter:
                raise ValidationError(
                    "filter contains SQL comment syntax", field="filter"
                )
            # Detect subqueries: nested parens containing SELECT
            if re.search(r"\(.*\bSELECT\b.*\)", filter, re.IGNORECASE | re.DOTALL):
                raise ValidationError("filter contains a subquery", field="filter")

        # Distance operators: <=> (cosine), <-> (L2), <#> (inner product)
        distance_ops = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}

        if distance_type not in distance_ops:
            raise ValueError(
                f"Invalid distance_type. Must be one of: {list(distance_ops.keys())}"
            )

        operator = distance_ops[distance_type]

        # Build SQL query
        where_clause = f"WHERE {filter}" if filter else ""
        sql = f"""
        SELECT {select_columns},
               {vector_column} {operator} $1::vector AS distance
        FROM {table}
        {where_clause}
        ORDER BY distance
        LIMIT {top_k}
        """

        # Convert vector to string format for pgvector
        vector_str = f"[{','.join(map(str, query_vector))}]"

        results = await self.query(sql, [vector_str])
        return results

    async def vector_upsert(
        self,
        table: str,
        id_column: str,
        vector_column: str,
        id: str,
        vector: List[float],
        extra_columns: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Insert or update a vector with ON CONFLICT handling.

        Args:
            table: Table name
            id_column: Primary key column name
            vector_column: Vector column name
            id: ID value for the row
            vector: Vector as list of floats
            extra_columns: Optional dictionary of additional columns to upsert

        Returns:
            Dictionary with operation results
        """
        # Only auto-connect if pool is None
        if self._pool is None:
            await self.connect()

        # Build column lists
        columns = [id_column, vector_column]
        values_placeholders = ["$1", "$2"]
        params = [id, f"[{','.join(map(str, vector))}]"]

        if extra_columns:
            for idx, (col, val) in enumerate(extra_columns.items(), start=3):
                columns.append(col)
                values_placeholders.append(f"${idx}")
                params.append(val)

        # Build update clause for ON CONFLICT
        update_clauses = [f"{vector_column} = EXCLUDED.{vector_column}"]
        if extra_columns:
            for col in extra_columns.keys():
                update_clauses.append(f"{col} = EXCLUDED.{col}")

        sql = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(values_placeholders)})
        ON CONFLICT ({id_column})
        DO UPDATE SET {', '.join(update_clauses)}
        RETURNING {id_column}
        """

        pool = self.pool
        async with pool.acquire() as conn:
            result = await conn.fetchrow(sql, *params)
            if result:
                return {"id": str(result[id_column]), "upserted": True}
            return {"upserted": False}

    async def create_vector_index(
        self,
        table: str,
        vector_column: str,
        index_type: str = "hnsw",
        distance_type: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Create a vector index for faster similarity search.

        Args:
            table: Table name
            vector_column: Vector column name
            index_type: Index type - "hnsw" (faster queries) or "ivfflat" (faster builds)
            distance_type: Distance metric - "cosine", "l2", or "inner_product"

        Returns:
            Dictionary with index creation results
        """
        # Only auto-connect if pool is None
        if self._pool is None:
            await self.connect()

        # Distance operators for index: vector_cosine_ops, vector_l2_ops, vector_ip_ops
        ops_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }

        if distance_type not in ops_map:
            raise ValueError(
                f"Invalid distance_type. Must be one of: {list(ops_map.keys())}"
            )

        if index_type not in ["hnsw", "ivfflat"]:
            raise ValueError("Invalid index_type. Must be 'hnsw' or 'ivfflat'")

        ops = ops_map[distance_type]
        index_name = f"{table}_{vector_column}_{index_type}_idx"

        sql = f"CREATE INDEX {index_name} ON {table} USING {index_type} ({vector_column} {ops})"

        try:
            await self.execute(sql)
            return {
                "index_name": index_name,
                "table": table,
                "column": vector_column,
                "index_type": index_type,
                "distance_type": distance_type,
            }
        except Exception as e:
            raise PluginError(
                f"Failed to create vector index: {e}",
                plugin_name="PostgreSQL",
            ) from e

    async def _tool_list_tables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for postgres_list_tables (kept for backward compat, not in get_tools)"""
        tables = await self.tables()
        total = len(tables)
        truncated = total > 50
        return {
            "tables": tables[:50],
            "count": len(tables[:50]),
            "total_tables": total,
            "truncated": truncated,
        }

    async def _tool_get_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for postgres_get_schema (kept for backward compat, not in get_tools)"""
        table_name = args.get("table_name")
        if not isinstance(table_name, str) or not table_name:
            raise ValidationError("table_name is required", field="table_name")
        columns = await self.describe(table_name)

        return {
            "table": table_name,
            "columns": [self._compact_column(c) for c in columns],
        }

    async def _tool_inspect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for postgres_inspect — fetch all table schemas in parallel."""
        filter_tables = args.get("tables")

        all_tables = await self.tables()
        total_tables = len(all_tables)
        targets = (
            [t for t in all_tables if t in filter_tables]
            if filter_tables
            else all_tables[:50]
        )
        truncated = not filter_tables and total_tables > 50

        schemas = await asyncio.gather(*[self.describe(t) for t in targets])

        return {
            "tables": [
                {"name": t, "columns": [self._compact_column(c) for c in s]}
                for t, s in zip(targets, schemas)
            ],
            "count": len(targets),
            "total_tables": total_tables,
            "truncated": truncated,
        }

    async def _tool_count(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for postgres_count"""
        table = args.get("table")
        if not isinstance(table, str) or not table:
            raise ValidationError("table is required", field="table")
        filter_clause = args.get("filter")
        count = await self.count_rows(table, filter_clause)
        return {"table": table, "count": count}

    async def _tool_sample(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for postgres_sample"""
        table = args.get("table")
        if not isinstance(table, str) or not table:
            raise ValidationError("table is required", field="table")
        n = args.get("n", 5)
        rows = await self.sample_rows(table, n)
        return {"table": table, "rows": rows}

    async def _tool_vector_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for postgres_vector_search"""
        table = args.get("table")
        if not isinstance(table, str) or not table:
            raise ValidationError("table is required", field="table")
        vector_column = args.get("vector_column")
        if not isinstance(vector_column, str) or not vector_column:
            raise ValidationError("vector_column is required", field="vector_column")
        raw_query_vector = args.get("query_vector")
        if not isinstance(raw_query_vector, list) or not all(
            isinstance(value, (int, float)) for value in raw_query_vector
        ):
            raise ValidationError(
                "query_vector must be a list of numbers", field="query_vector"
            )
        query_vector = [float(value) for value in raw_query_vector]
        top_k = args.get("top_k", 10)
        filter = args.get("filter")
        distance_type = args.get("distance_type", "cosine")

        results = await self.vector_search(
            table=table,
            vector_column=vector_column,
            query_vector=query_vector,
            top_k=top_k,
            filter=filter,
            distance_type=distance_type,
        )

        return {"results": results, "count": len(results)}

    async def _tool_vector_upsert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for postgres_vector_upsert"""
        table = args.get("table")
        if not isinstance(table, str) or not table:
            raise ValidationError("table is required", field="table")
        id_column = args.get("id_column")
        if not isinstance(id_column, str) or not id_column:
            raise ValidationError("id_column is required", field="id_column")
        vector_column = args.get("vector_column")
        if not isinstance(vector_column, str) or not vector_column:
            raise ValidationError("vector_column is required", field="vector_column")
        row_id = args.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValidationError("id is required", field="id")
        raw_vector = args.get("vector")
        if not isinstance(raw_vector, list) or not all(
            isinstance(value, (int, float)) for value in raw_vector
        ):
            raise ValidationError("vector must be a list of numbers", field="vector")
        vector = [float(value) for value in raw_vector]
        extra_columns = args.get("extra_columns")

        result = await self.vector_upsert(
            table=table,
            id_column=id_column,
            vector_column=vector_column,
            id=row_id,
            vector=vector,
            extra_columns=extra_columns,
        )

        return result


def postgresql(**kwargs) -> PostgreSQLPlugin:
    """Create PostgreSQL plugin with simplified interface."""
    return PostgreSQLPlugin(**kwargs)


def _json_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _postgresql_table_parts(
    table: str,
    *,
    schema: Any = None,
) -> tuple[str, str]:
    raw_schema = str(schema or "").strip()
    raw_table = table.strip()
    if "." in raw_table:
        parts = [part.strip('" ') for part in raw_table.split(".") if part.strip()]
        if len(parts) != 2:
            raise ValidationError("Invalid PostgreSQL table identifier", field="table")
        raw_schema, raw_table = parts
    schema_name = _validate_postgresql_identifier(raw_schema or "public")
    table_name = _validate_postgresql_identifier(raw_table)
    return schema_name, table_name


def _validate_postgresql_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or ""):
        raise ValidationError("Invalid PostgreSQL identifier", field="identifier")
    return value


def _quote_postgresql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_postgresql_table(schema: str, table: str) -> str:
    return (
        f"{_quote_postgresql_identifier(schema)}."
        f"{_quote_postgresql_identifier(table)}"
    )


def _looks_sensitive_column(column: str) -> bool:
    lowered = column.lower()
    sensitive = {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "credential",
        "email",
        "phone",
        "address",
        "ssn",
        "comment",
        "message",
        "body",
        "notes",
        "note",
    }
    return any(term in lowered for term in sensitive)


def _postgresql_source_fingerprint(
    schema: str,
    table: str,
    column: str,
    *,
    max_values: int,
    max_distinct: int,
    max_value_length: int,
    source_revision: str | None = None,
) -> str:
    payload = (
        f"postgresql:{schema.lower()}.{table.lower()}.{column.lower()}:"
        f"{max_values}:{max_distinct}:{max_value_length}"
    )
    if source_revision:
        payload = f"{payload}:{source_revision}"
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _postgresql_live_source_revision(
    plugin: PostgreSQLPlugin,
    schema: str,
    table: str,
    *,
    timeout_seconds: int,
) -> dict[str, str | None]:
    safe_schema = schema.replace("'", "''")
    safe_table = table.replace("'", "''")
    sql = (
        "SELECT c.oid::text AS table_oid, "
        "c.relfilenode::text AS relfilenode, "
        "c.relpages::bigint AS relpages, "
        "c.reltuples::bigint AS reltuples, "
        "COALESCE(s.n_tup_ins, 0)::bigint AS n_tup_ins, "
        "COALESCE(s.n_tup_upd, 0)::bigint AS n_tup_upd, "
        "COALESCE(s.n_tup_del, 0)::bigint AS n_tup_del "
        "FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "LEFT JOIN pg_stat_all_tables s ON s.relid = c.oid "
        f"WHERE n.nspname = '{safe_schema}' AND c.relname = '{safe_table}' "
        "LIMIT 1"
    )
    try:
        rows = await asyncio.wait_for(plugin.query(sql), timeout=timeout_seconds)
    except Exception:
        return {
            "revision": None,
            "status": "unavailable",
            "reason": "postgresql_stats_unavailable",
        }
    if not rows:
        return {
            "revision": None,
            "status": "unavailable",
            "reason": "postgresql_stats_missing",
        }
    row = rows[0]
    return {
        "revision": "|".join(
            f"{key}:{row.get(key)}"
            for key in (
                "table_oid",
                "relfilenode",
                "relpages",
                "reltuples",
                "n_tup_ins",
                "n_tup_upd",
                "n_tup_del",
            )
        ),
        "status": "best_effort",
        "reason": "postgresql_catalog_stats",
    }
