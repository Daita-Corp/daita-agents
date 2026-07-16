"""
SQLite plugin for Daita Agents.

File-based (or in-memory) async SQLite access via aiosqlite.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import PluginContext
from .base_db import BaseDatabasePlugin
from daita.runtime import EvidenceWrappingExecutor
from .sql_params import coerce_sql_params, param_specs_from_payload
from .sqlite_extensions import (
    SQLITE_MANIFEST,
    sqlite_capabilities,
    sqlite_evidence_schemas,
    sqlite_tool_views,
)

if TYPE_CHECKING:
    from aiosqlite import Connection

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

    @property
    def schema(self) -> str:
        return self._schema

    @schema.setter
    def schema(self, value: Any) -> None:
        schema = str(value or "main").strip().lower()
        if schema != "main":
            raise ValueError("SQLite source schema must be 'main'")
        self._schema = schema

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
        self._db: Optional["Connection"] = None

        logger.debug(f"SQLitePlugin configured for path={path!r}")

    # ------------------------------------------------------------------
    # is_connected — override base since we use _db, not _connection/_pool
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._db is not None

    @property
    def db(self) -> "Connection":
        """Return the active connection owned by this plugin."""
        if self._db is None:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                "SQLitePlugin is not connected to database",
                field="connection_state",
            )
        return self._db

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
        except ImportError as exc:
            raise ImportError(
                "aiosqlite is required. Install with: pip install 'daita-agents[sqlite]'"
            ) from exc

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
            EvidenceWrappingExecutor(
                id="sqlite.schema.inspect",
                owner="sqlite",
                capability_ids=frozenset({"db.schema.inspect"}),
                evidence_kind="schema.asset_profile",
                handler=self._execute_schema_inspect,
            ),
            EvidenceWrappingExecutor(
                id="sqlite.source.revision",
                owner="sqlite",
                capability_ids=frozenset({"db.source.revision"}),
                evidence_kind="source.revision",
                handler=self._execute_source_revision,
            ),
            EvidenceWrappingExecutor(
                id="sqlite.sql.validate",
                owner="sqlite",
                capability_ids=frozenset({"db.sql.validate"}),
                evidence_kind="sql.validation",
                handler=self._execute_sql_validate,
            ),
            EvidenceWrappingExecutor(
                id="sqlite.sql.execute_read",
                owner="sqlite",
                capability_ids=frozenset({"db.sql.execute_read"}),
                evidence_kind="query.result",
                handler=self._execute_sql_read,
            ),
            EvidenceWrappingExecutor(
                id="sqlite.sql.execute_write",
                owner="sqlite",
                capability_ids=frozenset({"db.sql.execute_write"}),
                evidence_kind="write.execution",
                handler=self._execute_sql_write,
            ),
            EvidenceWrappingExecutor(
                id="sqlite.sql.explain",
                owner="sqlite",
                capability_ids=frozenset({"db.sql.explain"}),
                evidence_kind="sql.explain.plan",
                handler=self._execute_sql_explain,
            ),
            EvidenceWrappingExecutor(
                id="sqlite.column_values.profile",
                owner="sqlite",
                capability_ids=frozenset({"db.column_values.profile"}),
                evidence_kind="column_values.profile",
                handler=self._execute_column_values_profile,
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

    async def _execute_source_revision(self, _payload: Any) -> Dict[str, Any]:
        """Return the declared SQLite structural revision."""

        cursor = await self.db.execute("PRAGMA schema_version")
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        revision = int(row[0]) if row else None
        return {
            "revision": f"sqlite-schema:{revision}" if revision is not None else None,
            "status": "authoritative" if revision is not None else "unavailable",
            "reason": "sqlite_schema_version",
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
                dialect="sqlite",
                analysis=analysis,
            )
            if preflight.get("ok") is not True:
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
        statement_facts = sql_statement_facts(sql, analysis)
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
            "statement_facts": statement_facts,
        }

    async def _execute_sql_read(self, payload: Any) -> Dict[str, Any]:
        from daita.db.query_sql_validation import sql_fingerprint

        args = dict(payload or {})
        sql = str(args.get("sql") or "")
        params = coerce_sql_params(
            list(args.get("params") or []),
            param_specs_from_payload(args),
            dialect="sqlite",
            json_binding="text",
        )
        result = await self._run_guarded_tool_query(
            sql,
            params,
            args.get("focus"),
        )
        return {
            **result,
            "sql_fingerprint": str(args.get("sql_fingerprint") or sql_fingerprint(sql)),
            "executed_sql_fingerprint": sql_fingerprint(str(result.get("sql") or "")),
        }

    async def _execute_sql_write(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        sql = self._prepare_tool_execute_sql(str(args.get("sql") or ""))
        params = coerce_sql_params(
            list(args.get("params") or []),
            param_specs_from_payload(args),
            dialect="sqlite",
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
            dialect="sqlite",
            json_binding="text",
        )
        rows = await self.query(f"EXPLAIN QUERY PLAN {sql}", params)
        return {"sql": sql, "plan": rows}

    async def _execute_column_values_profile(self, payload: Any) -> Dict[str, Any]:
        from datetime import datetime, timezone

        args = dict(payload or {})
        table = _validate_sqlite_identifier(str(args.get("table") or ""))
        column = _validate_sqlite_identifier(str(args.get("column") or ""))
        max_values = max(1, min(int(args.get("max_values") or 25), 100))
        max_distinct = max(1, int(args.get("max_distinct_count") or 100))
        max_value_length = max(1, int(args.get("max_value_length") or 80))
        max_profile_rows = max(1, int(args.get("max_profile_rows") or 1_000_000))
        timeout_seconds = max(1, min(int(args.get("profile_timeout_seconds") or 5), 60))
        fingerprint_only = bool(args.get("fingerprint_only", False))
        include_source_revision = bool(
            args.get("include_source_revision") or fingerprint_only
        )

        blocked_tables = {
            item.lower() for item in getattr(self, "blocked_tables", set())
        }
        blocked_columns = {
            item.lower() for item in getattr(self, "blocked_columns", set())
        }
        include_sample_values = bool(self.include_sample_values)
        redact_pii_columns = bool(self.redact_pii_columns)
        profile: Dict[str, Any] = {
            "table": table,
            "column": column,
            "profile_kind": "categorical_values",
            "profile_status": "profiled",
            "max_values": max_values,
            "sampled": False,
            "truncated": False,
            "redacted": False,
            "top_values": [],
            "policy": {
                "policy_owner": "sqlite",
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
                "fingerprint_only_supported": True,
                "include_source_revision": include_source_revision,
                "include_sample_values": include_sample_values,
                "redact_pii_columns": redact_pii_columns,
            },
            "profiled_at": datetime.now(timezone.utc).isoformat(),
        }
        if table.lower() in blocked_tables:
            return {
                **profile,
                "profile_status": "skipped",
                "skipped_reason": "blocked_table",
            }
        column_refs = {column.lower(), f"{table}.{column}".lower()}
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
            await _sqlite_live_source_revision(self, timeout_seconds=timeout_seconds)
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
            profile["source_fingerprint"] = _sqlite_source_fingerprint(
                self.path,
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

        quoted_table = _quote_sqlite_identifier(table)
        quoted_column = _quote_sqlite_identifier(column)
        try:
            stats_rows = await asyncio.wait_for(
                self.query(
                    "SELECT COUNT(*) AS row_count, "
                    f"SUM(CASE WHEN {quoted_column} IS NULL THEN 1 ELSE 0 END) AS null_count, "
                    f"COUNT(DISTINCT {quoted_column}) AS distinct_count, "
                    f"MAX(LENGTH(CAST({quoted_column} AS TEXT))) AS max_value_length "
                    f"FROM {quoted_table}"
                ),
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

        try:
            rows = await asyncio.wait_for(
                self.query(
                    f"SELECT {quoted_column} AS value, COUNT(*) AS count "
                    f"FROM {quoted_table} "
                    f"WHERE {quoted_column} IS NOT NULL "
                    f"GROUP BY {quoted_column} "
                    f"ORDER BY COUNT(*) DESC, {quoted_column} ASC "
                    "LIMIT ?",
                    [max_values],
                ),
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

        db = self.db
        async with db.execute(sql, params or []) as cursor:
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

        db = self.db
        async with db.execute(sql, params or []) as cursor:
            await db.commit()
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

        db = self.db
        await db.executescript(sql)
        # executescript issues a COMMIT internally, but commit again for safety
        await db.commit()

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

        db = self.db
        await db.executemany(sql, rows)
        await db.commit()
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
            db = self.db
            await db.execute(f"PRAGMA {key} = {value}")
            await db.commit()
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
        if not isinstance(table_name, str) or not table_name:
            from ..core.exceptions import ValidationError

            raise ValidationError("table_name is required", field="table_name")
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
        if not isinstance(table, str) or not table:
            from ..core.exceptions import ValidationError

            raise ValidationError("table is required", field="table")
        filter_clause = args.get("filter")
        count = await self.count_rows(table, filter_clause)
        return {"table": table, "count": count}

    async def _tool_sample(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table = args.get("table")
        if not isinstance(table, str) or not table:
            from ..core.exceptions import ValidationError

            raise ValidationError("table is required", field="table")
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


def _validate_sqlite_identifier(value: str) -> str:
    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or ""):
        from ..core.exceptions import ValidationError

        raise ValidationError("Invalid SQLite identifier", field="identifier")
    return value


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


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


def _sqlite_source_fingerprint(
    path: str,
    table: str,
    column: str,
    *,
    max_values: int,
    max_distinct: int,
    max_value_length: int,
    source_revision: str | None = None,
) -> str:
    import hashlib

    payload = (
        f"sqlite:{path}:{table.lower()}.{column.lower()}:"
        f"{max_values}:{max_distinct}:{max_value_length}"
    )
    if source_revision:
        payload = f"{payload}:{source_revision}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _sqlite_live_source_revision(
    plugin: SQLitePlugin,
    *,
    timeout_seconds: int,
) -> dict[str, str | None]:
    import os

    parts: list[str] = []
    status = "best_effort"
    reason = "sqlite_data_version"
    if plugin.path and plugin.path != ":memory:":
        try:
            stat = os.stat(plugin.path)
            parts.append(f"file:{stat.st_mtime_ns}:{stat.st_size}")
            status = "authoritative"
            reason = "sqlite_file_metadata"
        except OSError:
            reason = "sqlite_file_metadata_unavailable"
    else:
        reason = "sqlite_in_memory_data_version"
    try:
        rows = await asyncio.wait_for(
            plugin.query("PRAGMA data_version"),
            timeout=timeout_seconds,
        )
        if rows:
            value = next(iter(rows[0].values()))
            parts.append(f"data_version:{value}")
    except Exception:
        if not parts:
            return {
                "revision": None,
                "status": "unavailable",
                "reason": "sqlite_data_version_unavailable",
            }
    if not parts:
        return {
            "revision": None,
            "status": "unavailable",
            "reason": "sqlite_source_revision_unavailable",
        }
    return {"revision": "|".join(parts), "status": status, "reason": reason}
