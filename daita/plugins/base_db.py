"""
Base class for database plugins.

Provides common connection management, error handling, and context manager
support for all database plugins in the Daita framework.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict
from ..core.exceptions import (
    PluginError,
    ConnectionError as DaitaConnectionError,
    ValidationError,
)
from .base import BasePlugin

logger = logging.getLogger(__name__)


class BaseDatabasePlugin(BasePlugin):
    """
    Base class for all database plugins with common connection management.

    This class provides:
    - Standardized connection/disconnection lifecycle
    - Context manager support for automatic cleanup
    - Common error handling patterns
    - Consistent configuration patterns

    Database-specific plugins should inherit from this class and implement
    the abstract methods for their specific database requirements.
    """

    # Subclasses set this to their dialect: "postgresql", "mysql", "snowflake", "sqlite"
    sql_dialect: str = "standard"

    def __init__(self, **kwargs):
        """
        Initialize base database plugin.

        Args:
            **kwargs: Database-specific configuration parameters
        """
        # Common connection state
        self._connection = None
        self._pool = None
        self._client = None
        self._db = None

        # Connection configuration
        self.config = kwargs
        self.timeout = kwargs.get("timeout", 30)
        self.max_retries = kwargs.get("max_retries", 3)
        self.read_only = kwargs.get("read_only", False)

        logger.debug(
            f"{self.__class__.__name__} initialized with config keys: {list(kwargs.keys())}"
        )

    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the database.

        This method must be implemented by each database plugin to handle
        the specific connection logic for that database type.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the database and clean up resources.

        This method must be implemented by each database plugin to handle
        the specific disconnection and cleanup logic for that database type.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """
        Check if the database connection is active.

        Returns:
            True if connected, False otherwise
        """
        return (
            self._connection is not None
            or self._pool is not None
            or self._client is not None
        )

    async def __aenter__(self):
        """Async context manager entry - automatically connect."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - automatically disconnect."""
        await self.disconnect()

    def _validate_connection(self) -> None:
        """
        Validate that the database connection is available.

        Raises:
            ValidationError: If not connected to database
        """
        if not self.is_connected:
            raise ValidationError(
                f"{self.__class__.__name__} is not connected to database",
                field="connection_state",
            )

    def _handle_connection_error(self, error: Exception, operation: str) -> None:
        """
        Handle database connection errors with consistent logging.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed

        Raises:
            PluginError: Wrapped database error with context
        """
        error_msg = f"{self.__class__.__name__} {operation} failed: {str(error)}"
        logger.error(error_msg)

        # Choose appropriate exception type based on the original error
        if isinstance(error, ImportError):
            # Missing dependency - permanent error
            raise PluginError(
                error_msg,
                plugin_name=self.__class__.__name__,
                retry_hint="permanent",
                context={"operation": operation, "original_error": str(error)},
            ) from error
        else:
            # Connection issues - typically transient
            raise DaitaConnectionError(
                error_msg,
                context={"plugin": self.__class__.__name__, "operation": operation},
            ) from error

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Strip trailing whitespace and semicolons from a single SQL statement.

        LLMs commonly append trailing semicolons to generated SQL. These break
        query manipulation (LIMIT appending, subquery wrapping) and some drivers
        reject single-statement calls that end with a semicolon.

        Do NOT use this on multi-statement scripts (execute_script), only for
        single-statement query/execute calls.
        """
        return sql.rstrip().rstrip(";").rstrip()

    async def query_checked(
        self,
        sql: str,
        params=None,
        assertions=None,
    ) -> list:
        """
        Run a query and enforce ItemAssertions against every returned row.

        Identical to query() when assertions is None or empty. Raises
        DataQualityError (permanent, not retried) if any assertion has
        violations, with the full violation list attached.

        Args:
            sql: SQL query string.
            params: Optional query parameters, passed through to query().
            assertions: Optional list of ItemAssertion objects.

        Returns:
            List of result rows (same as query()).

        Raises:
            DataQualityError: If one or more assertions have violations.
        """
        rows = await self.query(sql, params)
        if assertions:
            from ..core.assertions import _evaluate_assertions

            _evaluate_assertions(rows, assertions, source=sql)
        return rows

    async def _run_focus_query(self, sql: str, params: list, focus_dsl: str) -> list:
        """
        Parse *focus_dsl*, push LIMIT and WHERE into SQL (safe mode), execute,
        then let the Python evaluator handle SELECT, ORDER BY, GROUP BY, and
        any remaining clauses.

        If the DB-level pushdown raises an exception (e.g. a WHERE filter
        references a column absent in an aggregated subquery), the original
        query is executed without modification and the full focus expression is
        applied in Python — so focus never crashes the agent.
        """
        from ..core.focus import parse as _parse
        from ..core.focus.backends.sql import compile_focus_to_sql
        from ..core.focus.evaluator import evaluate_remaining

        fq = _parse(focus_dsl)
        mod_sql, extra_params, applied = compile_focus_to_sql(
            sql, fq, dialect=self.sql_dialect, param_offset=len(params), mode="safe"
        )
        try:
            rows = await self.query(mod_sql, params + extra_params or None)
        except Exception as db_err:
            logger.warning(
                f"Focus SQL pushdown failed ({db_err}); "
                "falling back to Python-only focus evaluation"
            )
            rows = await self.query(sql, params or None)
            applied = set()  # nothing was applied — Python evaluator handles everything

        return evaluate_remaining(rows, fq, applied)

    @staticmethod
    def _compact_column(col_dict: Dict[str, Any]) -> str:
        """Return a compact column representation: 'colname:datatype:null_status'.

        null_status is 'nn' (not nullable) or 'null' (nullable).
        Works with the column shape returned by all SQL plugins' describe() methods.
        """
        name = col_dict.get("column_name", col_dict.get("name", ""))
        dtype = col_dict.get("data_type", col_dict.get("type", ""))
        nullable = col_dict.get("is_nullable", "YES")
        nn = (
            "nn"
            if str(nullable).upper() in ("NO", "NOT NULL", "0", "FALSE")
            else "null"
        )
        return f"{name}:{dtype}:{nn}"

    @staticmethod
    def _truncate_result(
        rows: list, max_rows: int = 200, max_chars: int = 50000
    ) -> Dict[str, Any]:
        """Truncate a list of rows to fit within guardrails.

        Returns:
            dict with keys: rows, total_rows, truncated (bool)
        """
        import json

        total_rows = len(rows)
        truncated = False

        # Row count cap
        if total_rows > max_rows:
            rows = rows[:max_rows]
            truncated = True

        # Character cap — serialize and truncate if over limit
        try:
            serialized = json.dumps(rows)
            if len(serialized) > max_chars:
                # Binary-search to find how many rows fit
                lo, hi = 0, len(rows)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if len(json.dumps(rows[:mid])) <= max_chars:
                        lo = mid
                    else:
                        hi = mid - 1
                rows = rows[:lo]
                truncated = True
        except (TypeError, ValueError):
            pass

        return {"rows": rows, "total_rows": total_rows, "truncated": truncated}

    @property
    def info(self) -> Dict[str, Any]:
        """
        Get information about the database plugin.

        Returns:
            Dictionary with plugin information
        """
        return {
            "plugin_type": self.__class__.__name__,
            "connected": self.is_connected,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "config_keys": list(self.config.keys()),
        }
