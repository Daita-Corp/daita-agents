"""
DataQualityPlugin — analytical data quality for agents.

Provides statistical profiling, anomaly detection, freshness checks, and
consolidated quality reporting. Works with any BaseDatabasePlugin.

Enforcement (asserting that query results meet quality standards at consumption
time) is handled separately via ItemAssertion / query_checked() on the DB
plugin — see daita.core.assertions.

Features:
- Statistical profiling per column (null rates, cardinality, min/max/avg)
- Statistical anomaly detection (numpy; scipy z-score if available)
- Data freshness checks against a timestamp column
- Consolidated quality report persisted as a stable METRIC graph node

Usage::

    from daita.plugins import postgresql, data_quality

    db = postgresql(host="localhost", database="mydb")
    await db.connect()

    dq = data_quality(db=db)
    report = await dq.dq_report("orders")

    # As agent plugins
    agent = Agent(name="quality_checker", plugins=[db, data_quality(db=db)])
"""

import logging
import re
from datetime import datetime, timezone
from inspect import iscoroutinefunction
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BasePlugin, PluginContext
from .data_quality_extensions import (
    DATA_QUALITY_MANIFEST,
    DataQualityExecutor,
    data_quality_capabilities,
    data_quality_evidence_schemas,
)

if TYPE_CHECKING:
    from ..core.tools import LocalTool
    from .base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_.]+$")


def _validate_identifier(name: str) -> str:
    """Validate SQL identifier (table/column name) to prevent injection."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid identifier {name!r}. Only alphanumeric, underscore, and dot allowed."
        )
    return name


def _quote_identifier(name: str, dialect: str = "standard") -> str:
    """Validate and quote a SQL identifier for generated quality queries."""
    _validate_identifier(name)
    quote = "`" if dialect == "mysql" else '"'
    return ".".join(f"{quote}{part}{quote}" for part in name.split("."))


async def _query_rows(db: Any, sql: str, params: Any = None) -> List[Any]:
    """Run generated DQ SQL through DB guardrails when the plugin exposes them."""
    guarded_query = getattr(db, "_run_guarded_tool_query", None)
    if guarded_query is not None and iscoroutinefunction(guarded_query):
        guarded = await guarded_query(sql, list(params or []))
        return guarded.get("rows", [])
    prepare_query = getattr(db, "_prepare_tool_query_sql", None)
    if prepare_query is not None and not _is_mock_attr(prepare_query):
        sql = prepare_query(sql)
    return await db.query(sql, params)


def _is_mock_attr(value: Any) -> bool:
    return value.__class__.__module__.startswith("unittest.mock")


# ---------------------------------------------------------------------------
# Minimal dialect helpers (DQ-17 partial — enough for profile + discovery)
# ---------------------------------------------------------------------------


def _dialect(db: Any) -> str:
    return getattr(db, "sql_dialect", "standard")


def _placeholder(dialect: str) -> str:
    if dialect == "sqlite":
        return "?"
    if dialect == "postgresql":
        return "$1"
    return "%s"


def _column_discovery_sql(table_name: str, dialect: str) -> tuple:
    """Return (sql, params) to list column names for a table."""
    if dialect == "sqlite":
        return (
            "SELECT name AS column_name FROM pragma_table_info(?) ORDER BY cid",
            (table_name,),
        )
    ph = _placeholder(dialect)
    return (
        f"SELECT column_name FROM information_schema.columns "
        f"WHERE table_name = {ph} ORDER BY ordinal_position",
        (table_name,),
    )


class DataQualityPlugin(BasePlugin):
    """
    Plugin for analytical data quality: profiling, anomaly detection,
    freshness checks, and quality reporting.

    Pair with ItemAssertion / query_checked() on the DB plugin for
    enforcement at query time.
    """

    manifest = DATA_QUALITY_MANIFEST

    def __init__(
        self,
        db: Optional[Any] = None,
        backend: Optional[Any] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            db: Optional database plugin. Required at tool execution time.
            backend: Optional graph backend for persisting quality reports.
            thresholds: Anomaly detection sensitivity.
                        Keys: "z_score" (float, default 3.0),
                              "iqr_multiplier" (float, default 1.5).
        """
        self._db = db
        self._graph_backend = backend
        self._agent_id: Optional[str] = None
        self._thresholds = thresholds or {"z_score": 3.0, "iqr_multiplier": 1.5}

    async def setup(self, context: PluginContext) -> None:
        """Set up data-quality runtime context without taking DB ownership."""
        self._configure_runtime(context.agent_id or context.runtime_id)

    def declare_capabilities(self):
        return data_quality_capabilities()

    def get_executors(self):
        return (
            DataQualityExecutor(
                id="data_quality.profile",
                capability_ids=frozenset({"quality.profile"}),
                evidence_kind="quality.profile",
                handler=self._execute_quality_profile,
            ),
            DataQualityExecutor(
                id="data_quality.anomaly.detect",
                capability_ids=frozenset({"quality.anomaly.detect"}),
                evidence_kind="quality.anomaly",
                handler=self._execute_anomaly_detect,
            ),
            DataQualityExecutor(
                id="data_quality.freshness.check",
                capability_ids=frozenset({"quality.freshness.check"}),
                evidence_kind="quality.freshness",
                handler=self._execute_freshness_check,
            ),
            DataQualityExecutor(
                id="data_quality.report.generate",
                capability_ids=frozenset({"quality.report.generate"}),
                evidence_kind="quality.report",
                handler=self._execute_quality_report,
            ),
        )

    def declare_evidence_schemas(self):
        return data_quality_evidence_schemas()

    def _configure_runtime(self, agent_id: str) -> None:
        self._agent_id = agent_id
        if self._graph_backend is None:
            from daita.core.graph.backend import auto_select_backend

            try:
                self._graph_backend = auto_select_backend(graph_type="quality")
                logger.debug(
                    "DataQualityPlugin: using graph backend %s",
                    type(self._graph_backend).__name__,
                )
            except ImportError as exc:
                logger.debug(
                    "DataQualityPlugin: graph backend unavailable; reports will not be persisted: %s",
                    exc,
                )

    def _validate_db(self) -> Any:
        if self._db is None:
            raise ValueError(
                "No database plugin configured. Pass db=<plugin> to data_quality()."
            )
        return self._db

    async def _resolve_table_node_id(
        self, table: str, store: Optional[str] = None
    ) -> str:
        """Return the qualified Table node ID for ``table``.

        Ambiguous references (multiple stores, no ``store`` arg) raise
        ``AmbiguousReferenceError``. Unknown tables fall through to a
        canonical qualified ID when ``store`` is provided, otherwise to the
        ``__unresolved__`` sentinel which the catalog promotes on next
        discovery.
        """
        from daita.core.graph.resolution import resolve_or_placeholder

        return await resolve_or_placeholder(
            self._graph_backend, table, store=store, agent_id=self._agent_id
        )

    # -------------------------------------------------------------------------
    # Tool handlers
    # -------------------------------------------------------------------------

    async def _tool_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        db = self._validate_db()
        table = _validate_identifier(args["table"])
        columns = [_validate_identifier(c) for c in args.get("columns", [])]
        sample_size = args.get("sample_size")
        return await self.profile(
            db, table, columns=columns or None, sample_size=sample_size
        )

    async def _tool_detect_anomaly(self, args: Dict[str, Any]) -> Dict[str, Any]:
        db = self._validate_db()
        table = _validate_identifier(args["table"])
        column = _validate_identifier(args["column"])
        method = args.get("method", "zscore")
        sample_size = args.get("sample_size")
        return await self.detect_anomaly(
            db, table, column, method=method, sample_size=sample_size
        )

    async def _tool_check_freshness(self, args: Dict[str, Any]) -> Dict[str, Any]:
        db = self._validate_db()
        table = _validate_identifier(args["table"])
        ts_col = _validate_identifier(args["timestamp_column"])
        interval_hours = args.get("expected_interval_hours", 24)
        return await self.check_freshness(
            db, table, ts_col, expected_interval_hours=interval_hours
        )

    async def _tool_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        db = self._validate_db()
        table = _validate_identifier(args["table"])
        store = args.get("store")
        sample_size = args.get("sample_size")
        return await self.report(db, table, sample_size=sample_size, store=store)

    async def _execute_quality_profile(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return await self._tool_profile(args)

    async def _execute_anomaly_detect(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return await self._tool_detect_anomaly(args)

    async def _execute_freshness_check(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return await self._tool_check_freshness(args)

    async def _execute_quality_report(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return await self._tool_report(args)

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    async def profile(
        self,
        db: Any,
        table: str,
        columns: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate statistical profile for columns in a table.

        Issues at most 2 queries per column (1 for counts + 1 for numeric
        stats), down from the naive 4-query approach. The stats query is
        attempted and silently skipped with a debug log for non-numeric columns.
        """
        # Discover columns if not specified
        dialect = _dialect(db)
        table = _validate_identifier(table)
        table_ident = _quote_identifier(table, dialect)
        if not columns:
            table_name = table.split(".", 1)[-1]
            disc_sql, disc_params = _column_discovery_sql(table_name, dialect)
            rows = await db.query(disc_sql, disc_params)
            columns = []
            for row in rows:
                col = row.get("column_name") if isinstance(row, dict) else row[0]
                columns.append(col)

        if not columns:
            return {"success": False, "error": f"No columns found for table {table}"}

        # Build the subquery wrapper for sample_size paths
        if sample_size:
            sample_expr = (
                f"(SELECT {{col}} FROM {table_ident} LIMIT {int(sample_size)}) _sample"
            )
            sample_expr_nn = (
                f"(SELECT {{col}} FROM {table_ident} WHERE {{col}} IS NOT NULL "
                f"LIMIT {int(sample_size)}) _sample"
            )
        else:
            sample_expr = table_ident
            sample_expr_nn = f"{table_ident} WHERE {{col}} IS NOT NULL"

        col_profiles: Dict[str, Any] = {}

        for col in columns:
            _validate_identifier(col)
            col_ident = _quote_identifier(col, dialect)
            try:
                # Single query: total rows, non-null count, distinct count
                from_clause = sample_expr.format(col=col_ident)
                count_sql = (
                    f"SELECT COUNT(*) AS total, "
                    f"COUNT({col_ident}) AS non_null, "
                    f"COUNT(DISTINCT {col_ident}) AS distinct_count "
                    f"FROM {from_clause}"
                )
                count_rows = await _query_rows(db, count_sql)
                if count_rows:
                    r = count_rows[0]
                    if isinstance(r, dict):
                        total = int(r.get("total") or 0)
                        non_null = int(r.get("non_null") or 0)
                        distinct_count = int(r.get("distinct_count") or 0)
                    else:
                        total, non_null, distinct_count = (
                            int(r[0] or 0),
                            int(r[1] or 0),
                            int(r[2] or 0),
                        )
                else:
                    total = non_null = distinct_count = 0

                null_count = total - non_null
                null_rate = null_count / total if total > 0 else 0.0

                # Separate stats query — may fail for non-numeric columns
                min_v = max_v = avg_v = None
                try:
                    nn_from = sample_expr_nn.format(col=col_ident)
                    stats_sql = (
                        f"SELECT MIN({col_ident}) AS min_val, "
                        f"MAX({col_ident}) AS max_val, "
                        f"AVG(CAST({col_ident} AS FLOAT)) AS avg_val "
                        f"FROM {nn_from}"
                    )
                    stat_rows = await _query_rows(db, stats_sql)
                    if stat_rows:
                        sr = stat_rows[0]
                        if isinstance(sr, dict):
                            min_v = sr.get("min_val")
                            max_v = sr.get("max_val")
                            avg_v = sr.get("avg_val")
                        else:
                            min_v, max_v, avg_v = sr[0], sr[1], sr[2]
                        if avg_v is not None:
                            avg_v = float(avg_v)
                except Exception as exc:
                    logger.debug(
                        "profile: skipping numeric stats for %s.%s: %s",
                        table,
                        col,
                        exc,
                    )

                col_profiles[col] = {
                    "total_rows": total,
                    "non_null_count": non_null,
                    "null_count": null_count,
                    "null_rate": round(null_rate, 4),
                    "distinct_count": distinct_count,
                    "min": min_v,
                    "max": max_v,
                    "avg": avg_v,
                }
            except Exception as exc:
                col_profiles[col] = {"error": str(exc)}

        return {
            "success": True,
            "table": table,
            "columns_profiled": len(col_profiles),
            "profile": col_profiles,
        }

    async def detect_anomaly(
        self,
        db: Any,
        table: str,
        column: str,
        method: str = "zscore",
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect statistical outliers in a numeric column.

        Uses scipy.stats for z-score if available; falls back to numpy.
        IQR method uses numpy only.
        """
        limit_clause = f"LIMIT {int(sample_size)}" if sample_size else ""
        dialect = _dialect(db)
        table_ident = _quote_identifier(table, dialect)
        column_ident = _quote_identifier(column, dialect)
        sql = (
            f"SELECT {column_ident} FROM {table_ident} "
            f"WHERE {column_ident} IS NOT NULL {limit_clause}"
        )
        rows = await _query_rows(db, sql)

        values = []
        for row in rows:
            v = row.get(column) if isinstance(row, dict) else row[0]
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass

        if len(values) < 4:
            return {
                "success": True,
                "table": table,
                "column": column,
                "method": method,
                "anomaly_count": 0,
                "anomalies": [],
                "note": f"Insufficient data ({len(values)} rows) for anomaly detection",
            }

        import numpy as np

        arr = np.array(values)

        if method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            multiplier = self._thresholds.get("iqr_multiplier", 1.5)
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            flags = (arr < lower) | (arr > upper)
            threshold_info = {"lower": float(lower), "upper": float(upper)}
        else:
            try:
                from scipy import stats

                z_scores = np.abs(stats.zscore(arr))
            except ImportError:
                mean = np.mean(arr)
                std = np.std(arr)
                z_scores = np.abs((arr - mean) / std) if std > 0 else np.zeros_like(arr)

            z_thresh = self._thresholds.get("z_score", 3.0)
            flags = z_scores > z_thresh
            threshold_info = {"z_score_threshold": z_thresh}

        anomaly_indices = list(np.where(flags)[0])
        anomaly_values = [float(arr[i]) for i in anomaly_indices]

        return {
            "success": True,
            "table": table,
            "column": column,
            "method": method,
            "total_rows_scanned": len(values),
            "anomaly_count": len(anomaly_indices),
            "anomaly_values": anomaly_values[:100],
            "thresholds": threshold_info,
        }

    async def check_freshness(
        self,
        db: Any,
        table: str,
        timestamp_column: str,
        expected_interval_hours: float = 24.0,
    ) -> Dict[str, Any]:
        """Check that the most recent timestamp is within the expected interval."""
        dialect = _dialect(db)
        table_ident = _quote_identifier(table, dialect)
        column_ident = _quote_identifier(timestamp_column, dialect)
        sql = f"SELECT MAX({column_ident}) as latest FROM {table_ident}"
        rows = await _query_rows(db, sql)

        if not rows:
            return {
                "success": False,
                "table": table,
                "column": timestamp_column,
                "error": "No rows returned from freshness query",
            }

        row = rows[0]
        latest = row.get("latest") if isinstance(row, dict) else row[0]

        if latest is None:
            return {
                "success": True,
                "table": table,
                "column": timestamp_column,
                "is_fresh": False,
                "latest_timestamp": None,
                "detail": "Table is empty or all timestamps are NULL",
            }

        # handle DB adapters that return timestamps as ISO strings
        if isinstance(latest, str):
            latest = datetime.fromisoformat(latest.replace("Z", "+00:00"))

        if hasattr(latest, "tzinfo") and latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_hours = (now - latest).total_seconds() / 3600.0
        is_fresh = age_hours <= expected_interval_hours

        return {
            "success": True,
            "table": table,
            "column": timestamp_column,
            "is_fresh": is_fresh,
            "latest_timestamp": (
                latest.isoformat() if hasattr(latest, "isoformat") else str(latest)
            ),
            "age_hours": round(age_hours, 2),
            "expected_interval_hours": expected_interval_hours,
            "detail": f"Data is {age_hours:.1f}h old (limit: {expected_interval_hours}h)",
        }

    async def report(
        self,
        db: Any,
        table: str,
        sample_size: Optional[int] = None,
        store: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a consolidated quality report: column profiles + completeness score.

        Persists results to the graph backend using a stable, upsertable node ID
        (quality_latest:{table}) so callers can always retrieve the latest report
        without scanning the full graph.

        Args:
            db: Database plugin to profile against.
            table: Bare table name. Qualified against ``store`` (or the
                resolution layer if ``store`` is None) so the METRIC node
                attaches to the correct multi-store Table.
            sample_size: Optional row sample passed to profiling.
            store: Optional store qualifier (e.g. ``postgres:host/db``). When
                omitted, the resolution layer picks the matching Table node;
                ambiguous names raise unless the graph contains exactly one
                match.
        """
        profile_result = await self.profile(db, table, sample_size=sample_size)

        # Completeness: average non-null rate across successfully profiled columns
        profile_data = profile_result.get("profile", {})
        completeness_scores = [
            1.0 - col_stats["null_rate"]
            for col_stats in profile_data.values()
            if isinstance(col_stats, dict) and "null_rate" in col_stats
        ]
        completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 1.0
        )

        report_data = {
            "success": True,
            "table": table,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "completeness_score": round(completeness, 4),
            "profile": profile_data,
        }

        # Persist to graph backend using a stable node ID (DQ-12)
        if self._graph_backend:
            try:
                from daita.core.graph.models import (
                    AgentGraphNode,
                    AgentGraphEdge,
                    NodeType,
                    EdgeType,
                )

                table_node_id = await self._resolve_table_node_id(table, store=store)
                table_node = AgentGraphNode(
                    node_id=table_node_id,
                    node_type=NodeType.TABLE,
                    name=table,
                    created_by_agent=self._agent_id,
                )
                await self._graph_backend.add_node(table_node)

                # Stable node ID — upsertable, always points to latest report.
                # Qualified against the resolved Table so reports for the same
                # table name in different stores don't collide.
                qualified = table_node_id.split(":", 1)[1]
                metric_name = f"quality_latest:{qualified}"
                metric_node_id = AgentGraphNode.make_id(NodeType.METRIC, metric_name)
                metric_node = AgentGraphNode(
                    node_id=metric_node_id,
                    node_type=NodeType.METRIC,
                    name=metric_name,
                    created_by_agent=self._agent_id,
                    health_score=completeness,
                    properties={
                        "completeness_score": completeness,
                        "table": table,
                        "generated_at": report_data["generated_at"],
                    },
                )
                await self._graph_backend.add_node(metric_node)

                edge = AgentGraphEdge(
                    edge_id=AgentGraphEdge.make_id(
                        table_node_id, EdgeType.PRODUCES, metric_node_id
                    ),
                    from_node_id=table_node_id,
                    to_node_id=metric_node_id,
                    edge_type=EdgeType.PRODUCES,
                    created_by_agent=self._agent_id,
                )
                await self._graph_backend.add_edge(edge)

                if hasattr(self._graph_backend, "flush"):
                    await self._graph_backend.flush()

                report_data["graph_persisted"] = True
                report_data["metric_node_id"] = metric_node_id
            except Exception as exc:
                logger.warning(
                    "DataQualityPlugin: failed to persist report to graph: %s", exc
                )
                report_data["graph_persisted"] = False

        return report_data

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract_count(self, rows: List[Any]) -> int:
        """Extract a count integer from the first row of a query result."""
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            for key in ("cnt", "count", "COUNT(*)", "count(*)"):
                if key in row:
                    v = row[key]
                    return int(v) if v is not None else 0
            # last-resort path — guard against non-numeric first value
            try:
                return int(next(iter(row.values()), 0) or 0)
            except (ValueError, TypeError):
                logger.debug("_extract_count: unexpected value in row: %r", row)
                return 0
        try:
            return int(row[0] or 0)
        except (ValueError, TypeError):
            logger.debug("_extract_count: unexpected row[0]: %r", row[0])
            return 0


def data_quality(
    db: Optional[Any] = None,
    backend: Optional[Any] = None,
    thresholds: Optional[Dict[str, Any]] = None,
) -> DataQualityPlugin:
    """
    Create a DataQualityPlugin.

    Args:
        db: Database plugin to run quality checks against.
        backend: Optional graph backend for persisting results.
        thresholds: Optional dict with 'z_score' and/or 'iqr_multiplier' keys.

    Returns:
        DataQualityPlugin instance
    """
    return DataQualityPlugin(db=db, backend=backend, thresholds=thresholds)
