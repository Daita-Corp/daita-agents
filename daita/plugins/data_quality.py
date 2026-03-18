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

    # As agent tools
    agent = Agent(name="quality_checker", tools=[db, data_quality(db=db)])
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool
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


# ---------------------------------------------------------------------------
# Minimal dialect helpers (DQ-17 partial — enough for profile + discovery)
# ---------------------------------------------------------------------------


def _dialect(db: Any) -> str:
    return getattr(db, "sql_dialect", "standard")


def _placeholder(dialect: str) -> str:
    return "?" if dialect == "sqlite" else "%s"


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

    def initialize(self, agent_id: str) -> None:
        self._agent_id = agent_id
        if self._graph_backend is None:
            from daita.core.graph.backend import auto_select_backend

            self._graph_backend = auto_select_backend(graph_type="quality")
            logger.debug(
                "DataQualityPlugin: using graph backend %s",
                type(self._graph_backend).__name__,
            )

    def _validate_db(self) -> Any:
        if self._db is None:
            raise ValueError(
                "No database plugin configured. Pass db=<plugin> to data_quality()."
            )
        return self._db

    def get_tools(self) -> List["AgentTool"]:
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="dq_profile",
                description=(
                    "Generate statistical profile for each column in a table: "
                    "row count, null rate, cardinality, min, max, avg."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name to profile",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific columns to profile. Profiles all columns if omitted.",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Optional row sample size to limit scan cost.",
                        },
                    },
                    "required": ["table"],
                },
                handler=self._tool_profile,
                category="data_quality",
                source="plugin",
                plugin_name="DataQuality",
                timeout_seconds=120,
            ),
            AgentTool(
                name="dq_detect_anomaly",
                description=(
                    "Detect statistical outliers in a numeric column using z-score "
                    "(scipy if available, numpy fallback) or IQR method."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"},
                        "column": {
                            "type": "string",
                            "description": "Numeric column to analyse",
                        },
                        "method": {
                            "type": "string",
                            "description": "'zscore' (default) or 'iqr'",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Optional row limit to reduce scan cost",
                        },
                    },
                    "required": ["table", "column"],
                },
                handler=self._tool_detect_anomaly,
                category="data_quality",
                source="plugin",
                plugin_name="DataQuality",
                timeout_seconds=120,
            ),
            AgentTool(
                name="dq_check_freshness",
                description=(
                    "Check that data in a table is recent by comparing MAX(timestamp_column) "
                    "to an expected maximum age."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"},
                        "timestamp_column": {
                            "type": "string",
                            "description": "Column containing event/update timestamps",
                        },
                        "expected_interval_hours": {
                            "type": "number",
                            "description": "Maximum acceptable age in hours (default: 24)",
                        },
                    },
                    "required": ["table", "timestamp_column"],
                },
                handler=self._tool_check_freshness,
                category="data_quality",
                source="plugin",
                plugin_name="DataQuality",
                timeout_seconds=30,
            ),
            AgentTool(
                name="dq_report",
                description=(
                    "Generate a consolidated data quality report for a table: "
                    "column profiles and an overall completeness score. "
                    "Persists results to graph backend as a stable METRIC node."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"},
                        "sample_size": {
                            "type": "integer",
                            "description": "Optional row sample size passed to profiling.",
                        },
                    },
                    "required": ["table"],
                },
                handler=self._tool_report,
                category="data_quality",
                source="plugin",
                plugin_name="DataQuality",
                timeout_seconds=180,
            ),
        ]

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
        sample_size = args.get("sample_size")
        return await self.report(db, table, sample_size=sample_size)

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
        if not columns:
            dialect = _dialect(db)
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
                f"(SELECT {{col}} FROM {table} LIMIT {int(sample_size)}) _sample"
            )
            sample_expr_nn = (
                f"(SELECT {{col}} FROM {table} WHERE {{col}} IS NOT NULL "
                f"LIMIT {int(sample_size)}) _sample"
            )
        else:
            sample_expr = table
            sample_expr_nn = f"{table} WHERE {{col}} IS NOT NULL"

        col_profiles: Dict[str, Any] = {}

        for col in columns:
            _validate_identifier(col)
            try:
                # Single query: total rows, non-null count, distinct count
                from_clause = sample_expr.format(col=col)
                count_sql = (
                    f"SELECT COUNT(*) AS total, "
                    f"COUNT({col}) AS non_null, "
                    f"COUNT(DISTINCT {col}) AS distinct_count "
                    f"FROM {from_clause}"
                )
                count_rows = await db.query(count_sql)
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
                    nn_from = sample_expr_nn.format(col=col)
                    stats_sql = (
                        f"SELECT MIN({col}) AS min_val, "
                        f"MAX({col}) AS max_val, "
                        f"AVG(CAST({col} AS FLOAT)) AS avg_val "
                        f"FROM {nn_from}"
                    )
                    stat_rows = await db.query(stats_sql)
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
        sql = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL {limit_clause}"
        rows = await db.query(sql)

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
        sql = f"SELECT MAX({timestamp_column}) as latest FROM {table}"
        rows = await db.query(sql)

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
    ) -> Dict[str, Any]:
        """
        Generate a consolidated quality report: column profiles + completeness score.

        Persists results to the graph backend using a stable, upsertable node ID
        (quality_latest:{table}) so callers can always retrieve the latest report
        without scanning the full graph.
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

                table_node_id = AgentGraphNode.make_id(NodeType.TABLE, table)
                table_node = AgentGraphNode(
                    node_id=table_node_id,
                    node_type=NodeType.TABLE,
                    name=table,
                    created_by_agent=self._agent_id,
                )
                await self._graph_backend.add_node(table_node)

                # Stable node ID — upsertable, always points to latest report
                metric_name = f"quality_latest:{table}"
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
