"""
detect_anomalies — statistical outlier detection tool.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from ....core.tools import AgentTool
from ._helpers import ensure_pandas, ensure_numpy, safe_query, to_serializable

if TYPE_CHECKING:
    from ....plugins.base_db import BaseDatabasePlugin


def create_detect_anomalies_tool(plugin: "BaseDatabasePlugin", schema: Dict[str, Any]) -> AgentTool:
    """Return an AgentTool that detects statistical outliers in a column."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pd = ensure_pandas()
            np = ensure_numpy()
        except ImportError as e:
            return {"success": False, "error": str(e)}

        sql = args.get("sql", "").strip()
        column = args.get("column", "").strip()
        method = args.get("method", "zscore")
        limit = int(args.get("limit", 20))

        if not sql:
            return {"success": False, "error": "sql parameter is required"}
        if not column:
            return {"success": False, "error": "column parameter is required"}

        # Default thresholds differ by method
        if "threshold" in args:
            threshold = float(args["threshold"])
        else:
            threshold = 2.5 if method == "zscore" else 1.5

        try:
            rows = await safe_query(plugin, sql)
            if not rows:
                return {"success": True, "anomalies": [], "anomaly_count": 0, "total_rows": 0}

            df = pd.DataFrame([{k: to_serializable(v) for k, v in r.items()} for r in rows])

            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in query results"}

            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if len(series) == 0:
                return {"success": False, "error": f"No numeric values found in column '{column}'"}

            values = series.values
            mean = float(np.mean(values))
            std = float(np.std(values))
            median = float(np.median(values))
            min_val = float(np.min(values))
            max_val = float(np.max(values))

            anomaly_rows = []

            if method == "zscore":
                if std == 0:
                    return {
                        "success": True,
                        "anomalies": [],
                        "anomaly_count": 0,
                        "total_rows": len(df),
                        "stats": {"mean": mean, "std": std, "median": median, "min": min_val, "max": max_val},
                    }
                zscores = (values - mean) / std
                anomaly_mask = np.abs(zscores) > threshold
                for idx, (is_anomaly, z) in enumerate(zip(anomaly_mask, zscores)):
                    if is_anomaly:
                        orig_idx = series.index[idx]
                        row = {k: to_serializable(v) for k, v in df.loc[orig_idx].items()}
                        row["_zscore"] = round(float(z), 4)
                        row["_direction"] = "high" if z > 0 else "low"
                        anomaly_rows.append(row)

                anomaly_rows.sort(key=lambda r: abs(r["_zscore"]), reverse=True)
                anomaly_rows = anomaly_rows[:limit]

                return {
                    "success": True,
                    "anomalies": anomaly_rows,
                    "anomaly_count": int(np.sum(anomaly_mask)),
                    "total_rows": len(df),
                    "stats": {"mean": mean, "std": std, "median": median, "min": min_val, "max": max_val},
                }

            elif method == "iqr":
                q1 = float(np.percentile(values, 25))
                q3 = float(np.percentile(values, 75))
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr

                anomaly_mask = (values < lower) | (values > upper)
                for idx, (is_anomaly, val) in enumerate(zip(anomaly_mask, values)):
                    if is_anomaly:
                        orig_idx = series.index[idx]
                        row = {k: to_serializable(v) for k, v in df.loc[orig_idx].items()}
                        if val > upper:
                            row["_iqr_distance"] = round(float(val - upper), 4)
                            row["_direction"] = "high"
                        else:
                            row["_iqr_distance"] = round(float(lower - val), 4)
                            row["_direction"] = "low"
                        anomaly_rows.append(row)

                anomaly_rows.sort(key=lambda r: r["_iqr_distance"], reverse=True)
                anomaly_rows = anomaly_rows[:limit]

                return {
                    "success": True,
                    "anomalies": anomaly_rows,
                    "anomaly_count": int(np.sum(anomaly_mask)),
                    "total_rows": len(df),
                    "bounds": {"lower": lower, "upper": upper, "q1": q1, "q3": q3, "iqr": iqr},
                    "stats": {"mean": mean, "std": std, "median": median, "min": min_val, "max": max_val},
                }

            else:
                return {"success": False, "error": f"Unknown method '{method}'. Use 'zscore' or 'iqr'"}

        except Exception as e:
            return {"success": False, "error": f"Anomaly detection failed: {e}"}

    return AgentTool(
        name="detect_anomalies",
        description=(
            "Detect statistical outliers in a numeric column using z-score or IQR method. "
            "Returns flagged rows sorted by deviation magnitude, along with summary statistics. "
            "Use this to find unusual transactions, spikes, or data quality issues. "
            "IMPORTANT: write SQL that computes a meaningful metric column — if the value you "
            "want to analyse (e.g. order total) is spread across tables, JOIN them first "
            "and alias the computed column (e.g. SUM(quantity*unit_price) AS total)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query returning rows to analyse"},
                "column": {"type": "string", "description": "Numeric column to check for anomalies"},
                "method": {
                    "type": "string",
                    "enum": ["zscore", "iqr"],
                    "description": "Detection method: zscore (default) or iqr",
                },
                "threshold": {
                    "type": "number",
                    "description": "Deviation threshold (default: 2.5 for zscore, 1.5 for iqr)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum anomalies to return (default: 20)",
                },
            },
            "required": ["sql", "column"],
        },
        handler=handler,
        category="analysis",
        source="analyst_toolkit",
    )
