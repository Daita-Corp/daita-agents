"""
correlate — column correlation tool.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ....core.tools import AgentTool
from ._helpers import ensure_pandas, safe_query, to_serializable

if TYPE_CHECKING:
    from ....plugins.base_db import BaseDatabasePlugin

_STRENGTH_THRESHOLDS = [
    (0.9, "very strong"),
    (0.7, "strong"),
    (0.5, "moderate"),
    (0.3, "weak"),
    (0.0, "negligible"),
]


def _strength_label(r: float) -> str:
    abs_r = abs(r)
    for threshold, label in _STRENGTH_THRESHOLDS:
        if abs_r >= threshold:
            return label
    return "negligible"


def create_correlate_tool(plugin: "BaseDatabasePlugin", schema: Dict[str, Any]) -> AgentTool:
    """Return an AgentTool that computes column correlations."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pd = ensure_pandas()
        except ImportError as e:
            return {"success": False, "error": str(e)}

        sql = args.get("sql", "").strip()
        columns: Optional[List[str]] = args.get("columns")
        method = args.get("method", "pearson")
        min_corr = float(args.get("min_correlation", 0.3))

        if not sql:
            return {"success": False, "error": "sql parameter is required"}

        try:
            rows = await safe_query(plugin, sql)
            if not rows:
                return {"success": True, "correlations": [], "matrix": {}, "sample_size": 0}

            df = pd.DataFrame([{k: to_serializable(v) for k, v in r.items()} for r in rows])

            # Select columns
            if columns:
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    return {"success": False, "error": f"Columns not found: {missing}"}
                num_df = df[columns].apply(pd.to_numeric, errors="coerce")
            else:
                num_df = df.select_dtypes(include="number")

            if num_df.shape[1] < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 numeric columns to compute correlations",
                }

            corr_matrix = num_df.corr(method=method)
            cols = list(corr_matrix.columns)

            # Extract upper triangle pairs
            pairs = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    r = corr_matrix.iloc[i, j]
                    if pd.isna(r):
                        continue
                    r = float(r)
                    if abs(r) >= min_corr:
                        pairs.append({
                            "column_a": cols[i],
                            "column_b": cols[j],
                            "correlation": round(r, 4),
                            "strength": _strength_label(r),
                        })

            pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            # Build serializable matrix
            matrix = {
                col: {
                    other: round(float(corr_matrix.loc[col, other]), 4)
                    for other in cols
                    if not pd.isna(corr_matrix.loc[col, other])
                }
                for col in cols
            }

            return {
                "success": True,
                "correlations": pairs,
                "matrix": matrix,
                "method": method,
                "sample_size": len(df),
            }
        except Exception as e:
            return {"success": False, "error": f"Correlation failed: {e}"}

    return AgentTool(
        name="correlate",
        description=(
            "Compute pairwise correlations between numeric columns in a query result. "
            "Useful for finding which variables are related. Returns pairs sorted by "
            "absolute correlation strength, filtered by a minimum threshold."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query returning numeric columns to correlate"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to correlate (default: all numeric)",
                },
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman", "kendall"],
                    "description": "Correlation method (default: pearson)",
                },
                "min_correlation": {
                    "type": "number",
                    "description": "Minimum absolute correlation to include in results (default: 0.3)",
                },
            },
            "required": ["sql"],
        },
        handler=handler,
        category="analysis",
        source="analyst_toolkit",
    )
