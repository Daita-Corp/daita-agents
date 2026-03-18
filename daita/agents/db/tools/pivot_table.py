"""
pivot_table — cross-tabulation tool.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from ....core.tools import AgentTool
from ._helpers import ensure_pandas, safe_query, to_serializable

if TYPE_CHECKING:
    from ....plugins.base_db import BaseDatabasePlugin


def create_pivot_table_tool(plugin: "BaseDatabasePlugin", schema: Dict[str, Any]) -> AgentTool:
    """Return an AgentTool that cross-tabulates query results."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pd = ensure_pandas()
        except ImportError as e:
            return {"success": False, "error": str(e)}

        sql = args.get("sql", "").strip()
        rows_col = args.get("rows", "")
        cols_col = args.get("columns", "")
        values_col = args.get("values", "")
        aggfunc = args.get("aggfunc", "sum")
        fill_value = args.get("fill_value", 0)

        if not sql:
            return {"success": False, "error": "sql parameter is required"}
        if not rows_col or not cols_col or not values_col:
            return {"success": False, "error": "rows, columns, and values parameters are required"}

        try:
            rows = await safe_query(plugin, sql)
            if not rows:
                return {"success": True, "pivot": [], "row_count": 0, "column_count": 0}

            df = pd.DataFrame([{k: to_serializable(v) for k, v in r.items()} for r in rows])

            for col in [rows_col, cols_col, values_col]:
                if col not in df.columns:
                    return {"success": False, "error": f"Column '{col}' not found in query results"}

            pivot = pd.pivot_table(
                df,
                index=rows_col,
                columns=cols_col,
                values=values_col,
                aggfunc=aggfunc,
                fill_value=fill_value,
            )
            pivot = pivot.reset_index()
            pivot.columns = [str(c) for c in pivot.columns]

            result_rows = pivot.to_dict(orient="records")
            serialized = [
                {k: to_serializable(v) for k, v in row.items()}
                for row in result_rows
            ]

            return {
                "success": True,
                "pivot": serialized,
                "row_count": len(serialized),
                "column_count": len(pivot.columns),
                "dimensions": {"rows": rows_col, "columns": cols_col, "values": values_col, "aggfunc": aggfunc},
            }
        except Exception as e:
            return {"success": False, "error": f"Pivot failed: {e}"}

    return AgentTool(
        name="pivot_table",
        description=(
            "Cross-tabulate query results. Provide a SQL query that returns raw data, "
            "then specify which column to use as rows, which as columns, which as values, "
            "and the aggregation function (sum/mean/count/min/max). Returns a pivot matrix."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query returning raw rows to pivot"},
                "rows": {"type": "string", "description": "Column name to use as row labels"},
                "columns": {"type": "string", "description": "Column name to use as column labels"},
                "values": {"type": "string", "description": "Column name to aggregate"},
                "aggfunc": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "min", "max"],
                    "description": "Aggregation function (default: sum)",
                },
                "fill_value": {"type": "number", "description": "Value for missing cells (default: 0)"},
            },
            "required": ["sql", "rows", "columns", "values"],
        },
        handler=handler,
        category="analysis",
        source="analyst_toolkit",
    )
