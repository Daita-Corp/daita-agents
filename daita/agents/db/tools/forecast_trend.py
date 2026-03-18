"""
forecast_trend — time-series trend and projection tool.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ....core.tools import AgentTool
from ._helpers import ensure_pandas, ensure_numpy, safe_query, to_serializable

if TYPE_CHECKING:
    from ....plugins.base_db import BaseDatabasePlugin

_FREQ_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "quarterly": 91,
}


def _detect_frequency(median_gap_days: float) -> str:
    if median_gap_days <= 1.5:
        return "daily"
    if median_gap_days <= 10:
        return "weekly"
    if median_gap_days <= 50:
        return "monthly"
    return "quarterly"


def create_forecast_trend_tool(
    plugin: "BaseDatabasePlugin", schema: Dict[str, Any]
) -> AgentTool:
    """Return an AgentTool that fits a linear trend and projects forward."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pd = ensure_pandas()
            np = ensure_numpy()
        except ImportError as e:
            return {"success": False, "error": str(e)}

        sql = args.get("sql", "").strip()
        date_column = args.get("date_column", "").strip()
        metric_column = args.get("metric_column", "").strip()
        periods = int(args.get("periods", 3))
        explicit_freq: Optional[str] = args.get("frequency")

        if not sql:
            return {"success": False, "error": "sql parameter is required"}
        if not date_column:
            return {"success": False, "error": "date_column parameter is required"}
        if not metric_column:
            return {"success": False, "error": "metric_column parameter is required"}

        try:
            rows = await safe_query(plugin, sql)
            if not rows:
                return {
                    "success": True,
                    "historical": [],
                    "forecast": [],
                    "trend": None,
                }

            df = pd.DataFrame(
                [{k: to_serializable(v) for k, v in r.items()} for r in rows]
            )

            if date_column not in df.columns:
                return {"success": False, "error": f"Column '{date_column}' not found"}
            if metric_column not in df.columns:
                return {
                    "success": False,
                    "error": f"Column '{metric_column}' not found",
                }

            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
            df = df[[date_column, metric_column]].dropna().sort_values(date_column)

            if len(df) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 data points to compute trend",
                }

            dates = df[date_column].values
            values = df[metric_column].values.astype(float)

            # Determine frequency
            gaps_days = np.diff(dates.astype("datetime64[D]").astype(float))
            median_gap = float(np.median(gaps_days)) if len(gaps_days) > 0 else 30.0
            frequency = explicit_freq or _detect_frequency(median_gap)

            # Convert dates to ordinal float for regression
            ordinals = dates.astype("datetime64[D]").astype(float)
            coeffs = np.polyfit(ordinals, values, deg=1)
            slope, intercept = coeffs

            fitted = np.polyval(coeffs, ordinals)
            ss_res = np.sum((values - fitted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else 1.0

            # Trend direction
            direction = "up" if slope > 0 else ("down" if slope < 0 else "flat")

            # Growth rate: compare first vs last fitted value
            if fitted[0] != 0:
                total_growth = (fitted[-1] - fitted[0]) / abs(fitted[0]) * 100
            else:
                total_growth = 0.0

            # Confidence rating based on R²
            if r_squared >= 0.9:
                confidence = "high"
            elif r_squared >= 0.7:
                confidence = "medium"
            else:
                confidence = "low"

            # Forecast future periods
            freq_days = _FREQ_DAYS.get(frequency, 30)
            last_date = pd.Timestamp(dates[-1])
            forecast = []
            for i in range(1, periods + 1):
                future_date = last_date + pd.Timedelta(days=freq_days * i)
                future_ordinal = float(
                    np.datetime64(future_date.date(), "D").astype(float)
                )
                predicted = float(np.polyval(coeffs, future_ordinal))
                forecast.append(
                    {
                        "date": future_date.date().isoformat(),
                        "predicted": round(predicted, 4),
                    }
                )

            historical = [
                {
                    "date": pd.Timestamp(d).date().isoformat(),
                    "value": round(float(v), 4),
                }
                for d, v in zip(dates, values)
            ]

            # slope_per_period is slope in original units per freq_days
            slope_per_period = float(slope) * freq_days

            return {
                "success": True,
                "trend": {
                    "direction": direction,
                    "slope_per_period": round(slope_per_period, 6),
                    "r_squared": round(r_squared, 4),
                    "confidence": confidence,
                    "frequency": frequency,
                },
                "historical": historical,
                "forecast": forecast,
                "growth_rate_pct": round(total_growth, 2),
            }

        except Exception as e:
            return {"success": False, "error": f"Trend forecast failed: {e}"}

    return AgentTool(
        name="forecast_trend",
        description=(
            "Fit a linear trend to time-series data and project forward. "
            "Provide a SQL query returning date and metric columns; the tool handles "
            "frequency detection, R² confidence scoring, and future-period projection."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query returning at least two columns: a date and a numeric metric",
                },
                "date_column": {
                    "type": "string",
                    "description": "Name of the date/timestamp column",
                },
                "metric_column": {
                    "type": "string",
                    "description": "Name of the numeric metric column",
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of future periods to forecast (default: 3)",
                },
                "frequency": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly", "quarterly"],
                    "description": "Data frequency — auto-detected from gaps if omitted",
                },
            },
            "required": ["sql", "date_column", "metric_column"],
        },
        handler=handler,
        category="analysis",
        source="analyst_toolkit",
    )
