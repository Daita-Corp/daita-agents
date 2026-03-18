"""
Numeric column auto-calibration — infer units (cents vs dollars, etc.) on cold start.
"""

import decimal
import json
import logging
from typing import Any, Dict, List, TYPE_CHECKING

from .sampling import NUMERIC_TYPES

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


async def calibrate_numerics(
    agent: "Agent",
    schema: Dict[str, Any],
    memory_plugin: Any,
) -> None:
    """
    Ask the LLM to infer the units of numeric columns and persist the result via
    *memory_plugin*. Called once on cold start (cache miss) after memory plugin
    attachment. Uses ``memory_plugin.remember()`` directly — bypasses
    ``agent.run()`` to avoid audit log pollution.

    No-ops silently if the schema has no numeric columns or if the LLM call fails.
    """
    numeric_cols: List[Dict[str, Any]] = []
    for table in schema.get("tables", []):
        for col in table.get("columns", []):
            col_type = col.get("type", "").lower()
            if any(nt in col_type for nt in NUMERIC_TYPES):
                entry: Dict[str, Any] = {
                    "table": table["name"],
                    "column": col["name"],
                    "type": col["type"],
                }
                if col.get("_samples"):
                    entry["samples"] = col["_samples"]
                if col.get("column_comment"):
                    entry["comment"] = col["column_comment"]
                numeric_cols.append(entry)

    if not numeric_cols:
        return

    prompt = (
        "You are a database analyst. For each numeric column below, infer whether "
        "values are stored in smallest units (e.g. cents, pence) or whole units "
        "(e.g. dollars, full amounts). Use the column name, type, and sample values "
        "as evidence. Respond with a JSON array where each item has keys: "
        "\"table\", \"column\", \"unit\" (a short string like \"cents\", \"dollars\", "
        "\"grams\", \"unknown\"), and \"confidence\" (\"high\", \"medium\", or \"low\").\n\n"
        f"Columns: {json.dumps(numeric_cols, default=lambda o: float(o) if isinstance(o, decimal.Decimal) else str(o))}"
    )

    try:
        calibration_result = await agent.run(prompt)
        if hasattr(memory_plugin, "remember"):
            await memory_plugin.remember("numeric_column_units", calibration_result)
    except Exception as exc:
        logger.debug(f"Numeric calibration failed: {exc}")
