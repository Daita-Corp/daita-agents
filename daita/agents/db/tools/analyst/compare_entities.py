"""
compare_entities — side-by-side entity comparison tool.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .....core.tools import AgentTool
from ._helpers import (
    AnalystCatalogContext,
    build_entity_profiles,
    combined_source_metadata,
    get_pk_column,
    make_analysis_tool,
)

if TYPE_CHECKING:
    from .....plugins.base_db import BaseDatabasePlugin


def create_compare_entities_tool(
    plugin: "BaseDatabasePlugin", catalog_context: AnalystCatalogContext
) -> AgentTool:
    """Return an AgentTool that compares entities side-by-side."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        entity_table = args.get("entity_table", "").strip()
        entity_ids = args.get("entity_ids", [])
        id_column = args.get("id_column") or get_pk_column(
            catalog_context, entity_table
        )
        custom_dimensions: Optional[List[Dict]] = args.get("dimensions")

        if not entity_table:
            return {"success": False, "error": "entity_table parameter is required"}
        if len(entity_ids) < 2:
            return {"success": False, "error": "entity_ids must contain at least 2 IDs"}
        if not id_column:
            return {
                "success": False,
                "error": f"Could not detect primary key for '{entity_table}'. "
                "Pass id_column explicitly.",
            }

        try:
            profile_result = await build_entity_profiles(
                plugin,
                catalog_context,
                entity_table=entity_table,
                id_column=id_column,
                dimensions=custom_dimensions,
                entity_ids=list(entity_ids),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

        entity_profiles = profile_result.profiles

        if not any(entity_profiles.values()):
            return {
                "success": False,
                "error": "Could not retrieve any dimension data for the entities",
                **combined_source_metadata(profile_result.query_results),
            }

        # Build comparison table
        all_dim_names = profile_result.dimensions

        comparison = []
        for dim in all_dim_names:
            entry: Dict[str, Any] = {"dimension": dim}
            values = []
            for eid in entity_ids:
                val = entity_profiles[eid].get(dim)
                entry[str(eid)] = val
                values.append(val)

            # Compute delta / pct_diff for exactly 2 entities
            if len(entity_ids) == 2:
                a, b = values
                if a is not None and b is not None:
                    try:
                        delta = float(b) - float(a)
                        pct = (delta / float(a) * 100) if float(a) != 0 else None
                        entry["delta"] = round(delta, 4)
                        entry["pct_diff"] = round(pct, 2) if pct is not None else None
                    except (TypeError, ValueError):
                        pass

            comparison.append(entry)

        # Biggest differences (top 5 by |pct_diff|)
        biggest = sorted(
            [r for r in comparison if r.get("pct_diff") is not None],
            key=lambda r: abs(r["pct_diff"]),
            reverse=True,
        )[:5]

        return {
            "success": True,
            "entities": list(entity_ids),
            "comparison": comparison,
            "biggest_differences": biggest,
            "skipped_dimensions": profile_result.skipped_dimensions,
            **combined_source_metadata(profile_result.query_results),
        }

    return make_analysis_tool(
        name="compare_entities",
        description=(
            "Compare two or more entities (e.g. customers, products, employees) side by side "
            "across automatically-inferred dimensions derived from related tables. "
            "Highlights deltas and percentage differences between entities."
        ),
        parameters={
            "type": "object",
            "properties": {
                "entity_table": {
                    "type": "string",
                    "description": "Table whose rows represent the entities to compare",
                },
                "entity_ids": {
                    "type": "array",
                    "items": {},
                    "description": "List of 2+ entity IDs (primary key values) to compare",
                },
                "id_column": {
                    "type": "string",
                    "description": "Primary key column name (auto-detected from schema if omitted)",
                },
                "dimensions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "alias": {"type": "string"},
                            "child_table": {"type": "string"},
                        },
                        "required": ["expression", "alias", "child_table"],
                    },
                    "description": "Custom aggregate expressions (auto-inferred from FK relationships if omitted)",
                },
            },
            "required": ["entity_table", "entity_ids"],
        },
        handler=handler,
    )
