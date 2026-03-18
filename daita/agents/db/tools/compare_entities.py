"""
compare_entities — side-by-side entity comparison tool.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ....core.tools import AgentTool
from ._helpers import (
    ensure_pandas,
    safe_query,
    to_serializable,
    get_pk_column,
    infer_dimensions,
)

if TYPE_CHECKING:
    from ....plugins.base_db import BaseDatabasePlugin


def create_compare_entities_tool(
    plugin: "BaseDatabasePlugin", schema: Dict[str, Any]
) -> AgentTool:
    """Return an AgentTool that compares entities side-by-side."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pd = ensure_pandas()
        except ImportError as e:
            return {"success": False, "error": str(e)}

        entity_table = args.get("entity_table", "").strip()
        entity_ids = args.get("entity_ids", [])
        id_column = args.get("id_column") or get_pk_column(schema, entity_table)
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

        dims = custom_dimensions or infer_dimensions(schema, entity_table)
        if not dims:
            return {
                "success": False,
                "error": (
                    f"No FK relationships found for '{entity_table}'. "
                    "Pass explicit dimensions as [{expression, alias, child_table}]."
                ),
            }

        # Build one query per unique child_table
        child_tables = list(dict.fromkeys(d["child_table"] for d in dims))
        entity_profiles: Dict[Any, Dict[str, Any]] = {eid: {} for eid in entity_ids}

        id_list = ", ".join(
            f"'{eid}'" if isinstance(eid, str) else str(eid) for eid in entity_ids
        )

        for child in child_tables:
            child_dims = [d for d in dims if d["child_table"] == child]
            fk_col = child_dims[0].get("fk_col")
            if not fk_col:
                continue

            # Collect any grandchild joins required by 2-hop dims
            gc_joins: Dict[str, tuple] = {}
            for d in child_dims:
                if d.get("grandchild_table"):
                    gc_alias = d["grandchild_alias"]
                    if gc_alias not in gc_joins:
                        gc_joins[gc_alias] = (
                            d["grandchild_table"],
                            d["grandchild_fk_col"],
                            d.get("child_pk") or get_pk_column(schema, child),
                        )

            exprs = ", ".join(f"{d['expression']} AS {d['alias']}" for d in child_dims)
            join_clause = (
                f"FROM {entity_table} e "
                f"LEFT JOIN {child} c ON e.{id_column} = c.{fk_col}"
            )
            for gc_alias, (gc_table, gc_fk_col, child_pk_col) in gc_joins.items():
                join_clause += (
                    f" LEFT JOIN {gc_table} {gc_alias}"
                    f" ON c.{child_pk_col} = {gc_alias}.{gc_fk_col}"
                )
            sql = (
                f"SELECT e.{id_column} AS _entity_id, {exprs} "
                f"{join_clause} "
                f"WHERE e.{id_column} IN ({id_list}) "
                f"GROUP BY e.{id_column}"
            )

            try:
                rows = await safe_query(plugin, sql)
                for row in rows:
                    eid = row.get("_entity_id")
                    if eid in entity_profiles:
                        for d in child_dims:
                            entity_profiles[eid][d["alias"]] = to_serializable(
                                row.get(d["alias"])
                            )
            except Exception:
                # Skip dimension set that fails; don't abort entire comparison
                continue

        if not any(entity_profiles.values()):
            return {
                "success": False,
                "error": "Could not retrieve any dimension data for the entities",
            }

        # Build comparison table
        all_dim_names = list(
            dict.fromkeys(alias for d in dims for alias in [d["alias"]])
        )

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
        }

    return AgentTool(
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
        category="analysis",
        source="analyst_toolkit",
    )
