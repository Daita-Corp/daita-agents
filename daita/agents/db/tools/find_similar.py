"""
find_similar — entity similarity via normalised Euclidean distance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ....core.tools import AgentTool
from ._helpers import (
    ensure_pandas,
    ensure_numpy,
    safe_query,
    to_serializable,
    get_pk_column,
    infer_dimensions,
)

if TYPE_CHECKING:
    from ....plugins.base_db import BaseDatabasePlugin


def create_find_similar_tool(
    plugin: "BaseDatabasePlugin", schema: Dict[str, Any]
) -> AgentTool:
    """Return an AgentTool that finds entities similar to a reference entity."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pd = ensure_pandas()
            np = ensure_numpy()
        except ImportError as e:
            return {"success": False, "error": str(e)}

        entity_table = args.get("entity_table", "").strip()
        entity_id = args.get("entity_id")
        id_column = args.get("id_column") or get_pk_column(schema, entity_table)
        custom_dimensions: Optional[List[Dict]] = args.get("dimensions")
        top_k = int(args.get("top_k", 5))

        if not entity_table:
            return {"success": False, "error": "entity_table parameter is required"}
        if entity_id is None:
            return {"success": False, "error": "entity_id parameter is required"}
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

        # Build one query per child_table, fetching ALL entities
        child_tables = list(dict.fromkeys(d["child_table"] for d in dims))
        all_profiles: Dict[Any, Dict[str, float]] = {}

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
                f"GROUP BY e.{id_column}"
            )

            try:
                rows = await safe_query(plugin, sql)
                for row in rows:
                    eid = row.get("_entity_id")
                    if eid not in all_profiles:
                        all_profiles[eid] = {}
                    for d in child_dims:
                        raw = row.get(d["alias"])
                        try:
                            all_profiles[eid][d["alias"]] = float(
                                to_serializable(raw) or 0
                            )
                        except (TypeError, ValueError):
                            all_profiles[eid][d["alias"]] = 0.0
            except Exception:
                continue

        if not all_profiles:
            return {"success": False, "error": "Could not retrieve dimension data"}

        if entity_id not in all_profiles:
            return {
                "success": False,
                "error": f"Entity '{entity_id}' not found in '{entity_table}'",
            }

        # Build matrix
        all_dim_names = list(
            dict.fromkeys(alias for d in dims for alias in [d["alias"]])
        )

        eids = list(all_profiles.keys())
        matrix = np.array(
            [[all_profiles[eid].get(dim, 0.0) for dim in all_dim_names] for eid in eids]
        )

        # Min-max normalise per column
        col_min = matrix.min(axis=0)
        col_max = matrix.max(axis=0)
        ranges = col_max - col_min
        ranges[ranges == 0] = 1.0  # avoid division by zero
        normed = (matrix - col_min) / ranges

        ref_idx = eids.index(entity_id)
        ref_vec = normed[ref_idx]

        distances = np.sqrt(((normed - ref_vec) ** 2).sum(axis=1))

        # Sort by distance, exclude reference entity
        order = np.argsort(distances)
        similar = []
        for idx in order:
            if eids[idx] == entity_id:
                continue
            profile = {dim: all_profiles[eids[idx]].get(dim) for dim in all_dim_names}
            similar.append(
                {
                    "entity_id": to_serializable(eids[idx]),
                    "distance": round(float(distances[idx]), 6),
                    "profile": {k: to_serializable(v) for k, v in profile.items()},
                }
            )
            if len(similar) >= top_k:
                break

        ref_profile = {dim: all_profiles[entity_id].get(dim) for dim in all_dim_names}

        return {
            "success": True,
            "reference": {
                "id": to_serializable(entity_id),
                "profile": {k: to_serializable(v) for k, v in ref_profile.items()},
            },
            "similar": similar,
            "dimensions_used": all_dim_names,
            "top_k": top_k,
        }

    return AgentTool(
        name="find_similar",
        description=(
            "Find entities most similar to a reference entity using normalised Euclidean "
            "distance across automatically-inferred dimensions (counts, sums, averages from "
            "related tables). Returns the top_k closest matches ranked by similarity."
        ),
        parameters={
            "type": "object",
            "properties": {
                "entity_table": {
                    "type": "string",
                    "description": "Table whose rows represent entities",
                },
                "entity_id": {
                    "description": "Primary key value of the reference entity",
                },
                "id_column": {
                    "type": "string",
                    "description": "Primary key column name (auto-detected if omitted)",
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
                    "description": "Custom dimensions (auto-inferred from FK relationships if omitted)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar entities to return (default: 5)",
                },
            },
            "required": ["entity_table", "entity_id"],
        },
        handler=handler,
        category="analysis",
        source="analyst_toolkit",
    )
