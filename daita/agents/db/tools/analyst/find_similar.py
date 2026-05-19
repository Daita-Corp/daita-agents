"""
find_similar — entity similarity via normalised Euclidean distance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .....core.tools import AgentTool
from ._helpers import (
    AnalystCatalogContext,
    ensure_numpy,
    build_entity_profiles,
    combined_source_metadata,
    safe_query,
    to_serializable,
    get_pk_column,
    make_analysis_tool,
)

if TYPE_CHECKING:
    from .....plugins.base_db import BaseDatabasePlugin


def create_find_similar_tool(
    plugin: "BaseDatabasePlugin", catalog_context: AnalystCatalogContext
) -> AgentTool:
    """Return an AgentTool that finds entities similar to a reference entity."""

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            np = ensure_numpy()
        except ImportError as e:
            return {"success": False, "error": str(e)}

        entity_table = args.get("entity_table", "").strip()
        entity_id = args.get("entity_id")
        id_column = args.get("id_column") or get_pk_column(
            catalog_context, entity_table
        )
        custom_dimensions: Optional[List[Dict]] = args.get("dimensions")
        top_k = int(args.get("top_k", 5))
        candidate_sql = (args.get("candidate_sql") or "").strip()
        candidate_limit = int(args.get("candidate_limit", 1000))

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

        query_results = []
        try:
            candidate_ids = None
            candidate_scope = "limited"
            if candidate_sql:
                candidate_result = await safe_query(plugin, candidate_sql)
                query_results.append(candidate_result)
                candidate_ids = _candidate_ids_from_rows(
                    candidate_result.rows,
                    id_column=id_column,
                )
                if entity_id not in candidate_ids:
                    candidate_ids.append(entity_id)
                candidate_scope = "candidate_sql"

            profile_result = await build_entity_profiles(
                plugin,
                catalog_context,
                entity_table=entity_table,
                id_column=id_column,
                dimensions=custom_dimensions,
                entity_ids=candidate_ids,
                candidate_limit=max(top_k + 1, candidate_limit),
            )
            query_results.extend(profile_result.query_results)
        except Exception as e:
            return {"success": False, "error": str(e)}

        all_profiles: Dict[Any, Dict[str, float]] = {}
        for eid, profile in profile_result.profiles.items():
            all_profiles[eid] = {}
            for key, value in profile.items():
                try:
                    all_profiles[eid][key] = float(to_serializable(value) or 0)
                except (TypeError, ValueError):
                    all_profiles[eid][key] = 0.0

        if not all_profiles:
            return {
                "success": False,
                "error": "Could not retrieve dimension data",
                **combined_source_metadata(query_results),
            }

        if entity_id not in all_profiles:
            return {
                "success": False,
                "error": f"Entity '{entity_id}' not found in '{entity_table}'",
                **combined_source_metadata(query_results),
            }

        # Build matrix
        all_dim_names = profile_result.dimensions

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
            "candidate_count": len(eids),
            "candidate_scope": candidate_scope,
            "skipped_dimensions": profile_result.skipped_dimensions,
            **combined_source_metadata(query_results),
        }

    return make_analysis_tool(
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
                "candidate_sql": {
                    "type": "string",
                    "description": "Optional SQL returning candidate entity IDs to compare against.",
                },
                "candidate_limit": {
                    "type": "integer",
                    "description": "Maximum candidate entities to profile when candidate_sql is omitted (default: 1000).",
                },
            },
            "required": ["entity_table", "entity_id"],
        },
        handler=handler,
    )


def _candidate_ids_from_rows(
    rows: List[Dict[str, Any]], *, id_column: str
) -> List[Any]:
    ids = []
    for row in rows:
        if "_entity_id" in row:
            ids.append(row["_entity_id"])
        elif id_column in row:
            ids.append(row[id_column])
        elif row:
            ids.append(next(iter(row.values())))
    return list(dict.fromkeys(ids))
