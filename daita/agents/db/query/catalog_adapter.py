"""SQL-planner adapters over CatalogPlugin payloads.

These helpers do not own discovery, persistence, or catalog state. They only
project catalog search/inspection/path responses into the compact shapes the
from_db query planner already consumes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .metadata import column_name as _column_name

MAX_SEARCH_LIMIT = 50
MAX_JOIN_HOPS = 6
MAX_JOIN_PATHS = 8


def has_likely_catalog_match(agent: Any, text: str) -> bool:
    """Return True when catalog search finds likely assets/fields for text."""
    catalog = vars(agent).get("_db_catalog")
    store_id = vars(agent).get("_db_catalog_store_id")
    if catalog is None or not store_id:
        return False
    result = catalog.catalog_search_schema(store_id, text, limit=3)
    return any(float(item.get("score") or 0) > 0 for item in result.get("tables", []))


def catalog_schema_snapshot(agent: Any) -> Dict[str, Any]:
    """Return the active catalog schema snapshot."""
    catalog = vars(agent).get("_db_catalog")
    store_id = vars(agent).get("_db_catalog_store_id")
    if catalog is None or not store_id:
        return {}
    schema = catalog.get_schema(store_id)
    if schema is None:
        return {}
    return schema.to_dict() if hasattr(schema, "to_dict") else dict(schema)


def primary_key_or_identity(table: Dict[str, Any]) -> Optional[str]:
    columns = table.get("columns") or []
    for column in columns:
        if isinstance(column, dict) and column.get("is_primary_key"):
            return _column_name(column)
    for column in columns:
        name = _column_name(column)
        if name and (name == "id" or name.endswith("_id")):
            return name
    return None


def search_tables(
    _schema: Dict[str, Any],
    *,
    query: str,
    limit: int = 20,
    catalog: Any = None,
    store_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return planner-shaped table candidates from catalog search."""
    limit = _clamp_int(limit, default=20, minimum=1, maximum=MAX_SEARCH_LIMIT)
    if catalog is not None and store_id:
        result = catalog.catalog_search_schema(store_id, query, limit=limit)
        tables = [_catalog_asset_to_table(item) for item in result.get("tables", [])]
        return {
            "query": query,
            "tokens": result.get("tokens", _query_tokens(query)),
            "total_matches": result.get("total_matches", len(tables)),
            "limit": limit,
            "truncated": result.get("truncated", False),
            "tables": tables,
            "source": "catalog",
        }

    return {
        "query": query,
        "tokens": _query_tokens(query),
        "total_matches": 0,
        "limit": limit,
        "truncated": False,
        "tables": [],
        "source": "missing_catalog",
        "error": "catalog and store_id are required for table search",
    }


def find_relationship_paths(
    _schema: Dict[str, Any],
    *,
    from_tables: List[str],
    to_tables: List[str],
    max_hops: int = 4,
    max_paths: int = 5,
    catalog: Any = None,
    store_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return planner-shaped join paths from catalog relationships."""
    max_hops = _clamp_int(max_hops, default=4, minimum=1, maximum=MAX_JOIN_HOPS)
    max_paths = _clamp_int(max_paths, default=5, minimum=1, maximum=MAX_JOIN_PATHS)
    if catalog is not None and store_id:
        result = catalog.find_relationship_paths(
            store_id,
            from_tables,
            to_tables,
            relationship_types=["foreign_key", "references"],
            max_hops=max_hops,
            max_paths=max_paths,
        )
        return {
            **result,
            "from_tables": result.get("from_assets", from_tables),
            "to_tables": result.get("to_assets", to_tables),
            "source": "catalog",
        }

    schema_result = _schema_relationship_paths(
        _schema,
        from_tables=from_tables,
        to_tables=to_tables,
        max_hops=max_hops,
        max_paths=max_paths,
    )
    if schema_result.get("reachable"):
        return schema_result

    return {
        "success": False,
        "from_tables": from_tables,
        "to_tables": to_tables,
        "max_hops": max_hops,
        "path_count": 0,
        "reachable": False,
        "paths": [],
        "errors": ["catalog and store_id are required for relationship paths"],
        "source": "missing_catalog",
    }


def _schema_relationship_paths(
    schema: Dict[str, Any],
    *,
    from_tables: List[str],
    to_tables: List[str],
    max_hops: int,
    max_paths: int,
) -> Dict[str, Any]:
    adjacency = _schema_relationship_adjacency(schema)
    starts = [table for table in from_tables if table in adjacency]
    targets = set(to_tables)
    paths = []
    for start in starts:
        queue = [(start, [])]
        visited = {start}
        while queue and len(paths) < max_paths:
            table, joins = queue.pop(0)
            if len(joins) >= max_hops:
                continue
            for next_table, join in adjacency.get(table, []):
                if next_table in visited:
                    continue
                next_joins = joins + [join]
                if next_table in targets:
                    paths.append(
                        {
                            "tables": [start]
                            + [item["right_table"] for item in next_joins],
                            "joins": next_joins,
                            "predicate": " AND ".join(
                                item["predicate"] for item in next_joins
                            ),
                            "confidence": 0.9,
                        }
                    )
                    if len(paths) >= max_paths:
                        break
                visited.add(next_table)
                queue.append((next_table, next_joins))
    return {
        "success": bool(paths),
        "from_tables": from_tables,
        "to_tables": to_tables,
        "max_hops": max_hops,
        "path_count": len(paths),
        "reachable": bool(paths),
        "paths": paths,
        "errors": [] if paths else ["no schema relationship path found"],
        "source": "schema",
    }


def _schema_relationship_adjacency(
    schema: Dict[str, Any],
) -> Dict[str, List[tuple[str, Dict[str, Any]]]]:
    adjacency: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for fk in schema.get("foreign_keys", []) or []:
        source_table = str(fk.get("source_table") or "")
        source_column = str(fk.get("source_column") or "")
        target_table = str(fk.get("target_table") or "")
        target_column = str(fk.get("target_column") or "")
        if not all((source_table, source_column, target_table, target_column)):
            continue
        forward = {
            "left_table": source_table,
            "left_column": source_column,
            "right_table": target_table,
            "right_column": target_column,
            "predicate": (
                f"{source_table}.{source_column} = " f"{target_table}.{target_column}"
            ),
        }
        reverse = {
            "left_table": target_table,
            "left_column": target_column,
            "right_table": source_table,
            "right_column": source_column,
            "predicate": (
                f"{target_table}.{target_column} = " f"{source_table}.{source_column}"
            ),
        }
        adjacency.setdefault(source_table, []).append((target_table, forward))
        adjacency.setdefault(target_table, []).append((source_table, reverse))
    return adjacency


def _catalog_asset_to_table(asset: Dict[str, Any]) -> Dict[str, Any]:
    matched_columns = []
    for field in asset.get("matched_fields") or []:
        matched_columns.append(
            {
                "name": field.get("name"),
                "type": field.get("type"),
                "score": field.get("score", 0),
                "reasons": field.get("reasons", []),
            }
        )
    return {
        "name": asset.get("name") or asset.get("asset_ref"),
        "row_count": asset.get("row_count"),
        "column_count": asset.get("column_count") or asset.get("field_count", 0),
        "score": asset.get("score", 0),
        "matched_columns": matched_columns,
        "match_reasons": asset.get("match_reasons", []),
        "relationships": asset.get("relationships", []),
    }


def _query_tokens(query: str) -> List[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", (query or "").lower())
    tokens = []
    for raw in raw_tokens:
        tokens.extend(_split_identifier(raw))
    seen = set()
    return [
        token
        for token in tokens
        if len(token) > 1 and token not in seen and not seen.add(token)
    ]


def _split_identifier(value: str) -> List[str]:
    return [part for part in re.split(r"[^a-zA-Z0-9]+|_", value.lower()) if part]


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))
