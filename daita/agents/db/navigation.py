"""
Metadata-only schema navigation for large ``from_db()`` agents.

These helpers operate on the already-normalized schema captured during
``from_db`` construction. They never query live rows and never execute SQL;
query execution remains owned by the database plugins and their guardrails.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_NAVIGATION_TABLE_THRESHOLD = 30
SCHEMA_NAVIGATION_WIDE_TABLE_THRESHOLD = 100
MAX_TABLE_LIST_LIMIT = 100
MAX_SEARCH_LIMIT = 50
MAX_INSPECT_COLUMNS_LIMIT = 200
MATCHED_COLUMNS_LIMIT = 12


def should_register_schema_navigation(schema: Dict[str, Any]) -> bool:
    """Return True when prompt-only schema context is likely insufficient."""
    tables = schema.get("tables", []) or []
    table_count = int(schema.get("table_count") or len(tables))
    short_names = [_short_table_name(str(table.get("name", ""))) for table in tables]
    return (
        table_count > SCHEMA_NAVIGATION_TABLE_THRESHOLD
        or any(
            len(table.get("columns", []) or []) > SCHEMA_NAVIGATION_WIDE_TABLE_THRESHOLD
            for table in tables
        )
        or len(set(short_names)) < len(short_names)
    )


def list_tables(
    schema: Dict[str, Any],
    *,
    pattern: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List bounded table summaries, optionally filtered by substring/glob."""
    tables = schema.get("tables", []) or []
    normalized_pattern = (pattern or "").strip()
    matches = [
        table for table in tables if _matches_table_pattern(table, normalized_pattern)
    ]
    limit = _clamp_int(limit, default=50, minimum=1, maximum=MAX_TABLE_LIST_LIMIT)
    offset = _clamp_int(offset, default=0, minimum=0, maximum=max(len(matches), 0))
    page = matches[offset : offset + limit]
    return {
        "database_type": schema.get("database_type"),
        "table_count": int(schema.get("table_count") or len(tables)),
        "pattern": normalized_pattern or None,
        "total_matches": len(matches),
        "offset": offset,
        "limit": limit,
        "truncated": offset + len(page) < len(matches),
        "tables": [_table_summary(table) for table in page],
    }


def search_schema(
    schema: Dict[str, Any],
    *,
    query: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search table and column metadata with bounded, explainable results."""
    tokens = _query_tokens(query)
    limit = _clamp_int(limit, default=20, minimum=1, maximum=MAX_SEARCH_LIMIT)
    scored = []
    for table in schema.get("tables", []) or []:
        score, matched_columns, matched_reasons = _score_table(table, tokens)
        rels = _relationships_for_table(schema, table.get("name", ""))
        if score <= 0 and tokens:
            continue
        scored.append(
            {
                **_table_summary(table),
                "score": round(score, 3),
                "matched_columns": matched_columns[:MATCHED_COLUMNS_LIMIT],
                "match_reasons": matched_reasons[:8],
                "relationships": rels[:8],
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["name"]))
    return {
        "query": query,
        "tokens": tokens,
        "total_matches": len(scored),
        "limit": limit,
        "truncated": len(scored) > limit,
        "tables": scored[:limit],
    }


def inspect_table(
    schema: Dict[str, Any],
    *,
    table_name: str,
    column_pattern: Optional[str] = None,
    include_columns: bool = True,
    limit: int = 100,
    offset: int = 0,
    blocked_columns: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return bounded metadata for one table, including optional column page."""
    table = _find_table(schema, table_name)
    if table is None:
        candidates = search_schema(schema, query=table_name, limit=10)["tables"]
        return {
            "success": False,
            "error": f"Table not found: {table_name}",
            "candidates": candidates,
        }

    blocked = {col.lower() for col in blocked_columns or []}
    columns = table.get("columns", []) or []
    filtered_columns = [
        col for col in columns if _matches_column_pattern(col, column_pattern)
    ]
    limit = _clamp_int(limit, default=100, minimum=1, maximum=MAX_INSPECT_COLUMNS_LIMIT)
    offset = _clamp_int(
        offset, default=0, minimum=0, maximum=max(len(filtered_columns), 0)
    )
    page = filtered_columns[offset : offset + limit]
    result = {
        "success": True,
        "table": _table_summary(table),
        "relationships": _relationships_for_table(schema, table["name"]),
    }
    if include_columns:
        result["columns"] = [
            _column_summary(col, blocked_columns=blocked) for col in page
        ]
        result["column_count"] = len(columns)
        result["matched_column_count"] = len(filtered_columns)
        result["column_pattern"] = column_pattern or None
        result["offset"] = offset
        result["limit"] = limit
        result["truncated"] = offset + len(page) < len(filtered_columns)
    return result


def describe_relationships(
    schema: Dict[str, Any],
    *,
    table_name: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Return bounded FK metadata, optionally scoped to one table."""
    rels = schema.get("foreign_keys", []) or []
    if table_name:
        wanted = table_name.lower()
        rels = [
            fk
            for fk in rels
            if str(fk.get("source_table", "")).lower() == wanted
            or str(fk.get("target_table", "")).lower() == wanted
        ]
    limit = _clamp_int(limit, default=50, minimum=1, maximum=100)
    page = rels[:limit]
    return {
        "table_name": table_name,
        "relationship_count": len(rels),
        "limit": limit,
        "truncated": len(rels) > limit,
        "relationships": [_relationship_summary(fk) for fk in page],
    }


def _score_table(table: Dict[str, Any], tokens: List[str]) -> tuple[float, list, list]:
    if not tokens:
        return 0.0, [], []
    table_name = str(table.get("name", "")).lower()
    columns = table.get("columns", []) or []
    score = 0.0
    reasons = []
    matched_columns = []

    for token in tokens:
        if token == table_name:
            score += 6.0
            reasons.append(f"exact table:{token}")
        elif token in table_name or token in _split_identifier(table_name):
            score += 3.0
            reasons.append(f"table:{token}")

    for col in columns:
        col_name = str(col.get("name", "")).lower()
        col_comment = str(col.get("column_comment") or "").lower()
        col_score = 0.0
        col_reasons = []
        for token in tokens:
            if token == col_name:
                col_score += 4.0
                col_reasons.append(f"exact:{token}")
            elif token in col_name or token in _split_identifier(col_name):
                col_score += 2.0
                col_reasons.append(token)
            elif col_comment and token in col_comment:
                col_score += 0.5
                col_reasons.append(f"comment:{token}")
        if col_score:
            score += min(col_score, 8.0)
            matched_columns.append(
                {
                    "name": col.get("name"),
                    "type": col.get("type"),
                    "score": round(col_score, 3),
                    "reasons": col_reasons[:5],
                }
            )

    matched_columns.sort(key=lambda item: (-item["score"], item["name"]))
    return score, matched_columns, reasons


def _table_summary(table: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": table.get("name"),
        "row_count": table.get("row_count"),
        "column_count": len(table.get("columns", []) or []),
    }


def _column_summary(
    col: Dict[str, Any],
    *,
    blocked_columns: set[str],
) -> Dict[str, Any]:
    out = {
        "name": col.get("name"),
        "type": col.get("type"),
        "nullable": col.get("nullable"),
        "is_primary_key": bool(col.get("is_primary_key")),
    }
    if col.get("column_comment"):
        out["comment"] = _truncate(str(col["column_comment"]), 160)
    if str(col.get("name", "")).lower() in blocked_columns:
        out["blocked_by_policy"] = True
    return out


def _relationship_summary(fk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_table": fk.get("source_table"),
        "source_column": fk.get("source_column"),
        "target_table": fk.get("target_table"),
        "target_column": fk.get("target_column"),
    }


def _relationships_for_table(
    schema: Dict[str, Any],
    table_name: str,
) -> List[Dict[str, Any]]:
    wanted = table_name.lower()
    relationships = []
    for fk in schema.get("foreign_keys", []) or []:
        source = str(fk.get("source_table", "")).lower()
        target = str(fk.get("target_table", "")).lower()
        if source == wanted or target == wanted:
            direction = "outgoing" if source == wanted else "incoming"
            relationships.append({**_relationship_summary(fk), "direction": direction})
    return relationships


def _find_table(schema: Dict[str, Any], table_name: str) -> Optional[Dict[str, Any]]:
    wanted = table_name.strip().lower()
    for table in schema.get("tables", []) or []:
        name = str(table.get("name", ""))
        if name.lower() == wanted:
            return table
    return None


def _short_table_name(name: str) -> str:
    return name.split(".")[-1].lower()


def _matches_table_pattern(table: Dict[str, Any], pattern: str) -> bool:
    if not pattern:
        return True
    name = str(table.get("name", ""))
    lowered_name = name.lower()
    lowered_pattern = pattern.lower()
    if any(ch in lowered_pattern for ch in "*?[]"):
        return fnmatch.fnmatch(lowered_name, lowered_pattern)
    return lowered_pattern in lowered_name


def _matches_column_pattern(col: Dict[str, Any], pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    name = str(col.get("name", "")).lower()
    lowered_pattern = pattern.lower()
    if any(ch in lowered_pattern for ch in "*?[]"):
        return fnmatch.fnmatch(name, lowered_pattern)
    if lowered_pattern in name:
        return True
    pattern_parts = _split_identifier(lowered_pattern)
    name_parts = _split_identifier(name)
    return bool(pattern_parts) and all(
        any(part in name_part or name_part.startswith(part) for name_part in name_parts)
        for part in pattern_parts
    )


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
    return [part for part in re.split(r"[_\W]+", value.lower()) if part]


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."
