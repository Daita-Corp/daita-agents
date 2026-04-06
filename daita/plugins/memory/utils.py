"""
Shared utilities for the memory plugin.

Consolidates repeated patterns: metadata parsing, chunk ID generation,
SQL WHERE clause building, and metadata merging.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

_METADATA_DEFAULTS = {
    "importance": 0.5,
    "source": "agent_inferred",
    "pinned": False,
}


def parse_metadata_json(metadata_json: Optional[str]) -> Dict[str, Any]:
    """Parse metadata JSON string with sensible defaults on failure."""
    if not metadata_json:
        return dict(_METADATA_DEFAULTS)
    try:
        return json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return dict(_METADATA_DEFAULTS)


def generate_chunk_id(content: str) -> str:
    """Generate a deterministic chunk ID from content.

    Content-based hash only (no timestamp) so identical content
    submitted in the same session produces the same chunk_id.
    """
    return hashlib.md5(f"direct:{content.strip()}".encode()).hexdigest()


def merge_metadata_json(metadata, extra: Optional[Dict[str, Any]] = None) -> str:
    """Merge extra fields into MemoryMetadata and return JSON string.

    Args:
        metadata: MemoryMetadata instance
        extra: Optional dict of additional fields to merge
    """
    if not extra:
        return metadata.to_json()
    data = metadata.to_dict()
    data.update(extra)
    return json.dumps(data)


def build_where_clause(
    category: Optional[str] = None,
    since: Optional[str] = None,
    before: Optional[str] = None,
    table_alias: Optional[str] = None,
) -> Tuple[str, List]:
    """Build a SQL WHERE clause for metadata filtering.

    Args:
        category: Filter by category
        since: Filter by created_at >= value
        before: Filter by created_at < value
        table_alias: Optional table alias (e.g. "c" for "c.metadata")

    Returns:
        (where_sql, params) tuple. where_sql includes leading " WHERE " if non-empty.
    """
    meta_col = f"{table_alias}.metadata" if table_alias else "metadata"
    clauses = []
    params = []

    if category:
        clauses.append(f"json_extract({meta_col}, '$.category') = ?")
        params.append(category)
    if since:
        clauses.append(f"json_extract({meta_col}, '$.created_at') >= ?")
        params.append(since)
    if before:
        clauses.append(f"json_extract({meta_col}, '$.created_at') < ?")
        params.append(before)

    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def serialize_results(results: Any) -> Any:
    """JSON round-trip to convert datetime and other non-serializable types."""
    return json.loads(json.dumps(results, default=str))
