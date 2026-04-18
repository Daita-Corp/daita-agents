"""
Shared helpers for normalizers.

Holds:
  * ``build_store_metadata`` — canonical mapping from raw-discovery keys
    (``host`` / ``port`` / ``region`` / ``arn`` / ``project`` / ``instance``)
    to the ``metadata`` dict the persister's ``_derive_store`` reads.
    Every normalizer that wants collision-safe store IDs goes through here.
  * ``_normalize_relational`` — body shared by Postgres and MySQL normalizers.
  * Store-level dedup + environment inference utilities used after discovery.
"""

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base_discoverer import DiscoveredStore


# ---------------------------------------------------------------------------
# Store metadata builder — canonical place the persister's _derive_store
# reads from. Keep the key names in sync with persistence._derive_store.
# ---------------------------------------------------------------------------

# Keys the persister cares about when building store identifiers. Anything
# outside this list is normalizer-specific context (e.g. Kinesis shard count,
# S3 storage classes) and goes in the normalizer's own metadata merge.
_STORE_ID_KEYS: tuple[str, ...] = (
    "host",
    "port",
    "region",
    "arn",
    "project",
    "instance",
)


def build_store_metadata(
    raw: Dict[str, Any],
    *,
    keys: Iterable[str] = _STORE_ID_KEYS,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a metadata dict containing only truthy values for the given keys.

    Empty strings, ``None``, and missing keys are dropped so downstream
    ``_derive_store`` can use simple ``meta.get("host")`` truthiness checks
    without worrying about sentinel empty values.

    ``extra`` is merged last and takes precedence — normalizers use it to
    attach store-type-specific context (version, shard count, etc.) without
    polluting the canonical key set.
    """
    result: Dict[str, Any] = {}
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            result[key] = value
    if extra:
        result.update({k: v for k, v in extra.items() if v not in (None, "")})
    return result


# ---------------------------------------------------------------------------
# Relational helper (shared by PostgreSQL + MySQL)
# ---------------------------------------------------------------------------


def _normalize_relational(
    raw: Dict[str, Any],
    db_type: str,
    is_primary_key_fn: Callable[[str, Dict[str, Any]], bool],
    default_schema: str = "",
) -> Dict[str, Any]:
    """Shared normalizer body for relational databases."""
    col_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for col in raw.get("columns", []):
        col_by_table.setdefault(col["table_name"], []).append(col)

    tables = []
    for t in raw.get("tables", []):
        tname = t["table_name"]
        cols = []
        for c in col_by_table.get(tname, []):
            col: Dict[str, Any] = {
                "name": c["column_name"],
                "type": c.get("data_type", ""),
                "nullable": c.get("is_nullable", "YES") == "YES",
                "is_primary_key": is_primary_key_fn(tname, c),
            }
            if c.get("column_comment"):
                col["column_comment"] = c["column_comment"]
            cols.append(col)
        tables.append({"name": tname, "row_count": t.get("row_count"), "columns": cols})

    fks = [
        {
            "source_table": fk["source_table"],
            "source_column": fk["source_column"],
            "target_table": fk["target_table"],
            "target_column": fk["target_column"],
        }
        for fk in raw.get("foreign_keys", [])
    ]

    result: Dict[str, Any] = {
        "database_type": db_type,
        "database_name": raw.get("schema", default_schema),
        "tables": tables,
        "foreign_keys": fks,
        "table_count": len(tables),
    }
    metadata = build_store_metadata(raw)
    if metadata:
        result["metadata"] = metadata
    return result


# ---------------------------------------------------------------------------
# Store-level deduplication + environment inference
# ---------------------------------------------------------------------------

_ENV_PATTERNS = {
    "production": re.compile(r"(^|[\s_\-\.])prod(uction)?([\s_\-\.]|$)", re.IGNORECASE),
    "staging": re.compile(r"(^|[\s_\-\.])stag(ing|e)?([\s_\-\.]|$)", re.IGNORECASE),
    "development": re.compile(
        r"(^|[\s_\-\.])(dev(elopment)?|local)([\s_\-\.]|$)", re.IGNORECASE
    ),
    "test": re.compile(r"(^|[\s_\-\.])test(ing)?([\s_\-\.]|$)", re.IGNORECASE),
}


def deduplicate_stores(stores: List["DiscoveredStore"]) -> List["DiscoveredStore"]:
    """Merge stores with matching fingerprints. Higher confidence wins for display fields."""
    seen: Dict[str, "DiscoveredStore"] = {}
    for store in stores:
        if store.id in seen:
            seen[store.id] = merge_store_sources(seen[store.id], store)
        else:
            seen[store.id] = store
    return list(seen.values())


def merge_store_sources(
    existing: "DiscoveredStore", new: "DiscoveredStore"
) -> "DiscoveredStore":
    """Merge two records for the same store from different discoverers."""
    winner = new if new.confidence > existing.confidence else existing
    loser = existing if winner is new else new

    seen_by = winner.metadata.get("seen_by", [winner.source])
    if loser.source not in seen_by:
        seen_by.append(loser.source)

    merged_tags = list(set(winner.tags + loser.tags))
    merged_metadata = {**loser.metadata, **winner.metadata}
    merged_metadata["seen_by"] = seen_by

    winner.tags = merged_tags
    winner.metadata = merged_metadata
    winner.last_seen = new.last_seen or existing.last_seen

    if not winner.environment and loser.environment:
        winner.environment = loser.environment

    return winner


def infer_environment(store: "DiscoveredStore") -> str:
    """Infer prod/staging/dev from name patterns, tags, cloud metadata."""
    search_text = f"{store.display_name} {' '.join(store.tags)}"

    for env, pattern in _ENV_PATTERNS.items():
        if pattern.search(search_text):
            return env

    cloud_env = store.metadata.get("environment") or store.metadata.get("env")
    if cloud_env:
        return cloud_env.lower()

    return "unknown"
