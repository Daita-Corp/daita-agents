"""Catalog profile freshness and drift helpers for ``from_db``.

Catalog persistence remains owned by ``CatalogPlugin`` and its backing store.
This module only applies the DB-agent construction policy for reusing a
persisted catalog profile, detecting staleness, and comparing a refreshed
profile with the previously persisted one.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
_CATALOG_PROFILE_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}


def _memory_key(profile_key: str) -> str:
    return f"{Path.cwd().resolve()}:{profile_key}"


def catalog_profile_key(source: Union[str, Any]) -> str:
    """Compute a stable profile key for *source*, stripping credentials."""
    if isinstance(source, str):
        try:
            parsed = urlparse(source)
            clean = parsed._replace(
                netloc=f"{parsed.hostname or ''}{(':' + str(parsed.port)) if parsed.port else ''}"
            )
            key_text = clean.geturl()
        except Exception:
            key_text = source
    else:
        cls_name = type(source).__name__
        attrs = ":".join(
            str(getattr(source, attr, ""))
            for attr in ("host", "port", "database_name", "path")
        )
        key_text = f"{cls_name}:{attrs}"

    return hashlib.sha256(key_text.encode()).hexdigest()[:16]


def load_catalog_profile_snapshot(
    profile_key: str,
    *,
    catalog_keys: Optional[List[str]] = None,
    ttl: Optional[int] = None,
) -> Optional[Tuple[Dict[str, Any], bool]]:
    """Load a persisted catalog profile and report whether it is stale."""
    profile = (
        _load_profile_from_catalog(catalog_keys)
        if ttl is not None
        else _load_catalog_profile(profile_key, catalog_keys=catalog_keys)
    )
    if profile is None:
        return None
    _CATALOG_PROFILE_MEMORY_CACHE[_memory_key(profile_key)] = profile
    return profile, _is_expired(profile, ttl)


def detect_profile_drift(
    old_profile: Dict[str, Any], new_profile: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Compare two normalized catalog profiles and return a drift summary."""
    old_tables = {t["name"]: t for t in old_profile.get("tables", [])}
    new_tables = {t["name"]: t for t in new_profile.get("tables", [])}

    added_tables = sorted(set(new_tables) - set(old_tables))
    removed_tables = sorted(set(old_tables) - set(new_tables))

    column_changes: List[Dict[str, Any]] = []
    for table_name in set(old_tables) & set(new_tables):
        old_cols = {c["name"] for c in old_tables[table_name].get("columns", [])}
        new_cols = {c["name"] for c in new_tables[table_name].get("columns", [])}
        added = sorted(new_cols - old_cols)
        removed = sorted(old_cols - new_cols)
        if added or removed:
            column_changes.append(
                {
                    "table": table_name,
                    "added_columns": added,
                    "removed_columns": removed,
                }
            )

    if not added_tables and not removed_tables and not column_changes:
        return None

    return {
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "column_changes": column_changes,
    }


def _load_catalog_profile(
    profile_key: str, *, catalog_keys: Optional[List[str]]
) -> Optional[Dict[str, Any]]:
    memory_key = _memory_key(profile_key)
    if memory_key in _CATALOG_PROFILE_MEMORY_CACHE:
        return _CATALOG_PROFILE_MEMORY_CACHE[memory_key]

    profile = _load_profile_from_catalog(catalog_keys)
    if profile is not None:
        _CATALOG_PROFILE_MEMORY_CACHE[memory_key] = profile
        return profile
    return None


def _is_expired(profile: Dict[str, Any], ttl: Optional[int]) -> bool:
    if ttl is None:
        return False
    if ttl <= 0:
        return True

    timestamp = (
        profile.get("last_seen")
        or profile.get("profiled_at")
        or profile.get("first_seen")
    )
    if not timestamp:
        return True
    try:
        seen_at = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return True
    if seen_at.tzinfo is None:
        seen_at = seen_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - seen_at).total_seconds() > ttl


def _load_profile_from_catalog(
    catalog_keys: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    catalog_path = Path(".daita") / "catalog.json"
    if not catalog_path.exists():
        return None

    try:
        catalog = json.loads(catalog_path.read_text())
    except Exception as exc:
        logger.debug(f"Failed to load catalog profile {catalog_path}: {exc}")
        return None

    if not isinstance(catalog, dict):
        return None

    for key in catalog_keys or []:
        profile = catalog.get(key)
        if isinstance(profile, dict) and isinstance(profile.get("tables"), list):
            return profile

    if len(catalog) == 1:
        profile = next(iter(catalog.values()))
        if isinstance(profile, dict) and isinstance(profile.get("tables"), list):
            return profile

    return None
