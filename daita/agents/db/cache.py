"""
Schema caching and drift detection.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def cache_key(source: Union[str, Any]) -> str:
    """Compute a stable cache key for *source*, stripping credentials."""
    if isinstance(source, str):
        try:
            parsed = urlparse(source)
            # Rebuild without password so credential rotation doesn't change the key
            clean = parsed._replace(
                netloc=f"{parsed.hostname or ''}{(':' + str(parsed.port)) if parsed.port else ''}"
            )
            key_str = clean.geturl()
        except Exception:
            key_str = source
    else:
        cls_name = type(source).__name__
        attrs = ":".join(
            str(getattr(source, a, ""))
            for a in ("host", "port", "database_name", "path")
        )
        key_str = f"{cls_name}:{attrs}"

    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def load_cached_schema(
    cache_key_str: str, ttl: int
) -> Optional[Tuple[Dict[str, Any], bool]]:
    """Load cached schema from ``.daita/schema_cache/{cache_key_str}.json``.

    Returns ``(schema, is_expired)`` or ``None`` if no cache file exists.
    """
    cache_path = Path(".daita") / "schema_cache" / f"{cache_key_str}.json"
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        now = datetime.now(timezone.utc)
        is_expired = (now - cached_at).total_seconds() > ttl
        return data["schema"], is_expired
    except Exception as exc:
        logger.debug(f"Failed to load schema cache {cache_path}: {exc}")
        return None


def save_schema_cache(cache_key_str: str, schema: Dict[str, Any]) -> None:
    """Persist *schema* to ``.daita/schema_cache/{cache_key_str}.json``."""
    cache_dir = Path(".daita") / "schema_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key_str}.json"
    payload = {
        "schema": schema,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key_str,
    }
    cache_path.write_text(json.dumps(payload, indent=2))


def detect_drift(
    old_schema: Dict[str, Any], new_schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Compare two normalized schemas and return a drift summary, or ``None``."""
    old_tables = {t["name"]: t for t in old_schema.get("tables", [])}
    new_tables = {t["name"]: t for t in new_schema.get("tables", [])}

    added_tables = sorted(set(new_tables) - set(old_tables))
    removed_tables = sorted(set(old_tables) - set(new_tables))

    column_changes: List[Dict[str, Any]] = []
    for tname in set(old_tables) & set(new_tables):
        old_cols = {c["name"] for c in old_tables[tname].get("columns", [])}
        new_cols = {c["name"] for c in new_tables[tname].get("columns", [])}
        added = sorted(new_cols - old_cols)
        removed = sorted(old_cols - new_cols)
        if added or removed:
            column_changes.append(
                {"table": tname, "added_columns": added, "removed_columns": removed}
            )

    if not added_tables and not removed_tables and not column_changes:
        return None

    return {
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "column_changes": column_changes,
    }
