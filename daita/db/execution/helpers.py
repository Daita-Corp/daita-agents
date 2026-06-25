"""Small generic helpers for DB execution."""

from __future__ import annotations

from typing import Any

from ..models import DbRequest


def _store_id_for_request(request: DbRequest, runtime: Any | None = None) -> str:
    value = (
        request.metadata.get("store_id")
        or request.constraints.get("store_id")
        or (request.source_scope[0] if request.source_scope else None)
        or _runtime_from_db_option(runtime, "catalog_store_id")
    )
    return str(value or "runtime_source")


def _short_table_name(table: str) -> str:
    return table.split(".")[-1].lower() if table else ""


def _runtime_from_db_option(runtime: Any | None, key: str) -> Any | None:
    if runtime is None:
        return None
    options = getattr(getattr(runtime, "config", None), "metadata", {}).get(
        "from_db_options"
    )
    if isinstance(options, dict):
        return options.get(key)
    return None


def _stable_hash(value: Any) -> str:
    import hashlib
    import json

    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lineage_entity_for_request(request: DbRequest, table: str | None) -> str:
    value = (
        request.metadata.get("entity_id")
        or request.constraints.get("entity_id")
        or request.metadata.get("table")
        or request.constraints.get("table")
    )
    if value:
        entity = str(value)
        return entity if ":" in entity else f"table:{entity}"
    return f"table:{table}" if table else "table:unknown"
