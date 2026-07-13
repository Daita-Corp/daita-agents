"""Structured storage boundary for DB-specific memory records."""

from __future__ import annotations

from inspect import getattr_static, isawaitable
from typing import Any, Callable

from .records import DB_MARKER_CATEGORY, DBMemoryRecord, normalize_db_memory_record
from .safety import db_memory_pii_error


STRUCTURED_DB_MEMORY_UNSUPPORTED = (
    "DB memory requires a structured backend implementing {operation!r}; "
    "generic memory fallbacks are unsupported"
)

_STRUCTURED_BACKEND_OPERATIONS = frozenset(
    {
        "list_db_records",
        "recall_db_records",
        "upsert_db_record",
    }
)


def require_structured_db_backend_method(
    plugin: Any,
    operation: str,
) -> Callable[..., Any]:
    """Return one declared structured backend method or fail explicitly."""
    if operation not in _STRUCTURED_BACKEND_OPERATIONS:
        raise ValueError(f"Unknown structured DB memory operation {operation!r}")
    backend = getattr(plugin, "backend", None)
    if backend is not None:
        try:
            getattr_static(backend, operation)
        except AttributeError:
            pass
        else:
            method = getattr(backend, operation, None)
            if callable(method):
                return method
    raise TypeError(STRUCTURED_DB_MEMORY_UNSUPPORTED.format(operation=operation))


async def write_db_memory_record(plugin: Any, raw: Any) -> dict[str, Any]:
    """Validate and write one DB record through structured backend upsert."""
    try:
        record = normalize_db_memory_record(raw)
        pii_error = db_memory_pii_error(
            key=record.key,
            text=record.text,
            metadata=record.metadata,
        )
        if pii_error:
            return {"success": False, "error": pii_error}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    upsert = require_structured_db_backend_method(plugin, "upsert_db_record")
    try:
        stored = upsert(record.to_dict())
        if isawaitable(stored):
            stored = await stored
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    status = stored.get("status", "created") if isinstance(stored, dict) else "created"
    return {
        "success": True,
        "content": record.to_memory_content(),
        "result": stored,
        "kind": record.kind,
        "key": record.key,
        "category": record.category,
        "status": status,
        "updated": 1 if status == "updated" else 0,
        "stored": stored,
    }


async def db_memory_record_ids_by_key(plugin: Any, raw: Any) -> list[str]:
    """Return structured record IDs for one exact source/key identity."""
    record = normalize_db_memory_record(raw)
    source_identity = _required_source_identity(record)
    results = await _list_exact_records(
        plugin,
        category=record.category,
        key=record.key,
        source_identity=source_identity,
        limit=1000,
    )
    record_ids: list[str] = []
    for result in results:
        record_id = result.get("record_id") or result.get("chunk_id")
        if record_id is not None and str(record_id) not in record_ids:
            record_ids.append(str(record_id))
    return record_ids


async def has_db_memory_marker(
    plugin: Any,
    key: str,
    *,
    source_identity: str,
) -> bool:
    """Return whether one exact source-scoped calibration marker exists."""
    normalized_source = str(source_identity).strip()
    if not normalized_source:
        raise ValueError("DB memory exact lookup requires source_identity")
    results = await _list_exact_records(
        plugin,
        category=DB_MARKER_CATEGORY,
        key=str(key),
        source_identity=normalized_source,
        limit=1,
    )
    return bool(results)


async def _list_exact_records(
    plugin: Any,
    *,
    category: str,
    key: str,
    source_identity: str,
    limit: int,
) -> list[dict[str, Any]]:
    list_records = require_structured_db_backend_method(plugin, "list_db_records")
    results = list_records(
        category=category,
        key=key,
        source_identity=source_identity,
        limit=limit,
    )
    if isawaitable(results):
        results = await results
    if not isinstance(results, list):
        raise TypeError("Structured DB memory list_db_records() must return a list")
    return [result for result in results if isinstance(result, dict)]


def _required_source_identity(record: DBMemoryRecord) -> str:
    source_identity = str(record.metadata.get("source_identity") or "").strip()
    if not source_identity:
        raise ValueError("DB memory exact lookup requires source_identity")
    return source_identity
