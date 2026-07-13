"""DB-specific memory record contracts and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
import decimal
import json
import re
from typing import Any

from .contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    normalize_db_memory_semantic_contract,
)
from .safety import validate_value_alias_metadata

DB_SEMANTIC_CATEGORY = "db_semantics"
DB_MARKER_CATEGORY = "db_cache_marker"

DB_MEMORY_KINDS = frozenset(
    {
        "unit_convention",
        "metric_definition",
        "business_rule",
        "data_contract_note",
        "schema_interpretation",
        "value_alias",
        "cache_marker",
    }
)
DB_SEMANTIC_MEMORY_KINDS = (
    "unit_convention",
    "metric_definition",
    "business_rule",
    "data_contract_note",
    "schema_interpretation",
    "value_alias",
)
DB_PLANNING_MEMORY_KINDS = frozenset(DB_SEMANTIC_MEMORY_KINDS)


@dataclass(frozen=True)
class DBMemoryRecord:
    """Structured DB memory record stored through a MemoryPlugin backend."""

    kind: str
    key: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.7

    @property
    def category(self) -> str:
        if self.kind == "cache_marker":
            return DB_MARKER_CATEGORY
        return DB_SEMANTIC_CATEGORY

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "text": self.text,
            "metadata": json_safe(self.metadata),
            "importance": self.importance,
            "category": self.category,
        }

    def to_memory_content(self) -> str:
        return f"DB memory record:\n{json.dumps(self.to_dict(), sort_keys=True)}"


def db_memory_record_from_payload(
    payload: dict[str, Any],
    prompt: str,
    *,
    task_metadata: dict[str, Any] | None = None,
) -> DBMemoryRecord:
    """Build a DB memory record from runtime request metadata and constraints."""
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    metadata = _direct_write_contract_metadata(
        dict(metadata),
        task_metadata=task_metadata or {},
    )

    kind = str(payload.get("kind") or "business_rule").strip()
    text = str(payload.get("text") or payload.get("content") or prompt).strip()
    key = str(payload.get("key") or _default_key(kind, text)).strip()
    importance = float(payload.get("importance", 0.7))
    return normalize_db_memory_record(
        {
            "kind": kind,
            "key": key,
            "text": text,
            "metadata": metadata,
            "importance": importance,
        }
    )


def normalize_db_memory_record(raw: Any) -> DBMemoryRecord:
    """Validate and normalize a DB memory record."""
    if isinstance(raw, DBMemoryRecord):
        record = raw
    elif isinstance(raw, dict):
        record = DBMemoryRecord(
            kind=str(raw.get("kind") or "").strip(),
            key=str(raw.get("key") or "").strip(),
            text=str(raw.get("text") or raw.get("content") or "").strip(),
            metadata=dict(raw.get("metadata") or {}),
            importance=float(raw.get("importance", 0.7)),
        )
    else:
        raise TypeError("DB memory records must be dictionaries or DBMemoryRecord")

    if record.kind not in DB_MEMORY_KINDS:
        raise ValueError(
            f"Unsupported DB memory kind {record.kind!r}; expected one of "
            f"{sorted(DB_MEMORY_KINDS)}"
        )
    if not record.key:
        raise ValueError("DB memory record requires a key")
    if not record.text:
        raise ValueError("DB memory record requires text")
    if record.kind == "value_alias":
        validate_value_alias_metadata(record.metadata)
    metadata = json_safe(record.metadata)
    if DB_MEMORY_SEMANTIC_CONTRACT_KEY in metadata:
        try:
            metadata[DB_MEMORY_SEMANTIC_CONTRACT_KEY] = (
                normalize_db_memory_semantic_contract(
                    metadata.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
                )
            )
        except Exception as exc:
            metadata.pop(DB_MEMORY_SEMANTIC_CONTRACT_KEY, None)
            metadata["semantic_contract_diagnostics"] = {
                "valid": False,
                "reason": str(exc),
            }
    return DBMemoryRecord(
        kind=record.kind,
        key=record.key,
        text=record.text,
        metadata=metadata,
        importance=max(0.0, min(1.0, record.importance)),
    )


def json_default(value: Any) -> Any:
    """Return the deterministic JSON fallback used by DB-memory records."""
    if isinstance(value, decimal.Decimal):
        return float(value)
    return str(value)


def json_safe(value: Any) -> Any:
    """Recursively normalize a value for deterministic JSON serialization."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json_default(value)


def _direct_write_contract_metadata(
    metadata: dict[str, Any],
    *,
    task_metadata: dict[str, Any],
) -> dict[str, Any]:
    if DB_MEMORY_SEMANTIC_CONTRACT_KEY not in metadata:
        return metadata
    if metadata.get("semantic_contract_status") == "validated" and task_metadata.get(
        "reason"
    ) in {"db_memory_commit_update", "db_memory_learning_promotion"}:
        return metadata
    metadata.pop(DB_MEMORY_SEMANTIC_CONTRACT_KEY, None)
    metadata.pop("semantic_contract_status", None)
    metadata["semantic_contract_diagnostics"] = {
        "created": False,
        "reason": "direct_write_unvalidated",
    }
    return metadata


def _default_key(kind: str, text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:8]
    slug = "_".join(words) or "memory"
    return f"{kind}:{slug}"
