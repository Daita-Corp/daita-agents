"""PostgreSQL normalizer."""

from typing import Any, Dict

from ._common import _normalize_relational


def normalize_postgresql(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin PostgreSQL discovery output."""
    pk_set = {
        (pk["table_name"], pk["column_name"]) for pk in raw.get("primary_keys", [])
    }
    return _normalize_relational(
        raw,
        "postgresql",
        is_primary_key_fn=lambda tname, c: (tname, c["column_name"]) in pk_set,
        default_schema="public",
    )
