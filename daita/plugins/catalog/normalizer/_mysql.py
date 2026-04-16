"""MySQL normalizer."""

from typing import Any, Dict

from ._common import _normalize_relational


def normalize_mysql(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin MySQL discovery output."""
    return _normalize_relational(
        raw,
        "mysql",
        is_primary_key_fn=lambda _tname, c: c.get("column_key", "") == "PRI",
    )
