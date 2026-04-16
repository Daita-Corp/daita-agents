"""DocumentDB normalizer.

DocumentDB is MongoDB wire-compatible, so we reuse the MongoDB normalizer
and only override the ``database_type`` tag.
"""

from typing import Any, Dict

from ._mongodb import normalize_mongodb


def normalize_documentdb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DocumentDB discover output."""
    result = normalize_mongodb(raw)
    result["database_type"] = "documentdb"
    return result
