"""DocumentDB normalizer.

DocumentDB is MongoDB wire-compatible, so we reuse the MongoDB normalizer
and only override the ``database_type`` tag.
"""

from typing import Any, Dict

from ._mongodb import normalize_mongodb


def normalize_documentdb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DocumentDB discover output.

    ``normalize_mongodb`` already forwards ``host`` / ``port`` from the raw
    discovery dict into ``metadata`` — we only need to overwrite the
    ``database_type`` tag so the persister's store-derivation match arm
    picks the ``documentdb`` case and emits IDs of the form
    ``documentdb:<host>/<database>``.
    """
    result = normalize_mongodb(raw)
    result["database_type"] = "documentdb"
    return result
