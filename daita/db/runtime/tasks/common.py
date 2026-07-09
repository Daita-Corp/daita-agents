"""Pure helpers shared by DB runtime task modules."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from daita.runtime import Evidence


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError("DB runtime task mappings must be JSON serializable") from exc
    return copied


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return _stable_hash(payload)


def _evidence_payload_fingerprint(evidence: Evidence) -> str:
    return str(
        evidence.metadata.get("payload_fingerprint")
        or _payload_fingerprint(evidence.payload)
    )


def _payload_contains(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True
