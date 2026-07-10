"""Pure helpers shared by DB runtime task modules."""

from __future__ import annotations

import json
from typing import Any, Mapping

from daita.runtime import Evidence

from ...fingerprints import persisted_fingerprint


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError("DB runtime task mappings must be JSON serializable") from exc
    return copied


def _evidence_payload_fingerprint(evidence: Evidence) -> str:
    return str(
        evidence.metadata.get("payload_fingerprint")
        or persisted_fingerprint(evidence.payload)
    )


def _payload_contains(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True
