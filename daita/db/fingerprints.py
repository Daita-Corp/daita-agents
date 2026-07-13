"""Canonical fingerprints for DB runtime identifiers and sensitive values."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping


def stable_fingerprint(value: Any) -> str:
    """Return an unkeyed deterministic fingerprint for non-sensitive values."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def persisted_fingerprint(value: Any) -> str:
    """Preserve the established JSON encoding used by persisted DB hashes."""
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def db_operation_contract_binding_fingerprint(operation: Any) -> str:
    """Fingerprint stable DB request and safety facts across planner stages."""
    request = getattr(operation, "request", None)
    request = request if isinstance(request, Mapping) else {}
    metadata = getattr(operation, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    source_scope = request.get("source_scope") or metadata.get("source_scope") or ()
    requested_capabilities = (
        request.get("requested_capabilities")
        or metadata.get("requested_capabilities")
        or ()
    )
    safety_frame = metadata.get("safety_frame")
    safety_frame = safety_frame if isinstance(safety_frame, Mapping) else {}
    return persisted_fingerprint(
        {
            "operation_type": str(getattr(operation, "operation_type", "") or ""),
            "source_scope": (
                [source_scope] if isinstance(source_scope, str) else list(source_scope)
            ),
            "mode": request.get("mode") or metadata.get("mode"),
            "requested_capabilities": (
                [requested_capabilities]
                if isinstance(requested_capabilities, str)
                else list(requested_capabilities)
            ),
            "required_evidence": sorted(
                str(item)
                for item in (getattr(operation, "required_evidence", ()) or ())
            ),
            "safety_frame": dict(safety_frame),
        }
    )


def text_fingerprint(value: str) -> str:
    """Return the established UTF-8 SHA-256 fingerprint for DB text values."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sensitive_fingerprint(value: Any, key: bytes) -> str:
    """Return a keyed deterministic fingerprint without exposing a preview."""
    if not isinstance(key, bytes):
        raise TypeError("sensitive fingerprint key must be bytes")
    if len(key) < 32:
        raise ValueError("sensitive fingerprint key must be at least 32 bytes")
    return hmac.new(key, _canonical_json(value), hashlib.sha256).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
