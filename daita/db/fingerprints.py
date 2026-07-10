"""Canonical fingerprints for DB runtime identifiers and sensitive values."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def stable_fingerprint(value: Any) -> str:
    """Return an unkeyed deterministic fingerprint for non-sensitive values."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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
