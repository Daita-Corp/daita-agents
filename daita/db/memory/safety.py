"""Safety validation for DB-specific semantic memory."""

from __future__ import annotations

import decimal
import json
import re
from typing import Any

PII_COLUMN_PATTERNS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "email",
    "phone",
    "mobile",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "cvv",
    "pin",
    "dob",
    "date_of_birth",
    "birth_date",
    "address",
    "street",
    "zip",
    "postal",
    "passport",
    "national_id",
)

SENSITIVE_METADATA_KEYS = frozenset(
    (
        *PII_COLUMN_PATTERNS,
        "address",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "credit_card",
        "email",
        "phone",
        "private_key",
        "ssn",
    )
)

PII_VALUE_PATTERNS = (
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("US SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit card number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    (
        "phone number",
        re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
)

FORBIDDEN_OBSERVED_VALUE_KEYS = frozenset(
    {
        "canonical_value",
        "observed_value",
        "observed_values",
        "top_values",
        "value",
        "values",
    }
)


def validate_value_alias_metadata(metadata: dict[str, Any]) -> None:
    """Require catalog citation and reject recursively stored observed values."""
    catalog_ref = str(metadata.get("catalog_profile_ref") or "").strip()
    catalog_evidence_id = str(metadata.get("catalog_evidence_id") or "").strip()
    if not catalog_ref and not catalog_evidence_id:
        raise ValueError(
            "value_alias memory requires catalog_profile_ref or catalog_evidence_id"
        )

    forbidden = find_forbidden_observed_value_key(metadata)
    if forbidden:
        raise ValueError(
            f"value_alias memory cannot store observed value field {forbidden!r}; "
            "cite catalog evidence instead"
        )


def find_forbidden_observed_value_key(
    value: Any,
    prefix: str = "",
) -> str | None:
    """Return the first recursively nested observed-value metadata key."""
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in FORBIDDEN_OBSERVED_VALUE_KEYS:
                return path
            nested = find_forbidden_observed_value_key(item, path)
            if nested:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = find_forbidden_observed_value_key(item, f"{prefix}[{index}]")
            if nested:
                return nested
    return None


def db_memory_pii_error(
    *,
    key: str,
    text: str,
    metadata: dict[str, Any],
) -> str | None:
    """Return a user-facing validation error when DB memory stores row values."""
    violation = detect_pii_value(text) or detect_pii_value(key)
    if violation:
        return (
            f"DB memory cannot store row-level or PII values ({violation}); "
            "store durable database semantics only."
        )

    sensitive_key = find_sensitive_metadata_key(metadata)
    if sensitive_key:
        return (
            f"DB memory metadata cannot include sensitive field {sensitive_key!r}; "
            "store durable database semantics only."
        )

    metadata_value = detect_pii_value(
        json.dumps(_json_safe_for_detection(metadata), sort_keys=True)
    )
    if metadata_value:
        return (
            "DB memory metadata cannot store row-level or PII values "
            f"({metadata_value}); store durable database semantics only."
        )
    return None


def detect_pii_value(value: str) -> str | None:
    """Return the detected PII value type, if any."""
    text = str(value or "")
    if not text:
        return None
    for label, pattern in PII_VALUE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0)
            if label == "credit card number" and not looks_like_credit_card(candidate):
                continue
            return label
    return None


def looks_like_credit_card(candidate: str) -> bool:
    """Return whether a numeric candidate passes credit-card Luhn validation."""
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def find_sensitive_metadata_key(
    metadata: dict[str, Any],
    prefix: str = "",
) -> str | None:
    """Return the first recursively nested sensitive metadata key."""
    for key, value in metadata.items():
        key_text = str(key).lower()
        if metadata_key_is_sensitive(key_text):
            return f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            nested = find_sensitive_metadata_key(
                value,
                prefix=f"{prefix}{key}." if prefix else f"{key}.",
            )
            if nested:
                return nested
    return None


def metadata_key_is_sensitive(key_text: str) -> bool:
    """Return whether a metadata key names a sensitive value."""
    if key_text in SENSITIVE_METADATA_KEYS:
        return True
    tokens = {token for token in re.split(r"[^a-z0-9]+", key_text) if token}
    if tokens & SENSITIVE_METADATA_KEYS:
        return True
    for pattern in SENSITIVE_METADATA_KEYS:
        pattern_tokens = {token for token in re.split(r"[^a-z0-9]+", pattern) if token}
        if pattern_tokens and pattern_tokens <= tokens:
            return True
    return False


def _json_safe_for_detection(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_for_detection(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_for_detection(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    return str(value)
