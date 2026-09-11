"""Private validation helpers for provider-native response fields."""

from __future__ import annotations

from collections.abc import Mapping

MISSING = object()


def field(value: object, name: str, default: object = MISSING) -> object:
    """Read one mapping or attribute field while preserving missing/null semantics."""
    if value is None:
        if default is not MISSING:
            return default
        raise KeyError(name)
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not MISSING:
        return default
    raise KeyError(name)


def safe_structural_token(value: object) -> str | None:
    if not isinstance(value, str) or not 1 <= len(value) <= 96:
        return None
    if (
        not value[0].isascii()
        or not value[0].isalnum()
        or any(
            not character.isascii() or not (character.isalnum() or character in "._:-")
            for character in value
        )
    ):
        return None
    return value


def required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return required_text(value, label)


def nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def usage_int(value: object, label: str) -> int:
    if value is None:
        return 0
    return nonnegative_int(value, label)
