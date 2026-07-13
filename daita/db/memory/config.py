"""DB-memory configuration extraction helpers."""

from __future__ import annotations

from typing import Any


def db_memory_options_from_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return normalized DB memory options from runtime metadata."""
    return db_memory_options_from_from_db_options(_from_db_options(metadata))


def db_memory_options_from_from_db_options(options: dict[str, Any]) -> dict[str, Any]:
    """Return normalized DB memory options from ``from_db_options``."""
    memory = options.get("memory")
    return memory if isinstance(memory, dict) else {}


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}
