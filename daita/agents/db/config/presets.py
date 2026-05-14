"""
Mode presets for ``Agent.from_db()``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

AUTO_TOOLKIT = "auto"


@dataclass(frozen=True)
class FromDbModePreset:
    """Default behavior bundle for a from_db mode."""

    toolkit: Optional[str]
    read_only: bool = True
    include_sample_values: bool = False
    query_default_limit: int = 50
    query_max_rows: int = 200
    query_max_chars: int = 50000
    query_timeout: Optional[float] = None
    lineage: bool = False
    memory: bool = False
    history: bool = False
    quality: bool = False


MODE_PRESETS: Dict[str, FromDbModePreset] = {
    "simple": FromDbModePreset(
        toolkit=None,
        query_max_rows=100,
        query_max_chars=25000,
    ),
    "analyst": FromDbModePreset(
        toolkit="analyst",
    ),
    "governed": FromDbModePreset(
        toolkit="analyst",
        query_default_limit=25,
        query_max_rows=100,
        query_max_chars=25000,
        query_timeout=30,
        lineage=True,
        history=True,
    ),
    "data_team": FromDbModePreset(
        toolkit="analyst",
        query_default_limit=50,
        query_max_rows=200,
        query_max_chars=50000,
        query_timeout=60,
        lineage=True,
        history=True,
        quality=True,
    ),
}


def resolve_mode_options(mode: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve mode defaults while preserving explicit caller overrides."""
    mode_name = (mode or "analyst").lower()
    if mode_name not in MODE_PRESETS:
        valid = ", ".join(sorted(MODE_PRESETS))
        raise ValueError(f"Unknown from_db mode {mode!r}. Valid modes: {valid}")

    preset = MODE_PRESETS[mode_name]
    resolved = dict(options)

    for name in (
        "read_only",
        "include_sample_values",
        "query_default_limit",
        "query_max_rows",
        "query_max_chars",
        "query_timeout",
        "lineage",
        "memory",
        "history",
        "quality",
    ):
        if resolved.get(name) is None:
            resolved[name] = getattr(preset, name)

    if resolved.get("toolkit", AUTO_TOOLKIT) == AUTO_TOOLKIT:
        resolved["toolkit"] = preset.toolkit

    resolved["mode"] = mode_name
    return resolved
