"""Read and change the host's user settings, top-level booleans in host.toml."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Final

from .paths import HostPaths

# Every setting and its default; a missing key means the default.
DEFAULTS: Final = {
    # Desktop notifications when no client watches an agent (§11).
    "notifications": True,
    # Offer the background service after the first routine (§12.2).
    "background_offer": True,
}
_TABLE: Final = re.compile(r"^[ \t]*\[", re.MULTILINE)


def read_setting(paths: HostPaths, key: str) -> bool:
    """Return one setting; a malformed host.toml raises ValueError."""

    value = _parse(paths.settings, _read(paths.settings)).get(key, DEFAULTS[key])
    if type(value) is not bool:
        raise ValueError(f"{paths.settings}: {key} must be true or false")
    return value


def write_setting(paths: HostPaths, key: str, value: bool) -> None:
    """Set one top-level boolean and keep every other line of the file."""

    if key not in DEFAULTS or type(value) is not bool:
        raise ValueError(f"unknown host setting or value: {key}")
    path = paths.settings
    text = _read(path)
    before = _parse(path, text)
    line = f"{key} = {'true' if value else 'false'}"
    # Top-level keys precede the first table header.
    table = _TABLE.search(text)
    top, rest = (
        (text, "") if table is None else (text[: table.start()], text[table.start() :])
    )
    existing = re.compile(rf"^[ \t]*{key}[ \t]*=.*$", re.MULTILINE)
    if existing.search(top):
        top = existing.sub(line, top, count=1)
    else:
        top = (top if not top or top.endswith("\n") else top + "\n") + line + "\n"
    updated = top + rest
    # shortcut: the line edit can't follow multi-line values; refuse rather
    # than guess, and parse the file properly if a setting ever needs that.
    after = _parse(path, updated)
    if after != {**before, key: value}:
        raise ValueError(f"{path} could not be updated safely; set {line} by hand")
    os.makedirs(paths.root, mode=0o700, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(updated)
    os.replace(temporary, path)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _parse(path: Path, text: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path} is not valid TOML: {error}") from None


__all__ = ["DEFAULTS", "read_setting", "write_setting"]
