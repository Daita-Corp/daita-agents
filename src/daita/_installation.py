"""Shared application installation repair guidance."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

PIPX_REPAIR_GUIDANCE = "Repair it with: pipx reinstall daita-agents"
MANAGED_REPAIR_GUIDANCE = (
    "Repair it with: curl -fsSL --proto '=https' --tlsv1.2 "
    "https://daita-tech.io/install.sh | "
    "bash -s -- --repair --no-onboard"
)

_MANAGED_ROOT_ENV = "DAITA_MANAGED_INSTALL_ROOT"
_OWNER_MARKER = "daita-managed-install-v1"


def repair_guidance() -> str:
    """Return repair guidance for the verified installation backend."""

    return (
        MANAGED_REPAIR_GUIDANCE
        if _is_trusted_managed_runtime()
        else PIPX_REPAIR_GUIDANCE
    )


def _is_trusted_managed_runtime(
    *,
    environ: Mapping[str, str] | None = None,
    executable: str | Path | None = None,
    home: str | Path | None = None,
) -> bool:
    """Validate launcher provenance without granting installer ownership."""

    environment = os.environ if environ is None else environ
    raw_root = environment.get(_MANAGED_ROOT_ENV)
    if raw_root is None or not raw_root or "\x00" in raw_root:
        return False
    try:
        expected_root = (
            (Path.home() if home is None else Path(home)).resolve(strict=True)
            / ".local"
            / "share"
            / "daita"
        )
        supplied_root = Path(raw_root)
        if not supplied_root.is_absolute() or supplied_root != expected_root:
            return False
        if supplied_root.is_symlink():
            return False
        root = supplied_root.resolve(strict=True)
        if root != expected_root:
            return False
        owner = root / "install-state" / "owner"
        if not owner.is_file() or owner.is_symlink():
            return False
        fields = _read_state_fields(owner)
        if fields != {
            "marker": _OWNER_MARKER,
            "root": str(root),
        }:
            return False
        current = root / "current"
        if not current.is_symlink():
            return False
        generation = current.resolve(strict=True)
        generations = (root / "generations").resolve(strict=True)
        if generation.parent != generations:
            return False
        manifest = generation / "manifest"
        if not manifest.is_file() or manifest.is_symlink():
            return False
        manifest_fields = _read_state_fields(manifest)
        if manifest_fields.get("marker") != _OWNER_MARKER:
            return False
        runtime = Path(sys.prefix if executable is None else executable).resolve(
            strict=True
        )
        runtime.relative_to(generation)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _read_state_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            return {}
        fields[key] = value
    return fields
