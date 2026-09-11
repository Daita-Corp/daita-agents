"""Deterministic explicit workspaces for tests that own local agent homes."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from daita import LocalWorkspace


def workspace_for(state_root: str | Path | None) -> LocalWorkspace:
    if state_root is None:
        base = Path(tempfile.gettempdir()) / "daita-test-workspaces"
        key = f"default-{os.getpid()}"
    else:
        state = Path(state_root).absolute()
        base = state.parent
        key = (
            state.name
            + "-"
            + hashlib.sha256(os.fspath(state).encode()).hexdigest()[:12]
        )
    root = base / f".{key}-workspace"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return LocalWorkspace(root)


__all__ = ["workspace_for"]
