"""Resolve the host's socket, lock, status and log paths under one state root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..hosting.agent_home import _resolve_state_root

# macOS sun_path holds 104 bytes including the terminating NUL; Linux allows 108.
MAX_SOCKET_PATH_BYTES: Final = 103


@dataclass(frozen=True, slots=True)
class HostPaths:
    """Fixed locations of one host's files; nothing here touches the filesystem."""

    root: Path

    @property
    def run_dir(self) -> Path:
        return self.root / "run"

    @property
    def socket(self) -> Path:
        return self.run_dir / "host.sock"

    @property
    def lock(self) -> Path:
        return self.run_dir / "host.lock"

    @property
    def status(self) -> Path:
        return self.run_dir / "host.json"

    @property
    def runs(self) -> Path:
        """Foreground runs in flight; left behind only when the host died."""
        return self.run_dir / "runs.json"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log(self) -> Path:
        return self.log_dir / "host.log"


def host_paths(root: str | Path | None = None) -> HostPaths:
    """Resolve the state root like ``Agent.open`` and admit its socket path."""

    paths = HostPaths(_resolve_state_root(root))
    length = len(os.fsencode(paths.socket))
    if length > MAX_SOCKET_PATH_BYTES:
        raise ValueError(
            f"The Daita host socket path is too long ({length} bytes, limit "
            f"{MAX_SOCKET_PATH_BYTES}): {paths.socket}. Use a shorter --root."
        )
    return paths


__all__ = ["MAX_SOCKET_PATH_BYTES", "HostPaths", "host_paths"]
