"""Install the host as a per-user login service: launchd on macOS, systemd on Linux.

The service runs the same ``daita host serve --auto`` that clients auto-spawn,
with this interpreter, so it picks up an upgraded package. ``--auto`` makes a
service start that finds a host already running exit quietly (code 0) instead
of entering a restart loop.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

from .paths import HostPaths
from .supervise import READY_TIMEOUT_SECONDS, host_status, stop_host

LAUNCHD_LABEL: Final = "ai.daita.host"
SYSTEMD_UNIT: Final = "daita-host.service"
LINGER_NOTE: Final = (
    "To keep the Daita host running after you log out, run: "
    "loginctl enable-linger $USER"
)
Runner = Callable[[list[str], bool], None]

_SYSTEMD_TEMPLATE: Final = """\
[Unit]
Description=Daita agent host

[Service]
ExecStart={exec_start}
Environment={search_path}
Restart=on-failure
RestartSec=3
StandardOutput=append:{log}
StandardError=append:{log}

[Install]
WantedBy=default.target
"""


def service_path(platform: str = sys.platform, home: Path | None = None) -> Path:
    """Where this platform's service definition lives for the current user."""

    home = Path.home() if home is None else home
    if platform == "darwin":
        return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    if platform.startswith("linux"):
        return home / ".config" / "systemd" / "user" / SYSTEMD_UNIT
    raise ValueError("the Daita host service is supported on macOS and Linux only")


def installed(platform: str = sys.platform, home: Path | None = None) -> bool:
    try:
        return service_path(platform, home).exists()
    except ValueError:
        return False


def serve_command(paths: HostPaths, interpreter: str) -> list[str]:
    root = os.fspath(paths.root)
    return [interpreter, "-m", "daita", "--root", root, "host", "serve", "--auto"]


def launchd_plist(paths: HostPaths, *, interpreter: str, search_path: str) -> bytes:
    """The LaunchAgent; PATH lets subscription CLIs such as claude resolve."""

    return plistlib.dumps(
        {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": serve_command(paths, interpreter),
            "EnvironmentVariables": {"PATH": search_path},
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            # Background throttles CPU about 2.7x, which clients would feel.
            "ProcessType": "Standard",
            "StandardOutPath": os.fspath(paths.log),
            "StandardErrorPath": os.fspath(paths.log),
        }
    )


def systemd_unit(paths: HostPaths, *, interpreter: str, search_path: str) -> str:
    command = serve_command(paths, interpreter)
    return _SYSTEMD_TEMPLATE.format(
        exec_start=" ".join(_systemd_quote(item, command=True) for item in command),
        search_path=_systemd_quote(f"PATH={search_path}"),
        # append: takes the bare path; only specifiers need escaping.
        log=os.fspath(paths.log).replace("%", "%%"),
    )


def install(
    paths: HostPaths,
    *,
    start: bool = True,
    run: Runner | None = None,
    platform: str = sys.platform,
    home: Path | None = None,
) -> list[str]:
    """Write the service definition and, with ``start``, hand the host to it.

    A host already running is stopped first, so the service owns the host.
    Returns notes for the user.
    """

    runner = _run if run is None else run
    path = service_path(platform, home)
    interpreter = sys.executable
    search_path = os.environ.get("PATH", os.defpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if platform == "darwin":
        content = launchd_plist(paths, interpreter=interpreter, search_path=search_path)
    else:
        content = systemd_unit(
            paths, interpreter=interpreter, search_path=search_path
        ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)
    notes = [f"Daita host service installed: {path}"]
    if platform == "darwin":
        if start:
            _stop_running(paths)
            domain = f"gui/{os.getuid()}"
            runner(["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"], False)
            runner(["launchctl", "bootstrap", domain, os.fspath(path)], True)
        else:
            notes.append("It starts at your next login.")
    else:
        runner(["systemctl", "--user", "daemon-reload"], True)
        if start:
            _stop_running(paths)
            runner(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT], True)
        else:
            runner(["systemctl", "--user", "enable", SYSTEMD_UNIT], True)
            notes.append("It starts at your next login.")
        notes.append(LINGER_NOTE)
    if start:
        notes.append(
            "The Daita host is running."
            if _ready(paths)
            else f"The service did not answer yet; see {paths.log}."
        )
    return notes


def uninstall(
    paths: HostPaths,
    *,
    run: Runner | None = None,
    platform: str = sys.platform,
    home: Path | None = None,
) -> bool:
    """Stop and remove the service; return whether one was installed."""

    runner = _run if run is None else run
    path = service_path(platform, home)
    if platform == "darwin":
        runner(["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"], False)
    else:
        runner(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT], False)
    existed = path.exists()
    path.unlink(missing_ok=True)
    if platform != "darwin":
        runner(["systemctl", "--user", "daemon-reload"], True)
    return existed


def _stop_running(paths: HostPaths) -> None:
    if host_status(paths) is not None:
        stop_host(paths, reason="user")


def _ready(paths: HostPaths) -> bool:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if host_status(paths) is not None:
            return True
        time.sleep(0.1)
    return False


def _run(command: list[str], check: bool) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")


def _systemd_quote(value: str, *, command: bool = False) -> str:
    """Quote one systemd value; % is a specifier and $ expands in ExecStart."""

    if any(character in value for character in "\r\n\x00"):
        raise ValueError("a systemd unit value must be a single line")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        escaped = escaped.replace("$", "$$")
    return f'"{escaped}"'


__all__ = [
    "LAUNCHD_LABEL",
    "LINGER_NOTE",
    "SYSTEMD_UNIT",
    "install",
    "installed",
    "launchd_plist",
    "serve_command",
    "service_path",
    "systemd_unit",
    "uninstall",
]
