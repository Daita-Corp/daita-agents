"""Start, stop and query the background host from a client process (§10.2)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from typing import Any

from .. import __version__
from .codec import decode
from .paths import HostPaths
from .protocol import PROTOCOL_HEADER, PROTOCOL_VERSION, VERSION_HEADER

READY_TIMEOUT_SECONDS = 10.0
STOP_TIMEOUT_SECONDS = 40.0
_POLL_SECONDS = 0.1


class HostStartError(RuntimeError):
    """The spawned host did not become ready; the message holds the log tail."""


def request(
    paths: HostPaths,
    method: str,
    path: str,
    *,
    body: object = None,
    timeout: float = 5.0,
) -> Any:
    """Send one request over the host socket and return the raw httpx response."""

    import httpx

    transport = httpx.HTTPTransport(uds=os.fspath(paths.socket))
    headers = {PROTOCOL_HEADER: str(PROTOCOL_VERSION), VERSION_HEADER: __version__}
    with httpx.Client(
        transport=transport,
        base_url="http://daita-host",
        headers=headers,
        timeout=timeout,
    ) as client:
        return client.request(method, path, json=body)


def host_status(paths: HostPaths) -> dict[str, object] | None:
    """Return the running host's status, or None when no host answers."""

    import httpx

    if not paths.socket.exists():
        return None
    try:
        response = request(paths, "GET", "/v1/host")
    except (httpx.TransportError, OSError):
        return None
    payload = response.json()
    if response.status_code == 409:
        attributes = payload["error"]["details"]["attributes"]
        return {
            "running": True,
            "version_mismatch": True,
            "version": decode(attributes["version"]),
            "protocol": decode(attributes["protocol"]),
        }
    status = dict(decode(payload))
    status["running"] = True
    return status


def start_host(
    paths: HostPaths, *, timeout: float = READY_TIMEOUT_SECONDS
) -> dict[str, object]:
    """Spawn ``daita host serve`` in the background unless a host already answers."""

    status = host_status(paths)
    if status is not None:
        return {**status, "started": False}
    command = [
        sys.executable,
        "-m",
        "daita",
        "--root",
        os.fspath(paths.root),
        "host",
        "serve",
        "--auto",
    ]
    os.makedirs(paths.log_dir, mode=0o700, exist_ok=True)
    with open(paths.log, "ab") as log:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = host_status(paths)
        if status is not None:
            return {**status, "started": True}
        time.sleep(_POLL_SECONDS)
    raise HostStartError(
        "The Daita host did not become ready.\n"
        + "".join(log_tail(paths, 20))
        + "\nRun it by hand to see why:\n  "
        + " ".join(command[:-1])
        + " --foreground"
    )


def stop_host(
    paths: HostPaths, *, timeout: float = STOP_TIMEOUT_SECONDS, reason: str = "user"
) -> bool:
    """Ask the host to drain and exit, falling back to SIGTERM; return if one ran."""

    import httpx

    try:
        request(paths, "POST", "/v1/host/shutdown", body={"reason": reason})
    except (httpx.TransportError, OSError):
        pid = _lock_holder_pid(paths)
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not paths.socket.exists() and _lock_holder_pid(paths) is None:
            return True
        time.sleep(_POLL_SECONDS)
    raise TimeoutError("the Daita host did not stop in time")


def log_tail(paths: HostPaths, lines: int = 200) -> list[str]:
    try:
        with open(paths.log, encoding="utf-8", errors="replace") as file:
            return list(deque(file, maxlen=lines))
    except FileNotFoundError:
        return []


def _lock_holder_pid(paths: HostPaths) -> int | None:
    """Return the pid recorded by a host that still holds the singleton lock."""

    import fcntl

    try:
        descriptor = os.open(paths.lock, os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            try:
                pid = json.loads(os.pread(descriptor, 4096, 0)).get("pid")
            except (ValueError, AttributeError):
                return None
            return pid if type(pid) is int else None
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return None
    finally:
        os.close(descriptor)


__all__ = [
    "HostStartError",
    "host_status",
    "log_tail",
    "request",
    "start_host",
    "stop_host",
]
