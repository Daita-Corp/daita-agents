"""Serve the host API over one owner-only Unix socket and run its lifecycle.

Startup (§10.3): admit the paths, take the singleton lock, open agents with
background work, bind the socket, serve, write host.json. Shutdown (§10.4):
refuse new runs, drain runs, end event streams, stop the server, close agents,
remove the socket and host.json, release the lock. There is no TCP listener.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import json
import logging
import os
import signal
import socket
import stat
import struct
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn.protocols.http.h11_impl import H11Protocol

from .. import __version__
from .events import EventBus
from .interactions import DEFAULT_EXPIRY, InteractionHub
from .paths import HostPaths
from .protocol import (
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    VERSION_HEADER,
    HostVersionMismatch,
)
from .registry import AgentRegistry
from .routes import (
    agents,
    artifacts,
    catalog,
    conversations,
    deliveries,
    effects,
    error_response,
    events,
    host as host_routes,
    interactions,
    jobs,
    mcp,
    memory,
    model,
    routines,
    runs,
    sources,
    work,
)
from .runs import RunManager

_LOGGER = logging.getLogger("daita.host")
# One module per area (§7.7); each lists its routes and the members they reach.
AREAS = (
    host_routes,
    agents,
    runs,
    events,
    interactions,
    conversations,
    sources,
    catalog,
    artifacts,
    deliveries,
    work,
    routines,
    jobs,
    effects,
    memory,
    mcp,
    model,
)
_SIGNALS = (signal.SIGTERM, signal.SIGINT)
PeerUid = Callable[[Any], int]


class HostAlreadyRunning(RuntimeError):
    """Another host holds the singleton lock for this state root."""

    def __init__(self, message: str, *, holder: Mapping[str, object]) -> None:
        super().__init__(message)
        self.holder = dict(holder)


def peer_uid(sock: Any) -> int:
    """Return the uid of the process at the other end of a Unix socket."""

    if sys.platform == "darwin":
        # struct xucred begins with u_int cr_version, uid_t cr_uid; SOL_LOCAL is 0.
        data = sock.getsockopt(0, getattr(socket, "LOCAL_PEERCRED", 1), 76)
        return int(struct.unpack_from("=II", data)[1])
    data = sock.getsockopt(
        socket.SOL_SOCKET, getattr(socket, "SO_PEERCRED"), struct.calcsize("3i")
    )
    return int(struct.unpack("3i", data)[1])


class Host:
    """The components one host process owns."""

    def __init__(
        self,
        paths: HostPaths,
        *,
        agent_options: Mapping[str, object] | None = None,
        interaction_expiry: timedelta = DEFAULT_EXPIRY,
        heartbeat_seconds: int = 15,
        drain_seconds: float = 30.0,
    ) -> None:
        self.paths = paths
        self.heartbeat_seconds = heartbeat_seconds
        self.drain_seconds = drain_seconds
        self.bus = EventBus()
        self.hub = InteractionHub(self.bus, expiry=interaction_expiry)
        self.runs = RunManager(self.bus, paths.runs)
        self.registry = AgentRegistry(
            paths.root, self.bus, self.hub, self.runs, agent_options=agent_options
        )
        self.started_at = datetime.now(UTC)
        self._started = time.monotonic()
        self._server: _Server | None = None
        self._shutdown: asyncio.Task[None] | None = None

    def status(self) -> dict[str, object]:
        agents = self.registry.open_names()
        return {
            "version": __version__,
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "uptime_seconds": round(time.monotonic() - self._started, 3),
            "socket": self.paths.socket,
            "log": self.paths.log,
            "agents": agents,
            "clients": sum(self.bus.attached(name) for name in agents),
        }

    def request_shutdown(self, reason: str) -> asyncio.Task[None]:
        if self._shutdown is None:
            _LOGGER.info("shutdown requested: %s", reason)
            self._shutdown = asyncio.create_task(self._drain())
        return self._shutdown

    async def _drain(self) -> None:
        await self.runs.drain(self.drain_seconds)
        self.hub.close()
        self.bus.close()
        if self._server is not None:
            self._server.should_exit = True


async def serve_host(
    paths: HostPaths,
    *,
    on_ready: Callable[[Host], None] | None = None,
    peer_uid: PeerUid = peer_uid,
    handle_signals: bool = False,
    **options: Any,
) -> None:
    """Run one host until shutdown; raise HostAlreadyRunning if one is live."""

    _admit_private_directory(paths.run_dir)
    _admit_private_directory(paths.log_dir)
    lock = _acquire_singleton(paths)
    sock: socket.socket | None = None
    host: Host | None = None
    loop = asyncio.get_running_loop()
    try:
        # After the lock: the run manager reads the previous host's run marker.
        host = Host(paths, **options)
        if handle_signals:
            for signum in _SIGNALS:
                loop.add_signal_handler(signum, host.request_shutdown, "signal")
        await host.registry.open_background_agents()
        sock = _bind(paths)
        server = _Server(_config(create_app(host), peer_uid))
        host._server = server

        def started() -> None:
            assert host is not None
            _write_private(paths.status, json.dumps(_status_record(paths, host)))
            _LOGGER.info("ready on %s", paths.socket)
            if on_ready is not None:
                on_ready(host)

        server.on_started = started
        await server.serve(sockets=[sock])
    finally:
        if host is not None:
            with contextlib.suppress(Exception):
                await host.request_shutdown("stopped")
            await host.registry.close_all()
        if sock is not None:
            sock.close()
            with contextlib.suppress(FileNotFoundError):
                paths.socket.unlink()
        with contextlib.suppress(FileNotFoundError):
            paths.status.unlink()
        os.close(lock)
        if handle_signals:
            for signum in _SIGNALS:
                loop.remove_signal_handler(signum)
        _LOGGER.info("stopped")


def create_app(host: Host) -> ASGIApp:
    app = Starlette(
        routes=[route for area in AREAS for route in area.ROUTES],
        exception_handlers={HTTPException: _http_error},  # type: ignore[dict-item]
    )
    app.state.host = host
    return _Handshake(app)


class _Handshake:
    """Refuse requests from another protocol or package version (§10.5)."""

    def __init__(self, app: Starlette) -> None:
        self.app = app
        self.state = app.state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # The client asks a mismatched host to drain, so shutdown is exempt.
        if scope["type"] == "http" and scope["path"] != "/v1/host/shutdown":
            headers = Headers(scope=scope)
            if (
                headers.get(PROTOCOL_HEADER) != str(PROTOCOL_VERSION)
                or headers.get(VERSION_HEADER) != __version__
            ):
                error = HostVersionMismatch(
                    f"the Daita host runs version {__version__} "
                    f"(protocol {PROTOCOL_VERSION})",
                    protocol=PROTOCOL_VERSION,
                    version=__version__,
                )
                request = Request(scope, receive)
                await error_response(request, error)(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def _http_error(request: Request, error: HTTPException) -> Response:
    missing = KeyError(f"no host route: {request.method} {request.url.path}")
    response = error_response(
        request, missing if error.status_code == 404 else ValueError(error.detail)
    )
    response.status_code = error.status_code
    return response


def _config(app: ASGIApp, check_peer: PeerUid) -> uvicorn.Config:
    class PeerCheckedProtocol(H11Protocol):
        def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
            super().connection_made(transport)
            try:
                allowed = check_peer(transport.get_extra_info("socket")) == os.getuid()
            except OSError:
                allowed = False
            if not allowed:
                _LOGGER.warning("refused a connection from another user")
                transport.close()

    return uvicorn.Config(
        app,
        http=PeerCheckedProtocol,
        ws="none",
        lifespan="off",
        log_config=None,
        timeout_graceful_shutdown=5,
    )


class _Server(uvicorn.Server):
    on_started: Callable[[], None] | None = None

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        # The host installs its own handlers; never re-raise signals here.
        yield

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started and self.on_started is not None:
            self.on_started()


def _admit_private_directory(path: os.PathLike[str]) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)
    facts = os.lstat(path)
    if not stat.S_ISDIR(facts.st_mode) or facts.st_uid != os.getuid():
        raise PermissionError(f"host directory must be a directory you own: {path}")
    if facts.st_mode & 0o022:
        raise PermissionError(
            f"host directory is writable by other users; refusing to run: {path}"
        )
    os.chmod(path, 0o700)


def _acquire_singleton(paths: HostPaths) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(paths.lock, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        content = os.pread(descriptor, 4096, 0)
        os.close(descriptor)
        if error.errno not in {errno.EACCES, errno.EAGAIN}:
            raise
        try:
            holder = json.loads(content)
        except ValueError:
            holder = {}
        raise HostAlreadyRunning(
            f"a Daita host is already running (pid {holder.get('pid', 'unknown')})",
            holder=holder if type(holder) is dict else {},
        ) from None
    os.fchmod(descriptor, 0o600)
    record = {
        "pid": os.getpid(),
        "version": __version__,
        "protocol": PROTOCOL_VERSION,
        "socket": str(paths.socket),
        "started_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    os.ftruncate(descriptor, 0)
    os.pwrite(descriptor, json.dumps(record).encode("utf-8"), 0)
    return descriptor


def _bind(paths: HostPaths) -> socket.socket:
    path = os.fspath(paths.socket)
    if os.path.lexists(path):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(path)
        except OSError:
            os.unlink(path)  # stale: the host that made it is gone
        else:
            raise HostAlreadyRunning(f"another Daita host answers on {path}", holder={})
        finally:
            probe.close()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    previous = os.umask(0o177)
    try:
        sock.bind(path)
    finally:
        os.umask(previous)
    os.chmod(path, 0o600)
    sock.listen(128)
    sock.setblocking(False)
    return sock


def _status_record(paths: HostPaths, host: Host) -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "version": __version__,
        "protocol": PROTOCOL_VERSION,
        "socket": str(paths.socket),
        "started_at": host.started_at.replace(microsecond=0).isoformat(),
        "agents": list(host.registry.open_names()),
    }


def _write_private(path: os.PathLike[str], text: str) -> None:
    temporary = f"{os.fspath(path)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(text)
    os.replace(temporary, path)


__all__ = ["Host", "HostAlreadyRunning", "create_app", "peer_uid", "serve_host"]
