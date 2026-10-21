"""Owner-only socket and files, peer uid checks, and secrets kept out of logs."""

from __future__ import annotations

import logging
import os
import socket
import stat

import httpx
import pytest

from daita.host.paths import host_paths
from daita.host.server import peer_uid, serve_host
from daita.security import SecretReference
from tests.host.conftest import StartHost
from tests.support.host import error_code, remove_root, short_root, value
from tests.support.mcp import MemoryKeychain

pytestmark = pytest.mark.integration

SENTINEL = "sentinel-Pa55w0rd-7f3e"


class RefusingKeychain(MemoryKeychain):
    """A keychain backend whose error echoes the value it was given."""

    async def set(self, reference: SecretReference, value: str) -> None:
        raise ValueError(f"keychain refused the value {value!r}")


async def test_socket_and_host_files_are_owner_only(start_host: StartHost) -> None:
    harness = await start_host()
    paths = harness.paths

    assert stat.S_IMODE(os.lstat(paths.run_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(paths.log_dir).st_mode) == 0o700
    assert stat.S_ISSOCK(os.lstat(paths.socket).st_mode)
    for path in (paths.socket, paths.lock, paths.status, paths.runs):
        assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600, path
    assert harness.host._server is not None
    listeners = [
        sock for server in harness.host._server.servers for sock in server.sockets
    ]
    assert listeners and all(sock.family == socket.AF_UNIX for sock in listeners)


async def test_a_run_directory_writable_by_others_is_refused() -> None:
    root = short_root()
    paths = host_paths(root)
    try:
        paths.run_dir.mkdir(mode=0o700)
        os.chmod(paths.run_dir, 0o770)
        with pytest.raises(PermissionError, match="writable by other users"):
            await serve_host(paths)
        assert not paths.socket.exists()
    finally:
        remove_root(root)


async def test_connections_from_another_uid_are_refused(start_host: StartHost) -> None:
    harness = await start_host(peer_uid=lambda _sock: os.getuid() + 1)

    with pytest.raises(httpx.TransportError):
        await harness.http.get("/v1/host")


@pytest.mark.unit
def test_the_peer_credential_check_reads_this_process_uid() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert peer_uid(left) == os.getuid()
    finally:
        left.close()
        right.close()


async def test_secrets_never_reach_the_log_and_are_redacted_from_errors(
    start_host: StartHost,
) -> None:
    accepted = await start_host()
    refused = await start_host(agent_options={"keychain": RefusingKeychain()})
    handler = logging.FileHandler(accepted.paths.log)
    logger = logging.getLogger()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        for harness in (accepted, refused):
            await harness.create_agent()
        stored = await accepted.http.post(
            "/v1/agents/sales/sources/postgresql/passwords",
            json={"password": SENTINEL},
        )
        failed = await refused.http.post(
            "/v1/agents/sales/sources/postgresql/passwords",
            json={"password": SENTINEL},
        )
        configured = await accepted.http.post(
            "/v1/agents/sales/model",
            json={"provider": "openai", "model": "gpt-unknown", "api_key": SENTINEL},
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
        handler.close()

    assert isinstance(value(stored), SecretReference)
    assert error_code(failed) == "invalid_value"
    assert "[redacted]" in failed.json()["error"]["message"]
    assert configured.status_code >= 400
    for response in (failed, configured):
        assert SENTINEL not in response.text
    log = accepted.paths.log.read_text(encoding="utf-8")
    assert "/sources/postgresql/passwords" in log
    assert SENTINEL not in log
