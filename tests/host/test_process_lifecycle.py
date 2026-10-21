"""A real background host process: start, kill -9, recover, stop."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import daita
from daita import Agent, LocalFileAccess, LocalWorkspace
from daita.host.codec import decode
from daita.host.paths import HostPaths, host_paths
from daita.host.supervise import host_status, request, stop_host
from tests.support.host import remove_root, short_root, work_folder
from tests.support.mcp import MemoryKeychain

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
# A host with a scripted model whose first turn waits for an approval.
_SCRIPTED_HOST = """
import asyncio, sys
from pathlib import Path
from daita.host.paths import host_paths
from daita.host.server import serve_host
from tests.support.approval_learning import (
    MockModelProvider, _call, _memory_call, _profile, _stop,
)
from tests.support.mcp import MemoryKeychain

model = MockModelProvider((_call(_memory_call(content="note")), _stop("Saved.")))
options = {"model": model, "model_profile": _profile(model), "keychain": MemoryKeychain()}
asyncio.run(serve_host(host_paths(Path(sys.argv[1])), agent_options=options))
"""


@pytest.fixture
def paths() -> Iterator[HostPaths]:
    root = short_root()
    paths = host_paths(root)
    try:
        yield paths
    finally:
        try:
            stop_host(paths)
        finally:
            assert not paths.socket.exists()
            remove_root(root)


def _daita(paths: HostPaths, *arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "daita", "--root", str(paths.root), *arguments],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _wait(probe: Callable[[], Any], timeout: float = 20.0) -> Any:
    deadline = time.monotonic() + timeout
    while not (result := probe()):
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        time.sleep(0.05)
    return result


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _pid(paths: HostPaths) -> int:
    status = host_status(paths)
    assert status is not None
    return int(status["pid"])  # type: ignore[call-overload]


def test_start_spawns_a_ready_host_and_a_second_start_is_a_no_op(
    paths: HostPaths,
) -> None:
    assert "stopped" in _daita(paths, "host", "status")

    assert _daita(paths, "host", "start").strip() == "Daita host started."
    first = _pid(paths)
    assert _daita(paths, "host", "start").strip() == "Daita host already running."

    assert _pid(paths) == first
    assert f"pid {first}" in _daita(paths, "host", "status")
    assert "ready on" in _daita(paths, "host", "logs")
    assert _daita(paths, "host", "stop").strip() == "Daita host stopped."
    assert _gone(first) or _wait(lambda: _gone(first))
    assert not paths.socket.exists() and not paths.status.exists()


def test_a_killed_host_leaves_nothing_that_blocks_the_next_start(
    paths: HostPaths,
) -> None:
    _daita(paths, "host", "start")
    killed = _pid(paths)
    os.kill(killed, signal.SIGKILL)
    _wait(lambda: _gone(killed))
    assert paths.socket.exists()  # stale: nobody removed it

    assert _daita(paths, "host", "start").strip() == "Daita host started."

    assert _pid(paths) != killed


def test_runs_of_a_killed_host_are_interrupted_never_active(paths: HostPaths) -> None:
    host = subprocess.Popen(
        [sys.executable, "-c", _SCRIPTED_HOST, str(paths.root)],
        cwd=REPOSITORY,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait(lambda: host_status(paths))
        request(paths, "POST", "/v1/agents", body={"name": "sales"})
        submitted = request(
            paths, "POST", "/v1/agents/sales/runs", body={"message": "Remember this."}
        )
        run_ref = decode(submitted.json())["run_ref"]
        _wait(
            lambda: decode(
                request(paths, "GET", f"/v1/agents/sales/runs/{run_ref}").json()
            )["state"]
            == "waiting_for_input"
        )
        host.send_signal(signal.SIGKILL)
        host.wait(timeout=20)
    finally:
        if host.poll() is None:
            host.kill()
            host.wait(timeout=20)

    _daita(paths, "host", "start")

    active = request(paths, "GET", "/v1/agents/sales/runs?active=true")
    run = request(paths, "GET", f"/v1/agents/sales/runs/{run_ref}")
    assert decode(active.json()) == ()
    assert decode(run.json())["state"] == "interrupted"


async def test_connect_starts_a_host_when_none_runs(paths: HostPaths) -> None:
    folder = work_folder(paths.root)
    created = await Agent.create(
        "sales",
        root=paths.root,
        workspace=LocalWorkspace(folder, access=LocalFileAccess.COMPUTER),
        keychain=MemoryKeychain(),
    )
    await created.close()
    assert host_status(paths) is None

    agent = await daita.connect("sales", cwd=folder, root=paths.root)
    try:
        status = await asyncio.to_thread(host_status, paths)
        assert agent.workspace.root == folder
        assert agent.name == "sales"
    finally:
        await agent.close()

    assert status is not None
    assert status["pid"] != os.getpid()
    assert status["agents"] == ("sales",)
    # Closing the client leaves the agent open in the host.
    assert (await asyncio.to_thread(host_status, paths) or {})["agents"] == ("sales",)
