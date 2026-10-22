from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from daita import Agent
from daita.agent import HostActiveError
from daita.cli import build_parser
from daita.hosting.resident import ResidentReady, run_resident_host
from tests.support.paths import REPO_ROOT
from tests.support.workspace import workspace_for


async def test_deprecated_resident_host_warns_and_owns_the_writer_lock(
    tmp_path: Path,
) -> None:
    workspace = workspace_for(tmp_path)
    agent = await Agent.create("resident", root=tmp_path, workspace=workspace)
    await agent.close()

    ready: asyncio.Future[ResidentReady] = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    def on_ready(value: ResidentReady) -> None:
        ready.set_result(value)

    with pytest.warns(DeprecationWarning, match="run_resident_host is deprecated"):
        hosting = run_resident_host(
            agent_name="resident",
            root=tmp_path,
            workspace=workspace,
            stop_event=stop,
            on_ready=on_ready,
        )
        task = asyncio.create_task(hosting)
        admitted = await ready
    assert admitted.agent_name == "resident"

    with pytest.raises(HostActiveError) as raised:
        await Agent.open("resident", root=tmp_path, workspace=workspace)
    assert raised.value.holder is not None
    assert (raised.value.holder["role"], raised.value.holder["pid"]) == (
        "script",
        os.getpid(),
    )

    stop.set()
    await task
    reopened = await Agent.open("resident", root=tmp_path, workspace=workspace)
    await reopened.close()


def test_host_cli_surface_still_parses_the_deprecated_agent_option() -> None:
    args = build_parser().parse_args(["host", "--agent", "resident"])
    assert args.command == "host"
    assert args.host_agent == "resident"


def test_resident_subprocess_runs_due_slot_and_restart_does_not_duplicate(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tests.hosting.resident_subprocess", str(tmp_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "delivery_count": 1,
        "restart_model_calls": 0,
    }
