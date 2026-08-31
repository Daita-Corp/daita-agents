from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent
from daita.agent import HostActiveError
from daita.cli import build_parser
from daita.hosting.resident import ResidentReady, run_resident_host


async def test_resident_host_owns_normal_writer_lock_and_hands_off(
    tmp_path: Path,
) -> None:
    workspace = workspace_for(tmp_path)
    agent = await Agent.create("resident", root=tmp_path, workspace=workspace)
    await agent.close()

    ready: asyncio.Future[ResidentReady] = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    def on_ready(value: ResidentReady) -> None:
        ready.set_result(value)

    task = asyncio.create_task(
        run_resident_host(
            agent_name="resident",
            root=tmp_path,
            workspace=workspace,
            stop_event=stop,
            on_ready=on_ready,
        )
    )
    admitted = await ready
    assert admitted.agent_name == "resident"

    with pytest.raises(HostActiveError, match="host_active"):
        await Agent.open("resident", root=tmp_path, workspace=workspace)

    stop.set()
    await task
    reopened = await Agent.open("resident", root=tmp_path, workspace=workspace)
    await reopened.close()


def test_host_cli_surface_requires_explicit_agent_option() -> None:
    args = build_parser().parse_args(["host", "--agent", "resident"])
    assert args.command == "host"
    assert args.host_agent == "resident"


def test_resident_subprocess_runs_due_slot_and_restart_does_not_duplicate(
    tmp_path: Path,
) -> None:
    helper = Path(__file__).with_name("_routine_resident_subprocess.py")
    completed = subprocess.run(
        [sys.executable, str(helper), str(tmp_path)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "inbox_count": 1,
        "restart_model_calls": 0,
    }
