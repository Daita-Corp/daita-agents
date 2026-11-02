"""The CLI reaches agents through the Daita host; --no-host keeps library mode."""

from __future__ import annotations

import asyncio
import json
import typing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from daita.host.paths import host_paths
from tests.support.approval_learning import (
    MockModelProvider,
    _call,
    _memory_call,
    _stop,
)
from tests.support.cli import invoke, json_lines
from tests.support.host import HostHarness, StartHost, remove_root, short_root

pytestmark = pytest.mark.integration

SENTINEL = "sentinel-secret-7f3a"


async def _daita(
    harness: HostHarness, *argv: str, stdin: str = ""
) -> tuple[int, str, str]:
    # cli.main runs its own event loop, so it runs in a thread beside the host.
    arguments = ["--root", str(harness.root), "--workspace", str(harness.work())]
    return await asyncio.to_thread(invoke, [*arguments, *argv], stdin=stdin)


def _without_clock(stdout: str) -> object:
    record = json.loads(stdout)
    if isinstance(record, dict):
        record.pop("generated_at", None)
    return record


async def test_agent_commands_reach_the_host_and_keep_their_output(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    code, stdout, stderr = await _daita(harness, "create", "sales")
    assert (code, stderr) == (0, "")
    assert set(json.loads(stdout)) == {"agent_id", "name", "home"}
    assert harness.host.registry.open_names() == ("sales",)
    seeded: tuple[tuple[str, ...], ...] = (
        ("memory", "set", "sales", "--target", "memory", "--file", "-"),
        ("skills", "save", "sales", "monthly", "--description", "Monthly.")
        + ("--instructions-file", "-"),
    )
    for argv in seeded:
        result = await _daita(harness, *argv, stdin="Paid invoices only.")
        assert result[0] == 0, result

    reads: tuple[tuple[str, ...], ...] = (
        ("sources", "sales"),
        ("memory", "read", "sales"),
        ("skills", "list", "sales"),
        ("work", "show", "sales"),
        ("routines", "list", "sales"),
        ("jobs", "list", "sales"),
        ("export-location", "get", "sales"),
    )
    for argv in reads:
        code, via_host, stderr = await _daita(harness, *argv)
        assert (code, stderr) == (0, ""), argv
        assert "sales" in harness.host.registry.open_names()
        # The host lets go, so the same command can run in this process.
        await harness.host.registry.close("sales")
        code, in_process, _notice = await _daita(harness, "--no-host", *argv)
        assert code == 0, argv
        assert _without_clock(via_host) == _without_clock(in_process), argv

    code, stdout, stderr = await _daita(harness, "delete", "sales", "--yes")
    assert (code, stderr) == (0, "")
    assert json.loads(stdout) == {"name": "sales", "deleted": True}
    assert harness.host.registry.open_names() == ()


async def test_no_host_fails_clearly_while_the_host_holds_the_agent(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()

    code, stdout, stderr = await _daita(harness, "--no-host", "sources", "sales")

    assert (code, stdout) == (1, "")
    message = json.loads(stderr)["error"]
    assert 'Agent "sales" is run by the Daita host' in message
    assert "daita host stop" in message


async def test_run_through_the_host_writes_one_record_and_this_runs_events(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=MockModelProvider((_stop("bounded answer"),)))
    await harness.create_agent()

    code, stdout, stderr = await _daita(
        harness, "run", "sales", "bounded question", "--events-jsonl"
    )

    assert code == 0, stderr
    (record,) = json_lines(stdout)
    assert (record["status"], record["text"]) == ("completed", "bounded answer")
    assert {"run_id", "conversation_id", "reason", "steps", "tool_results"} <= set(
        record
    )
    events = json_lines(stderr)
    assert tuple(event["kind"] for event in events) == (
        "run.started",
        "model.completed",
        "run.completed",
    )
    assert {event["run_id"] for event in events} == {record["run_id"]}


async def test_run_through_the_host_denies_approvals_without_prompting(
    start_host: StartHost,
) -> None:
    model = MockModelProvider(
        (_call(_memory_call(content=SENTINEL)), _stop("approval remains required"))
    )
    harness = await start_host(model=model)
    await harness.create_agent()

    code, stdout, stderr = await _daita(harness, "run", "sales", "remember this")

    assert (code, stderr) == (0, "")
    assert "Approve this exact change once?" not in stdout
    (record,) = json_lines(stdout)
    assert record["text"] == "approval remains required"
    results = typing.cast(list[dict[str, object]], record["tool_results"])
    (write,) = (item for item in results if item["call_id"] == "write")
    assert write["is_error"] is True
    model.assert_consumed()
    memory = await _daita(harness, "memory", "read", "sales")
    assert json.loads(memory[1])["content"] == ""
    assert harness.host.hub.pending("sales") == ()


@pytest.mark.parametrize(
    ("options", "refused"),
    [
        (("run", "sales", "q", "--model", "mock:scripted"), "--model"),
        (("run", "sales", "q", "--max-cost-usd", "1"), "--max-cost-usd"),
        (("run", "sales", "q", "--diagnostics-path", "d.jsonl"), "--diagnostics-path"),
        (("--workspace-sensitivity", "confidential", "sources", "sales"), "sensitiv"),
    ],
)
def test_options_that_compose_this_process_need_no_host(
    options: tuple[str, ...], refused: str
) -> None:
    root = short_root()
    try:
        code, stdout, stderr = invoke(["--root", str(root), *options])
        assert (code, stdout) == (1, "")
        message = json.loads(stderr)["error"]
        assert refused in message and "--no-host" in message
        # Refused before reaching for a host.
        assert not host_paths(root).socket.exists()
    finally:
        remove_root(root)


async def test_routine_create_confirms_at_the_cli_through_the_host(
    start_host: StartHost,
) -> None:
    model = MockModelProvider((_stop("Seven invoices are paid."),))
    harness = await start_host(model=model)
    await harness.create_agent()
    code, stdout, _ = await _daita(harness, "run", "sales", "How many are paid?")
    assert code == 0
    spec = _routine_spec(harness.work() / "routine.json")

    code, stdout, stderr = await _daita(
        harness, "routines", "create", "sales", "--spec", str(spec), stdin="n\n"
    )
    assert code == 1 and "Approval required" in stdout, stderr
    code, stdout, _ = await _daita(harness, "routines", "list", "sales")
    assert json.loads(stdout) == []

    code, stdout, stderr = await _daita(
        harness, "routines", "create", "sales", "--spec", str(spec), stdin="y\n"
    )
    assert code == 0, stderr
    review, _, result = stdout.rpartition("Approve this exact change once? [y/n]")
    assert review.startswith("Approval required\n")
    assert json.loads(result)["title"] == "Paid invoice count"
    code, stdout, _ = await _daita(harness, "routines", "list", "sales")
    assert [item["title"] for item in json.loads(stdout)] == ["Paid invoice count"]
    assert harness.host.hub.pending("sales") == ()


def _routine_spec(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "title": "Paid invoice count",
                "prompt": "Report the paid invoice count.",
                "schedule": {
                    "kind": "interval",
                    "interval_seconds": 3600,
                    "anchor_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("argv", [["--no-host"], ["--no-host", "chat", "sales"]])
def test_the_terminal_app_refuses_no_host(argv: list[str]) -> None:
    code, stdout, stderr = invoke(argv, tty=True)

    assert (code, stdout) == (1, "")
    assert json.loads(stderr) == {
        "error": "the terminal app always uses the Daita host"
    }
