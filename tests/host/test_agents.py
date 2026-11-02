"""The registry: client folders, background agents at startup and writer locks."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from daita import Agent, JobState, LocalFileAccess
from daita.hosting.agent_home import read_writer_holder
from daita.hosting.errors import HostActiveError
from daita.storage.sqlite import SQLiteStateStore
from tests.host.conftest import StartHost
from tests.support.approval_learning import (
    MockModelProvider,
    _call,
    _memory_call,
    _stop,
)
from tests.support.host import (
    error_code,
    finished_run,
    pending_interaction,
    short_root,
    value,
)
from tests.support.jobs import stored_job
from tests.support.mcp import MemoryKeychain
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration


async def test_without_a_client_folder_no_local_files_are_reachable(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=MockModelProvider(()))
    await harness.http.post("/v1/agents", json={"name": "sales"})

    status = value(await harness.http.get("/v1/agents/sales"))
    assert status["workspace"].access is LocalFileAccess.NONE
    assert status["session_source_id"] is None
    files = await harness.submit(message="List my files.", files_only=True)
    failed = await finished_run(harness.http, "sales", files["run_ref"])
    assert failed["state"] == "failed"

    folder = harness.work()
    opened = value(
        await harness.http.post("/v1/agents/sales/open", json={"cwd": str(folder)})
    )
    assert opened["workspace"].root == folder.resolve()
    assert opened["workspace"].access is LocalFileAccess.COMPUTER
    assert opened["session_source_id"] is not None


async def test_an_idle_agent_reopens_in_the_folder_of_the_next_run(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=MockModelProvider((_stop("Done."),)))
    await harness.create_agent()
    other = harness.work("other")

    async with harness.events("sales") as events:
        run = await harness.submit(message="Hi", cwd=str(other))
        await events.next("agent_state", lambda data: data["state"] == "reopened")
    await finished_run(harness.http, "sales", run["run_ref"])

    status = value(await harness.http.get("/v1/agents/sales"))
    assert status["workspace"].root == other.resolve()


async def test_a_busy_agent_keeps_its_folder_and_names_it(
    start_host: StartHost,
) -> None:
    script = (_call(_memory_call(content="note")), _stop("Saved."))
    harness = await start_host(model=MockModelProvider(script))
    await harness.create_agent()
    await harness.submit(message="Remember this.", cwd=str(harness.work()))
    await pending_interaction(harness.http, "sales")

    busy = await harness.http.post(
        "/v1/agents/sales/runs",
        json={"message": "Hi", "cwd": str(harness.work("other"))},
    )

    assert (busy.status_code, error_code(busy)) == (409, "workspace_busy")
    error = busy.json()["error"]
    assert error["message"].startswith(f"Daita is busy in {harness.work().resolve()}")
    assert error["details"]["attributes"]["folder"]["v"] == str(
        harness.work().resolve()
    )


async def test_startup_opens_only_agents_with_background_work(
    start_host: StartHost,
) -> None:
    root = short_root()
    for name in ("idle", "busy"):
        agent = await Agent.create(
            name, root=root, workspace=workspace_for(root), keychain=MemoryKeychain()
        )
        agent_id = agent.id
        await agent.close()
    # An unfinished job, waiting for its user, so no run starts.
    store = await SQLiteStateStore.open(root / "agents" / "busy" / "state.db")
    try:
        await store.admit_job(
            replace(
                stored_job("job-1", agent_id),
                state=JobState.WAITING_FOR_YOU,
                question="Which region?",
            )
        )
    finally:
        await store.close()

    harness = await start_host(root=root)

    assert harness.host.registry.open_names() == ("busy",)


async def test_a_lock_held_by_a_script_is_reported_and_never_taken(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()
    await harness.http.post("/v1/agents/sales/close")
    script = await Agent.open(
        "sales",
        root=harness.root,
        workspace=workspace_for(harness.root),
        keychain=MemoryKeychain(),
    )
    try:
        locked = await harness.http.get("/v1/agents/sales/memory")
        assert (locked.status_code, error_code(locked)) == (423, "agent_locked")
        assert locked.json()["error"]["message"] == (
            f"In use by a Python script (pid {os.getpid()})."
        )
        holder = locked.json()["error"]["details"]["attributes"]["holder"]["v"]
        assert (holder["role"], holder["pid"]) == ("script", os.getpid())
        assert holder["started_at"].endswith("Z")
    finally:
        await script.close()

    assert value(await harness.http.get("/v1/agents/sales/memory")) == ""


async def test_an_agent_run_by_the_host_refuses_a_direct_open_clearly(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()

    with pytest.raises(HostActiveError) as raised:
        await Agent.open(
            "sales",
            root=harness.root,
            workspace=workspace_for(harness.root),
            keychain=MemoryKeychain(),
        )

    assert str(raised.value) == (
        f'Agent "sales" is run by the Daita host (pid {os.getpid()}).\n'
        'Use daita.connect("sales") from a script, or run `daita host stop`\n'
        "to use it directly in this process."
    )
    assert raised.value.holder is not None
    assert raised.value.holder["role"] == "host"


@pytest.mark.unit
def test_lock_contents_name_the_holder_and_old_files_still_parse() -> None:
    assert read_writer_holder(b"pid=4121\n") == {
        "role": None,
        "pid": 4121,
        "started_at": None,
    }
    assert read_writer_holder(
        b"pid=4121\nrole=host\nstarted_at=2026-10-20T09:00:00Z\n"
    ) == {"role": "host", "pid": 4121, "started_at": "2026-10-20T09:00:00Z"}
