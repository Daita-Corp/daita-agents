"""RemoteAgent reaches the host with the Agent's own calls, callbacks and errors."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import os
import shutil
import sqlite3
import sys
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import daita
from daita import (
    Agent,
    AgentEvent,
    AgentEventKind,
    IntervalSchedule,
    LocalFileAccess,
    LocalWorkspace,
    RoutineIntent,
)
from daita._json import FrozenJsonObject
from daita.capabilities import ApprovalDecision, ApprovalRequest
from daita.host import (
    HostUnavailableError,
    HostVersionMismatch,
    client as host_client,
    server as host_server,
)
from daita.host.client import RemoteAgent
from daita.host.codec import supports
from daita.host.protocol import (
    CLIENT_ONLY_MEMBERS,
    CLIENT_SIDE_ANNOTATIONS,
    REMOTE_AGENT_SURFACE,
    ConversationBusyError,
    EventType,
    WorkspaceBusyError,
)
from daita.host.server import AREAS
from daita.host.supervise import host_status, stop_host
from daita.hosting.errors import AgentNotFoundError
from daita.hosting.sources import SourceEditPreview
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.llm.subscription_auth import CodexDevicePrompt, CodexOAuthCredential
from daita.loop.models import LoopExitKind
from tests.host.conftest import StartHost
from tests.support.approval_learning import (
    MockModelProvider,
    _call,
    _memory_call,
    _stop,
)
from tests.support.host import (
    bounded,
    fake_openai_device_login,
    finished_run,
    pending_interaction,
    remote_agent,
    remove_root,
    short_root,
    value,
    work_folder,
)
from tests.support.mcp import MemoryKeychain
from tests.support.toolbox_model import ToolboxAwareMockModelProvider

NAME = "parity"
NOW = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
USAGE = ModelUsage(
    input_tokens=100,
    output_tokens=10,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)


# The §8 guard: RemoteAgent is the Agent surface, typed for the wire.


def _public(cls: type) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


def _function(member: object) -> Callable[..., Any]:
    if isinstance(member, property):
        assert member.fget is not None
        return member.fget
    return typing.cast(Callable[..., Any], getattr(member, "__func__", member))


def _shape(member: object, client_side: frozenset[str]) -> list[object]:
    function = _function(member)
    hints = typing.get_type_hints(function)
    shape: list[object] = [
        type(member),
        inspect.iscoroutinefunction(function),
        *(
            (item.name, item.kind, item.default, hints.get(item.name))
            for item in inspect.signature(function).parameters.values()
            if item.name not in client_side
        ),
    ]
    if "return" not in client_side:
        shape.append(hints["return"])
    return shape


@pytest.mark.unit
def test_remote_agent_has_exactly_the_surface_members() -> None:
    assert _public(RemoteAgent) == REMOTE_AGENT_SURFACE | CLIENT_ONLY_MEMBERS
    assert not REMOTE_AGENT_SURFACE & CLIENT_ONLY_MEMBERS


@pytest.mark.unit
@pytest.mark.parametrize("member", sorted(REMOTE_AGENT_SURFACE))
def test_each_member_keeps_the_agent_signature(member: str) -> None:
    client_side = CLIENT_SIDE_ANNOTATIONS.get(member, frozenset())

    assert _shape(inspect.getattr_static(RemoteAgent, member), client_side) == (
        _shape(inspect.getattr_static(Agent, member), client_side)
    )


@pytest.mark.unit
@pytest.mark.parametrize("member", sorted(REMOTE_AGENT_SURFACE))
def test_every_remote_parameter_and_result_has_a_codec(member: str) -> None:
    client_side = CLIENT_SIDE_ANNOTATIONS.get(member, frozenset())
    hints = typing.get_type_hints(
        _function(inspect.getattr_static(RemoteAgent, member))
    )

    missing = {
        name: annotation
        for name, annotation in hints.items()
        if name not in client_side and not supports(annotation)
    }
    assert not missing


@pytest.mark.unit
def test_the_client_reaches_the_hosts_own_route_tables() -> None:
    tables = {
        route.member: route for area in AREAS for route in getattr(area, "TABLE", ())
    }

    assert host_client._routes() == tables


# Parity: the same operation, in process and through the host, gives one result.


@dataclass(frozen=True)
class Seeded:
    root: Path
    work: Path
    source_id: str
    resource_id: str
    conversation_id: str
    run_id: str
    artifact_id: str


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        usage=USAGE,
    )


def _answer(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=USAGE)


def _options(*script: ModelResponse) -> dict[str, object]:
    """Deterministic composition, so two homes in one state answer alike."""

    model = ToolboxAwareMockModelProvider(
        script, provider_id="mock:parity", complete_pricing=True
    )
    counter = itertools.count(1)
    return {
        "model": model,
        "model_profile": ModelProfile(
            id=model.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        "clock": lambda: NOW,
        "id_factory": lambda prefix: f"{prefix}-{next(counter):032x}",
        "keychain": MemoryKeychain(),
    }


def _database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, amount REAL)"
        )
        connection.execute("INSERT INTO orders VALUES (1, 'EMEA', 110.5)")
    return path


async def _seed(root: Path) -> Seeded:
    work = work_folder(root)
    database = _database(work / "orders.sqlite")
    shutil.copy(database, work / "orders-copy.sqlite")
    options = _options()
    model = typing.cast(ToolboxAwareMockModelProvider, options["model"])
    agent = await Agent.create(
        NAME,
        root=root,
        workspace=LocalWorkspace(work, access=LocalFileAccess.COMPUTER),
        **options,  # type: ignore[arg-type]
    )
    try:
        source = await agent.attach_sqlite(database, name="Orders")
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        model.replace_script(
            (
                _tool("describe", "catalog_describe", {"source_id": source.id}),
                _tool(
                    "note",
                    "artifact_create",
                    {
                        "from": "document",
                        "format": "txt",
                        "filename": "orders.txt",
                        "content": "EMEA has the largest order.\n",
                    },
                ),
                _answer("EMEA has the largest order."),
            )
        )
        origin = await agent.run("Which region has the largest order?")
        (artifact,) = await agent.list_artifacts()
        assert origin.conversation_id is not None
        return Seeded(
            root,
            work,
            source.id,
            resource.id,
            origin.conversation_id,
            origin.run_id,
            artifact.artifact_id,
        )
    finally:
        await agent.close()


async def _approve(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.APPROVE


async def _accept(_preview: SourceEditPreview) -> bool:
    return True


async def _agents(agent: Any, seeded: Seeded) -> object:
    kind = type(agent)
    return (
        agent.name,
        agent.id,
        agent.home,
        agent.workspace,
        agent.session_source_id,
        agent.model_profile,
        agent.model_route,
        await kind.list(root=seeded.root),
        await kind.inspect_home(NAME, root=seeded.root),
    )


async def _runs(agent: Any, seeded: Seeded) -> object:
    return (
        await agent.run("Which region leads?", conversation_id=seeded.conversation_id),
        await agent.learn("Remember that EMEA leads.", conversation_id=None),
    )


async def _conversations(agent: Any, seeded: Seeded) -> object:
    return (
        await agent.conversation_exists(seeded.conversation_id),
        await agent.conversation_runs(seeded.conversation_id),
        await agent.transcript(seeded.run_id),
        await agent.tool_evidence(seeded.run_id),
        await agent.analysis_usage(seeded.run_id),
    )


async def _sources(agent: Any, seeded: Seeded) -> object:
    edited = await agent.edit_sqlite_source(
        seeded.source_id,
        seeded.work / "orders-copy.sqlite",
        confirmation_handler=_accept,
        name="Orders copy",
    )
    return (
        edited,
        await agent.list_sources(),
        await agent.resolve_source("Orders copy"),
        await agent.inspect_source_permissions(edited.source.id),
    )


async def _catalog(agent: Any, seeded: Seeded) -> object:
    return (
        await agent.catalog_summary(),
        await agent.list_catalog_resources(source_id=seeded.source_id),
        await agent.inspect_catalog_resource(seeded.resource_id),
    )


async def _artifacts(agent: Any, seeded: Seeded) -> object:
    return (
        await agent.list_artifacts(),
        await agent.read_artifact(seeded.artifact_id),
        await agent.export_destination(),
    )


async def _deliveries(agent: Any, seeded: Seeded) -> object:
    del seeded
    return await agent.inspect_delivery("delivery-missing")


async def _work(agent: Any, _seeded: Seeded) -> object:
    return await agent.work_board()


async def _routines(agent: Any, seeded: Seeded) -> object:
    routine = await agent.create_routine(
        RoutineIntent(
            title="Order count report",
            prompt="Report the number of orders.",
            schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
            sources=("Orders",),
        ),
        conversation_id=seeded.conversation_id,
        confirmation_handler=_approve,
    )
    # The JSON schedule shape crosses the host; omitted fields keep their values.
    updated = await agent.update_routine(
        routine.routine_id,
        RoutineIntent(
            schedule=FrozenJsonObject.from_mapping(
                {"kind": "once", "exact_at": (NOW + timedelta(hours=2)).isoformat()}
            )
        ),
        confirmation_handler=_approve,
    )
    return (
        routine,
        updated,
        await agent.list_routines(),
        await agent.inspect_routine(routine.routine_id),
    )


async def _jobs(agent: Any, _seeded: Seeded) -> object:
    return (await agent.list_jobs(), await agent.inspect_job("job-missing"))


async def _effects(agent: Any, _seeded: Seeded) -> object:
    return (
        await agent.list_effects(),
        await agent.list_effects(unresolved_only=True, limit=5),
    )


async def _memory(agent: Any, _seeded: Seeded) -> object:
    await agent.set_memory("Revenue is paid subtotal.")
    return (
        await agent.read_memory(),
        await agent.save_skill("order-report", "Count orders.", "Count the rows."),
        await agent.list_skills(),
        await agent.read_skill("order-report"),
        await agent.list_semantic_annotations(),
    )


async def _mcp(agent: Any, _seeded: Seeded) -> object:
    return await agent.list_mcp_servers()


async def _model(agent: Any, _seeded: Seeded) -> object:
    return (
        agent.model_requires_explicit_limits(provider="openai", model="gpt-5"),
        agent.model_requires_explicit_limits(provider="mock", model="unreviewed"),
    )


async def _errors(agent: Any, seeded: Seeded) -> object:
    failures = []
    for call in (
        lambda: agent.read_artifact("artifact-missing"),
        lambda: agent.resolve_source("no such source"),
        lambda: agent.inspect_source_permissions("source-missing"),
        lambda: agent.run(""),
    ):
        try:
            await call()
        except Exception as error:
            failures.append((type(error), str(error)))
    return failures


SCENARIOS: dict[str, Callable[[Any, Seeded], Awaitable[object]]] = {
    "agents": _agents,
    "runs": _runs,
    "conversations": _conversations,
    "sources": _sources,
    "catalog": _catalog,
    "artifacts": _artifacts,
    "deliveries": _deliveries,
    "work": _work,
    "routines": _routines,
    "jobs": _jobs,
    "effects": _effects,
    "memory": _memory,
    "mcp": _mcp,
    "model": _model,
    "errors": _errors,
}
_SCRIPTS: dict[str, tuple[ModelResponse, ...]] = {
    "runs": (_answer("EMEA still leads."), _answer("Noted."))
}


async def _in_both(start_host: StartHost, area: str) -> tuple[object, object, Seeded]:
    """Seed one home, then run one scenario in process and through the host."""

    root = short_root()
    hosted = False
    try:
        seeded = await _seed(root)
        snapshot = root.with_name(f"{root.name}-snapshot")
        shutil.copytree(root, snapshot)
        script = _SCRIPTS.get(area, ())
        local = await Agent.open(
            NAME,
            root=root,
            workspace=LocalWorkspace(seeded.work, access=LocalFileAccess.COMPUTER),
            **_options(*script),  # type: ignore[arg-type]
        )
        try:
            expected = await SCENARIOS[area](local, seeded)
        finally:
            await local.close()
        shutil.rmtree(root)
        shutil.copytree(snapshot, root)
        hosted = True  # the host fixture removes the root after stopping
        await start_host(root=root, agent_options=_options(*script))
        remote = await daita.connect(NAME, cwd=seeded.work, root=root)
        try:
            actual = await SCENARIOS[area](remote, seeded)
        finally:
            await remote.close()
        return expected, actual, seeded
    finally:
        if not hosted:
            remove_root(root)


@pytest.mark.integration
@pytest.mark.parametrize("area", SCENARIOS)
async def test_remote_agent_gives_the_in_process_result(
    start_host: StartHost, area: str
) -> None:
    expected, actual, _seeded = await _in_both(start_host, area)

    assert actual == expected


# Runs, callbacks, the event stream and the host lifecycle, through RemoteAgent.

NOTE = "Revenue means paid invoice subtotal."
_ACCEPTANCE_SCRIPT = r"""
import asyncio, json, sys
from pathlib import Path

import daita
from daita.capabilities import ApprovalDecision


async def main(root: Path, cwd: Path) -> None:
    asked = []
    second_asked = asyncio.Event()

    async def approval_handler(request):
        asked.append(request.tool_name)
        if len(asked) == 1:
            return ApprovalDecision.APPROVE
        second_asked.set()
        await asyncio.Event().wait()

    agent = await daita.connect(
        "sales", cwd=cwd, root=root, approval_handler=approval_handler
    )
    try:
        first = await agent.run("Remember that revenue is paid subtotal.")
        second = asyncio.create_task(
            agent.run("Remember it again.", conversation_id=first.conversation_id)
        )
        await asyncio.wait_for(second_asked.wait(), 30)
        second.cancel()
        try:
            await second
        except asyncio.CancelledError:
            cancelled = True
        else:
            cancelled = False
        print(json.dumps({
            "answer": first.final_text,
            "asked": asked,
            "cancelled": cancelled,
            "memory": await agent.read_memory(),
        }))
    finally:
        await agent.close()


asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
"""


def _remembering(*answers: str) -> MockModelProvider:
    """A model that asks to write memory (needing approval), then answers."""

    script = []
    for index, answer in enumerate(answers):
        content = f"{NOTE} (note {index})"
        script += [
            _call(_memory_call(f"write-{index}", content=content)),
            _stop(answer),
        ]
    return MockModelProvider(tuple(script))


@pytest.mark.acceptance
async def test_a_script_connects_runs_approves_and_cancels(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved.", "Saved again."))
    await harness.create_agent()

    async with harness.events("sales") as events:
        script = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _ACCEPTANCE_SCRIPT,
            str(harness.root),
            str(harness.work()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(script.communicate(), 60)
        assert script.returncode == 0, stderr.decode()
        _seq, _event, cancelled = await events.next(
            "run_state", lambda data: data["state"] == "cancelled"
        )

    result = json_loads(stdout)
    assert result == {
        "answer": "Saved.",
        "asked": ["memory_set", "memory_set"],
        "cancelled": True,
        "memory": result["memory"],
    }
    assert NOTE in result["memory"]
    assert cancelled["mode"] == "run"
    assert value(await harness.http.get("/v1/agents/sales/runs?active=true")) == ()


def json_loads(raw: bytes) -> Any:
    import json

    return json.loads(raw.decode().strip().splitlines()[-1])


def _delta(index: int) -> AgentEvent:
    return AgentEvent(
        kind=AgentEventKind.MODEL_TEXT_DELTA,
        occurred_at=NOW,
        run_id="run-1",
        conversation_id="conversation-1",
        data=FrozenJsonObject.from_mapping(
            {"text": str(index), "model_call_index": index + 1}
        ),
    )


@pytest.mark.integration
async def test_a_dropped_stream_replays_the_missed_events_exactly_once(
    start_host: StartHost, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.access")
    harness = await start_host()
    await harness.create_agent()
    seen: list[AgentEvent] = []
    agent = await daita.connect(
        "sales", cwd=harness.work(), root=harness.root, observer=seen.append
    )
    resets: list[str] = []
    agent.on_reset = lambda: resets.append("reset")
    bus = harness.host.bus
    try:
        bus.publish("sales", EventType.AGENT_EVENT, _delta(0))
        await bounded(lambda: len(seen) == 1)
        cursor = bus.head("sales")

        assert harness.drop_connections() >= 1
        for index in (1, 2, 3):
            bus.publish("sales", EventType.AGENT_EVENT, _delta(index))
        await bounded(lambda: len(seen) == 4)
        # Anything replayed twice would arrive before this last event.
        bus.publish("sales", EventType.AGENT_EVENT, _delta(4))
        await bounded(lambda: len(seen) == 5)
    finally:
        await agent.close()

    assert [event.data["text"] for event in seen] == ["0", "1", "2", "3", "4"]
    assert f"/v1/agents/sales/events?after={cursor} " in caplog.text
    assert resets == []


@pytest.mark.integration
async def test_a_restarted_host_resets_the_stream_and_the_client_resyncs(
    start_host: StartHost,
) -> None:
    first = await start_host(model=_remembering("unused"))
    await first.create_agent()
    asked: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request)
        return ApprovalDecision.APPROVE

    agent = await daita.connect(
        "sales", cwd=first.work(), root=first.root, approval_handler=approve
    )
    resets: list[str] = []
    agent.on_reset = lambda: resets.append(agent.name)
    try:
        await first.stop()
        second = await start_host(root=first.root, model=_remembering("Saved."))
        try:
            await bounded(lambda: resets)
            exit = await asyncio.wait_for(agent.run("Remember revenue."), 20)
        finally:
            await second.stop()  # before the fixture checks the shared socket
    finally:
        await agent.close()

    assert resets == ["sales"]
    assert exit.final_text == "Saved."
    assert len(asked) == 1


@pytest.mark.integration
async def test_questions_opened_before_connecting_reach_the_approval_handler(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    run = await harness.submit(message="Remember revenue.", cwd=str(harness.work()))
    await pending_interaction(harness.http, "sales")
    holding = asyncio.Event()
    withdrawn = asyncio.Event()

    async def hold(request: ApprovalRequest) -> ApprovalDecision:
        holding.set()
        try:
            await asyncio.Event().wait()
        finally:
            withdrawn.set()
        raise AssertionError("unreachable")

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        await holding.wait()
        return ApprovalDecision.APPROVE

    cwd = harness.work()
    slow = await daita.connect(
        "sales", cwd=cwd, root=harness.root, approval_handler=hold
    )
    fast = await daita.connect(
        "sales", cwd=cwd, root=harness.root, approval_handler=approve
    )
    try:
        finished = await finished_run(harness.http, "sales", run["run_ref"])
        # The question was answered elsewhere, so the slow card is withdrawn.
        await asyncio.wait_for(withdrawn.wait(), 10)
    finally:
        await slow.close()
        await fast.close()

    assert finished["state"] == "completed"
    assert finished["exit"].final_text == "Saved."


@pytest.mark.integration
async def test_host_errors_are_raised_as_the_same_exception_types(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("unused"))
    await harness.create_agent()
    waiting = asyncio.Event()

    async def hold(request: ApprovalRequest) -> ApprovalDecision:
        waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    agent = await daita.connect(
        "sales", cwd=harness.work(), root=harness.root, approval_handler=hold
    )
    try:
        first = asyncio.create_task(agent.run("Remember revenue."))
        await asyncio.wait_for(waiting.wait(), 10)
        (active,) = value(await harness.http.get("/v1/agents/sales/runs?active=true"))

        with pytest.raises(ConversationBusyError) as busy:
            await agent.run("Again.", conversation_id=active["conversation_id"])
        with pytest.raises(WorkspaceBusyError) as moved:
            await daita.connect("sales", cwd=harness.work("other"), root=harness.root)
        with pytest.raises(AgentNotFoundError):
            await daita.connect("nobody", root=harness.root)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    finally:
        await agent.close()

    assert busy.value.conversation_id == active["conversation_id"]
    assert moved.value.folder == harness.work()
    await bounded(
        lambda: _no_active_runs(harness),
    )


@pytest.mark.parametrize("value", ["a/b", "Users/x/guide.md", "..", ".", "a%2Fb"])
async def test_a_path_value_reaches_its_route_whatever_it_contains(
    start_host: StartHost, value: str
) -> None:
    harness = await start_host()
    agent = await remote_agent(harness, "sales")
    try:
        # The same outcome as in process: not found, or the member's own refusal.
        assert await agent.inspect_job(value) is None
        assert await agent.inspect_routine(value) is None
        with pytest.raises(ValueError, match="skill name must match"):
            await agent.read_skill(value)
    finally:
        await agent.close()


async def _no_active_runs(harness: Any) -> bool:
    return value(await harness.http.get("/v1/agents/sales/runs?active=true")) == ()


@pytest.mark.integration
async def test_a_subscription_login_runs_its_callbacks_on_the_client(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorized = asyncio.Event()
    fake_openai_device_login(monkeypatch, authorized)
    harness = await start_host()
    await harness.create_agent()
    prompts: list[CodexDevicePrompt] = []
    progress: list[str] = []
    agent = await daita.connect("sales", cwd=harness.work(), root=harness.root)
    try:
        login = asyncio.create_task(
            agent.authenticate_model_subscription(
                provider="codex",
                on_verification=prompts.append,
                on_progress=progress.append,
            )
        )
        await bounded(lambda: "Waiting for ChatGPT authorization" in progress)
        authorized.set()
        credential = await asyncio.wait_for(login, 10)

        authorized.clear()
        abandoned = asyncio.create_task(
            agent.authenticate_model_subscription(
                provider="codex", on_verification=prompts.append
            )
        )
        await bounded(lambda: len(prompts) == 2)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        await bounded(lambda: _no_interactions(harness))
    finally:
        await agent.close()

    assert prompts[0].user_code == "ABCD-EFGH"
    assert CodexOAuthCredential.from_secret(credential).refresh_token == "refresh-1"


async def _no_interactions(harness: Any) -> bool:
    return value(await harness.http.get("/v1/agents/sales/interactions")) == ()


@pytest.mark.integration
async def test_a_host_of_another_version_is_drained_and_replaced(
    start_host: StartHost,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="daita.host")
    monkeypatch.setattr(host_server, "__version__", "0.0.0+old")
    harness = await start_host()
    try:
        names = await RemoteAgent.list(root=harness.root)
        status = await asyncio.to_thread(host_status, harness.paths)
    finally:
        await asyncio.to_thread(stop_host, harness.paths)

    assert names == ()
    assert harness.task.done()
    assert "shutdown requested: upgrade" in caplog.text
    assert status is not None
    assert (status["version"], status["protocol"]) == (daita.__version__, 1)
    assert status["pid"] != os.getpid()
    assert not harness.paths.socket.exists()


@pytest.mark.unit
async def test_a_host_that_cannot_start_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = short_root()
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    try:
        with pytest.raises((HostUnavailableError, OSError)):
            await RemoteAgent.list(root=root)
    finally:
        remove_root(root)


@pytest.mark.unit
def test_a_widened_workspace_is_refused_before_reaching_the_host(
    tmp_path: Path,
) -> None:
    contained = LocalWorkspace(tmp_path)

    with pytest.raises(ValueError, match="computer access"):
        asyncio.run(RemoteAgent.open(NAME, workspace=contained, root=tmp_path))
    assert not (tmp_path / "run").exists()


@pytest.mark.unit
def test_connect_and_the_host_errors_are_public() -> None:
    assert daita.connect is daita.host.connect
    assert (daita.host.HostUnavailableError, daita.host.HostVersionMismatch) == (
        HostUnavailableError,
        HostVersionMismatch,
    )


class _Unreviewable(Exception):
    """A client that cannot show the question, like a too-small terminal."""


async def _raises(_request: ApprovalRequest) -> ApprovalDecision:
    raise _Unreviewable("terminal is too small to review this change")


async def _closes(_request: ApprovalRequest) -> ApprovalDecision:
    raise asyncio.CancelledError  # the TUI's answer when a card is closed


async def _never(_request: ApprovalRequest) -> ApprovalDecision:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("handler", "options"),
    (
        (_raises, {}),
        (_closes, {}),
        (_never, {"interaction_expiry": timedelta(milliseconds=200)}),
    ),
    ids=("handler_raises", "handler_closes", "question_expires"),
)
async def test_an_unanswered_approval_interrupts_the_run_and_never_denies(
    start_host: StartHost,
    handler: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]],
    options: dict[str, object],
) -> None:
    harness = await start_host(model=_remembering("Saved."), **options)
    await harness.create_agent()
    agent = await daita.connect(
        "sales",
        cwd=harness.work(),
        root=harness.root,
        approval_handler=handler,  # type: ignore[arg-type]
    )
    try:
        exit = await asyncio.wait_for(agent.run("Remember revenue."), 20)
        memory = await agent.read_memory()
    finally:
        await agent.close()

    # A denial would let the model answer; an interruption ends the run unwritten.
    assert (exit.kind, exit.reason, exit.final_text) == (
        LoopExitKind.INTERRUPTED,
        "tool_batch_interrupted",
        None,
    )
    assert memory == ""


@pytest.mark.integration
async def test_configuring_the_model_reopens_an_idle_agent_on_the_new_route(
    start_host: StartHost,
) -> None:
    harness = await start_host(agent_options={"model_validator": _validator()})
    await harness.create_agent()
    agent = await daita.connect("sales", cwd=harness.work(), root=harness.root)
    try:
        async with harness.events("sales") as events:
            route = await agent.configure_model(**_OPENAI)
            await events.next("agent_state", lambda data: data["state"] == "reopened")
        current = agent.model_route
        live = value(await harness.http.get("/v1/agents/sales"))["model_route"]
    finally:
        await agent.close()

    # The host reopened the idle agent, so the new route is in effect at once.
    assert current == live == route
    assert route.candidates[0].provider_id == "openai:test-model"


@pytest.mark.integration
async def test_a_busy_agent_keeps_its_route_until_its_next_reopen(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorized = asyncio.Event()
    fake_openai_device_login(monkeypatch, authorized)
    harness = await start_host(agent_options={"model_validator": _validator()})
    await harness.create_agent()
    agent = await daita.connect("sales", cwd=harness.work(), root=harness.root)
    # A pending question makes the agent busy, like an active run.
    login = asyncio.create_task(
        harness.http.post(
            "/v1/agents/sales/model/subscription-login", json={"provider": "codex"}
        )
    )
    try:
        await pending_interaction(harness.http, "sales")
        route = await agent.configure_model(**_OPENAI)
        busy = value(await harness.http.get("/v1/agents/sales"))["model_route"]
        authorized.set()
        await asyncio.wait_for(login, 10)
        reopened = await harness.http.post(
            "/v1/agents/sales/open", json={"cwd": str(harness.work("other"))}
        )
    finally:
        authorized.set()
        await agent.close()

    assert busy is None
    assert value(reopened)["model_route"] == route


_OPENAI: dict[str, Any] = {
    "provider": "openai",
    "model": "test-model",
    "api_key": "sk-test-not-real",
    "context_window_tokens": 8_192,
    "max_output_tokens": 1_024,
}


def _validator() -> MockModelProvider:
    validation = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="validation-call", name="daita_validate_tool_support", arguments={}
            ),
        ),
        provider_id="openai:test-model",
    )
    return MockModelProvider((validation,), provider_id="openai:test-model")


@pytest.mark.integration
async def test_a_folder_switch_by_another_client_refreshes_the_workspace(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()
    here, there = harness.work(), harness.work("other")
    first = await daita.connect("sales", cwd=here, root=harness.root)
    try:
        before = (first.workspace.root, first.session_source_id)
        second = await daita.connect("sales", cwd=there, root=harness.root)
        try:
            await bounded(lambda: first.workspace.root == there)
            after = (first.workspace.root, first.session_source_id)
            expected = (second.workspace.root, second.session_source_id)
        finally:
            await second.close()
    finally:
        await first.close()

    assert before[0] == here
    assert after == expected
    assert after[1] != before[1]


@pytest.mark.integration
async def test_a_run_finishing_while_the_stream_is_down_still_returns(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    asked = asyncio.Event()
    release = asyncio.Event()

    async def approve_later(request: ApprovalRequest) -> ApprovalDecision:
        asked.set()
        await release.wait()
        return ApprovalDecision.APPROVE

    agent = await daita.connect(
        "sales", cwd=harness.work(), root=harness.root, approval_handler=approve_later
    )
    try:
        run = asyncio.create_task(agent.run("Remember revenue."))
        await asyncio.wait_for(asked.wait(), 10)
        assert harness.drop_connections() >= 1
        release.set()  # the run ends before the stream's first reconnect attempt
        exit = await asyncio.wait_for(run, 20)
    finally:
        await agent.close()

    assert exit.final_text == "Saved."
