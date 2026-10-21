"""Drive a real host over its Unix socket with raw httpx and bounded waits."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from daita import (
    LocalWorkspace,
    MisfirePolicy,
    OnceSchedule,
    ReportingMode,
    ScheduledRoutine,
    ScheduledRoutineDraft,
    __version__,
)
from daita.host.client import RemoteAgent, connect
from daita.host.codec import decode
from daita.host.paths import HostPaths, host_paths
from daita.host.protocol import PROTOCOL_HEADER, PROTOCOL_VERSION, VERSION_HEADER
from daita.host.server import Host, serve_host
from daita.llm import subscription_auth
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
)
from daita.llm.pricing import CostEstimate
from tests.llm._subscription_support import _jwt
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.mcp import MemoryKeychain

VERSION_HEADERS = {PROTOCOL_HEADER: str(PROTOCOL_VERSION), VERSION_HEADER: __version__}
ACTIVE = frozenset({"queued", "running", "waiting_for_input"})


def short_root() -> Path:
    """A state root under /tmp: pytest's tmp_path can exceed the socket limit."""

    return Path(tempfile.mkdtemp(prefix="dh-", dir="/tmp")).resolve()


def remove_root(root: Path) -> None:
    # Work folders sit beside the root; workspace_for() makes hidden ones there.
    siblings = (
        *root.parent.glob(f"{root.name}-*"),
        *root.parent.glob(f".{root.name}-*"),
    )
    for folder in (root, *siblings):
        shutil.rmtree(folder, ignore_errors=True)


def work_folder(root: Path, name: str = "work") -> Path:
    """A client working directory beside, never inside, the state root."""

    folder = root.parent / f"{root.name}-{name}"
    folder.mkdir(mode=0o700, exist_ok=True)
    return folder


def client(paths: HostPaths, **headers: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=str(paths.socket)),
        base_url="http://daita-host",
        headers={**VERSION_HEADERS, **headers},
        timeout=10,
    )


def value(response: httpx.Response) -> Any:
    """Decode a successful host response body."""

    assert response.status_code < 400, response.text
    return decode(response.json())


def error_code(response: httpx.Response) -> str:
    return str(response.json()["error"]["code"])


@dataclass
class HostHarness:
    host: Host
    paths: HostPaths
    task: asyncio.Task[None]
    http: httpx.AsyncClient

    @property
    def root(self) -> Path:
        return self.paths.root

    def work(self, name: str = "work") -> Path:
        return work_folder(self.root, name)

    def events(self, agent: str) -> EventStream:
        """Stream every event published from now on, without a subscribe race."""

        return EventStream(self.http, agent, after=self.host.bus.head(agent))

    def drop_connections(self) -> int:
        """Close every client connection from the host side, as a broken link would."""

        server = self.host._server
        assert server is not None
        connections = list(server.server_state.connections)
        for connection in connections:
            connection.transport.close()
        return len(connections)

    async def create_agent(self, name: str = "sales") -> dict[str, Any]:
        response = await self.http.post(
            "/v1/agents", json={"name": name, "cwd": str(self.work())}
        )
        return value(response)

    async def submit(self, agent: str = "sales", **body: Any) -> dict[str, Any]:
        response = await self.http.post(f"/v1/agents/{agent}/runs", json=body)
        assert response.status_code == 202, response.text
        return value(response)

    async def stop(self) -> None:
        self.host.request_shutdown("test")
        try:
            await asyncio.wait_for(asyncio.shield(self.task), 60)
        finally:
            await self.http.aclose()


StartHost = Callable[..., Awaitable[HostHarness]]


@contextlib.asynccontextmanager
async def hosts() -> AsyncIterator[StartHost]:
    """Start real in-process hosts on short sockets; stop and remove each after.

    ``model`` is injected into every agent the host opens, with an in-memory
    keychain so no test touches the OS keychain.
    """

    harnesses: list[HostHarness] = []

    async def start(
        *,
        model: object = None,
        root: Path | None = None,
        agent_options: Mapping[str, object] | None = None,
        **options: object,
    ) -> HostHarness:
        paths = host_paths(root or short_root())
        composed: dict[str, object] = {"keychain": MemoryKeychain()}
        if model is not None:
            composed["model"] = model
            composed["model_profile"] = ModelProfile(
                id=model.provider_id,  # type: ignore[attr-defined]
                context_window_tokens=32_000,
                max_output_tokens=1_000,
                supports_tools=True,
                supports_parallel_tools=True,
            )
        composed.update(agent_options or {})
        ready: asyncio.Future[Host] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            serve_host(
                paths,
                on_ready=ready.set_result,
                agent_options=composed,
                **options,  # type: ignore[arg-type]
            )
        )

        def stopped(done: asyncio.Task[None]) -> None:
            if not ready.done():
                error = done.exception() if not done.cancelled() else None
                ready.set_exception(error or RuntimeError("the host stopped early"))

        task.add_done_callback(stopped)
        host = await asyncio.wait_for(ready, 30)
        harness = HostHarness(host, paths, task, client(paths))
        harnesses.append(harness)
        return harness

    try:
        yield start
    finally:
        for harness in harnesses:
            try:
                await harness.stop()
                assert not harness.paths.socket.exists()
            finally:
                remove_root(harness.root)


async def remote_agent(
    harness: HostHarness, name: str, **callbacks: Any
) -> RemoteAgent:
    """Create one agent in the harness host and attach from its work folder."""

    await harness.create_agent(name)
    return await connect(name, cwd=harness.work(), root=harness.root, **callbacks)


def measured(text: str) -> ModelResponse:
    """A final answer with measured usage, which scheduled runs require."""

    usage = ModelUsage(
        input_tokens=100,
        output_tokens=10,
        cost_estimate=CostEstimate.complete(Decimal("0")),
    )
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=usage)


async def scheduled_routine(
    agent: RemoteAgent, model_id: str, *, due_in: timedelta
) -> ScheduledRoutine:
    """Ask one question, then schedule one reporting routine from its answer.

    The model must script one ``measured`` answer for the question and one per
    occurrence.
    """

    origin = await agent.run("How many invoices are paid?")
    assert origin.conversation_id is not None
    (destination,) = await agent.distribution_destinations(origin.conversation_id)
    now = datetime.now(UTC)
    draft = ScheduledRoutineDraft(
        origin_run_id=origin.run_id,
        title="Paid invoice count",
        authorized_instruction="Report the paid invoice count.",
        schedule=OnceSchedule(now + due_in),
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        allowed_source_ids=(),
        allowed_connector_binding_ids=(),
        allowed_resource_ids=(),
        allowed_capability_ids=("artifact.create_document",),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        outcome_contract=no_artifact_outcome_contract(),
        distribution_destination_id=destination.destination_id,
        eligible_model_routes=(model_id,),
        per_run_max_tokens=1_000,
        per_run_max_cost_usd=None,
        cumulative_max_tokens=10_000,
        cumulative_max_cost_usd=None,
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        maximum_consecutive_failures=3,
        expires_at=now + timedelta(days=30),
    )
    return await agent.create_routine(await agent.propose_routine(draft))


def host_workspace(harness: HostHarness) -> LocalWorkspace:
    """The app's folder: the work folder its remote agents were attached from."""

    return LocalWorkspace(harness.work())


async def bounded(
    probe: Callable[[], Any], *, timeout: float = 10.0, interval: float = 0.02
) -> Any:
    """Await probe() until it returns something truthy, or fail after timeout."""

    deadline = time.monotonic() + timeout
    while True:
        result = probe()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return result
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(interval)


async def finished_run(
    http: httpx.AsyncClient, agent: str, ref: str, timeout: float = 10.0
) -> dict[str, Any]:
    async def probe() -> dict[str, Any] | None:
        run = value(await http.get(f"/v1/agents/{agent}/runs/{ref}"))
        return None if run["state"] in ACTIVE else run

    return await bounded(probe, timeout=timeout)


async def pending_interaction(
    http: httpx.AsyncClient, agent: str, timeout: float = 10.0
) -> dict[str, Any]:
    async def probe() -> dict[str, Any] | None:
        pending = value(await http.get(f"/v1/agents/{agent}/interactions"))
        return pending[0] if pending else None

    return await bounded(probe, timeout=timeout)


def fake_openai_device_login(
    monkeypatch: pytest.MonkeyPatch, authorized: asyncio.Event
) -> None:
    """Fake only OpenAI's endpoints; the token poll waits for ``authorized``."""

    async def post_json(url: str, _body: object) -> subscription_auth._HttpResult:
        if url.endswith("/usercode"):
            payload: dict[str, object] = {
                "device_auth_id": "device-1",
                "user_code": "ABCD-EFGH",
                "interval": 1,
            }
        else:
            await authorized.wait()
            payload = {"authorization_code": "code-1", "code_verifier": "verifier-1"}
        return subscription_auth._HttpResult(200, json.dumps(payload).encode())

    async def post_form(_url: str, _body: object) -> subscription_auth._HttpResult:
        tokens = {
            "access_token": _jwt(),
            "refresh_token": "refresh-1",
            "expires_in": 3_600,
        }
        return subscription_auth._HttpResult(200, json.dumps(tokens).encode())

    monkeypatch.setattr(subscription_auth, "_post_json", post_json)
    monkeypatch.setattr(subscription_auth, "_post_form", post_form)


@dataclass
class EventStream:
    """Read one agent's SSE stream into decoded (seq, event, data) records."""

    http: httpx.AsyncClient
    agent: str
    after: int | None = None
    records: list[tuple[int, str, Any]] = field(default_factory=list)
    _queue: asyncio.Queue[tuple[int, str, Any]] = field(default_factory=asyncio.Queue)
    _task: asyncio.Task[None] | None = None
    _connected: asyncio.Event = field(default_factory=asyncio.Event)

    async def __aenter__(self) -> EventStream:
        self._task = asyncio.create_task(self._read())
        await asyncio.wait_for(self._connected.wait(), 10)
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                await self._task
            self._task = None

    async def next(
        self,
        event: str | None = None,
        where: Callable[[Any], bool] = lambda _data: True,
        timeout: float = 10.0,
    ) -> tuple[int, str, Any]:
        async with asyncio.timeout(timeout):
            while True:
                record = await self._queue.get()
                if (event is None or record[1] == event) and where(record[2]):
                    return record

    async def _read(self) -> None:
        params = {} if self.after is None else {"after": str(self.after)}
        url = f"/v1/agents/{self.agent}/events"
        async with self.http.stream(
            "GET", url, params=params, timeout=None
        ) as response:
            assert response.status_code == 200, await response.aread()
            self._connected.set()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    payload = json.loads(line[len("data:") :])
                    record = (payload["seq"], payload["event"], decode(payload["data"]))
                    self.records.append(record)
                    self._queue.put_nowait(record)


__all__ = [
    "ACTIVE",
    "EventStream",
    "HostHarness",
    "StartHost",
    "VERSION_HEADERS",
    "bounded",
    "client",
    "error_code",
    "fake_openai_device_login",
    "host_workspace",
    "hosts",
    "measured",
    "finished_run",
    "pending_interaction",
    "remote_agent",
    "remove_root",
    "scheduled_routine",
    "short_root",
    "value",
    "work_folder",
]
