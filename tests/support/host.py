"""Drive a real host over its Unix socket with raw httpx and bounded waits."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from daita import __version__
from daita.host.codec import decode
from daita.host.paths import HostPaths
from daita.host.protocol import PROTOCOL_HEADER, PROTOCOL_VERSION, VERSION_HEADER
from daita.host.server import Host

VERSION_HEADERS = {PROTOCOL_HEADER: str(PROTOCOL_VERSION), VERSION_HEADER: __version__}
ACTIVE = frozenset({"queued", "running", "waiting_for_input"})


def short_root() -> Path:
    """A state root under /tmp: pytest's tmp_path can exceed the socket limit."""

    return Path(tempfile.mkdtemp(prefix="dh-", dir="/tmp")).resolve()


def remove_root(root: Path) -> None:
    for folder in (root, *root.parent.glob(f"{root.name}-*")):
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
    "VERSION_HEADERS",
    "bounded",
    "client",
    "error_code",
    "finished_run",
    "pending_interaction",
    "remove_root",
    "short_root",
    "value",
    "work_folder",
]
