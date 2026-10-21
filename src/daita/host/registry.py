"""Open the agents the host serves, own their writer locks and their folders."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..agent import Agent
from ..errors import StateCompatibilityError
from ..hosting.agent_home import WRITER_ROLE
from ..hosting.errors import AgentAlreadyExistsError, AgentHomeError, HostActiveError
from ..jobs.graph.models import TERMINAL_GRAPH_STATES, GraphState
from ..observation import AgentEvent, AgentEventKind
from ..routines import RoutineState
from ..workspace import LocalFileAccess, LocalWorkspace, working_directory
from .events import EventBus
from .interactions import InteractionHub, InteractionKind, approval_handler
from .protocol import AgentLockedError, EventType, WorkspaceBusyError
from .runs import CURRENT_RUN, RunManager

if TYPE_CHECKING:
    from ..security import KeychainStore

_LOGGER = logging.getLogger("daita.host")
_UNFINISHED_JOBS = frozenset(GraphState) - TERMINAL_GRAPH_STATES


@dataclass(slots=True)
class _Entry:
    agent: Agent
    workspace: LocalWorkspace


class AgentRegistry:
    """One open Agent per name, opened on demand under the host writer role."""

    def __init__(
        self,
        root: Path,
        bus: EventBus,
        hub: InteractionHub,
        runs: RunManager,
        *,
        agent_options: Mapping[str, object] | None = None,
    ) -> None:
        self._root = root
        self._bus = bus
        self._hub = hub
        self._runs = runs
        # Composition the host supplies to every open, such as a model in tests.
        self._options = dict(agent_options or {})
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def open_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    async def get(self, name: str, cwd: Path | None = None) -> Agent:
        """Return the open agent, reopening it in the client's folder when idle."""

        entry = self._entries.get(name)
        if entry is not None and (cwd is None or entry.workspace == self._for(cwd)):
            return entry.agent
        async with self._lock(name):
            return await self._ensure(name, cwd)

    async def create(self, name: str, cwd: Path | None) -> Agent:
        async with self._lock(name):
            if name in self._entries:
                raise AgentAlreadyExistsError(f"agent already exists: {name}")
            workspace = self._for(cwd)
            agent = await self._compose(Agent.create, name, workspace)
            self._entries[name] = _Entry(agent, workspace)
            self._bus.publish(name, EventType.AGENT_STATE, {"state": "opened"})
            return agent

    async def delete(self, name: str) -> None:
        async with self._lock(name):
            await self._close(name)
            keychain = cast("KeychainStore | None", self._options.get("keychain"))
            await Agent.delete(name, root=self._root, keychain=keychain)

    async def close(self, name: str) -> None:
        async with self._lock(name):
            await self._close(name)

    async def close_all(self) -> None:
        for name in self.open_names():
            await self.close(name)

    async def open_background_agents(self) -> None:
        """Open every agent with scheduled, unfinished or unread work (§10.3)."""

        for name in await Agent.list(root=self._root):
            try:
                agent = await self.get(name)
                if not await _has_background_work(agent):
                    await self.close(name)
            except (AgentHomeError, StateCompatibilityError, ValueError) as error:
                _LOGGER.warning(
                    "agent skipped at startup: %s (%s)", name, type(error).__name__
                )

    async def _ensure(self, name: str, cwd: Path | None) -> Agent:
        entry = self._entries.get(name)
        if entry is not None and cwd is None:
            return entry.agent
        workspace = self._for(cwd)
        if entry is not None:
            if entry.workspace == workspace:
                return entry.agent
            if self._runs.active(name) or self._hub.pending(name):
                raise WorkspaceBusyError(
                    f"Daita is busy in {entry.workspace.root}. Wait for it to "
                    "finish, or ask without local files.",
                    folder=entry.workspace.root,
                )
            # shortcut: a non-run request still in flight on the old instance
            # fails when it closes; track in-flight calls if that shows up.
            del self._entries[name]
            await entry.agent.close()
        agent = await self._compose(Agent.open, name, workspace)
        self._entries[name] = _Entry(agent, workspace)
        state = "opened" if entry is None else "reopened"
        self._bus.publish(name, EventType.AGENT_STATE, {"state": state})
        return agent

    async def _compose(
        self,
        opener: Callable[..., Awaitable[Agent]],
        name: str,
        workspace: LocalWorkspace,
    ) -> Agent:
        token = WRITER_ROLE.set("host")
        try:
            return await opener(
                name,
                workspace=workspace,
                root=self._root,
                observer=lambda event: self._observe(name, event),
                approval_handler=approval_handler(
                    self._hub,
                    name,
                    InteractionKind.TOOL_APPROVAL,
                    run_ref=CURRENT_RUN.get,
                    waiting=self._runs.waiting,
                ),
                **self._options,
            )
        except HostActiveError as error:
            if type(error) is not HostActiveError:
                raise
            raise AgentLockedError(
                f"agent is in use by another process: {name}", holder=error.holder
            ) from error
        finally:
            WRITER_ROLE.reset(token)

    async def _close(self, name: str) -> None:
        entry = self._entries.pop(name, None)
        if entry is not None:
            await entry.agent.close()
            self._bus.publish(name, EventType.AGENT_STATE, {"state": "closed"})

    def _observe(self, name: str, event: AgentEvent) -> None:
        self._runs.observe(event)
        self._bus.publish(name, EventType.AGENT_EVENT, event)
        if event.kind is AgentEventKind.RUN_COMPLETED and event.run_origin != "user":
            self._bus.publish(name, EventType.WORK_CHANGED, {"run_id": event.run_id})

    def _for(self, cwd: Path | None) -> LocalWorkspace:
        if cwd is None:
            # No client folder yet: never silently grant home-directory access.
            return LocalWorkspace(Path.home(), access=LocalFileAccess.NONE)
        return LocalWorkspace(
            working_directory(cwd, self._root), access=LocalFileAccess.COMPUTER
        )

    def _lock(self, name: str) -> asyncio.Lock:
        return self._locks.setdefault(name, asyncio.Lock())


async def _has_background_work(agent: Agent) -> bool:
    return bool(
        await agent.list_routines(states=frozenset({RoutineState.ACTIVE}), limit=1)
        or await agent.list_jobs(states=_UNFINISHED_JOBS, limit=1)
        or await agent.list_deliveries(limit=1)
    )


__all__ = ["AgentRegistry"]
