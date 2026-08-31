"""Hold one ordinary agent composition open for scheduled work."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..agent import Agent
from ..llm.models import ModelProfile
from ..llm.protocols import ModelProvider
from ..workspace import LocalWorkspace


@dataclass(frozen=True, slots=True)
class ResidentReady:
    agent_id: str
    agent_name: str
    agent_home: Path


async def run_resident_host(
    *,
    agent_name: str,
    workspace: LocalWorkspace,
    root: str | Path | None = None,
    stop_event: asyncio.Event | None = None,
    on_ready: Callable[[ResidentReady], None] | None = None,
    clock: Callable[[], datetime] | None = None,
    id_factory: Callable[[str], str] | None = None,
    model: ModelProvider | None = None,
    model_profile: ModelProfile | None = None,
) -> None:
    """Run supervisors until termination while owning the normal writer lock."""

    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agent_name must be non-empty text")
    if not isinstance(workspace, LocalWorkspace):
        raise TypeError("workspace must be LocalWorkspace")
    event = stop_event or asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    if stop_event is None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, event.set)
            except (NotImplementedError, RuntimeError):
                continue
            installed_signals.append(signum)
    agent: Agent | None = None
    try:
        agent = await Agent.open(
            agent_name.strip(),
            workspace=workspace,
            root=root,
            clock=clock,
            id_factory=id_factory,
            model=model,
            model_profile=model_profile,
        )
        if on_ready is not None:
            on_ready(
                ResidentReady(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    agent_home=agent.home,
                )
            )
        await event.wait()
    finally:
        if agent is not None:
            await agent.close()
        for signum in installed_signals:
            loop.remove_signal_handler(signum)


__all__ = ["ResidentReady", "run_resident_host"]
