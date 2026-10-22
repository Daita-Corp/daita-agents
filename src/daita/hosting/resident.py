"""Deprecated: hold one agent open in this process.

The Daita host (``daita host serve``) serves every agent; a later release
removes this module.
"""

from __future__ import annotations

import asyncio
import signal
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..agent import Agent
from ..llm.models import ModelProfile
from ..llm.protocols import ModelProvider
from ..workspace import LocalWorkspace

_DEPRECATION = (
    "{name} is deprecated; the Daita host serves every agent. "
    "Use daita.connect() or `daita host serve`."
)


@dataclass(frozen=True, slots=True)
class ResidentReady:
    agent_id: str
    agent_name: str
    agent_home: Path

    def __post_init__(self) -> None:
        warnings.warn(
            _DEPRECATION.format(name="ResidentReady"), DeprecationWarning, stacklevel=3
        )


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
    diagnostics_path: Path | None = None,
) -> None:
    """Run supervisors until termination while owning the normal writer lock."""

    warnings.warn(
        _DEPRECATION.format(name="run_resident_host"), DeprecationWarning, stacklevel=2
    )
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
            diagnostics_path=diagnostics_path,
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
