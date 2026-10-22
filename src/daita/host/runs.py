"""Own foreground runs on the host: start, track, cancel and drain them."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .._json import FrozenJsonObject
from ..loop.models import LoopExit, LoopExitKind
from ..observation import AgentEvent, AgentEventKind
from .events import EventBus
from .protocol import (
    ConversationBusyError,
    EventType,
    HostShuttingDownError,
    error_envelope,
)

if TYPE_CHECKING:
    from ..agent import Agent

# The host run_ref of the run executing in this task, read by agent callbacks.
CURRENT_RUN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "daita_host_run", default=None
)
ACTIVE_STATES = frozenset({"queued", "running", "waiting_for_input"})
_EXIT_STATES = {
    LoopExitKind.COMPLETED: "completed",
    LoopExitKind.FAILED: "failed",
    LoopExitKind.MACHINE_TERMINATED: "failed",
    LoopExitKind.INTERRUPTED: "interrupted",
}
# shortcut: finished runs are kept in memory up to this count; persist them if
# clients need older run states.
_MAX_FINISHED = 1_000


@dataclass(slots=True)
class Run:
    run_ref: str
    agent: str
    mode: str
    conversation_id: str | None
    run_id: str | None = None
    state: str = "queued"
    exit: LoopExit | None = None
    error: FrozenJsonObject | None = None
    task: asyncio.Task[None] | None = None
    cancel_requested: bool = False

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def view(self) -> dict[str, object]:
        return {
            "run_ref": self.run_ref,
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "mode": self.mode,
            "state": self.state,
            "exit": self.exit,
            "error": self.error,
        }


class RunManager:
    """Runs belong to the host: a client disconnect never cancels one."""

    def __init__(self, bus: EventBus, marker: Path) -> None:
        self._bus = bus
        self._marker = marker
        self._runs: dict[str, Run] = {}
        self._shutting_down = False
        self._recover_interrupted()

    def submit(
        self,
        agent_name: str,
        agent: Agent,
        message: str,
        *,
        conversation_id: str | None,
        source_scope_ids: tuple[str, ...],
        files_only: bool,
        mode: str,
    ) -> Run:
        if self._shutting_down:
            raise HostShuttingDownError("the Daita host is shutting down")
        if mode not in ("run", "learn"):
            raise ValueError('run mode must be "run" or "learn"')
        if conversation_id is not None and any(
            run.active and run.conversation_id == conversation_id
            for run in self.active(agent_name)
        ):
            raise ConversationBusyError(
                "a run is already active in this conversation",
                conversation_id=conversation_id,
            )
        run = Run(f"runref-{uuid4().hex}", agent_name, mode, conversation_id)
        if mode == "learn":
            if files_only:
                raise ValueError("learn runs cannot be files_only")
            call = agent.learn(
                message,
                conversation_id=conversation_id,
                source_scope_ids=source_scope_ids,
            )
        else:
            call = agent.run(
                message,
                conversation_id=conversation_id,
                source_scope_ids=source_scope_ids,
                files_only=files_only,
            )
        context = contextvars.copy_context()
        context.run(CURRENT_RUN.set, run.run_ref)
        self._runs[run.run_ref] = run
        run.task = asyncio.create_task(self._execute(run, call), context=context)
        self._changed(run)
        return run

    def observe(self, event: AgentEvent) -> None:
        """Bind the engine run_id to the host run executing this event."""

        run = self._runs.get(CURRENT_RUN.get() or "")
        if (
            run is not None
            and event.kind is AgentEventKind.RUN_STARTED
            and run.run_id is None
        ):
            run.run_id = event.run_id
            run.conversation_id = event.conversation_id
            run.state = "running"
            self._changed(run)

    def waiting(self, run_ref: str | None, waiting: bool) -> None:
        run = self._runs.get(run_ref or "")
        if run is not None and run.active:
            run.state = "waiting_for_input" if waiting else "running"
            self._changed(run)

    def get(self, agent_name: str, ref: str) -> Run:
        for run in self._runs.values():
            if run.agent == agent_name and ref in (run.run_ref, run.run_id):
                return run
        raise KeyError(f"run is unknown to this host: {ref}")

    def active(self, agent_name: str) -> tuple[Run, ...]:
        return tuple(
            run for run in self._runs.values() if run.agent == agent_name and run.active
        )

    def cancel(self, agent_name: str, ref: str) -> Run:
        run = self.get(agent_name, ref)
        if run.active and run.task is not None and not run.cancel_requested:
            run.cancel_requested = True
            run.task.cancel()
        return run

    async def drain(self, seconds: float) -> None:
        """Refuse new runs, let active ones finish for a while, then cancel them."""

        self._shutting_down = True
        tasks = {run.task for run in self._runs.values() if run.active and run.task}
        if not tasks:
            return
        _done, pending = await asyncio.wait(tasks, timeout=seconds)
        for task in pending:
            run = next(item for item in self._runs.values() if item.task is task)
            self.cancel(run.agent, run.run_ref)
        if pending:
            await asyncio.wait(pending)

    async def _execute(self, run: Run, call: object) -> None:
        try:
            result = await call  # type: ignore[misc]
            run.exit = result
            run.state = _EXIT_STATES[result.kind]
        except asyncio.CancelledError:
            # Expired or dismissed interactions interrupt; only cancel cancels.
            run.state = "cancelled" if run.cancel_requested else "interrupted"
        except Exception as error:
            run.state = "failed"
            try:
                envelope = error_envelope(error)
            except TypeError:
                envelope = error_envelope(RuntimeError("the run failed in the host"))
            run.error = FrozenJsonObject.from_mapping(envelope)
        finally:
            if run.active:
                run.state = "interrupted"
            self._changed(run)
            self._forget_finished()

    def _changed(self, run: Run) -> None:
        self._bus.publish(run.agent, EventType.RUN_STATE, run.view())
        self._write_marker()

    def _forget_finished(self) -> None:
        finished = [ref for ref, run in self._runs.items() if not run.active]
        for ref in finished[: max(0, len(finished) - _MAX_FINISHED)]:
            del self._runs[ref]

    def _recover_interrupted(self) -> None:
        # Anything still listed belongs to a host that died mid-run (§10.6).
        try:
            entries = json.loads(self._marker.read_text(encoding="utf-8"))
        except FileNotFoundError:
            entries = []
        except (OSError, ValueError):
            entries = []
        for entry in entries if type(entries) is list else []:
            try:
                run = Run(
                    str(entry["run_ref"]),
                    str(entry["agent"]),
                    str(entry["mode"]),
                    entry["conversation_id"],
                    entry["run_id"],
                    state="interrupted",
                )
            except (KeyError, TypeError):
                continue
            self._runs[run.run_ref] = run
        self._write_marker()

    def _write_marker(self) -> None:
        entries = [
            {
                "run_ref": run.run_ref,
                "agent": run.agent,
                "mode": run.mode,
                "conversation_id": run.conversation_id,
                "run_id": run.run_id,
            }
            for run in self._runs.values()
            if run.active
        ]
        temporary = self._marker.with_name(f".{self._marker.name}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(entries, file)
        os.replace(temporary, self._marker)


__all__ = ["ACTIVE_STATES", "CURRENT_RUN", "Run", "RunManager"]
