"""Own one ordered transcript writer for one admitted loop run."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Protocol

from .._json import canonical_json
from ..llm.models import CanonicalMessage
from .models import ConversationRun, LoopExit, LoopExitKind, RunInput, Transcript

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ConversationPredecessor:
    """Exact terminal conversation revision used to prepare the next turn."""

    run_id: str
    turn_index: int
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("conversation predecessor run_id must be non-empty")
        if type(self.turn_index) is not int or self.turn_index < 0:
            raise ValueError("conversation predecessor turn_index must be non-negative")
        if (
            not isinstance(self.revision, str)
            or _SHA256.fullmatch(self.revision) is None
        ):
            raise ValueError("conversation predecessor revision must use sha256")

    @classmethod
    def from_run(cls, run: ConversationRun) -> ConversationPredecessor:
        if not isinstance(run, ConversationRun) or run.result is None:
            raise ValueError("conversation predecessor requires one terminal run")
        result = run.result
        material = {
            "run_id": result.run_id,
            "conversation_id": result.conversation_id,
            "turn_index": run.turn_index,
            "kind": result.kind.value,
            "reason": result.reason,
            "created_at": result.created_at.isoformat(),
        }
        return cls(
            run_id=result.run_id,
            turn_index=run.turn_index,
            revision="sha256:"
            + sha256(canonical_json(material).encode("utf-8")).hexdigest(),
        )


class TranscriptWriteStore(Protocol):
    async def start(
        self,
        run: RunInput,
        *,
        predecessor: ConversationPredecessor | None = None,
    ) -> Transcript: ...

    async def append_at(
        self,
        run_id: str,
        position: int,
        message: CanonicalMessage,
    ) -> None: ...

    async def complete(
        self,
        result: LoopExit,
        final_message: CanonicalMessage,
    ) -> None: ...

    async def finish(self, result: LoopExit) -> None: ...


class RunSessionWriterState(str, Enum):
    NEW = "new"
    STARTED = "started"
    TERMINAL = "terminal"


class RunSessionWriter:
    """Single-use ordered writer capability bound to exactly one run."""

    __slots__ = (
        "_bind_predecessor",
        "_next_position",
        "_state",
        "_store",
        "predecessor",
        "run",
    )

    def __init__(
        self,
        store: TranscriptWriteStore,
        run: RunInput,
        *,
        predecessor: ConversationPredecessor | None = None,
        bind_predecessor: bool = True,
    ) -> None:
        if not isinstance(run, RunInput):
            raise TypeError("run session writer requires RunInput")
        if predecessor is not None and not isinstance(
            predecessor, ConversationPredecessor
        ):
            raise TypeError(
                "writer predecessor must be ConversationPredecessor or None"
            )
        if not callable(getattr(store, "start", None)) or not callable(
            getattr(store, "append_at", None)
        ):
            raise TypeError("writer store must provide ordered transcript operations")
        if not isinstance(bind_predecessor, bool):
            raise TypeError("bind_predecessor must be bool")
        if predecessor is not None and not bind_predecessor:
            raise ValueError("an unbound writer cannot carry a predecessor")
        self._store = store
        self.run = run
        self.predecessor = predecessor
        self._bind_predecessor = bind_predecessor
        self._state = RunSessionWriterState.NEW
        self._next_position = 0

    @property
    def state(self) -> RunSessionWriterState:
        return self._state

    @property
    def next_position(self) -> int:
        return self._next_position

    async def start(self) -> Transcript:
        if self._state is not RunSessionWriterState.NEW:
            raise RuntimeError("run session writer may start exactly once")
        transcript = (
            await self._store.start(self.run, predecessor=self.predecessor)
            if self._bind_predecessor
            else await self._store.start(self.run)
        )
        if transcript.run != self.run or transcript.messages:
            raise RuntimeError("transcript store returned an invalid new transcript")
        self._state = RunSessionWriterState.STARTED
        return transcript

    async def append(self, message: CanonicalMessage, *, position: int) -> None:
        if self._state is not RunSessionWriterState.STARTED:
            raise RuntimeError("run session writer is not appendable")
        if type(position) is not int or position != self._next_position:
            raise ValueError("run session writer append position is out of order")
        if not isinstance(message, CanonicalMessage):
            raise TypeError("run session writer requires CanonicalMessage")
        await self._store.append_at(self.run.id, position, message)
        self._next_position += 1

    async def complete(
        self,
        result: LoopExit,
        final_message: CanonicalMessage,
    ) -> None:
        self._require_terminal_result(result, completed=True)
        await self._store.complete(result, final_message)
        self._state = RunSessionWriterState.TERMINAL

    async def finish(self, result: LoopExit) -> None:
        self._require_terminal_result(result, completed=False)
        await self._store.finish(result)
        self._state = RunSessionWriterState.TERMINAL

    def _require_terminal_result(self, result: LoopExit, *, completed: bool) -> None:
        if self._state is not RunSessionWriterState.STARTED:
            raise RuntimeError("run session writer is not terminalizable")
        if not isinstance(result, LoopExit) or result.run_id != self.run.id:
            raise ValueError("run session writer rejected a foreign result")
        if result.conversation_id != (self.run.conversation_id or self.run.id):
            raise ValueError("run session writer rejected a foreign conversation")
        if completed != (result.kind is LoopExitKind.COMPLETED):
            raise ValueError(
                "run session writer terminal operation differs from result"
            )


__all__ = [
    "ConversationPredecessor",
    "RunSessionWriter",
    "RunSessionWriterState",
    "TranscriptWriteStore",
]
