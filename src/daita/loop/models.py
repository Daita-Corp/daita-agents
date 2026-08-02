"""Small records used by the transcript-driven agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
import math

from ..artifacts.models import ArtifactDeliveryReceipt, ArtifactRef
from ..llm.models import CanonicalMessage, ModelUsage


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RunInput:
    """One user request and the small amount of identity needed to execute it."""

    id: str
    agent_id: str
    message: str
    created_at: datetime
    conversation_id: str | None = None
    source_id: str | None = None
    conversation_source_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "run id")
        _required_text(self.agent_id, "run agent_id")
        _required_text(self.message, "run message")
        _aware(self.created_at, "run created_at")
        if self.conversation_id is not None:
            _required_text(self.conversation_id, "run conversation_id")
        if self.source_id is not None:
            _required_text(self.source_id, "run source_id")
        if self.conversation_source_id is not None:
            _required_text(
                self.conversation_source_id,
                "run conversation_source_id",
            )


@dataclass(frozen=True, slots=True)
class LoopLimits:
    """Outer safety limits; normal tool recovery uses ordinary loop steps."""

    max_steps: int = 24
    max_total_tokens: int = 100_000
    max_wall_time_seconds: float = 300.0
    max_estimated_cost_usd: Decimal | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.max_steps, "max_steps"),
            (self.max_total_tokens, "max_total_tokens"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            not isinstance(self.max_wall_time_seconds, (int, float))
            or isinstance(self.max_wall_time_seconds, bool)
            or not math.isfinite(self.max_wall_time_seconds)
            or self.max_wall_time_seconds <= 0
        ):
            raise ValueError("max_wall_time_seconds must be finite and positive")
        object.__setattr__(
            self, "max_wall_time_seconds", float(self.max_wall_time_seconds)
        )
        if self.max_estimated_cost_usd is not None and (
            not isinstance(self.max_estimated_cost_usd, Decimal)
            or not self.max_estimated_cost_usd.is_finite()
            or self.max_estimated_cost_usd < 0
        ):
            raise ValueError(
                "max_estimated_cost_usd must be a finite non-negative Decimal or None"
            )


class LoopExitKind(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class LoopExit:
    run_id: str
    conversation_id: str
    kind: LoopExitKind
    reason: str
    created_at: datetime
    final_text: str | None = None
    steps: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)
    artifacts: tuple[ArtifactRef, ...] = ()
    artifact_deliveries: tuple[ArtifactDeliveryReceipt, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.run_id, "loop-exit run_id")
        _required_text(self.conversation_id, "loop-exit conversation_id")
        _required_text(self.reason, "loop-exit reason")
        if not isinstance(self.kind, LoopExitKind):
            raise TypeError("loop-exit kind must be LoopExitKind")
        _aware(self.created_at, "loop-exit created_at")
        if self.final_text is not None:
            _required_text(self.final_text, "loop-exit final_text")
        if self.kind is LoopExitKind.COMPLETED and self.final_text is None:
            raise ValueError("completed loop exit requires final text")
        if (
            not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or self.steps < 0
        ):
            raise ValueError("loop-exit steps must be a non-negative integer")
        if not isinstance(self.usage, ModelUsage):
            raise TypeError("loop-exit usage must be ModelUsage")
        artifacts = tuple(self.artifacts)
        deliveries = tuple(self.artifact_deliveries)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise TypeError("loop-exit artifacts must contain ArtifactRef records")
        if any(not isinstance(item, ArtifactDeliveryReceipt) for item in deliveries):
            raise TypeError(
                "loop-exit artifact_deliveries must contain receipt records"
            )
        if any(item.run_id != self.run_id for item in artifacts):
            raise ValueError("loop-exit artifacts must belong to its run")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "artifact_deliveries", deliveries)


@dataclass(frozen=True, slots=True)
class Transcript:
    """Canonical user, assistant, and tool messages persisted for one run.

    Prior conversation and system context may be visible to a model request but
    are never copied into this immutable per-run record.
    """

    run: RunInput
    messages: tuple[CanonicalMessage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunInput):
            raise TypeError("transcript run must be RunInput")
        messages = tuple(self.messages)
        if any(not isinstance(message, CanonicalMessage) for message in messages):
            raise TypeError("transcript messages must be CanonicalMessage records")
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class ConversationRun:
    """One inspectable run in an agent-scoped conversation."""

    turn_index: int
    transcript: Transcript
    result: LoopExit | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.turn_index, int)
            or isinstance(self.turn_index, bool)
            or self.turn_index < 0
        ):
            raise ValueError("conversation turn_index must be non-negative")
        if not isinstance(self.transcript, Transcript):
            raise TypeError("conversation transcript must be Transcript")
        if self.result is not None and not isinstance(self.result, LoopExit):
            raise TypeError("conversation result must be LoopExit or None")
        if self.result is not None and self.result.run_id != self.transcript.run.id:
            raise ValueError("conversation result must belong to its transcript")
