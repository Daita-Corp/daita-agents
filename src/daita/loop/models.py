"""Small records used by the transcript-driven agent loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ..artifacts.models import ArtifactDeliveryReceipt, ArtifactRef
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelUsage,
    TextBlock,
    ToolResultBlock,
)


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
    max_tool_calls_per_response: int = 16
    max_tool_calls_per_run: int = 64
    max_tool_result_bytes: int = 256 * 1_024
    max_tool_result_depth: int = 16
    max_parallel_reads: int = 8
    max_parallel_reads_per_source: int = 4
    max_context_evidence_bytes: int = 512 * 1_024
    side_effect_recovery_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.max_steps, "max_steps"),
            (self.max_total_tokens, "max_total_tokens"),
            (self.max_tool_calls_per_response, "max_tool_calls_per_response"),
            (self.max_tool_calls_per_run, "max_tool_calls_per_run"),
            (self.max_tool_result_bytes, "max_tool_result_bytes"),
            (self.max_tool_result_depth, "max_tool_result_depth"),
            (self.max_parallel_reads, "max_parallel_reads"),
            (self.max_parallel_reads_per_source, "max_parallel_reads_per_source"),
            (self.max_context_evidence_bytes, "max_context_evidence_bytes"),
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
        if self.max_tool_calls_per_response > self.max_tool_calls_per_run:
            raise ValueError(
                "max_tool_calls_per_response cannot exceed max_tool_calls_per_run"
            )
        if self.max_parallel_reads_per_source > self.max_parallel_reads:
            raise ValueError(
                "max_parallel_reads_per_source cannot exceed max_parallel_reads"
            )
        if (
            not isinstance(self.side_effect_recovery_timeout_seconds, (int, float))
            or isinstance(self.side_effect_recovery_timeout_seconds, bool)
            or not math.isfinite(self.side_effect_recovery_timeout_seconds)
            or not 0 < float(self.side_effect_recovery_timeout_seconds) <= 60
        ):
            raise ValueError(
                "side_effect_recovery_timeout_seconds must be positive and at most 60"
            )
        object.__setattr__(
            self,
            "side_effect_recovery_timeout_seconds",
            float(self.side_effect_recovery_timeout_seconds),
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


def validate_completed_transcript(
    transcript: Transcript,
    result: LoopExit,
) -> None:
    """Validate the exact per-run message state accepted as completion."""

    if not isinstance(transcript, Transcript):
        raise TypeError("completed transcript must be a Transcript")
    if not isinstance(result, LoopExit):
        raise TypeError("completed transcript result must be a LoopExit")
    if result.kind is not LoopExitKind.COMPLETED:
        raise ValueError("completed transcript validation requires a completed exit")
    if result.run_id != transcript.run.id:
        raise ValueError("completed transcript result belongs to another run")
    if result.conversation_id != (transcript.run.conversation_id or transcript.run.id):
        raise ValueError("completed transcript result belongs to another conversation")

    messages = transcript.messages
    if not messages or messages[0].role is not MessageRole.USER:
        raise ValueError("completed transcript must begin with its user message")
    if any(message.role is MessageRole.USER for message in messages[1:]):
        raise ValueError("completed transcript must contain exactly one user message")

    position = 1
    while position < len(messages):
        assistant = messages[position]
        if assistant.role is not MessageRole.ASSISTANT:
            raise ValueError("completed transcript expected an assistant message")
        position += 1
        if assistant.tool_calls:
            for call in assistant.tool_calls:
                if position >= len(messages):
                    raise ValueError(
                        "completed transcript is missing an ordered tool result"
                    )
                tool_message = messages[position]
                if (
                    tool_message.role is not MessageRole.TOOL
                    or len(tool_message.content) != 1
                    or not isinstance(tool_message.content[0], ToolResultBlock)
                    or tool_message.content[0].call_id != call.id
                ):
                    raise ValueError(
                        "completed transcript has an invalid ordered tool result"
                    )
                position += 1
            continue

        if position != len(messages):
            raise ValueError(
                "completed transcript has messages after its final assistant"
            )
        if (
            len(assistant.content) != 1
            or not isinstance(assistant.content[0], TextBlock)
            or assistant.content[0].text != result.final_text
        ):
            raise ValueError(
                "completed transcript final assistant must match LoopExit text"
            )
        return

    raise ValueError("completed transcript must end with final assistant text")


class ToolBatchInterruption(str, Enum):
    CANCELLED = "cancelled"
    DEADLINE = "deadline"


class ToolBatchCertainty(str, Enum):
    DEFINITE = "definite"
    OUTCOME_UNKNOWN = "outcome_unknown"


@dataclass(frozen=True, slots=True)
class ToolBatchOutcome:
    """One ordered result for every call plus any bounded interruption state."""

    ordered_results: tuple[ToolResultBlock, ...]
    interruption_kind: ToolBatchInterruption | None = None
    outcome_certainty: ToolBatchCertainty = ToolBatchCertainty.DEFINITE

    def __post_init__(self) -> None:
        results = tuple(self.ordered_results)
        if any(not isinstance(item, ToolResultBlock) for item in results):
            raise TypeError("tool batch outcome requires tool-result records")
        if self.interruption_kind is not None and not isinstance(
            self.interruption_kind, ToolBatchInterruption
        ):
            raise TypeError("tool batch interruption kind is invalid")
        if not isinstance(self.outcome_certainty, ToolBatchCertainty):
            raise TypeError("tool batch outcome certainty is invalid")
        if (
            self.interruption_kind is None
            and self.outcome_certainty is ToolBatchCertainty.OUTCOME_UNKNOWN
        ):
            raise ValueError("unknown tool outcome requires an interruption")
        object.__setattr__(self, "ordered_results", results)

    def __iter__(self):
        return iter(self.ordered_results)

    def __len__(self) -> int:
        return len(self.ordered_results)

    def __getitem__(self, index: int) -> ToolResultBlock:
        return self.ordered_results[index]


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
