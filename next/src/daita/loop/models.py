"""Canonical records owned by the generic agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class LoopPhase(str, Enum):
    CREATED = "created"
    PREPARING_CONTEXT = "preparing_context"
    AWAITING_MODEL = "awaiting_model"
    VALIDATING_ACTION = "validating_action"
    AWAITING_EXECUTION = "awaiting_execution"
    AWAITING_APPROVAL = "awaiting_approval"
    OBSERVING = "observing"
    SYNTHESIZING = "synthesizing"
    TERMINAL = "terminal"


class LoopExitKind(str, Enum):
    COMPLETED = "completed"
    WAITING = "waiting"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class LoopState:
    phase: LoopPhase = LoopPhase.CREATED
    turn_count: int = 0
    action_count: int = 0
    repair_count: int = 0
    identical_failure_count: int = 0
    observation_characters: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: Decimal = Decimal("0")
    waiting_approval_id: str | None = None
    interruption_reason: str | None = None
    final_answer_candidate: str | None = None
    no_progress_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.phase, LoopPhase):
            raise TypeError("loop phase must be a LoopPhase")
        counters = (
            self.turn_count,
            self.action_count,
            self.repair_count,
            self.identical_failure_count,
            self.observation_characters,
            self.input_tokens,
            self.output_tokens,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counters
        ):
            raise ValueError("loop counters must be non-negative integers")
        if not isinstance(self.estimated_cost_usd, Decimal):
            raise TypeError("estimated_cost_usd must be a Decimal")
        if not self.estimated_cost_usd.is_finite() or self.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be finite and non-negative")
        for value, name in (
            (self.waiting_approval_id, "waiting_approval_id"),
            (self.interruption_reason, "interruption_reason"),
            (self.final_answer_candidate, "final_answer_candidate"),
        ):
            if value is not None:
                _required_text(value, name)
        if isinstance(self.no_progress_fingerprints, str):
            raise TypeError("no-progress fingerprints must be a sequence of strings")
        fingerprints = tuple(self.no_progress_fingerprints)
        for fingerprint in fingerprints:
            _required_text(fingerprint, "no-progress fingerprint")
        object.__setattr__(self, "no_progress_fingerprints", fingerprints)


@dataclass(frozen=True, slots=True)
class Turn:
    id: str
    operation_id: str
    number: int
    created_at: datetime
    model_request_id: str | None = None
    model_response_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "turn id")
        _required_text(self.operation_id, "turn operation_id")
        if (
            not isinstance(self.number, int)
            or isinstance(self.number, bool)
            or self.number < 1
        ):
            raise ValueError("turn number must be a positive integer")
        _aware(self.created_at, "turn created_at")
        if self.model_request_id is not None:
            _required_text(self.model_request_id, "turn model_request_id")
        if self.model_response_id is not None:
            _required_text(self.model_response_id, "turn model_response_id")


@dataclass(frozen=True, slots=True)
class Readiness:
    allowed: bool
    code: str
    message: str
    evaluated_at: datetime
    missing_facts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise TypeError("readiness allowed must be a boolean")
        _required_text(self.code, "readiness code")
        _required_text(self.message, "readiness message")
        _aware(self.evaluated_at, "readiness evaluated_at")
        if isinstance(self.missing_facts, str):
            raise TypeError("readiness missing facts must be a sequence of strings")
        missing_facts = tuple(self.missing_facts)
        for fact in missing_facts:
            _required_text(fact, "readiness missing fact")
        if self.allowed and missing_facts:
            raise ValueError("allowed readiness cannot contain missing facts")
        object.__setattr__(self, "missing_facts", missing_facts)


@dataclass(frozen=True, slots=True)
class LoopExit:
    operation_id: str
    kind: LoopExitKind
    reason: str
    created_at: datetime
    final_text: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "loop-exit operation_id")
        if not isinstance(self.kind, LoopExitKind):
            raise TypeError("loop-exit kind must be a LoopExitKind")
        _required_text(self.reason, "loop-exit reason")
        _aware(self.created_at, "loop-exit created_at")
        if self.final_text is not None:
            _required_text(self.final_text, "loop-exit final text")
        if self.kind is LoopExitKind.COMPLETED and self.final_text is None:
            raise ValueError("completed loop exit requires final text")
