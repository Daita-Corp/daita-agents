from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from daita.loop.models import (
    LoopExit,
    LoopExitKind,
    LoopPhase,
    LoopState,
    Readiness,
    Turn,
)

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def test_loop_state_keeps_phase_and_budget_counters_separate() -> None:
    state = LoopState(
        phase=LoopPhase.AWAITING_MODEL,
        turn_count=2,
        action_count=1,
        repair_count=1,
        identical_failure_count=0,
        observation_characters=40,
        input_tokens=10,
        output_tokens=4,
        estimated_cost_usd=Decimal("0.01"),
        no_progress_fingerprints=("sha256:abc",),
    )

    assert state.phase is LoopPhase.AWAITING_MODEL
    assert state.turn_count == 2
    assert state.no_progress_fingerprints == ("sha256:abc",)

    with pytest.raises(ValueError, match="non-negative"):
        LoopState(turn_count=-1)

    with pytest.raises(TypeError, match="sequence"):
        LoopState(no_progress_fingerprints="sha256:not-a-sequence")  # type: ignore[arg-type]


def test_turn_requires_positive_number_and_stable_operation_linkage() -> None:
    turn = Turn(
        id="turn-1",
        operation_id="op-1",
        number=1,
        created_at=NOW,
    )
    assert turn.operation_id == "op-1"

    with pytest.raises(ValueError, match="positive"):
        Turn(id="turn-0", operation_id="op-1", number=0, created_at=NOW)


def test_readiness_and_exit_records_fail_closed() -> None:
    allowed = Readiness(
        allowed=True,
        code="ready",
        message="Current evidence supports the answer.",
        evaluated_at=NOW,
    )
    failed = LoopExit(
        operation_id="op-1",
        kind=LoopExitKind.FAILED,
        reason="turn_budget_exhausted",
        created_at=NOW,
    )

    assert allowed.missing_facts == ()
    assert failed.final_text is None

    with pytest.raises(ValueError, match="missing facts"):
        Readiness(
            allowed=True,
            code="ready",
            message="Contradictory.",
            missing_facts=("evidence",),
            evaluated_at=NOW,
        )

    with pytest.raises(TypeError, match="sequence"):
        Readiness(
            allowed=False,
            code="not_ready",
            message="Missing evidence.",
            missing_facts="evidence",  # type: ignore[arg-type]
            evaluated_at=NOW,
        )

    with pytest.raises(ValueError, match="final text"):
        LoopExit(
            operation_id="op-1",
            kind=LoopExitKind.COMPLETED,
            reason="completed",
            created_at=NOW,
        )
