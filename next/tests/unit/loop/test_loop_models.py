from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from daita._json import FrozenJsonObject
from daita.loop.models import (
    LoopBudgets,
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


def test_loop_budgets_bound_repairs_and_identical_failures() -> None:
    budgets = LoopBudgets(
        max_turns=6,
        max_actions=8,
        max_repairs=3,
        max_identical_failures=2,
        max_observation_characters=4096,
        max_total_tokens=2048,
        max_wall_time_seconds=30.0,
        task_timeout_seconds=5.0,
        max_estimated_cost_usd=Decimal("0.25"),
    )

    assert budgets.max_turns == 6
    assert budgets.max_actions == 8
    assert budgets.max_repairs == 3
    assert budgets.max_identical_failures == 2
    assert budgets.max_observation_characters == 4096
    assert budgets.max_total_tokens == 2048
    assert budgets.max_wall_time_seconds == 30.0
    assert budgets.task_timeout_seconds == 5.0
    assert budgets.max_estimated_cost_usd == Decimal("0.25")

    with pytest.raises(ValueError, match="positive"):
        LoopBudgets(max_repairs=0)

    with pytest.raises(ValueError, match="positive"):
        LoopBudgets(max_identical_failures=0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_turns", 0),
        ("max_actions", False),
        ("max_observation_characters", -1),
        ("max_total_tokens", 0),
        ("max_wall_time_seconds", 0.0),
        ("max_wall_time_seconds", float("inf")),
        ("task_timeout_seconds", -1.0),
        ("task_timeout_seconds", float("nan")),
    ],
)
def test_loop_budget_limits_must_be_finite_and_positive(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="positive|finite"):
        LoopBudgets(**{field_name: value})  # type: ignore[arg-type]


def test_optional_cost_budget_requires_a_finite_non_negative_decimal() -> None:
    assert LoopBudgets(max_estimated_cost_usd=None).max_estimated_cost_usd is None
    assert LoopBudgets(max_estimated_cost_usd=Decimal("0")).max_estimated_cost_usd == 0

    with pytest.raises(TypeError, match="Decimal"):
        LoopBudgets(max_estimated_cost_usd=0.1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite|non-negative"):
        LoopBudgets(max_estimated_cost_usd=Decimal("NaN"))

    with pytest.raises(ValueError, match="finite|non-negative"):
        LoopBudgets(max_estimated_cost_usd=Decimal("-0.01"))


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
    source_details = {
        "required_citations": [
            {
                "evidence_id": "evidence-1",
                "citation": "[evidence:evidence-1]",
            }
        ]
    }
    allowed = Readiness(
        allowed=True,
        code="ready",
        message="Current evidence supports the answer.",
        evaluated_at=NOW,
        repair_details=source_details,
    )
    failed = LoopExit(
        operation_id="op-1",
        kind=LoopExitKind.FAILED,
        reason="turn_budget_exhausted",
        created_at=NOW,
        post_operation_notices=("learning.correction_failed",),
    )

    assert allowed.missing_facts == ()
    source_details["required_citations"][0]["citation"] = "mutated"
    repair_details = allowed.repair_details
    assert isinstance(repair_details, FrozenJsonObject)
    assert repair_details.to_dict() == {
        "required_citations": [
            {
                "citation": "[evidence:evidence-1]",
                "evidence_id": "evidence-1",
            }
        ]
    }
    assert failed.final_text is None
    assert failed.post_operation_notices == ("learning.correction_failed",)

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

    with pytest.raises(ValueError, match="duplicated"):
        LoopExit(
            operation_id="op-1",
            kind=LoopExitKind.FAILED,
            reason="failed",
            created_at=NOW,
            post_operation_notices=("learning.failed", "learning.failed"),
        )


@pytest.mark.parametrize(
    ("code", "message", "missing_facts"),
    [
        ("x" * 129, "Missing evidence.", ()),
        ("not_ready", "x" * 513, ()),
        ("not_ready", "Missing evidence.", tuple("x" for _ in range(17))),
        ("not_ready", "Missing evidence.", ("x" * 257,)),
        (
            "not_ready",
            "Missing evidence.",
            tuple("x" * 256 for _ in range(16)),
        ),
    ],
)
def test_readiness_correction_fields_are_bounded(
    code: str,
    message: str,
    missing_facts: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="bounded"):
        Readiness(
            allowed=False,
            code=code,
            message=message,
            missing_facts=missing_facts,
            evaluated_at=NOW,
        )


def test_readiness_repair_details_are_strict_bounded_json() -> None:
    with pytest.raises(TypeError, match="JSON object"):
        Readiness(
            allowed=False,
            code="not_ready",
            message="Missing evidence.",
            evaluated_at=NOW,
            repair_details=["not", "an", "object"],  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="bounded"):
        Readiness(
            allowed=False,
            code="not_ready",
            message="Missing evidence.",
            evaluated_at=NOW,
            repair_details={"detail": "x" * 4_096},
        )
