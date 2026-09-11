"""Component-owned tests split from ``test_candidate_controls.py``."""

from __future__ import annotations

from tests.learning._candidate_control_support import (
    LEARNING_REVIEW_MAX_MODEL_CALLS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
    Agent,
    AgentConfig,
    Any,
    CandidateReviewMeasurement,
    Decimal,
    LearningReviewStatus,
    Path,
    PresentationController,
    SimpleNamespace,
    _route,
    cast,
    pytest,
    workspace_for,
)


async def test_embedded_reviewer_configuration_is_direct_and_token_bounded(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "bounded-reviewer",
        root=tmp_path,
        config=AgentConfig(model_route=_route()),
        reviewer_max_estimated_cost_usd=Decimal("0.05"),
        workspace=workspace_for(tmp_path),
    )
    try:
        reviewer = agent._embedded._candidate_reviewer
        assert reviewer._model is not None
        assert reviewer._model.provider_id == "openai:gpt-5.6-sol"
        assert reviewer._profile is not None
        assert (
            reviewer._profile.max_output_tokens == LEARNING_REVIEW_MAX_TOTAL_TOKENS // 4
        )
        assert reviewer._profile.supports_structured_output is True
    finally:
        await agent.close()


async def test_explicit_review_cost_authorization_uses_persisted_primary_route_once(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "on-demand-reviewer",
        root=tmp_path,
        config=AgentConfig(model_route=_route()),
        workspace=workspace_for(tmp_path),
    )
    try:
        disabled = await agent.review_learning_candidates()
        authorized = await agent.review_learning_candidates(
            max_estimated_cost_usd=Decimal("0.05"),
        )
        disabled_again = await agent.review_learning_candidates()

        assert disabled.status is LearningReviewStatus.DISABLED
        assert authorized.status is LearningReviewStatus.NO_ELIGIBLE_RUNS
        assert authorized.model_calls == 0
        assert disabled_again.status is LearningReviewStatus.DISABLED
        assert agent._embedded._candidate_reviewer.enabled is False
    finally:
        await agent.close()


@pytest.mark.unit
async def test_candidate_acceptance_controller_output_is_bounded_and_sanitized() -> (
    None
):
    unsafe = "accepted\x1b[31m\x00\n" + ("x" * 20_000)
    result = SimpleNamespace(
        final_text=unsafe,
        kind=SimpleNamespace(value="completed"),
        reason="done",
    )

    class Agent:
        async def accept_learning_candidate(self, candidate_id: str):
            assert candidate_id == "candidate-1"
            return result

    controller = PresentationController(root=None, workspace=workspace_for(None))
    controller.agent = Agent()  # type: ignore[assignment]
    rendered = (await controller.dispatch_command("/memory accept candidate-1")).message
    assert "\x1b" not in rendered
    assert "\x00" not in rendered
    assert len(rendered) <= 16_450


def test_candidate_review_measurement_accepts_exact_fixed_bounds() -> None:
    measurement = CandidateReviewMeasurement(
        proposed_candidates=LEARNING_REVIEW_MAX_PROPOSALS,
        accepted_candidates=2,
        rejected_candidates=2,
        false_positive_candidates=1,
        duplicate_candidates_suppressed=LEARNING_REVIEW_MAX_PROPOSALS,
        background_model_calls=LEARNING_REVIEW_MAX_MODEL_CALLS,
        background_total_tokens=LEARNING_REVIEW_MAX_TOTAL_TOKENS,
        background_duration_ms=int(LEARNING_REVIEW_MAX_WALL_TIME_SECONDS * 1_000),
    )

    assert measurement.proposed_candidates == LEARNING_REVIEW_MAX_PROPOSALS


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("proposed_candidates", LEARNING_REVIEW_MAX_PROPOSALS + 1),
        (
            "duplicate_candidates_suppressed",
            LEARNING_REVIEW_MAX_PROPOSALS + 1,
        ),
        ("background_model_calls", LEARNING_REVIEW_MAX_MODEL_CALLS + 1),
        ("background_total_tokens", LEARNING_REVIEW_MAX_TOTAL_TOKENS + 1),
        (
            "background_duration_ms",
            int(LEARNING_REVIEW_MAX_WALL_TIME_SECONDS * 1_000) + 1,
        ),
    ),
)
def test_candidate_review_measurement_rejects_values_above_fixed_bounds(
    field_name: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match="bound"):
        CandidateReviewMeasurement(**cast(Any, {field_name: value}))
