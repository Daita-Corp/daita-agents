"""Component-owned tests split from ``test_candidate_controls.py``."""

from __future__ import annotations

from tests.learning._candidate_control_support import (
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    AsyncMock,
    Decimal,
    LearningReviewResult,
    LearningReviewStatus,
    Path,
    cli,
    pytest,
)


def test_cli_requires_explicit_bounded_review_cost_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DAITA_CANDIDATE_REVIEW_MAX_COST_USD", raising=False)
    assert cli._candidate_review_cost_limit_from_environment() is None

    monkeypatch.setenv("DAITA_CANDIDATE_REVIEW_MAX_COST_USD", "0.05")
    assert cli._candidate_review_cost_limit_from_environment() == Decimal("0.05")

    monkeypatch.setenv("DAITA_CANDIDATE_REVIEW_MAX_COST_USD", "NaN")
    with pytest.raises(ValueError, match="finite"):
        cli._candidate_review_cost_limit_from_environment()


@pytest.mark.unit
async def test_headless_review_uses_bounded_reviewer_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ReviewAgent:
        async def review_learning_candidates(self) -> LearningReviewResult:
            return LearningReviewResult(status=LearningReviewStatus.NO_ELIGIBLE_RUNS)

        async def close(self) -> None:
            return None

    opened = AsyncMock(return_value=ReviewAgent())
    monkeypatch.setattr(cli.Agent, "open", opened)
    args = cli.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "memory",
            "review",
            "atlas",
            "--model",
            "openai:gpt-5.6-sol",
            "--cost-limit",
            "0.01",
        ]
    )

    result = await cli._execute(args)

    assert isinstance(result, dict)
    assert result["status"] == "no_eligible_runs"
    open_call = opened.await_args
    assert open_call is not None
    kwargs = open_call.kwargs
    assert (
        kwargs["reviewer_profile"].max_output_tokens
        == LEARNING_REVIEW_MAX_TOTAL_TOKENS // 4
    )
    assert kwargs["reviewer_max_estimated_cost_usd"] == Decimal("0.01")
