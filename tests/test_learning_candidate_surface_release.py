from __future__ import annotations

import io
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from daita import Agent, AgentConfig, cli
from daita.cli_text import _write_learning_review_result
from daita.evaluation import CandidateReviewMeasurement
from daita.learning_candidates import (
    LEARNING_REVIEW_MAX_MODEL_CALLS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
    LearningReviewResult,
    LearningReviewStatus,
)
from daita.llm.models import ModelProfile
from daita.llm.routing import (
    ModelRoute,
    ModelRouteCandidate,
    RetryPolicy,
)
from daita.tui.controller import PresentationController


def _profile(
    provider_id: str = "openai:gpt-5.6-sol",
    *,
    maximum_output: int = 128_000,
) -> ModelProfile:
    return ModelProfile(
        id=provider_id,
        context_window_tokens=1_050_000,
        max_output_tokens=maximum_output,
        supports_tools=True,
        supports_structured_output=True,
        supports_reasoning=True,
    )


def _route() -> ModelRoute:
    return ModelRoute(
        (
            ModelRouteCandidate(
                provider_id="openai:gpt-5.6-sol",
                profile=_profile(),
            ),
            ModelRouteCandidate(
                provider_id="gemini:gemini-3.6-flash",
                profile=_profile(
                    "gemini:gemini-3.6-flash",
                    maximum_output=65_536,
                ),
            ),
        ),
        retry_policy=RetryPolicy(attempts=5, backoff_seconds=1),
    )


async def test_embedded_reviewer_configuration_is_direct_and_token_bounded(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "bounded-reviewer",
        root=tmp_path,
        config=AgentConfig(model_route=_route()),
        reviewer_max_estimated_cost_usd=Decimal("0.05"),
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
async def test_terminal_review_prompts_for_one_call_authorization() -> None:
    class ReviewAgent:
        def __init__(self) -> None:
            self.cost_limits: list[Decimal | None] = []

        async def review_learning_candidates(
            self,
            *,
            max_estimated_cost_usd: Decimal | None = None,
        ) -> LearningReviewResult:
            self.cost_limits.append(max_estimated_cost_usd)
            return LearningReviewResult(
                status=(
                    LearningReviewStatus.DISABLED
                    if max_estimated_cost_usd is None
                    else LearningReviewStatus.NO_ELIGIBLE_RUNS
                )
            )

    agent = ReviewAgent()
    output = io.StringIO()

    controller = PresentationController(root=None)
    controller.agent = agent  # type: ignore[assignment]
    outcome = await controller.dispatch_command("/review")
    assert outcome.kind == "screen"
    assert outcome.screen == "review_cost"
    assert agent.cost_limits == [None]


@pytest.mark.unit
async def test_terminal_review_authorization_can_be_cancelled() -> None:
    class ReviewAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def review_learning_candidates(self) -> LearningReviewResult:
            self.calls += 1
            return LearningReviewResult(status=LearningReviewStatus.DISABLED)

    agent = ReviewAgent()
    output = io.StringIO()

    controller = PresentationController(root=None)
    controller.agent = agent  # type: ignore[assignment]
    outcome = await controller.dispatch_command("/review")
    assert outcome.kind == "screen"
    assert agent.calls == 1


@pytest.mark.unit
async def test_terminal_review_accepts_inline_one_call_cost_limit() -> None:
    class ReviewAgent:
        def __init__(self) -> None:
            self.cost_limits: list[Decimal] = []

        async def review_learning_candidates(
            self,
            *,
            max_estimated_cost_usd: Decimal,
        ) -> LearningReviewResult:
            self.cost_limits.append(max_estimated_cost_usd)
            return LearningReviewResult(status=LearningReviewStatus.NO_ELIGIBLE_RUNS)

    agent = ReviewAgent()
    output = io.StringIO()

    controller = PresentationController(root=None)
    controller.agent = agent  # type: ignore[assignment]
    outcome = await controller.dispatch_command("/review 0.02")
    assert outcome.kind == "notice"
    assert agent.cost_limits == [Decimal("0.02")]
    assert "Status: no_eligible_runs" in outcome.message


@pytest.mark.unit
def test_terminal_review_reports_unreadable_history_without_provider_blame() -> None:
    output = io.StringIO()

    _write_learning_review_result(
        LearningReviewResult(
            status=LearningReviewStatus.HISTORY_UNAVAILABLE,
            skipped_run_count=3,
        ),
        output,
    )

    text = output.getvalue()
    assert "Status: history_unavailable" in text
    assert "Skipped unreadable runs: 3" in text
    assert "provider_failed" not in text


@pytest.mark.unit
async def test_terminal_reopen_passes_explicit_reviewer_and_cost_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConfiguredAgent:
        name = "atlas"
        model_route = _route()

        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    configured = ConfiguredAgent()

    class ReplacementAgent:
        model_route = None

        async def list_sources(self) -> tuple[object, ...]:
            return ()

        async def close(self) -> None:
            return None

    replacement = ReplacementAgent()
    open_agent = AsyncMock(return_value=replacement)
    monkeypatch.setattr("daita.tui.controller.Agent.open", open_agent)
    controller = PresentationController(
        root=Path("/tmp/daita-test-root"),
        reviewer_max_estimated_cost_usd=Decimal("0.05"),
    )
    controller.agent = configured  # type: ignore[assignment]
    result = await controller.reopen_agent(observer=None, approval_handler=None)

    assert result is replacement
    assert configured.closed is True
    open_call = open_agent.await_args
    assert open_call is not None
    kwargs = open_call.kwargs
    assert "reviewer_model" not in kwargs
    assert "reviewer_profile" not in kwargs
    assert kwargs["reviewer_max_estimated_cost_usd"] == Decimal("0.05")


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


@pytest.mark.unit
async def test_candidate_acceptance_output_is_bounded_and_sanitized(
    capsys: pytest.CaptureFixture[str],
) -> None:
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

    assert await cli._handle_knowledge_chat_command(
        ["/memory", "accept", "candidate-1"],
        Agent(),  # type: ignore[arg-type]
    )
    legacy_output = capsys.readouterr().out
    assert "\x1b" not in legacy_output
    assert "\x00" not in legacy_output
    assert len(legacy_output) <= 16_400

    controller = PresentationController(root=None)
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


def test_readme_documents_explicit_inactive_review_lifecycle() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    normalized_words = normalized.replace("-", " ")

    assert "Candidate review is disabled by default." in readme
    assert "one tool free model request outside `AgentLoop`" in normalized_words
    assert "`/memory accept <id>` handles exactly one candidate" in readme
    assert "There is no bulk acceptance." in normalized
    assert "Daita performs no post-run review, auxiliary model call" not in readme
