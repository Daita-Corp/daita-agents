"""Component-owned tests split from ``test_candidate_controls.py``."""

from __future__ import annotations

from tests.learning._candidate_control_support import (
    AsyncMock,
    Decimal,
    LearningReviewResult,
    LearningReviewStatus,
    Path,
    PresentationController,
    _route,
    _write_learning_review_result,
    io,
    pytest,
    workspace_for,
)


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

    controller = PresentationController(root=None, workspace=workspace_for(None))
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

    controller = PresentationController(root=None, workspace=workspace_for(None))
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

    controller = PresentationController(root=None, workspace=workspace_for(None))
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
        workspace=workspace_for(Path("/tmp/daita-test-root")),
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
