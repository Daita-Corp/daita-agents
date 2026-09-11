"""Component-owned tests split from ``test_review.py``."""

from __future__ import annotations

from tests.learning._candidate_support import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    LEARNING_CANDIDATE_MAX_SUPPORTING_RUNS,
    LEARNING_REVIEW_MAX_MESSAGES,
    LEARNING_REVIEW_MAX_MODEL_CALLS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_RUNS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_TRANSCRIPT_UTF8_BYTES,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
    Agent,
    LearningReviewStatus,
    MockModelProvider,
    _ids,
    _response,
    _review_response,
    workspace_for,
)


def test_candidate_review_fixed_bounds_are_deliberately_small():
    assert LEARNING_CANDIDATE_MAX_RECORDS == 64
    assert LEARNING_REVIEW_MAX_RUNS == 8
    assert LEARNING_REVIEW_MAX_MESSAGES == 40
    assert LEARNING_REVIEW_MAX_TRANSCRIPT_UTF8_BYTES == 24_000
    assert LEARNING_REVIEW_MAX_PROPOSALS == 4
    assert LEARNING_CANDIDATE_MAX_SUPPORTING_RUNS == 8
    assert LEARNING_REVIEW_MAX_MODEL_CALLS == 1
    assert LEARNING_REVIEW_MAX_WALL_TIME_SECONDS == 60.0
    assert LEARNING_REVIEW_MAX_TOTAL_TOKENS == 24_000


async def test_assistant_only_and_transient_proposals_are_deterministically_dropped(
    tmp_path,
):
    foreground = MockModelProvider([_response("Booked revenue means today's total.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Booked revenue is today's total.")]
    )
    agent = await Agent.create(
        "phase4-grounding",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("What is booked revenue?")
        review = await agent.review_learning_candidates()
        assert review.status is LearningReviewStatus.COMPLETED
        assert review.candidates == ()
        assert await agent.list_learning_candidates() == ()
    finally:
        await agent.close()
