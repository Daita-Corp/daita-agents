"""Component-owned tests split from ``test_review.py``."""

from __future__ import annotations

from tests.learning._candidate_support import (
    Agent,
    MockModelProvider,
    PresentationController,
    _ids,
    _response,
    _review_response,
    _write_memory_surface,
    io,
    workspace_for,
)


async def test_memory_terminal_surface_lists_shows_and_rejects_one_candidate(
    tmp_path,
):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Our fiscal year begins in February.")]
    )
    agent = await Agent.create(
        "phase4-terminal",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        candidate = (await agent.review_learning_candidates()).candidates[0]
        output = io.StringIO()
        await _write_memory_surface(agent, "", output)
        assert "Pending candidates:" in output.getvalue()
        assert candidate.candidate.id in output.getvalue()

        controller = PresentationController(
            root=tmp_path, workspace=workspace_for(tmp_path)
        )
        controller.agent = agent
        shown = await controller.dispatch_command(
            f"/memory show {candidate.candidate.id}"
        )
        assert "Learning candidate:" in shown.message

        rejected = await controller.dispatch_command(
            f"/memory reject {candidate.candidate.id}"
        )
        assert "rejected" in rejected.message
    finally:
        await agent.close()
