"""Component-owned tests split from ``test_review.py``."""

from __future__ import annotations

from tests.learning._candidate_support import (
    EAGER_LIMITS,
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    DocumentCandidateContent,
    FinishReason,
    LearningCandidateError,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    Mapping,
    MockModelProvider,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    _ids,
    _response,
    _review_response,
    _sqlite_file,
    json,
    pytest,
    workspace_for,
)


async def test_acceptance_uses_fresh_foreground_approval_and_marks_only_on_success(
    tmp_path,
):
    content = "Booked revenue excludes completed refunds."
    foreground = MockModelProvider(
        [
            _response("Understood."),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="write-1",
                        name="memory_set",
                        arguments={"target": "memory", "content": content},
                    ),
                ),
            ),
            _response("Saved after exact approval."),
        ]
    )
    reviewer = MockModelProvider([_review_response("run-1", text=content)])
    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "phase4-accept",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        limits=EAGER_LIMITS,
        reviewer_model=reviewer,
        approval_handler=approve,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that booked revenue excludes completed refunds.")
        review = await agent.review_learning_candidates()
        candidate_id = review.candidates[0].candidate.id

        result = await agent.accept_learning_candidate(candidate_id)
        assert result.kind.value == "completed"
        assert await agent.read_memory() == content
        accepted = await agent.read_learning_candidate(candidate_id)
        assert accepted is not None
        assert accepted.status is LearningCandidateStatus.ACCEPTED
        assert len(approvals) == 1
        assert approvals[0].tool_name == "memory_set"
        acceptance_prompt = "\n".join(
            block.text
            for message in foreground.requests[1].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "<untrusted-learning-candidate>" in acceptance_prompt
        assert "not active memory" in acceptance_prompt
    finally:
        await agent.close()


async def test_denied_acceptance_has_no_active_effect_and_remains_awaiting(tmp_path):
    content = "Booked revenue excludes completed refunds."
    foreground = MockModelProvider(
        [
            _response("Understood."),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="write-1",
                        name="memory_set",
                        arguments={"target": "memory", "content": content},
                    ),
                ),
            ),
            _response("Not saved."),
        ]
    )
    reviewer = MockModelProvider([_review_response("run-1", text=content)])

    async def deny(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.DENY

    agent = await Agent.create(
        "phase4-deny",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        approval_handler=deny,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that booked revenue excludes completed refunds.")
        candidate_id = (
            (await agent.review_learning_candidates()).candidates[0].candidate.id
        )
        await agent.accept_learning_candidate(candidate_id)
        assert await agent.read_memory() == ""
        view = await agent.read_learning_candidate(candidate_id)
        assert view is not None
        assert view.status is LearningCandidateStatus.AWAITING_REVIEW
    finally:
        await agent.close()


async def test_acceptance_without_approval_handler_fails_closed(tmp_path):
    content = "Booked revenue excludes completed refunds."
    foreground = MockModelProvider(
        [
            _response("Understood."),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="write-1",
                        name="memory_set",
                        arguments={"target": "memory", "content": content},
                    ),
                ),
            ),
            _response("Not saved."),
        ]
    )
    reviewer = MockModelProvider([_review_response("run-1", text=content)])
    agent = await Agent.create(
        "phase4-missing-approval",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        limits=EAGER_LIMITS,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that booked revenue excludes completed refunds.")
        candidate_id = (
            (await agent.review_learning_candidates()).candidates[0].candidate.id
        )
        result = await agent.accept_learning_candidate(candidate_id)
        transcript = await agent.transcript(result.run_id)
        errors = [
            block.output.get("error")
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.is_error
        ]
        assert any(
            isinstance(error, Mapping) and error.get("code") == "approval_required"
            for error in errors
        )
        assert await agent.read_memory() == ""
        view = await agent.read_learning_candidate(candidate_id)
        assert view is not None
        assert view.status is LearningCandidateStatus.AWAITING_REVIEW
    finally:
        await agent.close()


async def test_acceptance_run_cannot_mutate_content_other_than_selected_candidate(
    tmp_path,
):
    selected = "Booked revenue excludes completed refunds."
    foreground = MockModelProvider(
        [
            _response("Understood."),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="write-other",
                        name="memory_set",
                        arguments={
                            "target": "memory",
                            "content": "Unrelated assistant-authored content.",
                        },
                    ),
                ),
            ),
            _response("No change."),
        ]
    )
    reviewer = MockModelProvider([_review_response("run-1", text=selected)])
    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "phase4-mismatch",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        limits=EAGER_LIMITS,
        reviewer_model=reviewer,
        approval_handler=approve,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that booked revenue excludes completed refunds.")
        candidate_id = (
            (await agent.review_learning_candidates()).candidates[0].candidate.id
        )
        result = await agent.accept_learning_candidate(candidate_id)
        transcript = await agent.transcript(result.run_id)
        errors = [
            block.output.get("error")
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.is_error
        ]
        assert any(
            isinstance(error, Mapping) and error.get("code") == "candidate_mismatch"
            for error in errors
        )
        assert approvals == []
        assert await agent.read_memory() == ""
        view = await agent.read_learning_candidate(candidate_id)
        assert view is not None
        assert view.status is LearningCandidateStatus.AWAITING_REVIEW
    finally:
        await agent.close()


async def test_edit_reject_and_clear_are_individual_and_bounded(tmp_path):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Use fiscal years beginning in February.")]
    )
    agent = await Agent.create(
        "phase4-edit",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        created = (await agent.review_learning_candidates()).candidates[0]
        edited = await agent.edit_learning_candidate(
            created.candidate.id,
            DocumentCandidateContent("Our fiscal year begins in February."),
        )
        assert isinstance(edited.candidate.content, DocumentCandidateContent)
        assert edited.candidate.content.text == "Our fiscal year begins in February."
        assert (
            edited.candidate.candidate_fingerprint
            != created.candidate.candidate_fingerprint
        )
        with pytest.raises(LearningCandidateError, match="sensitive data"):
            await agent.edit_learning_candidate(
                created.candidate.id,
                DocumentCandidateContent("API key is EDITED_CANDIDATE_SECRET_123456."),
            )
        unchanged = await agent.read_learning_candidate(created.candidate.id)
        assert unchanged is not None
        assert unchanged.candidate.content == edited.candidate.content

        rejected = await agent.reject_learning_candidate(
            created.candidate.id,
            LearningCandidateRejectionReason.USER_DECLINED,
        )
        assert rejected.status is LearningCandidateStatus.REJECTED
        assert await agent.clear_rejected_learning_candidates() == 1
        assert await agent.list_learning_candidates() == ()
    finally:
        await agent.close()


async def test_source_scoped_candidate_cannot_be_accepted_through_another_source(
    tmp_path,
):
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    _sqlite_file(first_path)
    _sqlite_file(second_path)
    foreground = MockModelProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="catalog-1",
                        name="catalog_search",
                        arguments={"query": "invoices"},
                    ),
                ),
            ),
            _response("Validated the reusable procedure."),
        ]
    )
    reviewer = MockModelProvider([])
    agent = await Agent.create(
        "phase4-source-scope",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        source_a = await agent.attach_sqlite(first_path, name="first")
        source_b = await agent.attach_sqlite(second_path, name="second")
        await agent.run(
            "Run and retain a reusable monthly invoice procedure.",
            source_scope_ids=(source_a.id,),
        )
        reviewer.replace_script(
            (
                _response(
                    json.dumps(
                        {
                            "candidates": [
                                {
                                    "target": "skill",
                                    "source_ids": [source_a.id],
                                    "supporting_run_ids": ["run-1"],
                                    "content": {
                                        "action": "save",
                                        "name": "monthly-invoices",
                                        "description": (
                                            "Use for validated monthly invoice review."
                                        ),
                                        "instructions": (
                                            "# Purpose\nReview monthly invoices.\n\n"
                                            "# Procedure\nInspect current catalog first.\n\n"
                                            "# Verification\nRequire a validated result."
                                        ),
                                    },
                                }
                            ]
                        }
                    )
                ),
            )
        )
        candidate = (await agent.review_learning_candidates()).candidates[0]
        assert candidate.candidate.source_ids == (source_a.id,)
        with pytest.raises(ValueError, match="bound source"):
            await agent.accept_learning_candidate(
                candidate.candidate.id,
                source_id=source_b.id,
            )
        assert await agent.read_skill("monthly-invoices") is None
    finally:
        await agent.close()
