from __future__ import annotations

import asyncio
import io
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    DocumentCandidateContent,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningReviewStatus,
)
from daita.cli_text import _write_memory_surface
from daita.tui.controller import PresentationController
from daita.evaluation import CandidateReviewMeasurement, CandidateReviewReport
from daita.learning_candidates import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    LEARNING_CANDIDATE_MAX_SUPPORTING_RUNS,
    LEARNING_REVIEW_MAX_MESSAGES,
    LEARNING_REVIEW_MAX_MODEL_CALLS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_RUNS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_TRANSCRIPT_UTF8_BYTES,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
    LearningCandidate,
    LearningCandidateError,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateTarget,
    OneShotCandidateReviewer,
    learning_candidate_content_from_mapping,
    learning_candidate_content_to_mapping,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite import SQLiteStateStore


class _BlockingReviewer:
    provider_id = "mock:blocking-reviewer"

    def __init__(self):
        self.started = asyncio.Event()
        self.requests: list[ModelRequest] = []

    @property
    def model_profile(self):
        return ModelProfile(
            id=self.provider_id,
            context_window_tokens=32_000,
            max_output_tokens=1_024,
            supports_structured_output=True,
        )

    def supports_request_policy(self, request):
        return isinstance(request, ModelRequest)

    def has_complete_pricing(self, request):
        return False

    async def generate(self, request):
        self.requests.append(request)
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


def _ids():
    counters: defaultdict[str, int] = defaultdict(int)

    def value(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]}"

    return value


def _response(text: str, *, usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text=text,
        usage=usage or ModelUsage(),
    )


def _review_response(run_id: str, *, text: str) -> ModelResponse:
    return _response(
        json.dumps(
            {
                "candidates": [
                    {
                        "target": "memory",
                        "source_ids": [],
                        "supporting_run_ids": [run_id],
                        "content": {"text": text},
                    }
                ]
            }
        )
    )


def _sqlite_file(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)")


def _stored_candidate(index: int, *, agent_id: str = "agent-one"):
    digest = f"{index + 1:064x}"
    run_id = f"run-{index}"
    reference = LearningCandidateRunReference(run_id, digest)
    candidate = LearningCandidate(
        id=f"candidate-{index}",
        agent_id=agent_id,
        target=LearningCandidateTarget.MEMORY,
        content=DocumentCandidateContent(f"Durable definition {index}."),
        source_ids=(),
        reviewed_runs=(reference,),
        supporting_run_ids=(run_id,),
        review_fingerprint=digest,
        artifact_state_sha256="a" * 64,
        catalog_revisions=(),
        candidate_fingerprint=digest,
        status=LearningCandidateStatus.AWAITING_REVIEW,
        created_at=datetime(2026, 7, 28, tzinfo=UTC),
        updated_at=datetime(2026, 7, 28, tzinfo=UTC),
    )
    stamp = LearningCandidateReviewStamp(
        run_id=run_id,
        transcript_sha256=digest,
        artifact_state_sha256="a" * 64,
        catalog_state_sha256="b" * 64,
    )
    return candidate, stamp


def test_phase4_fixed_bounds_are_deliberately_small():
    assert LEARNING_CANDIDATE_MAX_RECORDS == 64
    assert LEARNING_REVIEW_MAX_RUNS == 8
    assert LEARNING_REVIEW_MAX_MESSAGES == 40
    assert LEARNING_REVIEW_MAX_TRANSCRIPT_UTF8_BYTES == 24_000
    assert LEARNING_REVIEW_MAX_PROPOSALS == 4
    assert LEARNING_CANDIDATE_MAX_SUPPORTING_RUNS == 8
    assert LEARNING_REVIEW_MAX_MODEL_CALLS == 1
    assert LEARNING_REVIEW_MAX_WALL_TIME_SECONDS == 60.0
    assert LEARNING_REVIEW_MAX_TOTAL_TOKENS == 24_000


@pytest.mark.parametrize("target", ["memory", "user"])
def test_document_candidate_content_is_immutable(target: str):
    content = DocumentCandidateContent("Durable definition.")
    with pytest.raises(AttributeError):
        content.text = target  # type: ignore[misc]


async def test_candidate_storage_round_trips_exactly_and_enforces_64_and_owner(
    tmp_path,
):
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    for index in range(LEARNING_CANDIDATE_MAX_RECORDS):
        candidate, stamp = _stored_candidate(index)
        inserted = await store.save_learning_candidate_review(
            "agent-one",
            stamps=(stamp,),
            candidates=(candidate,),
        )
        assert inserted == (candidate,)
    values = await store.list_learning_candidates("agent-one")
    assert len(values) == LEARNING_CANDIDATE_MAX_RECORDS
    assert await store.load_learning_candidate("agent-one", "candidate-0") == values[0]

    extra, extra_stamp = _stored_candidate(LEARNING_CANDIDATE_MAX_RECORDS)
    with pytest.raises(LearningCandidateError, match="capacity"):
        await store.save_learning_candidate_review(
            "agent-one",
            stamps=(extra_stamp,),
            candidates=(extra,),
        )

    foreign, foreign_stamp = _stored_candidate(0, agent_id="agent-two")
    with pytest.raises(LearningCandidateError, match="another agent"):
        await store.save_learning_candidate_review(
            "agent-one",
            stamps=(foreign_stamp,),
            candidates=(foreign,),
        )


async def test_explicit_review_creates_only_inactive_idempotent_candidate(tmp_path):
    foreground = MockModelProvider(
        [
            _response("Understood."),
            _response("Ordinary answer."),
        ]
    )
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Booked revenue excludes completed refunds.")]
    )
    agent = await Agent.create(
        "phase4-review",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        first = await agent.run(
            "Remember that booked revenue excludes completed refunds."
        )
        assert first.run_id == "run-1"
        assert await agent.read_memory() == ""

        review = await agent.review_learning_candidates()
        assert review.status is LearningReviewStatus.COMPLETED
        assert review.model_calls == 1
        assert len(review.candidates) == 1
        candidate = review.candidates[0]
        assert candidate.status is LearningCandidateStatus.AWAITING_REVIEW
        assert candidate.candidate.supporting_run_ids == ("run-1",)
        assert await agent.read_memory() == ""

        repeated = await agent.review_learning_candidates()
        assert repeated.status is LearningReviewStatus.ALREADY_REVIEWED
        assert repeated.model_calls == 0
        assert len(reviewer.requests) == 1

        ordinary = await agent.run("What is booked revenue?")
        assert ordinary.final_text == "Ordinary answer."
        still_inactive = await agent.read_learning_candidate(candidate.candidate.id)
        assert still_inactive is not None
        assert still_inactive.status is LearningCandidateStatus.AWAITING_REVIEW
        ordinary_prompt = "\n".join(
            block.text
            for message in foreground.requests[1].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "<untrusted-learning-candidate>" not in ordinary_prompt
        assert "Booked revenue excludes completed refunds." not in ordinary_prompt
    finally:
        await agent.close()


async def test_review_skips_unreadable_history_and_reviews_new_compatible_runs(
    tmp_path,
):
    ids = _ids()
    foreground = MockModelProvider(
        [
            _response("Historical answer."),
            _response("Current answer."),
        ]
    )
    reviewer = MockModelProvider(
        [_review_response("run-2", text="Booked revenue excludes completed refunds.")]
    )
    agent = await Agent.create(
        "phase4-unreadable-history",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=ids,
    )
    await agent.run("Historical request.")
    await agent.close()

    database = tmp_path / "agents" / "phase4-unreadable-history" / "state.db"
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT id, result FROM runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        run_id, result_data = row
        document = json.loads(result_data)
        usage = document["fields"]["usage"]
        assert usage["__record__"] == "ModelUsage"
        usage["fields"]["estimated_cost_usd"] = {"__decimal__": "0.01"}
        connection.execute(
            "UPDATE runs SET result = ? WHERE id = ?",
            (json.dumps(document), run_id),
        )

    agent = await Agent.open(
        "phase4-unreadable-history",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=ids,
    )
    try:
        unavailable = await agent.review_learning_candidates()
        assert unavailable.status is LearningReviewStatus.HISTORY_UNAVAILABLE
        assert unavailable.skipped_run_count == 1
        assert unavailable.model_calls == 0
        assert reviewer.requests == ()

        await agent.run("Remember that booked revenue excludes completed refunds.")
        reviewed = await agent.review_learning_candidates()

        assert reviewed.status is LearningReviewStatus.COMPLETED
        assert reviewed.reviewed_run_ids == ("run-2",)
        assert reviewed.skipped_run_count == 1
        assert reviewed.model_calls == 1
        assert len(reviewed.candidates) == 1
        assert len(reviewer.requests) == 1
    finally:
        await agent.close()


async def test_local_review_preparation_failure_is_not_a_provider_failure(
    tmp_path,
    monkeypatch,
):
    foreground = MockModelProvider([])
    reviewer = MockModelProvider([])
    agent = await Agent.create(
        "phase4-local-review-failure",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
    )

    async def fail_artifact_state(self):
        raise RuntimeError("local state unavailable")

    monkeypatch.setattr(
        OneShotCandidateReviewer,
        "_artifact_state",
        fail_artifact_state,
    )
    try:
        result = await agent.review_learning_candidates()

        assert result.status is LearningReviewStatus.LOCAL_FAILED
        assert result.model_calls == 0
        assert reviewer.requests == ()
    finally:
        await agent.close()


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
        reviewer_model=reviewer,
        approval_handler=approve,
        id_factory=_ids(),
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
        reviewer_model=reviewer,
        id_factory=_ids(),
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
        reviewer_model=reviewer,
        approval_handler=approve,
        id_factory=_ids(),
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


async def test_cancelled_review_writes_no_candidate_or_review_stamp(tmp_path):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = _BlockingReviewer()
    agent = await Agent.create(
        "phase4-cancelled-review",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        reviewer_profile=reviewer.model_profile,
        id_factory=_ids(),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        task = asyncio.create_task(agent.review_learning_candidates())
        await reviewer.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await agent.list_learning_candidates() == ()
        with sqlite3.connect(agent.home / "state.db") as connection:
            rows = connection.execute(
                "SELECT data FROM metadata " "WHERE key LIKE 'learning_review_stamps:%'"
            ).fetchall()
        assert rows == []
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
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        candidate = (await agent.review_learning_candidates()).candidates[0]
        output = io.StringIO()
        await _write_memory_surface(agent, "", output)
        assert "Pending candidates:" in output.getvalue()
        assert candidate.candidate.id in output.getvalue()

        controller = PresentationController(root=tmp_path)
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
    )
    try:
        await agent.run("What is booked revenue?")
        review = await agent.review_learning_candidates()
        assert review.status is LearningReviewStatus.COMPLETED
        assert review.candidates == ()
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
    )
    try:
        source_a = await agent.attach_sqlite(first_path, name="first")
        source_b = await agent.attach_sqlite(second_path, name="second")
        await agent.run(
            "Run and retain a reusable monthly invoice procedure.",
            source_id=source_a.id,
        )
        reviewer._script = (
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


async def test_malformed_provider_and_cost_preconditions_have_no_candidate_effect(
    tmp_path,
):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_response("not-json")],
        complete_pricing=True,
    )
    agent = await Agent.create(
        "phase4-fail-closed",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        required = await agent.review_learning_candidates()
        assert required.status is LearningReviewStatus.COST_LIMIT_REQUIRED
        assert len(reviewer.requests) == 0
        assert await agent.list_learning_candidates() == ()
    finally:
        await agent.close()


async def test_malformed_and_token_exhausted_reviews_fail_closed_without_retry(
    tmp_path,
):
    for name, response, expected in (
        (
            "malformed",
            _response("not-json"),
            LearningReviewStatus.MALFORMED_RESPONSE,
        ),
        (
            "tokens",
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=json.dumps({"candidates": []}),
                usage=ModelUsage(input_tokens=LEARNING_REVIEW_MAX_TOTAL_TOKENS + 1),
            ),
            LearningReviewStatus.TOKEN_LIMIT_EXCEEDED,
        ),
    ):
        foreground = MockModelProvider([_response("Understood.")])
        reviewer = MockModelProvider([response])
        agent = await Agent.create(
            f"phase4-{name}",
            root=tmp_path,
            model=foreground,
            model_profile=foreground.model_profile,
            reviewer_model=reviewer,
            id_factory=_ids(),
        )
        try:
            await agent.run("Remember that our fiscal year begins in February.")
            result = await agent.review_learning_candidates()
            assert result.status is expected
            assert result.model_calls == 1
            assert len(reviewer.requests) == 1
            assert await agent.list_learning_candidates() == ()
        finally:
            await agent.close()


async def test_duplicate_model_proposals_collapse_before_persistence(tmp_path):
    content = "Our fiscal year begins in February."
    proposal = {
        "target": "memory",
        "source_ids": [],
        "supporting_run_ids": ["run-1"],
        "content": {"text": content},
    }
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_response(json.dumps({"candidates": [proposal, proposal]}))]
    )
    agent = await Agent.create(
        "phase4-duplicate",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        result = await agent.review_learning_candidates()
        assert result.status is LearningReviewStatus.COMPLETED
        assert len(result.candidates) == 1
        assert result.duplicate_proposals_suppressed == 1
        assert len(await agent.list_learning_candidates()) == 1
    finally:
        await agent.close()


async def test_reviewer_never_accepts_more_than_four_proposals(tmp_path):
    proposals = [
        {
            "target": "memory",
            "source_ids": [],
            "supporting_run_ids": ["run-1"],
            "content": {"text": f"Durable convention {index}."},
        }
        for index in range(LEARNING_REVIEW_MAX_PROPOSALS + 1)
    ]
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider([_response(json.dumps({"candidates": proposals}))])
    agent = await Agent.create(
        "phase4-max-proposals",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        await agent.run("Remember these durable business conventions.")
        result = await agent.review_learning_candidates()
        assert result.status is LearningReviewStatus.MALFORMED_RESPONSE
        assert await agent.list_learning_candidates() == ()
    finally:
        await agent.close()


async def test_candidate_obsolescence_is_derived_without_mutating_candidate_row(
    tmp_path,
):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Our fiscal year begins in February.")]
    )
    agent = await Agent.create(
        "phase4-obsolete",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        candidate = (await agent.review_learning_candidates()).candidates[0]
        database = agent.home / "state.db"
        with sqlite3.connect(database) as connection:
            stored_before = connection.execute(
                "SELECT data FROM learning_candidates WHERE id = ?",
                (candidate.candidate.id,),
            ).fetchone()[0]

        await agent.set_memory("An unrelated active artifact changed.")
        obsolete = await agent.read_learning_candidate(candidate.candidate.id)
        assert obsolete is not None
        assert obsolete.status is LearningCandidateStatus.OBSOLETE
        assert "referenced artifacts changed" in obsolete.obsolete_reasons
        with pytest.raises(ValueError, match="not awaiting review"):
            await agent.accept_learning_candidate(candidate.candidate.id)

        with sqlite3.connect(database) as connection:
            stored_after = connection.execute(
                "SELECT data FROM learning_candidates WHERE id = ?",
                (candidate.candidate.id,),
            ).fetchone()[0]
        assert stored_after == stored_before
    finally:
        await agent.close()


async def test_candidate_rows_and_review_metadata_never_copy_transcript_material(
    tmp_path,
):
    marker = "RAW_PROMPT_TOOL_ARGUMENT_ROW_CREDENTIAL_MARKER"
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Our fiscal year begins in February.")]
    )
    agent = await Agent.create(
        "phase4-sanitized",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        await agent.run(
            "Remember that our fiscal year begins in February. "
            f"Do not persist this transcript marker: {marker}"
        )
        review = await agent.review_learning_candidates()
        assert len(review.candidates) == 1
        with sqlite3.connect(agent.home / "state.db") as connection:
            candidate_data = "\n".join(
                row[0]
                for row in connection.execute("SELECT data FROM learning_candidates")
            )
            stamp_data = "\n".join(
                row[0]
                for row in connection.execute(
                    "SELECT data FROM metadata WHERE key LIKE 'learning_review_stamps:%'"
                )
            )
        assert marker not in candidate_data
        assert marker not in stamp_data
        assert "tool_arguments" not in candidate_data
        assert '"rows"' not in candidate_data
        assert "credential" not in stamp_data.lower()
    finally:
        await agent.close()


async def test_reviewer_is_disabled_by_default_and_foreground_never_invokes_it(
    tmp_path,
):
    foreground = MockModelProvider([_response("Done.")])
    agent = await Agent.create(
        "phase4-disabled",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        id_factory=_ids(),
    )
    try:
        result = await agent.run("Remember this.")
        assert result.final_text == "Done."
        review = await agent.review_learning_candidates()
        assert review.status is LearningReviewStatus.DISABLED
        assert len(foreground.requests) == 1
    finally:
        await agent.close()


def test_candidate_review_evaluation_is_caller_owned_content_free_and_bounded():
    report = CandidateReviewReport(
        (
            CandidateReviewMeasurement(
                proposed_candidates=4,
                accepted_candidates=2,
                rejected_candidates=2,
                false_positive_candidates=1,
                duplicate_candidates_suppressed=3,
                background_model_calls=1,
                background_total_tokens=900,
                background_duration_ms=125,
                background_estimated_cost_usd=Decimal("0.02"),
                background_cost_complete=True,
            ),
        )
    )
    data = report.to_mapping()
    assert data["candidate_precision"] == "0.750000"
    assert data["acceptance_rate"] == "0.500000"
    assert data["rejection_rate"] == "0.500000"
    assert data["hard_safety_passed"] is True
    rendered = report.render_markdown()
    assert "Background model calls" in rendered
    assert "Booked revenue" not in rendered


def test_candidate_edit_projection_round_trips_frozen_mapping_values():
    original = DocumentCandidateContent("Use a February fiscal-year start.")
    projected = learning_candidate_content_to_mapping(original)
    assert (
        learning_candidate_content_from_mapping(
            LearningCandidateTarget.MEMORY,
            projected,
        )
        == original
    )


async def test_reviewer_request_excludes_secret_shaped_input_material(tmp_path):
    assistant_secret = "ASSISTANT_SECRET_123456"
    tool_argument_secret = "TOOL_ARGUMENT_SECRET_123456"
    prefixed_assistant_secret = "PREFIXED_ASSISTANT_SECRET_123456"
    generic_token_secret = "GENERIC_TOKEN_SECRET_123456"
    natural_language_secret = "NATURAL_LANGUAGE_SECRET_123456"
    foreground = MockModelProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="catalog-secret-probe",
                        name="catalog_search",
                        arguments={
                            "query": f"api_key={tool_argument_secret}",
                        },
                    ),
                ),
            ),
            _response(
                f"Understood. password={assistant_secret} "
                f"MY_ACCESS_TOKEN={prefixed_assistant_secret} "
                f"SESSION_TOKEN={generic_token_secret}. "
                f"API key is {natural_language_secret}"
            ),
        ]
    )
    reviewer = MockModelProvider([_response(json.dumps({"candidates": []}))])
    agent = await Agent.create(
        "phase4-reviewer-privacy",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        memory_secret = "MEMORY_SECRET_123456"
        profile_secret = "PROFILE_SECRET_123456"
        transcript_secret = "TRANSCRIPT_SECRET_123456"
        prefixed_memory_secret = "PREFIXED_MEMORY_SECRET_123456"
        prefixed_profile_secret = "PREFIXED_PROFILE_SECRET_123456"
        prefixed_transcript_secret = "PREFIXED_TRANSCRIPT_SECRET_123456"
        connection_secret = (
            "postgresql://reviewer_user:REVIEWER_PASSWORD@" "private.internal/reviewer"
        )
        secret_reference = "env:DAITA_PRIVATE_REVIEW_TOKEN"
        keychain_reference = "keychain:daita-private-review"
        await agent.set_memory(
            f"api_key: {memory_secret}\ncredential: {secret_reference}\n"
            f"DATABASE_PASSWORD={prefixed_memory_secret}\n"
            f"APP_CONNECTION_STRING={connection_secret}\n"
            "budget_token=24000"
        )
        await agent.set_user_profile(
            f"access_token={profile_secret}\nsecret={keychain_reference}\n"
            f"OPENAI_API_KEY={prefixed_profile_secret}"
        )
        await agent.run(
            "Remember that the fiscal year starts in February. "
            f"Bearer {transcript_secret} "
            f"AWS_SECRET_ACCESS_KEY={prefixed_transcript_secret}"
        )

        review = await agent.review_learning_candidates()

        assert review.status is LearningReviewStatus.COMPLETED
        assert len(reviewer.requests) == 1
        rendered_request = "\n".join(
            block.text
            for message in reviewer.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert memory_secret not in rendered_request
        assert profile_secret not in rendered_request
        assert transcript_secret not in rendered_request
        assert assistant_secret not in rendered_request
        assert tool_argument_secret not in rendered_request
        assert prefixed_assistant_secret not in rendered_request
        assert generic_token_secret not in rendered_request
        assert natural_language_secret not in rendered_request
        assert prefixed_memory_secret not in rendered_request
        assert prefixed_profile_secret not in rendered_request
        assert prefixed_transcript_secret not in rendered_request
        assert connection_secret not in rendered_request
        assert "private.internal/reviewer" not in rendered_request
        assert secret_reference not in rendered_request
        assert keychain_reference not in rendered_request
        assert "[redacted-secret]]" not in rendered_request
        assert "budget_token=24000" in rendered_request
    finally:
        await agent.close()


async def test_reviewer_redacts_secret_values_inside_bounded_tool_results(tmp_path):
    database = tmp_path / "reviewer-secrets.db"
    tool_result_secret = "TOOL_RESULT_SECRET_123456"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE credentials(api_key TEXT)")
        connection.execute(
            "INSERT INTO credentials(api_key) VALUES (?)",
            (tool_result_secret,),
        )
    foreground = MockModelProvider(())
    reviewer = MockModelProvider([_response(json.dumps({"candidates": []}))])
    agent = await Agent.create(
        "phase4-reviewer-result-privacy",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        source = await agent.attach_sqlite(database, name="credentials")
        foreground._script = (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="secret-row",
                        name="data_query_sqlite",
                        arguments={
                            "source_id": source.id,
                            "sql": "SELECT api_key FROM credentials",
                        },
                    ),
                ),
            ),
            _response("The requested record was inspected."),
        )
        run = await agent.run(
            "Inspect the credential record without retaining its value.",
            source_id=source.id,
        )
        transcript = await agent.transcript(run.run_id)
        tool_result = transcript.messages[2].content[0]
        assert isinstance(tool_result, ToolResultBlock)
        data = tool_result.output["data"]
        assert isinstance(data, Mapping)
        rows = data["rows"]
        assert isinstance(rows, tuple)
        assert isinstance(rows[0], Mapping)
        assert rows[0]["api_key"] == tool_result_secret

        review = await agent.review_learning_candidates()

        assert review.status is LearningReviewStatus.COMPLETED
        rendered_request = "\n".join(
            block.text
            for message in reviewer.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert tool_result_secret not in rendered_request
    finally:
        await agent.close()


async def test_review_measurements_count_a_model_call_before_persistence_failure(
    tmp_path,
    monkeypatch,
):
    usage = ModelUsage(input_tokens=17, output_tokens=3)
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [
            _response(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "target": "memory",
                                "source_ids": [],
                                "supporting_run_ids": ["run-1"],
                                "content": {
                                    "text": (
                                        "Booked revenue excludes completed " "refunds."
                                    )
                                },
                            }
                        ]
                    }
                ),
                usage=usage,
            )
        ]
    )
    agent = await Agent.create(
        "phase4-review-measurement",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        await agent.run("Remember that booked revenue excludes completed refunds.")

        async def fail_after_review(*args, **kwargs):
            raise LearningCandidateError("candidate capacity exhausted")

        monkeypatch.setattr(
            agent._embedded._candidate_reviewer._store,
            "save_learning_candidate_review",
            fail_after_review,
        )

        result = await agent.review_learning_candidates()

        assert result.status is LearningReviewStatus.CAPACITY_EXHAUSTED
        assert result.model_calls == 1
        assert result.reviewed_run_ids == ("run-1",)
        assert result.usage == usage
        assert len(reviewer.requests) == 1
    finally:
        await agent.close()
