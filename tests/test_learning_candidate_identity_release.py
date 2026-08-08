from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from daita import Agent
from daita.learning_candidates import (
    DocumentCandidateContent,
    LearningCandidate,
    LearningCandidateAction,
    LearningCandidateError,
    LearningCandidateRejectionReason,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateStatus,
    LearningCandidateTarget,
    SemanticCandidateContent,
    SkillCandidateContent,
    _Proposal,
    _validate_candidate_support,
    candidate_matches_mutation_call,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import (
    ConversationRun,
    LoopExit,
    LoopExitKind,
    RunInput,
    Transcript,
)
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticKind,
    SemanticSubject,
)
from daita.storage.sqlite import SQLiteStateStore

_NOW = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _ids():
    counters: defaultdict[str, int] = defaultdict(int)

    def value(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]}"

    return value


def _memory_candidate(
    *,
    candidate_id: str,
    run_id: str,
    text: str,
    review_digest: str,
    artifact_digest: str,
    candidate_digest: str,
) -> LearningCandidate:
    return LearningCandidate(
        id=candidate_id,
        agent_id="agent-one",
        target=LearningCandidateTarget.MEMORY,
        content=DocumentCandidateContent(text),
        source_ids=(),
        reviewed_runs=(LearningCandidateRunReference(run_id, review_digest),),
        supporting_run_ids=(run_id,),
        review_fingerprint=review_digest,
        artifact_state_sha256=artifact_digest,
        catalog_revisions=(),
        candidate_fingerprint=candidate_digest,
        status=LearningCandidateStatus.AWAITING_REVIEW,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _stamp(run_id: str, digest: str) -> LearningCandidateReviewStamp:
    return LearningCandidateReviewStamp(
        run_id=run_id,
        transcript_sha256=digest,
        artifact_state_sha256=digest,
        catalog_state_sha256=digest,
    )


async def test_normalized_identity_retains_rejection_across_review_contexts_until_clear(
    tmp_path,
):
    first = _memory_candidate(
        candidate_id="candidate-first",
        run_id="run-first",
        text="Our   fiscal year begins in February.",
        review_digest="1" * 64,
        artifact_digest="a" * 64,
        candidate_digest="c" * 64,
    )
    repeated = _memory_candidate(
        candidate_id="candidate-repeated",
        run_id="run-repeated",
        text="  OUR fiscal year begins in FEBRUARY.  ",
        review_digest="2" * 64,
        artifact_digest="b" * 64,
        candidate_digest="d" * 64,
    )
    assert first.candidate_fingerprint != repeated.candidate_fingerprint
    assert first.candidate_identity_sha256 == repeated.candidate_identity_sha256

    store = await SQLiteStateStore.open(tmp_path / "state.db")
    assert await store.save_learning_candidate_review(
        "agent-one",
        stamps=(_stamp("run-first", "1" * 64),),
        candidates=(first,),
    ) == (first,)
    rejected = await store.reject_learning_candidate(
        "agent-one",
        first.id,
        expected_fingerprint=first.candidate_fingerprint,
        reason=LearningCandidateRejectionReason.USER_DECLINED,
        rejected_at=_NOW,
    )
    assert rejected.status is LearningCandidateStatus.REJECTED

    assert (
        await store.save_learning_candidate_review(
            "agent-one",
            stamps=(_stamp("run-repeated", "2" * 64),),
            candidates=(repeated,),
        )
        == ()
    )
    assert await store.list_learning_candidates("agent-one") == (rejected,)

    assert await store.clear_rejected_learning_candidates("agent-one") == 1
    assert await store.save_learning_candidate_review(
        "agent-one",
        stamps=(_stamp("run-repeated", "2" * 64),),
        candidates=(repeated,),
    ) == (repeated,)


def _semantic_candidate() -> LearningCandidate:
    binding = ResourceRevisionBinding("resource-invoices", "revision-one")
    content = SemanticCandidateContent(
        action=LearningCandidateAction.SAVE,
        subject=SemanticSubject(
            source_ids=("source-finance",),
            resource_ids=("resource-invoices",),
        ),
        kind=SemanticKind.METRIC_DEFINITION,
        statement="Booked revenue excludes completed refunds.",
        catalog_revisions=(binding,),
        annotation_id="annotation-current",
        supersedes_id="annotation-prior",
    )
    return LearningCandidate(
        id="candidate-semantic",
        agent_id="agent-one",
        target=LearningCandidateTarget.SEMANTIC,
        content=content,
        source_ids=("source-finance",),
        reviewed_runs=(LearningCandidateRunReference("run-semantic", "1" * 64),),
        supporting_run_ids=("run-semantic",),
        review_fingerprint="2" * 64,
        artifact_state_sha256="3" * 64,
        catalog_revisions=(binding,),
        candidate_fingerprint="4" * 64,
        status=LearningCandidateStatus.AWAITING_REVIEW,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _semantic_arguments() -> dict[str, object]:
    return {
        "id": "annotation-current",
        "subject": {
            "source_ids": ["source-finance"],
            "resource_ids": ["resource-invoices"],
            "fields": [],
        },
        "kind": "metric_definition",
        "statement": "Booked revenue excludes completed refunds.",
        "evidence": [{"kind": "user_assertion"}],
        "catalog_revisions": [
            {
                "resource_id": "resource-invoices",
                "revision": "revision-one",
            }
        ],
        "supersedes_id": "annotation-prior",
        "expected_sha256": "f" * 64,
    }


def test_semantic_mutation_match_is_exact_except_for_runtime_owned_evidence():
    candidate = _semantic_candidate()
    arguments = _semantic_arguments()
    assert candidate_matches_mutation_call(
        candidate,
        ToolCall("save", "semantic_save", arguments),
    )

    different_evidence = {
        **arguments,
        "evidence": [
            {
                "kind": "tool_result",
                "tool_call_id": "runtime-call",
                "note": "Fresh runtime evidence.",
            }
        ],
        "expected_sha256": "e" * 64,
    }
    assert candidate_matches_mutation_call(
        candidate,
        ToolCall("save-runtime", "semantic_save", different_evidence),
    )

    for changed in (
        {key: value for key, value in arguments.items() if key != "id"},
        {**arguments, "id": "annotation-other"},
        {key: value for key, value in arguments.items() if key != "supersedes_id"},
        {**arguments, "supersedes_id": "annotation-other"},
        {
            **arguments,
            "catalog_revisions": [
                {
                    "resource_id": "resource-invoices",
                    "revision": "revision-two",
                }
            ],
        },
    ):
        assert not candidate_matches_mutation_call(
            candidate,
            ToolCall("mismatch", "semantic_save", changed),
        )


def test_semantic_content_cannot_be_edited_outside_its_frozen_scope():
    candidate = _semantic_candidate()
    content = candidate.content
    assert isinstance(content, SemanticCandidateContent)
    assert content.subject is not None

    with pytest.raises(LearningCandidateError, match="source scope"):
        replace(
            candidate,
            content=replace(
                content,
                subject=SemanticSubject(
                    source_ids=("source-other",),
                    resource_ids=("resource-invoices",),
                ),
            ),
        )
    with pytest.raises(LearningCandidateError, match="catalog scope"):
        replace(
            candidate,
            content=replace(
                content,
                catalog_revisions=(
                    ResourceRevisionBinding(
                        "resource-invoices",
                        "revision-two",
                    ),
                ),
            ),
        )


def _conversation(
    *,
    user_text: str,
    calls: tuple[ToolCall, ...] = (),
    results: tuple[ToolResultBlock, ...] = (),
) -> ConversationRun:
    messages = [CanonicalMessage(MessageRole.USER, (TextBlock(user_text),))]
    if calls:
        messages.append(CanonicalMessage(MessageRole.ASSISTANT, tool_calls=calls))
        messages.append(CanonicalMessage(MessageRole.TOOL, content=results))
    messages.append(
        CanonicalMessage(
            MessageRole.ASSISTANT,
            (TextBlock("Completed the requested foreground run."),),
        )
    )
    run = RunInput(
        id="run-grounding",
        agent_id="agent-one",
        message=user_text,
        created_at=_NOW,
        conversation_id="conversation-one",
    )
    return ConversationRun(
        turn_index=0,
        transcript=Transcript(run, tuple(messages)),
        result=LoopExit(
            run_id=run.id,
            conversation_id="conversation-one",
            kind=LoopExitKind.COMPLETED,
            reason="completed",
            created_at=_NOW,
            final_text="Completed the requested foreground run.",
        ),
    )


def test_unrelated_assistant_claim_is_not_grounded_by_remember_instruction():
    proposal = _Proposal(
        target=LearningCandidateTarget.MEMORY,
        content=DocumentCandidateContent("Our fiscal year always ends in December."),
        source_ids=(),
        supporting_run_ids=("run-grounding",),
    )
    support = _conversation(
        user_text="Remember that our fiscal year begins in February."
    )
    with pytest.raises(LearningCandidateError, match="explicit user grounding"):
        _validate_candidate_support(proposal, (support,))

    generic_support = _conversation(
        user_text="Remember these durable business conventions."
    )
    with pytest.raises(LearningCandidateError, match="explicit user grounding"):
        _validate_candidate_support(proposal, (generic_support,))

    negated_support = _conversation(
        user_text=("Remember that our fiscal year does not always end in December.")
    )
    with pytest.raises(LearningCandidateError, match="explicit user grounding"):
        _validate_candidate_support(proposal, (negated_support,))


def test_unrelated_success_does_not_rescue_a_failed_candidate_procedure():
    proposal = _Proposal(
        target=LearningCandidateTarget.SKILL,
        content=SkillCandidateContent(
            action=LearningCandidateAction.SAVE,
            name="monthly-invoices",
            description="Review monthly invoices using a validated query.",
            instructions=(
                "Query the monthly invoices, inspect the result, and report "
                "whether validation succeeded."
            ),
        ),
        source_ids=("source-finance",),
        supporting_run_ids=("run-grounding",),
    )
    failed = ToolCall(
        "failed-invoices",
        "execute_sql",
        {"sql": "SELECT * FROM invoices WHERE month = :month"},
    )
    unrelated = ToolCall(
        "successful-customers",
        "catalog_search",
        {"query": "customer addresses"},
    )
    support = _conversation(
        user_text="Run and retain a reusable monthly invoice procedure.",
        calls=(failed, unrelated),
        results=(
            ToolResultBlock(
                failed.id,
                {"error": {"code": "invalid_query"}},
                is_error=True,
            ),
            ToolResultBlock(
                unrelated.id,
                {"data": {"resources": []}},
            ),
        ),
    )
    with pytest.raises(LearningCandidateError, match="failed related procedure"):
        _validate_candidate_support(proposal, (support,))

    corrected = ToolCall(
        "successful-invoices",
        "execute_sql",
        {"sql": "SELECT * FROM invoices WHERE month = :month"},
    )
    corrected_support = _conversation(
        user_text="Run and retain a reusable monthly invoice procedure.",
        calls=(failed, corrected),
        results=(
            ToolResultBlock(
                failed.id,
                {"error": {"code": "invalid_query"}},
                is_error=True,
            ),
            ToolResultBlock(corrected.id, {"data": {"returned_rows": 1}}),
        ),
    )
    _validate_candidate_support(proposal, (corrected_support,))


async def test_catalog_derived_skill_keeps_binding_for_read_time_obsolescence(
    tmp_path,
):
    database = tmp_path / "finance.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)")
    foreground = MockModelProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "catalog-invoices",
                        "catalog_search",
                        {"query": "invoices"},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Validated the monthly invoice procedure.",
            ),
        ]
    )
    reviewer = MockModelProvider([])
    agent = await Agent.create(
        "candidate-skill-obsolescence",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        source = await agent.attach_sqlite(database, name="finance")
        reviewer._script = (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "target": "skill",
                                "source_ids": [source.id],
                                "supporting_run_ids": ["run-1"],
                                "content": {
                                    "action": "save",
                                    "name": "monthly-invoices",
                                    "description": (
                                        "Review monthly invoices against the "
                                        "current catalog."
                                    ),
                                    "instructions": (
                                        "Inspect the current catalog, review "
                                        "monthly invoices, and verify the result."
                                    ),
                                },
                            }
                        ]
                    }
                ),
            ),
        )
        await agent.run(
            "Run and retain a reusable monthly invoice procedure.",
            source_id=source.id,
        )
        candidate = (await agent.review_learning_candidates()).candidates[0]
        assert candidate.candidate.catalog_revisions

        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE invoices ADD COLUMN posted_at TEXT")
        await agent.refresh_source(source.id)

        obsolete = await agent.read_learning_candidate(candidate.candidate.id)
        assert obsolete is not None
        assert obsolete.status is LearningCandidateStatus.OBSOLETE
        assert any(
            reason.startswith("catalog revision changed:")
            for reason in obsolete.obsolete_reasons
        )
    finally:
        await agent.close()


async def test_reviewer_discards_cross_source_semantic_delete_candidate(tmp_path):
    first_database = tmp_path / "first.db"
    second_database = tmp_path / "second.db"
    for database in (first_database, second_database):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)"
            )
    foreground = MockModelProvider(())
    reviewer = MockModelProvider(())
    agent = await Agent.create(
        "candidate-cross-source-delete",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        first_source = await agent.attach_sqlite(first_database, name="first")
        second_source = await agent.attach_sqlite(second_database, name="second")
        foreground._script = (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Recorded the second-source definition.",
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Understood.",
            ),
        )
        evidence_run = await agent.run(
            "Booked revenue uses the invoices table.",
            source_id=second_source.id,
        )
        evidence_transcript = await agent.transcript(evidence_run.run_id)
        second_resource = (
            await agent.list_catalog_resources(source_id=second_source.id)
        )[0]
        annotation = SemanticAnnotation(
            id="cross-source-definition",
            agent_id=agent.id,
            subject=SemanticSubject(
                source_ids=(second_source.id,),
                resource_ids=(second_resource.id,),
            ),
            kind=SemanticKind.METRIC_DEFINITION,
            statement="Booked revenue uses the invoices table.",
            evidence=(
                SemanticEvidence(
                    SemanticEvidenceKind.USER_ASSERTION,
                    evidence_run.run_id,
                    message_position=0,
                ),
            ),
            catalog_revisions=(
                ResourceRevisionBinding(
                    second_resource.id,
                    second_resource.current_revision,
                ),
            ),
            created_at=evidence_transcript.run.created_at,
            confirmed_at=evidence_transcript.run.created_at,
        )
        await agent.save_semantic_annotation(annotation)
        support = await agent.run(
            "Remember to delete semantic annotations first-source-definition "
            "and cross-source-definition.",
            source_id=first_source.id,
        )
        assert support.run_id == "run-2"
        support_transcript = await agent.transcript(support.run_id)
        first_resource = (
            await agent.list_catalog_resources(source_id=first_source.id)
        )[0]
        first_annotation = SemanticAnnotation(
            id="first-source-definition",
            agent_id=agent.id,
            subject=SemanticSubject(
                source_ids=(first_source.id,),
                resource_ids=(first_resource.id,),
            ),
            kind=SemanticKind.GLOSSARY,
            statement="First-source definition.",
            evidence=(
                SemanticEvidence(
                    SemanticEvidenceKind.USER_ASSERTION,
                    support.run_id,
                    message_position=0,
                ),
            ),
            catalog_revisions=(
                ResourceRevisionBinding(
                    first_resource.id,
                    first_resource.current_revision,
                ),
            ),
            created_at=support_transcript.run.created_at,
            confirmed_at=support_transcript.run.created_at,
        )
        await agent.save_semantic_annotation(first_annotation)
        reviewer._script = (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "target": "semantic",
                                "source_ids": [first_source.id],
                                "supporting_run_ids": [support.run_id],
                                "content": {
                                    "action": "delete",
                                    "annotation_id": annotation.id,
                                },
                            },
                            {
                                "target": "semantic",
                                "source_ids": [first_source.id],
                                "supporting_run_ids": [support.run_id],
                                "content": {
                                    "action": "delete",
                                    "annotation_id": first_annotation.id,
                                },
                            },
                        ]
                    }
                ),
            ),
        )

        result = await agent.review_learning_candidates()

        assert len(result.candidates) == 1
        candidate = result.candidates[0].candidate
        assert isinstance(candidate.content, SemanticCandidateContent)
        assert candidate.content.annotation_id == first_annotation.id
        with pytest.raises(LearningCandidateError, match="another source scope"):
            await agent.edit_learning_candidate(
                candidate.id,
                SemanticCandidateContent(
                    action=LearningCandidateAction.DELETE,
                    annotation_id=annotation.id,
                ),
            )
        assert await agent.read_semantic_annotation(annotation.id) is not None
    finally:
        await agent.close()


async def test_scoped_candidate_cannot_pool_grounding_from_another_source(tmp_path):
    first_database = tmp_path / "grounding-first.db"
    second_database = tmp_path / "grounding-second.db"
    for database in (first_database, second_database):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)"
            )
    foreground = MockModelProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Recorded the first-source reporting convention.",
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Understood.",
            ),
        ]
    )
    reviewer = MockModelProvider(())
    agent = await Agent.create(
        "candidate-cross-source-grounding",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    try:
        first_source = await agent.attach_sqlite(first_database, name="first")
        second_source = await agent.attach_sqlite(second_database, name="second")
        first_run = await agent.run(
            "Remember that first-source reporting uses monthly invoices.",
            source_id=first_source.id,
        )
        second_run = await agent.run(
            "Remember to delete semantic annotation source-a-definition.",
            source_id=second_source.id,
        )
        first_transcript = await agent.transcript(first_run.run_id)
        first_resource = (
            await agent.list_catalog_resources(source_id=first_source.id)
        )[0]
        annotation = SemanticAnnotation(
            id="source-a-definition",
            agent_id=agent.id,
            subject=SemanticSubject(
                source_ids=(first_source.id,),
                resource_ids=(first_resource.id,),
            ),
            kind=SemanticKind.GLOSSARY,
            statement="First-source reporting uses monthly invoices.",
            evidence=(
                SemanticEvidence(
                    SemanticEvidenceKind.USER_ASSERTION,
                    first_run.run_id,
                    message_position=0,
                ),
            ),
            catalog_revisions=(
                ResourceRevisionBinding(
                    first_resource.id,
                    first_resource.current_revision,
                ),
            ),
            created_at=first_transcript.run.created_at,
            confirmed_at=first_transcript.run.created_at,
        )
        await agent.save_semantic_annotation(annotation)
        reviewer._script = (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "target": "semantic",
                                "source_ids": [first_source.id],
                                "supporting_run_ids": [
                                    first_run.run_id,
                                    second_run.run_id,
                                ],
                                "content": {
                                    "action": "delete",
                                    "annotation_id": annotation.id,
                                },
                            }
                        ]
                    }
                ),
            ),
        )

        result = await agent.review_learning_candidates()

        assert result.candidates == ()
        current = await agent.read_semantic_annotation(annotation.id)
        assert current is not None
        assert current.annotation == annotation
    finally:
        await agent.close()
