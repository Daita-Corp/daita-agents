from __future__ import annotations

from _workspace_support import workspace_for

import asyncio
import json
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from _capability_runtime_support import ContextToolProjectionAdapter

from daita import (
    Agent,
    ApprovalDecision,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
)
from daita.hosting.embedded import AgentHomeError, EmbeddedAgent
from daita.learning_candidates import (
    LearningCandidate,
    LearningCandidateAction,
    LearningCandidateRunReference,
    LearningCandidateTarget,
    SemanticCandidateContent,
)
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopLimits, RunInput
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticKind,
    SemanticSubject,
)
from _toolbox_model_support import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)

EAGER_LIMITS = LoopLimits()


def _ids():
    counters: defaultdict[str, int] = defaultdict(int)

    def value(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]}"

    return value


def _response(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _review_response(run_id: str, text: str) -> ModelResponse:
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


async def _memory_candidate_agent(
    tmp_path,
    *,
    name: str,
    approval_handler,
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
            _response("Saved."),
        ]
    )
    reviewer = MockModelProvider([_review_response("run-1", content)])
    agent = await Agent.create(
        name,
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        limits=EAGER_LIMITS,
        reviewer_model=reviewer,
        approval_handler=approval_handler,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    await agent.run("Remember that booked revenue excludes completed refunds.")
    candidate_id = (await agent.review_learning_candidates()).candidates[0].candidate.id
    return agent, candidate_id, content


async def test_reject_waits_for_inflight_acceptance_and_cannot_split_state(tmp_path):
    approval_started = asyncio.Event()
    release_approval = asyncio.Event()

    async def approve(_request):
        approval_started.set()
        await release_approval.wait()
        return ApprovalDecision.APPROVE

    agent, candidate_id, content = await _memory_candidate_agent(
        tmp_path,
        name="candidate-reject-race",
        approval_handler=approve,
    )
    try:
        acceptance = asyncio.create_task(agent.accept_learning_candidate(candidate_id))
        await asyncio.wait_for(approval_started.wait(), timeout=2)
        rejection = asyncio.create_task(
            agent.reject_learning_candidate(
                candidate_id,
                LearningCandidateRejectionReason.USER_DECLINED,
            )
        )
        await asyncio.sleep(0)
        assert not rejection.done()

        release_approval.set()
        await acceptance
        with pytest.raises(ValueError, match="not awaiting review"):
            await rejection

        assert await agent.read_memory() == content
        view = await agent.read_learning_candidate(candidate_id)
        assert view is not None
        assert view.status is LearningCandidateStatus.ACCEPTED
    finally:
        release_approval.set()
        await agent.close()


async def test_concurrent_double_accept_executes_candidate_mutation_once(tmp_path):
    approval_started = asyncio.Event()
    release_approval = asyncio.Event()
    approvals = 0

    async def approve(_request):
        nonlocal approvals
        approvals += 1
        approval_started.set()
        await release_approval.wait()
        return ApprovalDecision.APPROVE

    agent, candidate_id, content = await _memory_candidate_agent(
        tmp_path,
        name="candidate-double-accept",
        approval_handler=approve,
    )
    try:
        first = asyncio.create_task(agent.accept_learning_candidate(candidate_id))
        await asyncio.wait_for(approval_started.wait(), timeout=2)
        second = asyncio.create_task(agent.accept_learning_candidate(candidate_id))
        await asyncio.sleep(0)
        assert not second.done()

        release_approval.set()
        await first
        with pytest.raises(ValueError, match="not awaiting review"):
            await second

        assert approvals == 1
        assert await agent.read_memory() == content
    finally:
        release_approval.set()
        await agent.close()


async def test_cancel_after_definite_mutation_finalizes_candidate_as_accepted(
    tmp_path,
    monkeypatch,
):
    async def approve(_request):
        return ApprovalDecision.APPROVE

    agent, candidate_id, content = await _memory_candidate_agent(
        tmp_path,
        name="candidate-cancel-finalize",
        approval_handler=approve,
    )
    memory = agent._embedded._memory_store
    original_write = memory._write_sync
    mutation_completed = threading.Event()
    release_executor = threading.Event()

    def blocked_write(name, text, data, max_characters, max_bytes):
        original_write(name, text, data, max_characters, max_bytes)
        mutation_completed.set()
        if not release_executor.wait(timeout=5):
            raise AssertionError("test did not release the memory executor")

    monkeypatch.setattr(memory, "_write_sync", blocked_write)
    task = asyncio.create_task(agent.accept_learning_candidate(candidate_id))
    try:
        completed = await asyncio.wait_for(
            asyncio.to_thread(mutation_completed.wait),
            timeout=2,
        )
        assert completed
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_executor.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert await agent.read_memory() == content
        view = await agent.read_learning_candidate(candidate_id)
        assert view is not None
        assert view.status is LearningCandidateStatus.ACCEPTED
    finally:
        release_executor.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await agent.close()


async def test_semantic_acceptance_projects_only_its_exact_write_tool(tmp_path):
    database = tmp_path / "semantic.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)")
    agent = await Agent.create(
        "candidate-semantic-projection",
        root=tmp_path,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_sqlite(database, name="semantic")
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        binding = ResourceRevisionBinding(
            resource_id=resource.id,
            revision=resource.current_revision,
        )
        now = datetime(2026, 7, 29, tzinfo=UTC)
        candidate = LearningCandidate(
            id="candidate-semantic",
            agent_id=agent.id,
            target=LearningCandidateTarget.SEMANTIC,
            content=SemanticCandidateContent(
                action=LearningCandidateAction.SAVE,
                subject=SemanticSubject(
                    source_ids=(source.id,),
                    resource_ids=(resource.id,),
                ),
                kind=SemanticKind.GLOSSARY,
                statement=(
                    "Booked revenue excludes completed refunds; memory_set and "
                    "skill_save are unrelated."
                ),
                catalog_revisions=(binding,),
            ),
            source_ids=(source.id,),
            reviewed_runs=(LearningCandidateRunReference("run-evidence", "1" * 64),),
            supporting_run_ids=("run-evidence",),
            review_fingerprint="2" * 64,
            artifact_state_sha256="3" * 64,
            catalog_revisions=(binding,),
            candidate_fingerprint="4" * 64,
            status=LearningCandidateStatus.AWAITING_REVIEW,
            created_at=now,
            updated_at=now,
        )
        run = RunInput(
            id="run-acceptance",
            agent_id=agent.id,
            message="Review the selected candidate and apply its eligible write.",
            created_at=now,
            source_id=source.id,
        )
        runtime = agent._embedded._capability_runtime
        guard = agent._embedded._learning_candidate_guard
        guard.select(run.id, candidate)
        try:
            catalog = await runtime.prepare_run(run)
            names = {entry.view.name for entry in catalog.entries}
        finally:
            guard.clear(run.id)
            guard.clear_outcome(run.id)

        write_tools = {
            "memory_set",
            "semantic_delete",
            "semantic_save",
            "skill_delete",
            "skill_save",
        }
        assert names & write_tools == {"semantic_save"}
    finally:
        await agent.close()


async def test_semantic_tools_cannot_cross_the_selected_source_boundary(tmp_path):
    first_database = tmp_path / "first-semantic.db"
    second_database = tmp_path / "second-semantic.db"
    for database in (first_database, second_database):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)"
            )
    foreground = MockModelProvider([_response("First."), _response("Second.")])
    approval_requests = []

    async def approve(request):
        approval_requests.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "semantic-source-boundary",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        limits=EAGER_LIMITS,
        approval_handler=approve,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        first_source = await agent.attach_sqlite(first_database, name="first")
        second_source = await agent.attach_sqlite(second_database, name="second")
        first_resource = (
            await agent.list_catalog_resources(source_id=first_source.id)
        )[0]
        second_resource = (
            await agent.list_catalog_resources(source_id=second_source.id)
        )[0]
        first_evidence = await agent.run(
            "Remember the first-source invoice definition.",
            source_id=first_source.id,
        )
        second_evidence = await agent.run(
            "Remember the second-source invoice definition.",
            source_id=second_source.id,
        )

        async def save_annotation(
            annotation_id,
            statement,
            source,
            resource,
            evidence,
        ):
            transcript = await agent.transcript(evidence.run_id)
            annotation = SemanticAnnotation(
                id=annotation_id,
                agent_id=agent.id,
                subject=SemanticSubject(
                    source_ids=(source.id,),
                    resource_ids=(resource.id,),
                ),
                kind=SemanticKind.GLOSSARY,
                statement=statement,
                evidence=(
                    SemanticEvidence(
                        SemanticEvidenceKind.USER_ASSERTION,
                        evidence.run_id,
                        message_position=0,
                    ),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(
                        resource.id,
                        resource.current_revision,
                    ),
                ),
                created_at=transcript.run.created_at,
                confirmed_at=transcript.run.created_at,
            )
            await agent.save_semantic_annotation(annotation)

        await save_annotation(
            "first-definition",
            "First-source invoice definition.",
            first_source,
            first_resource,
            first_evidence,
        )
        await save_annotation(
            "second-definition",
            "Second-source invoice definition.",
            second_source,
            second_resource,
            second_evidence,
        )
        second_view = await agent.read_semantic_annotation("second-definition")
        assert second_view is not None

        first_subject = {
            "source_ids": [first_source.id],
            "resource_ids": [first_resource.id],
            "fields": [],
        }
        first_revisions = [
            {
                "resource_id": first_resource.id,
                "revision": first_resource.current_revision,
            }
        ]
        second_subject = {
            "source_ids": [second_source.id],
            "resource_ids": [second_resource.id],
            "fields": [],
        }
        second_revisions = [
            {
                "resource_id": second_resource.id,
                "revision": second_resource.current_revision,
            }
        ]
        foreground.replace_script(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall("list", "semantic_list", {}),
                        ToolCall(
                            "view-cross-source",
                            "semantic_view",
                            {"id": "second-definition"},
                        ),
                        ToolCall(
                            "delete-cross-source",
                            "semantic_delete",
                            {
                                "id": "second-definition",
                                "expected_sha256": second_view.sha256,
                            },
                        ),
                        ToolCall(
                            "update-cross-source",
                            "semantic_save",
                            {
                                "id": "second-definition",
                                "subject": first_subject,
                                "kind": "glossary",
                                "statement": "Cross-source overwrite.",
                                "evidence": [{"kind": "user_assertion"}],
                                "catalog_revisions": first_revisions,
                                "expected_sha256": second_view.sha256,
                            },
                        ),
                        ToolCall(
                            "supersede-cross-source",
                            "semantic_save",
                            {
                                "id": "first-superseding-second",
                                "subject": first_subject,
                                "kind": "glossary",
                                "statement": "Cross-source supersession.",
                                "evidence": [{"kind": "user_assertion"}],
                                "catalog_revisions": first_revisions,
                                "supersedes_id": "second-definition",
                                "expected_sha256": second_view.sha256,
                            },
                        ),
                        ToolCall(
                            "create-cross-source",
                            "semantic_save",
                            {
                                "id": "second-created-from-first",
                                "subject": second_subject,
                                "kind": "glossary",
                                "statement": "Cross-source creation.",
                                "evidence": [{"kind": "user_assertion"}],
                                "catalog_revisions": second_revisions,
                            },
                        ),
                    ),
                ),
                _response("All cross-source requests were blocked."),
            )
        )

        result = await agent.learn(
            "Review, replace, and delete semantic definitions.",
            source_id=first_source.id,
        )
        transcript = await agent.transcript(result.run_id)
        results = {
            block.call_id: block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }

        listed = results["list"]
        assert not listed.is_error
        listed_data = listed.output["data"]
        assert isinstance(listed_data, Mapping)
        annotations = listed_data["annotations"]
        assert isinstance(annotations, tuple)
        assert {item["id"] for item in annotations if isinstance(item, Mapping)} == {
            "first-definition"
        }
        for call_id in (
            "view-cross-source",
            "delete-cross-source",
            "update-cross-source",
            "supersede-cross-source",
            "create-cross-source",
        ):
            blocked = results[call_id]
            assert blocked.is_error
            error = blocked.output["error"]
            assert isinstance(error, Mapping)
            assert error["code"] == "source_scope_violation"
        assert approval_requests == []
        assert await agent.read_semantic_annotation("second-definition") is not None
        assert await agent.read_semantic_annotation("first-superseding-second") is None
        assert await agent.read_semantic_annotation("second-created-from-first") is None
    finally:
        await agent.close()


class _TranscriptContext:
    async def prepare(self, run, messages, tool_context):
        del run
        return messages[:-1], tool_context.initial_provider_definitions

    def project(
        self,
        snapshot,
        messages,
        *,
        step,
        tool_context,
        final=False,
        previous_request_input_tokens=None,
    ):
        del step, previous_request_input_tokens, tool_context
        static, tools = snapshot
        return ModelRequest(
            messages=(*static, *messages),
            tools=() if final else tools,
        )


class _NoTools:
    def __init__(self) -> None:
        self._projection = ContextToolProjectionAdapter(())

    async def prepare_run(self, run):
        return await self._projection.prepare_run(run)

    def project(self, catalog, messages):
        return self._projection.project(catalog, messages)

    async def execute_all(self, run, calls, *, projection, messages, sensitivity):
        del run, calls, projection, messages, sensitivity
        raise AssertionError("the custom tool runtime must not execute")


async def test_acceptance_fails_closed_when_loop_uses_a_different_runtime(tmp_path):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", "Our fiscal year begins in February.")]
    )
    agent = Agent(
        await EmbeddedAgent.create(
            "candidate-custom-runtime",
            root=tmp_path,
            model=foreground,
            model_profile=foreground.model_profile,
            context_builder=_TranscriptContext(),
            tools=_NoTools(),
            reviewer_model=reviewer,
            id_factory=_ids(),
            workspace=workspace_for(tmp_path),
        )
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        candidate_id = (
            (await agent.review_learning_candidates()).candidates[0].candidate.id
        )

        with pytest.raises(AgentHomeError, match="built-in data context"):
            await agent.accept_learning_candidate(candidate_id)

        view = await agent.read_learning_candidate(candidate_id)
        assert view is not None
        assert view.status is LearningCandidateStatus.AWAITING_REVIEW
    finally:
        await agent.close()
