from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime

from _capability_runtime_support import execute_projected

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
    cli,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopLimits, RunInput
from daita.semantics import semantic_annotation_sha256
from daita.tui.commands import SLASH_COMMAND_COMPLETIONS, learning_invocation_message
from daita.tui.controller import PresentationController
from _toolbox_model_support import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)

NOW = datetime(2026, 7, 28, 16, tzinfo=UTC)
EAGER_LIMITS = LoopLimits()


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _ids(tag: str | None = None):
    counts: dict[str, int] = {}

    def create(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        if tag is not None and prefix in {"conversation", "run"}:
            return f"{prefix}-{tag}-{counts[prefix]}"
        return f"{prefix}-{counts[prefix]}"

    return create


def _semantic_call(
    *,
    call_id: str,
    annotation_id: str,
    source_id: str,
    resource_id: str,
    revision: str,
    field_name: str,
    statement: str,
    supersedes_id: str | None = None,
    expected_sha256: str | None = None,
) -> ToolCall:
    arguments: dict[str, object] = {
        "id": annotation_id,
        "subject": {
            "source_ids": [source_id],
            "resource_ids": [resource_id],
            "fields": [{"resource_id": resource_id, "field_name": field_name}],
        },
        "kind": "metric_definition",
        "statement": statement,
        "evidence": [
            {
                "kind": "user_assertion",
            }
        ],
        "catalog_revisions": [{"resource_id": resource_id, "revision": revision}],
    }
    if supersedes_id is not None:
        arguments["supersedes_id"] = supersedes_id
    if expected_sha256 is not None:
        arguments["expected_sha256"] = expected_sha256
    return ToolCall(
        id=call_id,
        name="semantic_save",
        arguments=arguments,
    )


def _tool_results(provider: MockModelProvider) -> tuple[ToolResultBlock, ...]:
    return tuple(
        block
        for request in provider.requests
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.output.get("kind")
        not in {"toolbox_load_receipt", "toolbox_search_results"}
    )


async def test_foreground_teaching_learn_supersession_reopen_and_skill_invocation(
    tmp_path,
):
    database = tmp_path / "semantic-acceptance.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE invoices("
            "id INTEGER PRIMARY KEY, booked_at TEXT, paid_at TEXT, "
            "refund_state TEXT)"
        )
    seed = await Agent.create("semantic-acceptance", root=tmp_path, clock=lambda: NOW)
    try:
        source = await seed.attach_sqlite(database)
        resource = (await seed.list_catalog_resources())[0]
        assert await seed.save_skill(
            "monthly-report",
            "Use for monthly booked-revenue reporting from invoices.",
            "Load current invoices schema, apply approved semantics, and verify totals.",
        )
        agent_id = seed.id
    finally:
        await seed.close()

    original = SemanticAnnotation(
        id="booked-revenue",
        agent_id=agent_id,
        subject=SemanticSubject(
            source_ids=(source.id,),
            resource_ids=(resource.id,),
            fields=(SemanticFieldReference(resource.id, "booked_at"),),
        ),
        kind=SemanticKind.METRIC_DEFINITION,
        statement="Booked revenue uses invoices.booked_at.",
        evidence=(
            SemanticEvidence(
                SemanticEvidenceKind.USER_ASSERTION,
                "run-1",
                message_position=0,
            ),
        ),
        catalog_revisions=(
            ResourceRevisionBinding(resource.id, resource.current_revision),
        ),
        created_at=NOW,
        confirmed_at=NOW,
    )
    original_digest = semantic_annotation_sha256(original)
    create = _semantic_call(
        call_id="teach",
        annotation_id=original.id,
        source_id=source.id,
        resource_id=resource.id,
        revision=resource.current_revision,
        field_name="booked_at",
        statement=original.statement,
    )
    correct = _semantic_call(
        call_id="correct",
        annotation_id="booked-revenue-corrected",
        source_id=source.id,
        resource_id=resource.id,
        revision=resource.current_revision,
        field_name="booked_at",
        statement=(
            "Booked revenue uses invoices.booked_at and excludes completed refunds."
        ),
        supersedes_id=original.id,
        expected_sha256=original_digest,
    )
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(create,),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="definition saved"),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(correct,),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="definition corrected"),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="view-skill",
                        name="skill_view",
                        arguments={"name": "monthly-report"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="skill applied"),
        )
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.open(
        "semantic-acceptance",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
    )
    try:
        first = await agent.learn("When we say booked revenue, use invoices.booked_at.")
        assert first.final_text == "definition saved"
        teaching = learning_invocation_message(
            "/learn Correct booked revenue to exclude completed refunds."
        )
        assert teaching is not None
        second = await agent.learn(teaching)
        assert second.final_text == "definition corrected"
        skill = await agent.run("/monthly-report Apply it to invoices.")
        assert skill.final_text == "skill applied"

        views = {
            view.annotation.id: view for view in await agent.list_semantic_annotations()
        }
        assert views["booked-revenue"].state is SemanticAnnotationState.SUPERSEDED
        assert views["booked-revenue-corrected"].state is SemanticAnnotationState.ACTIVE
        assert tuple(request.tool_name for request in approvals) == (
            "semantic_save",
            "semantic_save",
        )
        viewed_skill = next(
            result
            for result in _tool_results(provider)
            if result.call_id == "view-skill"
        )
        viewed_data = viewed_skill.output["data"]
        assert isinstance(viewed_data, Mapping)
        assert "apply approved semantics" in viewed_data["instructions"]
    finally:
        await agent.close()

    reopened_provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="recalled"),)
    )
    reopened = await Agent.open(
        "semantic-acceptance",
        root=tmp_path,
        model=reopened_provider,
        model_profile=_profile(reopened_provider),
        id_factory=_ids("reopen"),
        clock=lambda: NOW,
    )
    try:
        result = await reopened.run("How should invoices booked revenue be calculated?")
        assert result.final_text == "recalled"
        prompt = "\n".join(
            block.text
            for message in reopened_provider.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "excludes completed refunds" in prompt
        assert "Booked revenue uses invoices.booked_at.</semantic-annotation>" not in (
            prompt
        )
    finally:
        await reopened.close()


async def test_memory_terminal_surface_is_shared_by_cli_and_tui_and_shows_states(
    tmp_path,
    capsys,
):
    database = tmp_path / "semantic-memory-surface.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE invoices("
            "id INTEGER PRIMARY KEY, booked_at TEXT, refund_state TEXT)"
        )
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="evidence"),)
    )
    agent = await Agent.create(
        "semantic-memory-surface",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        clock=lambda: NOW,
    )
    try:
        source = await agent.attach_sqlite(database)
        resource = (await agent.list_catalog_resources())[0]
        run = await agent.run("Teach invoice meaning.")
        transcript = await agent.transcript(run.run_id)

        def annotation(
            annotation_id: str,
            statement: str,
            *,
            kind: SemanticKind,
            field_name: str,
        ) -> SemanticAnnotation:
            return SemanticAnnotation(
                id=annotation_id,
                agent_id=agent.id,
                subject=SemanticSubject(
                    source_ids=(source.id,),
                    resource_ids=(resource.id,),
                    fields=(SemanticFieldReference(resource.id, field_name),),
                ),
                kind=kind,
                statement=statement,
                evidence=(
                    SemanticEvidence(
                        SemanticEvidenceKind.USER_ASSERTION,
                        run.run_id,
                        message_position=0,
                    ),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(resource.id, resource.current_revision),
                ),
                created_at=transcript.run.created_at,
                confirmed_at=transcript.run.created_at,
            )

        await agent.save_semantic_annotation(
            annotation(
                "active-time",
                "Booking time uses booked_at.",
                kind=SemanticKind.TIME_SEMANTICS,
                field_name="booked_at",
            )
        )
        await agent.save_semantic_annotation(
            annotation(
                "conflict-a",
                "Refund state open means pending.",
                kind=SemanticKind.CODE_MAPPING,
                field_name="refund_state",
            )
        )
        await agent.save_semantic_annotation(
            annotation(
                "conflict-b",
                "Refund state open means incomplete.",
                kind=SemanticKind.CODE_MAPPING,
                field_name="refund_state",
            )
        )
        stale = replace(
            annotation(
                "aaa-stale-grain",
                "Invoice grain is one row per invoice.",
                kind=SemanticKind.GRAIN,
                field_name="booked_at",
            ),
            catalog_revisions=(ResourceRevisionBinding(resource.id, "stale-revision"),),
        )
        await agent._embedded._store.save_semantic_annotation(agent.id, stale)

        runtime = agent._embedded._capability_runtime
        semantic_domain = agent._embedded._semantic_domain
        read_run = RunInput(
            id="semantic-read-run",
            agent_id=agent.id,
            message="inspect current semantics",
            created_at=NOW,
            conversation_id="semantic-read-conversation",
            source_id=source.id,
        )
        semantic_domain.select_explicit_learning_run(read_run.id)
        listed = (
            await execute_projected(
                runtime,
                read_run,
                (
                    ToolCall(
                        id="list-semantics",
                        name="semantic_list",
                        arguments={"limit": 1},
                    ),
                ),
            )
        )[0]
        listed_data = listed.output["data"]
        assert isinstance(listed_data, Mapping)
        listed_annotations = listed_data["annotations"]
        assert isinstance(listed_annotations, tuple)
        assert tuple(item["id"] for item in listed_annotations) == ("active-time",)
        conflict_view = (
            await execute_projected(
                runtime,
                read_run,
                (
                    ToolCall(
                        id="view-conflict",
                        name="semantic_view",
                        arguments={"id": "conflict-a"},
                    ),
                ),
            )
        )[0]
        assert not conflict_view.is_error
        conflict_output = conflict_view.output["data"]
        assert isinstance(conflict_output, Mapping)
        conflict_maintenance = conflict_output["maintenance"]
        assert isinstance(conflict_maintenance, Mapping)
        assert conflict_maintenance["state"] == "conflicting"
        assert conflict_maintenance["usable_as_current_meaning"] is False
        assert conflict_maintenance["requires_revalidation"] is True
        semantic_domain.clear_explicit_learning_run(read_run.id)

        controller = PresentationController(root=None)
        controller.agent = agent
        rendered = (await controller.dispatch_command("/memory")).message
        assert "Global memory:" in rendered
        assert "Active data semantics:" in rendered
        assert "active-time" in rendered
        assert "Stale definitions:" in rendered
        assert "aaa-stale-grain" in rendered
        assert "Conflicts:" in rendered
        assert "conflict-a" in rendered
        assert "conflict-b" in rendered

        detail = (await controller.dispatch_command("/memory show active-time")).message
        assert "Verified revisions:" in detail
        assert "Confirmed:" in detail
        assert "Current SHA-256:" in detail
        assert "Evidence:" in detail
        assert "user_assertion" in detail

        assert await cli._handle_knowledge_chat_command(["/memory"], agent)
        cli_rendered = capsys.readouterr().out
        assert "Active data semantics:" in cli_rendered
        assert "Stale definitions:" in cli_rendered
        assert "Conflicts:" in cli_rendered

        memory_completion = next(
            item for item in SLASH_COMMAND_COMPLETIONS if item[0] == "/memory"
        )
        assert "duplicate, stale, conflicting, and superseded" in memory_completion[2]
    finally:
        await agent.close()
