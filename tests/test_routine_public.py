from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from _distribution_support import no_artifact_outcome_contract
from _workspace_support import workspace_for

from daita import (
    Agent,
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    RoutineState,
    ScheduledRoutineDraft,
    SQLiteSource,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=64_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


async def test_public_routine_surface_walks_create_and_lifecycle(
    tmp_path: Path,
) -> None:
    database = tmp_path / "current.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
    connection.execute("INSERT INTO current_value VALUES (7)")
    connection.commit()
    connection.close()

    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
        ),
        provider_id="mock:routines-public",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "routine-public",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Current"))
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        origin = await agent.run("Read the current value for a scheduled report.")
        assert origin.reason == "completed"
        assert origin.conversation_id is not None
        destinations = await agent.distribution_destinations(
            origin.conversation_id,
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
        )
        assert len(destinations) == 1
        assert destinations[0].kind == "conversation_inbox"
        assert destinations[0].selectable is True
        now = datetime.now(UTC)
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Current value report",
            authorized_instruction=(
                "Inspect the exact current_value resource and report its current value."
            ),
            schedule=IntervalSchedule(3_600, now + timedelta(hours=1)),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(source.id,),
            allowed_connector_binding_ids=(),
            allowed_resource_ids=(resource.id,),
            allowed_capability_ids=("catalog.inspect",),
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
            outcome_contract=no_artifact_outcome_contract(),
            distribution_destination_id=destinations[0].destination_id,
            eligible_model_routes=(provider.provider_id,),
            per_run_max_tokens=1_000,
            per_run_max_cost_usd=Decimal("0.01"),
            cumulative_max_tokens=10_000,
            cumulative_max_cost_usd=Decimal("0.10"),
            cumulative_max_attempts=10,
            cumulative_max_occurrences=10,
            maximum_consecutive_failures=3,
            expires_at=now + timedelta(days=30),
        )
        proposal = await agent.propose_routine(draft)
        assert proposal.next_due_at is None
        created = await agent.create_routine(proposal)
        assert created.state is RoutineState.ACTIVE
        assert (await agent.list_routines())[0].routine_id == created.routine_id
        assert await agent.inspect_routine(created.routine_id) is not None

        revision_origin = await agent.run(
            "Authorize the revised scheduled report definition.",
            conversation_id=origin.conversation_id,
        )
        revised = await agent.update_routine(
            created.routine_id,
            expected_revision=created.revision,
            draft=replace(
                draft,
                origin_run_id=revision_origin.run_id,
                title="Revised current value report",
            ),
        )
        assert revised.revision == created.revision + 1
        assert revised.title == "Revised current value report"
        assert revised.outcome_contract == draft.outcome_contract
        assert revised.distribution_plan == created.distribution_plan

        paused = await agent.pause_routine(
            revised.routine_id,
            expected_revision=revised.revision,
        )
        resumed = await agent.resume_routine(
            paused.routine_id,
            expected_revision=paused.revision,
        )
        await agent.set_memory("Unrelated private conversation note.")
        await agent.save_skill(
            "later-procedure",
            "Private later procedure",
            "Never part of this assignment.",
        )
        from daita import (
            ResourceRevisionBinding,
            SemanticAnnotation,
            SemanticEvidence,
            SemanticEvidenceKind,
            SemanticKind,
            SemanticSubject,
        )

        await agent.save_semantic_annotation(
            SemanticAnnotation(
                id="later-private-meaning",
                agent_id=agent.id,
                subject=SemanticSubject(
                    source_ids=(source.id,), resource_ids=(resource.id,), fields=()
                ),
                kind=SemanticKind.METRIC_DEFINITION,
                statement="Private later meaning of current_value.",
                evidence=(
                    SemanticEvidence(
                        SemanticEvidenceKind.USER_ASSERTION,
                        origin.run_id,
                        message_position=0,
                    ),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(resource.id, resource.current_revision),
                ),
                created_at=now,
                confirmed_at=now,
                sensitivity=ModelSensitivity.RESTRICTED,
            )
        )
        running = await agent.run_routine_now(
            resumed.routine_id,
            expected_revision=resumed.revision,
        )
        for _ in range(1_000):
            inbox = await agent.inbox(conversation_id=origin.conversation_id)
            if inbox:
                break
            await asyncio.sleep(0.005)
        inspection = await agent.inspect_routine(running.routine_id)
        assert inspection is not None
        assert len(inbox) == 1, (
            tuple(
                (
                    item.disposition,
                    item.failure_code,
                    item.reserved_run_id,
                    item.terminal_run_id,
                )
                for item in inspection.recent_occurrences
            ),
            len(provider.requests),
        )
        delivery = await agent.inspect_delivery(inbox[0].delivery_id)
        assert delivery is not None
        scheduled_request = provider.requests[-1]
        assert scheduled_request.sensitivity is ModelSensitivity.INTERNAL
        scheduled_text = repr(scheduled_request.messages)
        assert draft.authorized_instruction in scheduled_text
        assert "Authorize the revised scheduled report definition" not in scheduled_text
        assert "Unrelated private conversation note" not in scheduled_text
        assert "later-procedure" not in scheduled_text
        assert "Private later meaning" not in scheduled_text
        assert delivery.delivery.outcome.conclusion_digest == (
            inbox[0].conclusion_digest
        )
        disabled = await agent.disable_routine(
            running.routine_id,
            expected_revision=inspection.routine.revision,
        )
        assert disabled.state is RoutineState.DISABLED
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "classification", [ModelSensitivity.INTERNAL, ModelSensitivity.RESTRICTED]
)
async def test_public_source_free_once_routine_commits_required_document(
    tmp_path, classification
):
    from _toolbox_model_support import ToolboxAwareMockModelProvider
    from daita import OnceSchedule
    from daita.artifacts.models import ArtifactAuthorship
    from daita.distribution.models import ArtifactRequirement, OutcomeState
    from daita.llm.models import ToolCall

    usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
    provider = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The assignment is ready for approval.",
                usage=usage,
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                usage=usage,
                tool_calls=(
                    ToolCall(
                        "write-brief",
                        "artifact_create_document",
                        {
                            "format": "markdown",
                            "content": "# Local briefing\nNo outside research was requested.",
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The requested document is available.",
                usage=usage,
            ),
        ),
        complete_pricing=True,
    )
    agent = await Agent.create(
        "source-free-routine",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=replace(provider.model_profile, context_window_tokens=64000),
    )
    try:
        await agent.set_memory(
            "Background used to draft the assignment.", sensitivity=classification
        )
        origin = await agent.run("Prepare a one-time local briefing document.")
        assert origin.reason == "completed" and origin.conversation_id is not None
        destinations = await agent.distribution_destinations(
            origin.conversation_id, sensitivity_ceiling=classification
        )
        now = datetime.now(UTC)
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="One-time local briefing",
            authorized_instruction="Create a Markdown briefing documenting that no outside research was requested.",
            schedule=OnceSchedule(now),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(),
            allowed_resource_ids=(),
            allowed_connector_binding_ids=(),
            allowed_capability_ids=("artifact.create_document",),
            sensitivity_ceiling=classification,
            outcome_contract=replace(
                no_artifact_outcome_contract(),
                maximum_effective_sensitivity=classification,
                maximum_total_artifact_bytes=4096,
                artifact_requirements=(
                    ArtifactRequirement(
                        required=True,
                        minimum_count=1,
                        maximum_count=1,
                        allowed_media_types=("text/markdown",),
                        allowed_authorships=(
                            ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
                        ),
                        allowed_producer_capability_ids=("artifact.create_document",),
                        maximum_artifact_bytes=4096,
                        maximum_total_bytes=4096,
                        maximum_sensitivity=classification,
                    ),
                ),
            ),
            distribution_destination_id=destinations[0].destination_id,
            eligible_model_routes=(provider.provider_id,),
            per_run_max_tokens=5000,
            per_run_max_cost_usd=Decimal("0.10"),
            cumulative_max_tokens=5000,
            cumulative_max_cost_usd=Decimal("0.10"),
            cumulative_max_attempts=1,
            cumulative_max_occurrences=1,
            maximum_consecutive_failures=1,
            expires_at=now + timedelta(days=1),
        )
        from daita.routines.owner import RoutineError

        with pytest.raises(RoutineError, match="sensitivity"):
            await agent.propose_routine(
                replace(
                    draft,
                    sensitivity_ceiling=ModelSensitivity.PUBLIC,
                    outcome_contract=replace(
                        no_artifact_outcome_contract(),
                        maximum_effective_sensitivity=ModelSensitivity.PUBLIC,
                    ),
                )
            )
        await agent.set_memory("", sensitivity=ModelSensitivity.PUBLIC)
        created = await agent.create_routine(await agent.propose_routine(draft))
        for _ in range(1000):
            inbox = await agent.inbox(conversation_id=origin.conversation_id)
            if inbox:
                break
            await asyncio.sleep(0.005)
        assert len(inbox) == 1
        assert inbox[0].conclusion_state is OutcomeState.SUCCEEDED, tuple(
            item.result
            for item in await agent.conversation_runs(origin.conversation_id)
        )
        assert len(inbox[0].artifact_references) == 1
        assert provider.requests[-1].sensitivity is classification
        assert inbox[0].artifact_references[0].sensitivity.value == classification.value
        inspection = await agent.inspect_routine(created.routine_id)
        assert (
            inspection is not None
            and inspection.routine.state is RoutineState.COMPLETED
        )
        scope = inspection.recent_occurrences[0].execution_scope
        assert (
            scope is not None
            and scope.allowed_source_ids
            == scope.allowed_resource_ids
            == scope.allowed_connector_binding_ids
            == ()
        )
    finally:
        await agent.close()
