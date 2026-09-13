from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
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
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.workspace import workspace_for


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=64_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("mode", ["always", "changes_only"])
async def test_routine_reporting_input_error_precedes_binding_and_approval(
    tmp_path, monkeypatch, update, mode
):
    from daita.capabilities import ApprovalDecision
    from daita.distribution import outcome_contract_projection
    from daita.llm.models import ToolCall
    from tests.support.capability_runtime import execute_projected

    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        tuple(
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Authorize this report."
            )
            for _ in range(2)
        )
    )
    agent = await Agent.create(
        "reporting-input",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        origin = await agent.run("Create a scheduled report.")
        destinations = await agent.distribution_destinations(origin.conversation_id)
        now = datetime.now(UTC)
        arguments = {
            "title": "Report",
            "authorized_instruction": "Report completion.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
            "misfire_policy": "latest_only",
            "reporting_mode": "always",
            "allowed_source_ids": (),
            "allowed_resource_ids": (),
            "allowed_connector_binding_ids": (),
            "allowed_capability_ids": ("artifact.create_document",),
            "sensitivity_ceiling": "internal",
            "outcome_contract": outcome_contract_projection(
                no_artifact_outcome_contract()
            ),
            "distribution_destination_id": destinations[0].destination_id,
            "eligible_model_routes": (provider.provider_id,),
            "per_run_max_tokens": 1000,
            "per_run_max_cost_usd": "0.01",
            "cumulative_max_tokens": 10000,
            "cumulative_max_cost_usd": "0.10",
            "cumulative_max_attempts": 10,
            "cumulative_max_occurrences": 10,
            "maximum_consecutive_failures": 3,
            "expires_at": (now + timedelta(days=30)).isoformat(),
            "skill_names": (),
            "run_immediately": False,
        }
        runtime = agent._embedded._capability_runtime
        store = agent._embedded._store
        run = (await store.load(origin.run_id)).run
        if update:
            created = await execute_projected(
                runtime, run, (ToolCall("valid-create", "routine_create", arguments),)
            )
            assert not created.ordered_results[0].is_error, created.ordered_results[
                0
            ].output
            current = (await agent.list_routines())[0]
            arguments.update(
                routine_id=current.routine_id, expected_revision=current.revision
            )
            arguments.pop("run_immediately")
            origin = await agent.run(
                "Revise that report.", conversation_id=origin.conversation_id
            )
            run = (await store.load(origin.run_id)).run
        approvals.clear()
        arguments["reporting_mode"] = mode
        if mode == "always":
            arguments["precheck"] = {
                "capability_id": "catalog.schema",
                "contract_digest": "sha256:" + "a" * 64,
                "source_id": "source",
                "resource_id": "resource",
            }
        bindings = []
        owner = agent._embedded._routine_owner
        read_contracts = owner._execution_contract_reader

        async def record_binding(**kwargs):
            bindings.append(kwargs)
            return await read_contracts(**kwargs)

        monkeypatch.setattr(owner, "_execution_contract_reader", record_binding)
        result = await execute_projected(
            runtime,
            run,
            (
                ToolCall(
                    "invalid-reporting",
                    "routine_update" if update else "routine_create",
                    arguments,
                ),
            ),
        )
        error = result.ordered_results[0].output["error"]
        assert error["code"] == "routine_precheck_invalid", error
        assert "precheck" in error["message"]
        assert not bindings and not approvals
        assert len(await agent.list_routines()) == int(update)
        if update:
            assert (await agent.list_routines())[0].revision == current.revision
        assert not await agent.list_effects()
    finally:
        await agent.close()


async def test_public_routine_surface_walks_create_and_lifecycle(
    tmp_path: Path,
    monkeypatch,
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
        from daita.routines.owner import RoutineError

        async def forbidden_contract_read(**kwargs):
            raise AssertionError("invalid routes must fail before contract binding")

        for route in ("current", "unknown", provider.provider_id):
            with monkeypatch.context() as patch:
                owner = agent._embedded._routine_owner
                patch.setattr(
                    owner, "_execution_contract_reader", forbidden_contract_read
                )
                if route == provider.provider_id:
                    patch.setattr(owner, "_eligible_model_routes", ())
                with pytest.raises(RoutineError) as invalid:
                    await agent.propose_routine(
                        replace(draft, eligible_model_routes=(route,))
                    )
                assert invalid.value.code == "routine_model_route_revoked"
                assert not await agent.list_routines()
                assert not await agent.list_effects()
        proposal = await agent.propose_routine(draft)
        assert proposal.next_due_at is None
        created = await agent.create_routine(proposal)
        assert created.state is RoutineState.ACTIVE
        assert (await agent.list_routines())[0].routine_id == created.routine_id
        assert await agent.inspect_routine(created.routine_id) is not None

        from daita._json import canonical_json
        from daita.llm.models import TextBlock
        from daita.loop.models import RunInput
        from tests.capabilities._toolbox_support import _load

        authoring_run = RunInput(
            id="revision-facts",
            agent_id=agent.id,
            message="Revise the current_value report schedule.",
            created_at=now,
            conversation_id=origin.conversation_id,
        )
        runtime = agent._embedded._capability_runtime
        builder = agent._embedded._context_builder
        assert builder is not None
        messages = (authoring_run.start_message(),)
        catalog = await runtime.prepare_run(authoring_run)
        snapshot = await builder.prepare(authoring_run, messages, catalog)
        loaded, loaded_messages, projection = await _load(
            runtime,
            authoring_run,
            catalog,
            messages,
            ("routine_update",),
            call_id="load-revision",
        )
        assert not loaded.is_error
        request = builder.project(
            snapshot, loaded_messages, step=2, tool_context=projection
        )
        system = "\n".join(
            block.text
            for block in request.messages[0].content
            if isinstance(block, TextBlock)
        )
        assert canonical_json(owner.authoring_facts()) in system
        assert "routine_update" in {tool.name for tool in request.tools}

        revision_origin = await agent.run(
            "Authorize the revised current_value scheduled report definition.",
            conversation_id=origin.conversation_id,
        )

        async def forbidden_confirmation(
            request: ApprovalRequest,
        ) -> ApprovalDecision:
            raise AssertionError("an invalid revision must not request approval")

        for route in ("current", "unknown", provider.provider_id):
            with monkeypatch.context() as patch:
                patch.setattr(
                    owner, "_execution_contract_reader", forbidden_contract_read
                )
                if route == provider.provider_id:
                    patch.setattr(owner, "_eligible_model_routes", ())
                with pytest.raises(RoutineError) as invalid:
                    await agent.update_routine(
                        created.routine_id,
                        expected_revision=created.revision,
                        draft=replace(
                            draft,
                            origin_run_id=revision_origin.run_id,
                            eligible_model_routes=(route,),
                        ),
                        confirmation_handler=forbidden_confirmation,
                    )
                assert invalid.value.code == "routine_model_route_revoked"
                unchanged = await agent.inspect_routine(created.routine_id)
                assert unchanged is not None and unchanged.routine.revision == 1
                assert not await agent.list_effects()
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
    from daita import OnceSchedule
    from daita.artifacts.models import ArtifactAuthorship
    from daita.distribution.models import ArtifactRequirement, OutcomeState
    from daita.llm.models import ToolCall
    from tests.support.toolbox_model import ToolboxAwareMockModelProvider

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


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("budget", ["tokens", "cost_usd"])
async def test_routine_budget_error_precedes_preparation_and_next_step_corrects(
    tmp_path, monkeypatch, update, budget
):
    from daita.capabilities import ApprovalDecision
    from daita.distribution import outcome_contract_projection
    from daita.llm.models import ToolCall, ToolResultBlock
    from tests.support.capability_runtime import execute_projected
    from tests.support.toolbox_model import ToolboxAwareMockModelProvider

    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = ToolboxAwareMockModelProvider(
        [ModelResponse(finish_reason=FinishReason.STOP, text="Authorize the report.")]
    )
    agent = await Agent.create(
        "budget-input",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
        approval_handler=approve,
    )
    try:
        origin = await agent.run(
            "Use 100,000 tokens and $0.50 per run; 200,000 tokens and $1 total."
        )
        destinations = await agent.distribution_destinations(origin.conversation_id)
        now = datetime.now(UTC)
        arguments = {
            "title": "Report",
            "authorized_instruction": "Report completion.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
            "misfire_policy": "latest_only",
            "reporting_mode": "always",
            "allowed_source_ids": (),
            "allowed_resource_ids": (),
            "allowed_connector_binding_ids": (),
            "allowed_capability_ids": ("artifact.create_document",),
            "sensitivity_ceiling": "internal",
            "outcome_contract": outcome_contract_projection(
                no_artifact_outcome_contract()
            ),
            "distribution_destination_id": destinations[0].destination_id,
            "eligible_model_routes": (provider.provider_id,),
            "per_run_max_tokens": 100_000,
            "per_run_max_cost_usd": "0.50",
            "cumulative_max_tokens": 200_000,
            "cumulative_max_cost_usd": "1.00",
            "cumulative_max_attempts": 10,
            "cumulative_max_occurrences": 10,
            "maximum_consecutive_failures": 3,
            "expires_at": (now + timedelta(days=30)).isoformat(),
            "skill_names": (),
            "run_immediately": False,
        }

        runtime = agent._embedded._capability_runtime
        run = (await agent._embedded._store.load(origin.run_id)).run
        if update:
            created = await execute_projected(
                runtime, run, (ToolCall("seed", "routine_create", arguments),)
            )
            assert not created.ordered_results[0].is_error
            current = (await agent.list_routines())[0]
            arguments.update(
                routine_id=current.routine_id, expected_revision=current.revision
            )
            arguments.pop("run_immediately")
        approvals.clear()
        tool = "routine_update" if update else "routine_create"
        invalid = {
            **arguments,
            f"cumulative_max_{budget}": 91_936 if budget == "tokens" else "0.49",
        }
        owner = agent._embedded._routine_owner
        # These are the entry points before domain readiness or contract I/O.
        preparation = []
        with monkeypatch.context() as patch:

            async def forbidden(*args, **kwargs):
                preparation.append(True)
                raise AssertionError("invalid budgets must fail before preparation")

            patch.setattr(owner, "_prepare_requested_grants", forbidden)
            patch.setattr(owner, "_execution_contract_reader", forbidden)
            rejected = await execute_projected(
                runtime, run, (ToolCall("invalid", tool, invalid),)
            )
        error = rejected.ordered_results[0].output["error"]
        assert error["code"] == "routine_budget_invalid", error
        assert f"per_run_max_{budget}" in error["message"]
        assert f"cumulative_max_{budget}" in error["message"]
        assert "authorized" in error["message"]
        assert not preparation and not approvals
        # Unrelated owner failures must still remain internal errors.
        with monkeypatch.context() as patch:

            async def unrelated_failure(*args, **kwargs):
                raise ValueError("private unexpected owner failure")

            patch.setattr(owner, "_prepare_requested_grants", unrelated_failure)
            unexpected = await execute_projected(
                runtime, run, (ToolCall("unexpected", tool, arguments),)
            )
        assert (
            unexpected.ordered_results[0].output["error"]["code"]
            == "tool_execution_failed"
        )
        assert len(await agent.list_routines()) == int(update)
        assert not await agent.list_effects()

        provider.replace_script(
            [
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(ToolCall("bad-budget", tool, invalid),),
                ),
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(ToolCall("correct-budget", tool, arguments),),
                ),
                ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="The report is saved with the requested budgets.",
                ),
            ]
        )
        result = await agent.run(
            "Save the report with exactly 100,000 tokens and $0.50 per run, 200,000 tokens and $1 total.",
            conversation_id=origin.conversation_id,
        )
        assert result.kind.value == "completed"
        provider.assert_consumed()
        correction_request = provider.logical_requests[-2]
        errors = [
            block
            for message in correction_request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "bad-budget"
        ]
        assert len(errors) == 1 and errors[0].is_error
        next_error = errors[0].output["error"]
        assert isinstance(next_error, Mapping)
        assert next_error["code"] == "routine_budget_invalid"
        assert len(approvals) == 1
        routines = await agent.list_routines()
        assert len(routines) == 1
        inspection = await agent.inspect_routine(routines[0].routine_id)
        assert inspection is not None
        saved = inspection.routine
        approved = approvals[0].arguments["proposal"]
        for field in (
            "per_run_max_tokens",
            "cumulative_max_tokens",
            "per_run_max_cost_usd",
            "cumulative_max_cost_usd",
        ):
            assert str(approved[field]) == str(arguments[field])
        assert saved.revision == (2 if update else 1)
        assert (saved.per_run_max_tokens, saved.cumulative_max_tokens) == (
            100_000,
            200_000,
        )
        assert (saved.per_run_max_cost_usd, saved.cumulative_max_cost_usd) == (
            Decimal("0.50"),
            Decimal("1.00"),
        )
        assert not await agent.list_effects()
    finally:
        await agent.close()
