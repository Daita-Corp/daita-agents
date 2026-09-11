"""Production composition acceptance with deterministic model, MCP and database I/O."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from daita import (
    Agent,
    CalendarDaySelector,
    CalendarSchedule,
    EffectRequirement,
    MCPToolSelection,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    ScheduledRoutineDraft,
)
from daita._json import FrozenJsonObject, canonical_json
from daita.adapters import postgresql as pg, postgresql_write as native
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.adapters.models import DiscoveryRequest, SourceRegistration
from daita.capabilities import (
    ApprovalDecision,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectOutcome,
)
from daita.catalog.models import ResourceKind, Sensitivity, TabularColumn, TabularIndex
from daita.distribution.models import OutcomeState
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from tests.data.writes._upsert_support import Database
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.mcp import (
    conformance_identities,
    mock_transport,
)
from tests.support.native_writes import (
    NOW,
    ScriptedResearchModel,
    create_fixture,
    response,
)
from tests.support.workspace import workspace_for


async def test_initial_native_discovery_names_are_available_without_search_or_activation(
    tmp_path, monkeypatch
):
    (
        agent,
        provider,
        db,
        _server,
        _binding,
        _resource,
        _constraints,
        batch,
        _clock,
        _approvals,
    ) = await create_fixture(tmp_path, monkeypatch)
    try:
        provider.replace_script((response(text="Inspection only."),))
        await agent.run("Save the requested company.")
        request = provider.requests[-1]
        system = "\n".join(
            b.text for b in request.messages[0].content if isinstance(b, TextBlock)
        )
        assert "data_preview_upsert_rows" in system
        assert "data_upsert_rows" in system
        assert "data_update_rows" not in system
        assert "load the needed preview and execution tools together" in system
        assert "data_upsert_rows" not in {tool.name for tool in request.tools}
        query = next(tool for tool in request.tools if tool.name == "data_query")
        properties = query.input_schema["properties"]
        assert isinstance(properties, Mapping)
        parameters_schema = properties["parameters"]
        assert isinstance(parameters_schema, Mapping)
        parameters = parameters_schema["description"]
        assert isinstance(parameters, str)
        assert "$1" in parameters and "SQLite" in parameters
        assert not await agent.list_effects()
        permission = await agent.preview_source_permissions(
            source_id=batch["source_id"],
            read_mode="all",
            read_resource_ids=(),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=batch["source_id"],
            confirmation_fingerprint=permission.confirmation_fingerprint,
        )
        provider.replace_script((response(text="Inspection only."),))
        await agent.run("Save the requested company.")
        request = provider.requests[-1]
        system = "\n".join(
            b.text for b in request.messages[0].content if isinstance(b, TextBlock)
        )
        assert "Native row mutation is unavailable" in system
        assert "data_upsert_rows" not in system
        assert "data_update_rows" not in system
        assert not await agent.list_effects()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "review_action", ["approve", "deny", "drift", "short_id", "unknown_id"]
)
async def test_update_review_preserves_exact_selection_and_rechecks_without_replay(
    tmp_path, monkeypatch, review_action
):
    from daita.tui.projection import approval_review_document
    from tests.data.writes._update_runtime_support import _Connection, _row

    agent, provider, db, _, _, resource, _, batch, _, _ = await create_fixture(
        tmp_path, monkeypatch
    )
    connections, approvals = [], []
    changed = False

    async def connect(*args, **kwargs):
        connection = _Connection(
            (_row(1, before="Concurrent writer" if changed else "Before"),),
            update_status="UPDATE 1",
        )
        connections.append(connection)
        return connection

    async def approve(request):
        nonlocal changed
        approvals.append(request)
        if review_action == "drift":
            changed = True
        return (
            ApprovalDecision.DENY
            if review_action == "deny"
            else ApprovalDecision.APPROVE
        )

    try:
        permission = await agent.preview_source_permissions(
            source_id=batch["source_id"],
            read_mode="all",
            read_resource_ids=(),
            relational_write_scopes={
                resource.id: {
                    "allowed_operations": ("update",),
                    "allowed_insert_columns": (),
                    "allowed_update_columns": ("name",),
                    "key_columns": ("id",),
                    "generated_identity_columns": (),
                    "max_rows": 1,
                }
            },
        )
        await agent.apply_source_permissions(
            source_id=batch["source_id"],
            confirmation_fingerprint=permission.confirmation_fingerprint,
        )
        monkeypatch.setattr(native, "_connect", connect)
        agent._embedded._capability_runtime._approval_handler = approve
        arguments = {
            "source_id": batch["source_id"],
            "resource_id": (
                resource.id[:-3]
                if review_action == "short_id"
                else (
                    resource.id[:-1] + ("0" if resource.id[-1] != "0" else "1")
                    if review_action == "unknown_id"
                    else resource.id
                )
            ),
            "where": (
                {"column": "domain", "operator": "eq", "value": "existing.test"},
            ),
            "assignments": ({"column": "name", "value": "Updated"},),
        }

        def apply(request):
            preview = next(
                block
                for message in request.messages
                for block in message.content
                if isinstance(block, ToolResultBlock) and block.call_id == "preview"
            )
            if review_action in {"short_id", "unknown_id"}:
                assert preview.is_error
                error = preview.output["error"]
                assert isinstance(error, Mapping)
                assert error["code"] == (
                    "invalid_argument_value"
                    if review_action == "short_id"
                    else "resource_read_not_allowed"
                )
                if review_action == "unknown_id":
                    assert "without shortening" in str(error["message"])
                assert resource.id not in canonical_json(error)
                return response(text="The exact target is unresolved; no change made.")
            assert not preview.is_error, preview.output
            data = preview.output["data"]
            assert isinstance(data, Mapping)
            return response(
                ToolCall(
                    id="write",
                    name="data_update_rows",
                    arguments={
                        **arguments,
                        "preview_fingerprint": data["preview_fingerprint"],
                        "expected_affected_rows": data["matched_rows"],
                    },
                )
            )

        provider.replace_script(
            [
                response(
                    ToolCall(
                        id="load",
                        name="toolbox_load",
                        arguments={
                            "tool_names": (
                                "data_preview_update_rows",
                                "data_update_rows",
                            )
                        },
                    )
                ),
                response(
                    ToolCall(
                        id="preview",
                        name="data_preview_update_rows",
                        arguments=arguments,
                    )
                ),
                apply,
                response(text="Report the exact returned effect evidence."),
            ]
        )
        result = await agent.run(
            "Change the name of the company whose domain is existing.test."
        )
        assert result.reason == "completed"
        transcript = await agent.transcript(result.run_id)
        receipts = await agent.list_effects()
        mutations = [
            entry
            for connection in connections
            for entry in connection.log
            if entry[0] == "execute" and str(entry[1]).startswith("UPDATE")
        ]
        if review_action in {"short_id", "unknown_id"}:
            assert not approvals and not connections and not receipts
            return
        assert len(approvals) == 1, [
            block.output
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.is_error
        ]
        approval = approvals[0]
        call = next(
            call
            for message in transcript.messages
            for call in message.tool_calls
            if call.name == "data_update_rows"
        )
        assert approval.arguments["arguments"] == call.arguments
        document, reviewable = approval_review_document(
            tool_name=approval.tool_name,
            capability_id=approval.capability_id,
            arguments_text=approval.render_arguments_for_review(),
            reason=approval.reason,
        )
        assert reviewable and document is not None
        for text in (
            "Connection: Company research",
            "Table: companies",
            "existing.test",
            "Before",
            "Updated",
            "1 matching row(s)",
            "Bounded samples",
        ):
            assert text in document
        assert "EXCLUSIVE" not in document and "sequence gaps" not in document
        assert "locks matching rows" in document
        assert len(mutations) == (1 if review_action == "approve" else 0)
        assert len(receipts) == (1 if review_action == "approve" else 0)
        assert (
            not db.rows
        )  # The existing update I/O fixture owns this test's source calls.
    finally:
        await agent.close()


def script(
    provider: ScriptedResearchModel,
    binding,
    batch,
    *,
    write=True,
    duplicate=False,
    schema=False,
) -> None:
    research_name = binding.tools[0].local_name

    def apply(request):
        previews = [
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.capability_id == "data.preview_upsert_rows"
            and not block.is_error
        ]
        assert previews, [
            block.output
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        data = previews[-1].output["data"]
        assert isinstance(data, Mapping)
        return response(
            ToolCall(
                id="write",
                name="data_upsert_rows",
                arguments={**batch, "preview_fingerprint": data["preview_fingerprint"]},
            )
        )

    steps: list[ModelResponse | Callable[[ModelRequest], ModelResponse]] = [
        response(
            ToolCall(
                id="load",
                name="toolbox_load",
                arguments={
                    "tool_names": (
                        research_name,
                        "data_preview_upsert_rows",
                        "data_upsert_rows",
                    )
                },
            )
        ),
        response(
            ToolCall(
                id="research",
                name=research_name,
                arguments={
                    "query": "Companies doing business with Acme; cite sources and limitations"
                },
            )
        ),
        response(
            ToolCall(id="preview", name="data_preview_upsert_rows", arguments=batch)
        ),
    ]
    if schema:
        steps.insert(
            1,
            response(
                ToolCall(
                    id="schema",
                    name="catalog_schema",
                    arguments={"resource_ids": [batch["resource_id"]]},
                )
            ),
        )
    if write:
        steps.append(apply)
    if duplicate:
        steps.append(
            response(
                ToolCall(
                    id="preview-again", name="data_preview_upsert_rows", arguments=batch
                )
            )
        )

        def repeat(request):
            result = apply(request)
            return replace(
                result, tool_calls=(replace(result.tool_calls[0], id="write-again"),)
            )

        steps.append(repeat)
    steps.append(
        response(
            text="Research found one reported relationship with Acme, based on one cited source. Coverage is incomplete; transaction evidence establishes storage, not truth of the researched claim."
        )
    )
    provider.replace_script(steps)


async def wait_delivery(agent, conversation_id, count):
    for _ in range(1000):
        inbox = await agent.inbox(conversation_id=conversation_id)
        if len(inbox) >= count:
            return await agent.inspect_delivery(inbox[0].delivery_id)
        await asyncio.sleep(0.005)
    inspection = await agent.list_routines()
    pytest.fail(f"Native routine did not deliver: {inspection}")


async def test_native_insert_discovery_keeps_catalog_and_export_queries_distinct(
    tmp_path, monkeypatch
):
    agent, provider, db, *_ = await create_fixture(tmp_path, monkeypatch)
    cases = (
        (
            "Insert one row into an admitted catalog database table with exact supplied fields, preserving omitted columns, and return effect evidence.",
            "data_upsert_rows",
            3,
        ),
        ("catalog schema columns", "catalog_schema", 1),
        ("export all database rows csv", "data_export_tabular", 1),
    )
    provider.replace_script(
        (
            response(
                *(
                    ToolCall(f"search-{index}", "toolbox_search", {"query": query})
                    for index, (query, _, _) in enumerate(cases)
                )
            ),
            response(text="Discovery only."),
        )
    )
    try:
        result = await agent.run("Discover the admitted tools without executing work.")
        assert result.reason == "completed"
        transcript = await agent.transcript(result.run_id)
        results = dict((call.id, block) for call, block in transcript.tool_pairs)
        for index, (query, expected, rank) in enumerate(cases):
            block = results[f"search-{index}"]
            assert block is not None and not block.is_error
            data = block.output["data"]
            assert isinstance(data, Mapping)
            matches = data["matches"]
            assert isinstance(matches, tuple)
            names = []
            for item in matches:
                assert isinstance(item, Mapping)
                tool_name = item["tool_name"]
                assert isinstance(tool_name, str)
                names.append(tool_name)
            assert expected in names[:rank], (query, names)
        assert not db.rows and not await agent.list_effects()
    finally:
        await agent.close()


async def test_foreground_research_upsert_uses_authenticated_preview_and_one_receipt(
    tmp_path, monkeypatch
):
    (
        agent,
        provider,
        db,
        research,
        binding,
        resource,
        constraints,
        batch,
        clock,
        approvals,
    ) = await create_fixture(tmp_path, monkeypatch)
    try:
        script(provider, binding, batch, duplicate=True)
        result = await agent.run(
            "Research companies doing business with Acme and save cited findings now."
        )
        assert result.reason == "completed"
        receipts = await agent._embedded._store.list_effect_receipts(
            agent.id, run_id=result.run_id
        )
        assert len(receipts) == 1
        assert receipts[0].outcome is EffectOutcome.SUCCEEDED
        assert receipts[0].evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
        assert db.rows["new.test"]["evidence_url"] == "https://source.test/new"
        assert len(research.calls) == 1
        assert (
            len(
                [
                    item
                    for item in db.log
                    if item[0] == "fetch" and item[1].startswith("INSERT")
                ]
            )
            == 1
        )
        transcript = await agent._embedded._store.load(result.run_id)
        write_results = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "write"
        ]
        assert len(write_results) == 1 and not write_results[0].is_error
        assert isinstance(write_results[0].output["data"], Mapping)
        assert write_results[0].output["data"]["authorship"] == "model_derived"
        assert write_results[0].output["data"]["evidence_call_ids"] == ("research",)
        approval = next(
            item for item in approvals if item.capability_id == "data.upsert_rows"
        )
        write_call = next(
            call
            for message in transcript.messages
            for call in message.tool_calls
            if call.id == "write"
        )
        assert approval.arguments["arguments"] == write_call.arguments
        assert approval.arguments["target"]["name"] == "companies"
        assert approval.arguments["target"]["source_name"] == "Company research"
        assert approval.arguments["preview"]["inserted_count"] == 1
        assert approval.arguments["preview"]["updated_count"] == 0
        assert approval.arguments["preview"]["unchanged_count"] == 0
        assert canonical_json(
            approval.arguments["preview"]["classifications"]
        ) == canonical_json(({"key": {"domain": "new.test"}, "action": "insert"},))
        from daita.tui.projection import approval_review_document

        document, reviewable = approval_review_document(
            tool_name=approval.tool_name,
            capability_id=approval.capability_id,
            arguments_text=approval.render_arguments_for_review(),
            reason=approval.reason,
        )
        assert reviewable and document is not None
        assert "Table: companies" in document
        assert "Connection: Company research" in document
        assert "Preview: 1 insert, 0 update, 0 unchanged" in document
        assert "Exact validated details:" in document
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "natural_schema",
        "model_schema",
        "missing_effect",
        "commit_loss",
        "zero_budget",
        "two_calls",
        "two_native",
        "wrong_key",
        "row_ceiling",
        "stale_revision",
        "insert_column",
    ],
)
@pytest.mark.acceptance
async def test_immediate_and_weekly_research_upsert_production_path(
    tmp_path, monkeypatch, mode
):
    (
        agent,
        provider,
        db,
        research,
        binding,
        resource,
        constraints,
        batch,
        clock,
        approvals,
    ) = await create_fixture(tmp_path, monkeypatch)
    try:
        provider.replace_script(
            (
                response(
                    text="Prepare one immediate and weekly research assignment with explicit upsert permission."
                ),
            )
        )
        origin = await agent.run(
            "Research companies doing business with Acme, save cited findings now, and recheck every Monday. At least one verified write invocation is required."
        )
        destination = (
            await agent.distribution_destinations(
                origin.conversation_id, sensitivity_ceiling=ModelSensitivity.RESTRICTED
            )
        )[0]
        grant_constraints = {
            "source_id": batch["source_id"],
            "resource_id": resource.id,
            "resource_revision": resource.current_revision,
            **{
                key: value
                for key, value in constraints.items()
                if key != "allowed_operations"
            },
        }
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Acme relationship research",
            authorized_instruction="Research reported company relationships with Acme using the admitted research tool; preserve evidence URLs and describe source coverage. Preview and upsert one bounded batch into the exact companies target. A verified upsert invocation is required; no findings must report the missing effect.",
            schedule=CalendarSchedule(
                timezone="America/Chicago",
                hour=9,
                minute=0,
                day_selector=CalendarDaySelector.WEEKDAYS,
                weekdays=(1,),
            ),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(batch["source_id"],),
            allowed_resource_ids=(resource.id,),
            allowed_connector_binding_ids=(binding.binding_id,),
            allowed_capability_ids=(
                binding.tools[0].capability_id,
                "data.preview_upsert_rows",
                "data.upsert_rows",
                *(
                    ("catalog.schema",)
                    if mode in {"natural_schema", "model_schema"}
                    else ()
                ),
            ),
            sensitivity_ceiling=ModelSensitivity.RESTRICTED,
            outcome_contract=replace(
                no_artifact_outcome_contract(sensitivity=ModelSensitivity.RESTRICTED),
                effect_requirements=(
                    EffectRequirement(
                        "data.upsert_rows",
                        1,
                        frozenset({EffectEvidenceBasis.ADAPTER_VERIFIED}),
                    ),
                ),
            ),
            distribution_destination_id=destination.destination_id,
            eligible_model_routes=(provider.provider_id,),
            per_run_max_tokens=8000,
            per_run_max_cost_usd=(
                Decimal("0") if mode == "zero_budget" else Decimal("1")
            ),
            cumulative_max_tokens=40000,
            cumulative_max_cost_usd=Decimal("5"),
            cumulative_max_attempts=5,
            cumulative_max_occurrences=5,
            maximum_consecutive_failures=3,
            expires_at=NOW + timedelta(days=30),
            run_immediately=True,
            requested_capability_grants=(
                RequestedCapabilityGrant(
                    "data.upsert_rows",
                    FrozenJsonObject.from_mapping(grant_constraints),
                    1,
                ),
            ),
        )
        if mode in {
            "two_calls",
            "two_native",
            "wrong_key",
            "row_ceiling",
            "stale_revision",
            "insert_column",
        }:
            bad = dict(grant_constraints)
            if mode == "wrong_key":
                bad["key_columns"] = ("id",)
            elif mode == "row_ceiling":
                bad["max_rows"] = 11
            elif mode == "stale_revision":
                bad["resource_revision"] = "sha256:" + "0" * 64
            elif mode == "insert_column":
                bad["allowed_insert_columns"] = (
                    *bad["allowed_insert_columns"],
                    "notes",
                )
            draft = replace(
                draft,
                allowed_capability_ids=(
                    (*draft.allowed_capability_ids, "data.update_rows")
                    if mode == "two_native"
                    else draft.allowed_capability_ids
                ),
                requested_capability_grants=(
                    RequestedCapabilityGrant(
                        "data.upsert_rows",
                        FrozenJsonObject.from_mapping(bad),
                        2 if mode == "two_calls" else 1,
                    ),
                ),
            )
            with pytest.raises(CapabilityInputError) as error:
                await agent.propose_routine(draft)
            assert error.value.code in {
                "automation_grant_scope_invalid",
                "automation_grant_unsupported",
            }
            assert not await agent.list_routines()
            assert not db.rows and not research.calls
            return
        proposal = await agent.propose_routine(draft)
        script(
            provider,
            binding,
            batch,
            write=mode != "missing_effect",
            duplicate=mode == "commit_loss",
            schema=mode in {"natural_schema", "model_schema"},
        )
        if mode == "commit_loss":
            db.commit_error = ConnectionError("commit response lost")
        if mode == "model_schema":
            from daita.routines.capabilities import _spec_schema
            from daita.routines.owner import _routine_proposal_payload

            properties = _spec_schema(update=False)["properties"]
            assert isinstance(properties, Mapping)
            arguments = {
                key: value
                for key, value in _routine_proposal_payload(proposal).items()
                if key in properties and value is not None
            }
            arguments.update(
                skill_names=[],
                distribution_destination_id=destination.destination_id,
                requested_capability_grants=[
                    {
                        "capability_id": "data.upsert_rows",
                        "constraints": grant_constraints,
                        "max_calls_per_occurrence": 1,
                    }
                ],
            )

            def correct_route(request):
                errors = [
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "invalid-route"
                ]
                assert len(errors) == 1 and errors[0].is_error
                error = errors[0].output["error"]
                assert isinstance(error, Mapping)
                assert error["code"] == "routine_model_route_revoked"
                system = canonical_json(request.messages[0].content[0].text)
                assert provider.provider_id in system
                return response(ToolCall("create", "routine_create", arguments))

            def authoring_complete(request):
                created = [
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock) and block.call_id == "create"
                ]
                assert len(created) == 1 and not created[0].is_error, created
                script(provider, binding, batch, schema=True)
                return response(
                    text="The immediate and weekly assignment is saved; its hosted runs will report the results."
                )

            provider.replace_script(
                (
                    response(
                        ToolCall(
                            "load-authoring",
                            "toolbox_load",
                            {"tool_names": ["routine_create"]},
                        )
                    ),
                    response(
                        ToolCall(
                            "invalid-route",
                            "routine_create",
                            {**arguments, "eligible_model_routes": ["current"]},
                        )
                    ),
                    correct_route,
                    authoring_complete,
                )
            )
            authored = await agent.run(
                "Keep the researched companies current now and every Monday, using the approved table and permissions.",
                conversation_id=origin.conversation_id,
            )
            assert authored.kind.value == "completed"
            routines = await agent.list_routines()
            assert len(routines) == 1
            inspection = await agent.inspect_routine(routines[0].routine_id)
            assert inspection is not None
            routine = inspection.routine
            assert set(routine.allowed_capability_ids) == set(
                draft.allowed_capability_ids
            )
        else:
            routine = await agent.create_routine(proposal)
        delivery = await wait_delivery(agent, origin.conversation_id, 1)
        assert delivery is not None
        inspection = await agent.inspect_routine(routine.routine_id)
        assert inspection is not None
        occurrence = inspection.recent_occurrences[0]
        assert occurrence.reserved_run_id is not None
        if mode in {"success", "natural_schema", "model_schema"}:
            transcript = await agent._embedded._store.load(occurrence.reserved_run_id)
            assert (
                delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
            ), (
                occurrence,
                [
                    block.output
                    for message in transcript.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                ],
                len(provider.steps),
            )
            assert db.rows["new.test"]["name"] == "New Co"
            assert len(occurrence.effect_receipt_ids) == 1
            script(
                provider,
                binding,
                batch,
                schema=mode in {"natural_schema", "model_schema"},
            )
            clock[0] = datetime(2026, 9, 7, 14, tzinfo=UTC)
            agent._embedded._routine_supervisor.wake()
            weekly = await wait_delivery(agent, origin.conversation_id, 2)
            assert (
                weekly is not None
                and weekly.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
            )
            receipts = await agent._embedded._store.list_effect_receipts(agent.id)
            assert len(receipts) == 2
            unchanged_counts: list[int] = []
            for receipt in receipts:
                payload = receipt.payload
                assert payload is not None
                count = payload["unchanged_count"]
                assert type(count) is int
                unchanged_counts.append(count)
            assert sorted(unchanged_counts) == [0, 1]
            assert len(research.calls) == 2
            for request in provider.requests[(5 if mode == "model_schema" else 1) :]:
                system = "\n".join(
                    block.text
                    for message in request.messages
                    if message.role is MessageRole.SYSTEM
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
                assert "Routine authoring facts" not in system
                assert "catalog_inspect gives" not in system
                if mode == "success":
                    assert "catalog_schema first" not in system
                    assert "catalog_schema" not in {tool.name for tool in request.tools}
                    assert "Missing structure is unknown" in system
                else:
                    assert "catalog_schema first" in system
                    assert "catalog_schema" in {tool.name for tool in request.tools}
            inspection = await agent.inspect_routine(routine.routine_id)
            assert inspection is not None
            assert (
                len({item.reserved_run_id for item in inspection.recent_occurrences})
                == 2
            )
            for item in inspection.recent_occurrences:
                assert item.reserved_run_id is not None
                transcript = await agent._embedded._store.load(item.reserved_run_id)
                results = [
                    block
                    for message in transcript.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                ]
                previews = [
                    block
                    for block in results
                    if block.capability_id == "data.preview_upsert_rows"
                ]
                writes = [
                    block
                    for block in results
                    if block.capability_id == "data.upsert_rows"
                ]
                assert len(previews) == len(writes) == 1
                assert not previews[0].is_error and not writes[0].is_error
                schemas = [
                    block
                    for block in results
                    if block.capability_id == "catalog.schema"
                ]
                assert len(schemas) == (
                    1 if mode in {"natural_schema", "model_schema"} else 0
                )
                assert all(not block.is_error for block in schemas)
            assert (
                len(
                    [
                        item
                        for item in db.log
                        if item[0] == "fetch" and item[1].startswith("INSERT")
                    ]
                )
                == 1
            )
        else:
            assert (
                delivery.delivery.outcome.conclusion_state is not OutcomeState.SUCCEEDED
            )
            if mode in {"missing_effect", "zero_budget"}:
                if mode == "zero_budget":
                    terminal = await agent._embedded._store.result(
                        occurrence.reserved_run_id
                    )
                    assert terminal is not None
                    assert terminal.reason == "cost_limit_reached"
                assert not occurrence.effect_receipt_ids and not db.rows
            else:
                assert len(occurrence.effect_receipt_ids) == 1
                assert db.rows["new.test"]["name"] == "New Co"
                assert (
                    len(
                        [
                            item
                            for item in db.log
                            if item[0] == "fetch" and item[1].startswith("INSERT")
                        ]
                    )
                    == 1
                )
                with pytest.raises((ValueError, RuntimeError)):
                    await agent.run_routine_now(
                        routine.routine_id, expected_revision=routine.revision
                    )
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "invalid", ["missing_preview", "missing_evidence", "update_only"]
)
async def test_public_upsert_admission_cannot_invent_preview_evidence_or_insert_authority(
    tmp_path, monkeypatch, invalid
):
    (
        agent,
        provider,
        db,
        research,
        binding,
        resource,
        constraints,
        batch,
        clock,
        approvals,
    ) = await create_fixture(tmp_path, monkeypatch)
    try:
        if invalid == "update_only":
            read_only_update = {
                "allowed_operations": ("update",),
                "allowed_insert_columns": (),
                "allowed_update_columns": ("name",),
                "key_columns": ("id",),
                "generated_identity_columns": (),
                "max_rows": 10,
            }
            preview = await agent.preview_source_permissions(
                source_id=batch["source_id"],
                read_mode="all",
                read_resource_ids=(),
                relational_write_scopes={resource.id: read_only_update},
            )
            await agent.apply_source_permissions(
                source_id=batch["source_id"],
                confirmation_fingerprint=preview.confirmation_fingerprint,
            )
        name = (
            "data_upsert_rows"
            if invalid == "missing_preview"
            else "data_preview_upsert_rows"
        )
        arguments = (
            {**batch, "preview_fingerprint": "sha256:" + "0" * 64}
            if invalid == "missing_preview"
            else batch
        )
        provider.replace_script(
            (
                response(
                    ToolCall(
                        id="load",
                        name="toolbox_load",
                        arguments={"tool_names": (name,)},
                    )
                ),
                response(ToolCall(id="invalid-write", name=name, arguments=arguments)),
                response(text="No verified write occurred."),
            )
        )
        result = await agent.run("Save only admitted cited findings.")
        transcript = await agent._embedded._store.load(result.run_id)
        failures = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "invalid-write"
        ]
        assert len(failures) == 1 and failures[0].is_error
        assert not await agent._embedded._store.list_effect_receipts(agent.id)
        assert not db.rows and not research.calls
        assert not any(
            entry[0] == "fetch" and "upsert_target" in entry[1] for entry in db.log
        )
    finally:
        await agent.close()
