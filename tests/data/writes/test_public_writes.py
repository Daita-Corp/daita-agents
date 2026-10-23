"""Production composition acceptance with deterministic model, MCP and database I/O."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from daita import (
    CalendarDaySelector,
    CalendarSchedule,
    RoutineIntent,
)
from daita._json import canonical_json
from daita.adapters import postgresql_write as native
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    AutomationScopeProposal,
    CallTargetLane,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectOutcome,
    OperationalEffect,
)
from daita.capability_runtime import APPROVED_TARGET_CHANGED
from daita.distribution.models import OutcomeState
from daita.jobs import JobState
from daita.llm.models import (
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopLimits
from tests.support.data_write_approval import (
    TARGET_KEYS,
    UPDATE_PREVIEW_KEYS,
    UPSERT_PREVIEW_KEYS,
    data_write_approval,
)
from tests.support.distribution import inbox_distribution_plan
from tests.support.jobs import wait_for_job
from tests.support.native_writes import (
    NOW,
    ScriptedResearchModel,
    create_fixture,
    response,
)


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
        assert "data_preview_upsert_rows" not in system
        assert "data_upsert_rows" in system
        assert "data_update_rows" not in system
        assert "load the tool with toolbox_load" in system
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
                        "mode": "apply",
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
                        arguments={"tool_names": ("data_update_rows",)},
                    )
                ),
                response(
                    ToolCall(
                        id="preview",
                        name="data_update_rows",
                        arguments={**arguments, "mode": "preview"},
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
        approval_document = data_write_approval(approval)
        assert approval_document is not None
        assert set(approval_document["target"]) == TARGET_KEYS
        assert set(approval_document["preview"]) == UPDATE_PREVIEW_KEYS
        assert (
            approval_document["arguments"]["expected_affected_rows"]
            == approval_document["preview"]["matched_rows"]
        )
        call = next(
            call
            for message in transcript.messages
            for call in message.tool_calls
            if call.name == "data_update_rows" and call.arguments["mode"] == "apply"
        )
        assert dict(approval.arguments["arguments"]) == {
            key: value for key, value in call.arguments.items() if key != "mode"
        }
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
        if review_action == "drift":
            (write,) = [
                block
                for message in transcript.messages
                for block in message.content
                if isinstance(block, ToolResultBlock) and block.call_id == "write"
            ]
            assert write.is_error
            error = write.output["error"]
            assert isinstance(error, Mapping)
            assert error["code"] == "state_changed"
            assert dict(error["details"]["recovery"]) == (
                APPROVED_TARGET_CHANGED.to_details()
            )
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
    handles=False,
) -> None:
    research_name = binding.tools[0].local_name

    def named(request: ModelRequest) -> dict[str, object]:
        # With handles, the model names the table by its run-scoped refs.
        if not handles:
            return dict(batch)
        return {
            **batch,
            "source_id": request.scope_refs[batch["source_id"]],
            "resource_id": request.scope_refs[batch["resource_id"]],
        }

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
                arguments={
                    **named(request),
                    "mode": "apply",
                    "preview_fingerprint": data["preview_fingerprint"],
                },
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
        lambda request: response(
            ToolCall(
                id="preview",
                name="data_upsert_rows",
                arguments={**named(request), "mode": "preview"},
            )
        ),
    ]
    if schema:
        steps.insert(
            1,
            response(
                ToolCall(
                    id="schema",
                    name="catalog_describe",
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
                    id="preview-again",
                    name="data_upsert_rows",
                    arguments={**batch, "mode": "preview"},
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
        deliveries = await agent.list_deliveries(conversation_id=conversation_id)
        if len(deliveries) >= count:
            return await agent.inspect_delivery(deliveries[0].delivery_id)
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
        ("catalog schema columns", "catalog_describe", 1),
        ("export all database rows csv", "data_query", 1),
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
        script(provider, binding, batch, duplicate=True, handles=True)
        result = await agent.run(
            "Research companies doing business with Acme and save cited findings now."
        )
        assert result.reason == "completed"
        receipts = await agent._embedded._store.list_effect_receipts(
            agent.id, run_id=result.run_id
        )
        assert len(receipts) == 1
        assert receipts[0].call_id == "write"
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
        # Evidence is derived from earlier reads of the same table; the MCP
        # research read is not one, so the receipt cites none.
        assert write_results[0].output["data"]["evidence_call_ids"] == ()
        approval = next(
            item for item in approvals if item.capability_id == "data.upsert_rows"
        )
        approval_document = data_write_approval(approval)
        assert approval_document is not None
        assert set(approval_document["target"]) == TARGET_KEYS
        assert set(approval_document["preview"]) == UPSERT_PREVIEW_KEYS
        # The model named the table by handles; the executed record carries
        # the full identities the approval reviewed, in the tool's own shape.
        raw_write = next(
            call
            for message in transcript.messages
            for call in message.tool_calls
            if call.id == "write"
        )
        assert raw_write.arguments["source_id"] != batch["source_id"]
        write_call = next(
            call
            for call, _ in await agent.tool_evidence(result.run_id)
            if call.id == "write"
        )
        assert write_call.arguments["mode"] == "apply"
        assert write_call.arguments["source_id"] == batch["source_id"]
        assert write_call.arguments["resource_id"] == batch["resource_id"]
        assert dict(approval.arguments["arguments"]) == {
            key: value for key, value in write_call.arguments.items() if key != "mode"
        }
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


async def test_upsert_drift_during_approval_is_final_with_no_mutation(
    tmp_path, monkeypatch
):
    agent, provider, db, _, binding, _, _, batch, _, _ = await create_fixture(
        tmp_path, monkeypatch
    )
    concurrent = {
        "id": 42,
        "domain": "new.test",
        "name": "Concurrent writer",
        "evidence_url": "https://source.test/concurrent",
        "notes": None,
        "__daita_xmin": "7",
    }
    approvals = []

    async def approve_after_concurrent_write(request):
        approvals.append(request)
        if request.capability_id == "data.upsert_rows":
            db.rows["new.test"] = dict(concurrent)
        return ApprovalDecision.APPROVE

    try:
        agent._embedded._capability_runtime._approval_handler = (
            approve_after_concurrent_write
        )
        script(provider, binding, batch)
        result = await agent.run(
            "Research companies doing business with Acme and save cited findings now."
        )
        assert result.reason == "completed"
        transcript = await agent._embedded._store.load(result.run_id)
        (write,) = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "write"
        ]
        assert write.is_error
        error = write.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "state_changed"
        assert dict(error["details"]["recovery"]) == (
            APPROVED_TARGET_CHANGED.to_details()
        )
        assert [item.capability_id for item in approvals].count("data.upsert_rows") == 1
        assert not await agent.list_effects()
        assert db.rows == {"new.test": concurrent}
        assert not [
            item
            for item in db.log
            if item[0] in {"fetch", "execute"}
            and str(item[1]).startswith(("INSERT", "UPDATE", "LOCK TABLE"))
        ]
    finally:
        await agent.close()


async def test_upsert_evidence_is_derived_from_earlier_reads_of_the_same_table(
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
        script(provider, binding, batch, schema=True)
        result = await agent.run(
            "Research companies doing business with Acme and save cited findings now."
        )
        assert result.reason == "completed"
        transcript = await agent._embedded._store.load(result.run_id)
        outputs = {
            block.call_id: block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        assert not outputs["schema"].is_error
        assert not outputs["research"].is_error
        # Only the authenticated read of the written table is evidence; the
        # unrelated MCP research read and the write's own preview are not.
        for call_id in ("preview", "write"):
            data = outputs[call_id].output["data"]
            assert isinstance(data, Mapping)
            assert data["evidence_call_ids"] == ("schema",)
        approval = next(
            item for item in approvals if item.capability_id == "data.upsert_rows"
        )
        assert "evidence_call_ids" not in dict(approval.arguments["arguments"])
        definition = agent._embedded._capabilities.tool_definition(
            "data_upsert_rows", lane=CallTargetLane.FOREGROUND
        )
        properties = definition.input_schema["properties"]
        assert isinstance(properties, Mapping)
        assert "evidence_call_ids" not in properties
    finally:
        await agent.close()


async def test_exact_native_upsert_runs_once_from_a_job(tmp_path, monkeypatch) -> None:
    (
        agent,
        provider,
        db,
        _research,
        _binding,
        _resource,
        _constraints,
        batch,
        _clock,
        approvals,
    ) = await create_fixture(
        tmp_path,
        monkeypatch,
        limits=LoopLimits(max_estimated_cost_usd=Decimal("10")),
    )
    intent = {key: value for key, value in batch.items() if key != "evidence_call_ids"}
    objective = "Apply the exact company upsert once."

    def apply_preview(request: ModelRequest) -> ModelResponse:
        preview = next(
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "job-preview"
        )
        assert not preview.is_error, preview.output
        data = preview.output["data"]
        assert isinstance(data, Mapping)
        return response(
            ToolCall(
                "job-write",
                "data_upsert_rows",
                {
                    **intent,
                    "mode": "apply",
                    "preview_fingerprint": data["preview_fingerprint"],
                },
            )
        )

    # Only the job's main run calls the model.
    provider.replace_script(
        (
            response(
                ToolCall(
                    "load-write",
                    "toolbox_load",
                    {"tool_names": ("data_upsert_rows",)},
                )
            ),
            response(
                ToolCall(
                    "job-preview", "data_upsert_rows", {**intent, "mode": "preview"}
                )
            ),
            apply_preview,
            response(text="Applied the exact previewed company upsert."),
        )
    )
    try:
        job = await agent.start_job(objective, (batch["source_id"],))
        current = await wait_for_job(
            agent, job.job_id, lambda item: item.state is not JobState.RUNNING
        )
        assert current.state is JobState.DONE, canonical_json(
            {"state": current.state.value, "failure": current.failure_code}
        )
        receipts = await agent._embedded._store.job_effect_receipts(
            agent.id, job.job_id
        )
        assert len(receipts) == 1
        assert receipts[0].outcome is EffectOutcome.SUCCEEDED
        assert receipts[0].evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
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
        # The job's write asks at the time, naming the job that asked.
        assert [item.capability_id for item in approvals] == ["data.upsert_rows"]
        assert approvals[0].requester == objective
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
        intent = RoutineIntent(
            title="Acme relationship research",
            prompt="Research reported company relationships with Acme using the admitted research tool; preserve evidence URLs and describe source coverage. Preview and upsert one bounded batch into the exact companies target. If there are no findings, report that nothing was written.",
            schedule=CalendarSchedule(
                timezone="America/Chicago",
                hour=9,
                minute=0,
                day_selector=CalendarDaySelector.WEEKDAYS,
                weekdays=(1,),
            ),
            sources=(batch["source_id"],),
            run_immediately=True,
        )
        if mode in {
            "two_calls",
            "two_native",
            "wrong_key",
            "row_ceiling",
            "stale_revision",
            "insert_column",
        }:
            # Routines no longer carry standing native grants.
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
            capability_ids = (
                binding.tools[0].capability_id,
                "data.preview_upsert_rows",
                "data.upsert_rows",
                *(("data.update_rows",) if mode == "two_native" else ()),
            )
            proposal = AutomationScopeProposal(
                agent_id=agent.id,
                principal_id=f"agent:{agent.id}",
                allowed_source_ids=(batch["source_id"],),
                allowed_resource_ids=(resource.id,),
                allowed_connector_binding_ids=(binding.binding_id,),
                allowed_capability_ids=capability_ids,
                allowed_access_modes=frozenset(AccessMode),
                allowed_operational_effects=frozenset(OperationalEffect),
                sensitivity_ceiling=ModelSensitivity.RESTRICTED,
                eligible_model_routes=(provider.provider_id,),
                per_run_max_cost_usd=Decimal("1"),
                per_run_max_tokens=8000,
                expires_at=NOW + timedelta(days=30),
                distribution_plan_digest=inbox_distribution_plan(
                    "conversation-1", ModelSensitivity.RESTRICTED
                ).plan_digest,
            )
            with pytest.raises(CapabilityInputError) as error:
                await agent._embedded._capability_runtime.prepare_automation_grant(
                    "data.upsert_rows",
                    bad,
                    2 if mode == "two_calls" else 1,
                    proposal,
                )
            assert error.value.code in {
                "automation_grant_scope_invalid",
                "automation_grant_unsupported",
            }
            assert not await agent.list_routines()
            assert not db.rows and not research.calls
            return
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
            arguments = {
                "title": intent.title,
                "prompt": intent.prompt,
                "schedule": {
                    "kind": "calendar",
                    "timezone": "America/Chicago",
                    "hour": 9,
                    "minute": 0,
                    "weekdays": [1],
                },
                "sources": [batch["source_id"]],
                "run_immediately": True,
            }

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
                            {"tool_names": ["routine_save"]},
                        )
                    ),
                    response(ToolCall("create", "routine_save", arguments)),
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
            assert routine.source_ids == (batch["source_id"],)
        else:
            routine = await agent.create_routine(
                intent, conversation_id=origin.conversation_id
            )
        delivery = await wait_delivery(agent, origin.conversation_id, 1)
        assert delivery is not None
        inspection = await agent.inspect_routine(routine.routine_id)
        assert inspection is not None
        occurrence = inspection.recent_occurrences[0]
        assert occurrence.reserved_run_id is not None
        if mode == "missing_effect":
            # No write is a completed run that reports it; nothing was written.
            assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
            assert not occurrence.effect_receipt_ids and not db.rows
        elif mode in {"success", "natural_schema", "model_schema"}:
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
            for request in provider.requests[(4 if mode == "model_schema" else 1) :]:
                system = "\n".join(
                    block.text
                    for message in request.messages
                    if message.role is MessageRole.SYSTEM
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
                assert "Routine authoring facts" not in system
                # Scope is built at run start, so every run can describe its tables.
                assert "catalog_describe" in {tool.name for tool in request.tools}
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
                    if block.capability_id == "catalog.describe"
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
        name = "data_upsert_rows"
        arguments: dict[str, object] = (
            {**batch, "mode": "apply", "preview_fingerprint": "sha256:" + "0" * 64}
            if invalid == "missing_preview"
            else {**batch, "mode": "preview"}
        )
        if invalid == "missing_evidence":
            # Evidence is runtime-derived; a model cannot cite invented ids.
            arguments["evidence_call_ids"] = ("research",)
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
