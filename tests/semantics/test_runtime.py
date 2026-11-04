from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    SemanticEvidenceKind,
)
from daita.capabilities import CallTargetLane, OperationalEffect
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopLimits, RunInput
from daita.loop.session import RunSessionOptions
from daita.semantics import (
    SEMANTIC_DELETE_CAPABILITY_ID,
    SEMANTIC_LIST_CAPABILITY_ID,
    SEMANTIC_SAVE_CAPABILITY_ID,
    SEMANTIC_SAVE_TOOL_NAME,
    SEMANTIC_VIEW_CAPABILITY_ID,
    SEMANTIC_VIEW_TOOL_NAME,
    SemanticValidationError,
)
from daita.tui.commands import (
    learning_invocation_message as _learning_invocation_message,
)
from tests.support.capability_runtime import make_run_session
from tests.support.toolbox_model import (
    ToolboxAwareMockModelProvider as MockModelProvider,
    result_ref,
)
from tests.support.workspace import workspace_for

NOW = datetime(2026, 7, 28, 14, tzinfo=UTC)
EAGER_LIMITS = LoopLimits()


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=40_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _call(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _ids(run_tag: str | None = None):
    counts: dict[str, int] = {}

    def create(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        suffix = (
            f"{run_tag}-{counts[prefix]}"
            if prefix in {"conversation", "run"} and run_tag is not None
            else str(counts[prefix])
        )
        return f"{prefix}-{suffix}"

    return create


async def _seed_source(tmp_path, name: str):
    database = tmp_path / f"{name}.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE invoices("
            "id INTEGER PRIMARY KEY, booked_at TEXT, refund_state TEXT)"
        )
        connection.execute(
            "INSERT INTO invoices(booked_at, refund_state) VALUES ('2026-07-01', 'open')"
        )
    agent = await Agent.create(
        name, root=tmp_path, clock=lambda: NOW, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach_sqlite(database)
        resource = (await agent.list_catalog_resources())[0]
    finally:
        await agent.close()
    return database, source, resource


@pytest.mark.parametrize(
    "field_reference", ("invoices.booked_at", "booked_at", "`booked_at`")
)
async def test_schema_bound_memory_redirects_to_semantics_and_survives_restart(
    tmp_path,
    field_reference,
):
    database, source, resource = await _seed_source(tmp_path, "bound-memory")
    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    statement = f"Booked revenue uses {field_reference}."
    provider = MockModelProvider(
        (
            _call(
                ToolCall(
                    id="plain-memory",
                    name="memory_set",
                    arguments={
                        "target": "memory",
                        "action": "add",
                        "content": statement,
                    },
                )
            ),
            _call(
                _semantic_call(
                    source.id,
                    resource.id,
                    statement=statement,
                )
            ),
            _stop("Saved schema-bound meaning"),
        )
    )
    agent = await Agent.open(
        "bound-memory",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.learn("Remember: " + statement)
        assert (
            _error_code(_tool_results(provider)[0]) == "memory_schema_bound_definition"
        )
        assert await agent.read_memory() == ""
        assert len(approvals) == 1
        assert (await agent.list_semantic_annotations())[0].usable_as_current_meaning
    finally:
        await agent.close()
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE invoices RENAME COLUMN booked_at TO posted_at")
    agent = await Agent.open(
        "bound-memory", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        await agent.refresh_source(source.id)
        view = (await agent.list_semantic_annotations())[0]
        assert not view.usable_as_current_meaning
        assert view.requires_revalidation
        assert view.annotation.subject.fields[0].field_name == "booked_at"
    finally:
        await agent.close()


def _semantic_call(
    source_id: str,
    resource_id: str,
    *,
    call_id: str = "semantic-save",
    annotation_id: str = "booked-revenue",
    statement: str = "Booked revenue uses invoices.booked_at.",
    field_name: str = "booked_at",
    evidence_kind: str = "user_assertion",
    tool_call_id: str | None = None,
    supersedes_id: str | None = None,
) -> ToolCall:
    evidence: dict[str, object] = {"kind": evidence_kind}
    if tool_call_id is not None:
        evidence["result_ref"] = result_ref(tool_call_id)
    arguments: dict[str, object] = {
        "id": annotation_id,
        "subject": {
            "source_ids": [source_id],
            "resource_ids": [resource_id],
            "fields": [{"resource_id": resource_id, "field_name": field_name}],
        },
        "kind": "metric_definition",
        "statement": statement,
        "evidence": [evidence],
    }
    if supersedes_id is not None:
        arguments["supersedes_id"] = supersedes_id
    return ToolCall(id=call_id, name=SEMANTIC_SAVE_TOOL_NAME, arguments=arguments)


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


def _error_code(result: ToolResultBlock) -> str:
    error = result.output["error"]
    assert isinstance(error, Mapping)
    code = error["code"]
    assert isinstance(code, str)
    return code


async def test_semantic_tools_use_fixed_identities_and_the_existing_runtime_branch(
    tmp_path,
):
    _, source, resource = await _seed_source(tmp_path, "semantic-identities")
    provider = MockModelProvider((_stop(),))
    agent = await Agent.open(
        "semantic-identities",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        id_factory=_ids(),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        runtime = agent._embedded._capability_runtime
        projection_run = RunInput(
            id="projection-run",
            agent_id=agent.id,
            message="Teach this resource definition.",
            created_at=NOW,
            conversation_id="projection-conversation",
            source_scope_ids=(source.id,),
        )
        catalog = await runtime.prepare_session(
            make_run_session(
                projection_run,
                RunSessionOptions(explicit_learning=True),
            )
        )
        definitions = tuple(
            runtime._registry.tool_definition(
                entry.view.name, lane=CallTargetLane.FOREGROUND
            )
            for entry in catalog.entries
        )
        names = {item.name for item in definitions}
        assert {SEMANTIC_VIEW_TOOL_NAME, SEMANTIC_SAVE_TOOL_NAME} <= names
        registry = agent._embedded._capabilities
        # semantic_view lists without id; semantic_save deletes with action.
        expected: tuple[tuple[str, dict[str, str], str], ...] = (
            (SEMANTIC_VIEW_TOOL_NAME, {}, SEMANTIC_LIST_CAPABILITY_ID),
            (SEMANTIC_VIEW_TOOL_NAME, {"id": "x"}, SEMANTIC_VIEW_CAPABILITY_ID),
            (SEMANTIC_SAVE_TOOL_NAME, {}, SEMANTIC_SAVE_CAPABILITY_ID),
            (
                SEMANTIC_SAVE_TOOL_NAME,
                {"action": "delete"},
                SEMANTIC_DELETE_CAPABILITY_ID,
            ),
        )
        for name, arguments, capability_id in expected:
            capability = registry.call_capability(name, arguments)
            assert capability.id == capability_id
            assert capability.operational_effect is (
                OperationalEffect.CHANGE_ADVISORY_CONTEXT
                if name == SEMANTIC_SAVE_TOOL_NAME
                else OperationalEffect.NONE
            )
        save_definition = next(
            item for item in definitions if item.name == SEMANTIC_SAVE_TOOL_NAME
        )
        save_properties = save_definition.input_schema["properties"]
        assert isinstance(save_properties, Mapping)
        evidence_rule = save_properties["evidence"]
        assert isinstance(evidence_rule, Mapping)
        evidence_item = evidence_rule["items"]
        assert isinstance(evidence_item, Mapping)
        evidence_properties = evidence_item["properties"]
        assert isinstance(evidence_properties, Mapping)
        assert "run_id" not in evidence_properties
        assert "message_position" not in evidence_properties
        ordinary = await runtime.prepare_run(
            RunInput(
                id="ordinary-run",
                agent_id=agent.id,
                message="Which invoice has the largest value?",
                created_at=NOW,
                conversation_id="ordinary-conversation",
                source_scope_ids=(source.id,),
            )
        )
        assert not {
            SEMANTIC_VIEW_TOOL_NAME,
            SEMANTIC_SAVE_TOOL_NAME,
        }.intersection(item.view.name for item in ordinary.entries)
        incidental = await runtime.prepare_run(
            RunInput(
                id="incidental-run",
                agent_id=agent.id,
                message="Remember the definition while answering this semantic query.",
                created_at=NOW,
                conversation_id="incidental-conversation",
                source_scope_ids=(source.id,),
            )
        )
        assert not {
            SEMANTIC_VIEW_TOOL_NAME,
            SEMANTIC_SAVE_TOOL_NAME,
        }.intersection(item.view.name for item in incidental.entries)
    finally:
        await agent.close()


async def test_missing_approval_and_denial_bind_current_evidence_without_state(
    tmp_path,
):
    _, source, resource = await _seed_source(tmp_path, "semantic-denial")
    call = _semantic_call(source.id, resource.id)

    provider = MockModelProvider((_call(call), _stop("not saved")))
    agent = await Agent.open(
        "semantic-denial",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        id_factory=_ids(),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.learn("When we say booked revenue, use booked_at.")
        assert _error_code(_tool_results(provider)[0]) == "approval_required"
        assert await agent.list_semantic_annotations() == ()
    finally:
        await agent.close()

    approvals: list[ApprovalRequest] = []

    async def deny(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.DENY

    denied_call = _semantic_call(
        source.id,
        resource.id,
    )
    denied_arguments = dict(denied_call.arguments)
    denied_arguments["evidence"] = [
        {
            "kind": "user_assertion",
            "run_id": "current-run",
            "message_position": 99,
        }
    ]
    denied_call = ToolCall(
        id=denied_call.id,
        name=denied_call.name,
        arguments=denied_arguments,
    )
    provider = MockModelProvider((_call(denied_call), _stop("denied")))
    agent = await Agent.open(
        "semantic-denial",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=deny,
        id_factory=_ids("denied"),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.learn("Booked revenue means booked_at.")
        assert _error_code(_tool_results(provider)[0]) == "approval_denied"
        assert len(approvals) == 1
        assert approvals[0].tool_name == SEMANTIC_SAVE_TOOL_NAME
        approval_evidence = approvals[0].arguments["evidence"]
        assert isinstance(approval_evidence, tuple)
        assert len(approval_evidence) == 1
        assert isinstance(approval_evidence[0], Mapping)
        assert approval_evidence[0]["run_id"] == result.run_id
        assert approval_evidence[0]["message_position"] == 0
        assert await agent.list_semantic_annotations() == ()
    finally:
        await agent.close()


async def test_catalog_and_transcript_evidence_fail_before_approval(tmp_path):
    _, source, resource = await _seed_source(tmp_path, "semantic-invalid")
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    calls = (
        # Well-formed IDs the catalog does not hold; malformed IDs fail earlier
        # as schema input errors.
        _semantic_call(
            source.id,
            "catalog-resource:sha256:" + "e" * 64,
            annotation_id="missing-resource",
        ),
        _semantic_call(
            "source:sha256:" + "f" * 64,
            resource.id,
            annotation_id="source-mismatch",
        ),
        _semantic_call(
            source.id,
            resource.id,
            annotation_id="missing-field",
            field_name="missing",
        ),
        _semantic_call(
            source.id,
            resource.id,
            annotation_id="bad-evidence",
            evidence_kind="tool_result",
            tool_call_id="missing-call",
        ),
    )
    provider = MockModelProvider(
        tuple(item for call in calls for item in (_call(call), _stop("recovered")))
    )
    agent = await Agent.open(
        "semantic-invalid",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        for prompt in (
            "Teach the missing resource definition.",
            "Teach the source-mismatched definition.",
            "Teach missing.",
            "Teach invalid evidence.",
        ):
            await agent.learn(prompt)
        assert tuple(_error_code(item) for item in _tool_results(provider)) == (
            "semantic_unknown_resource",
            "source_scope_violation",
            "semantic_unknown_field",
            "result_ref_invalid",
        )
        assert approvals == []
        assert await agent.list_semantic_annotations() == ()
    finally:
        await agent.close()


async def test_semantic_replacement_and_deletion_require_current_digests(tmp_path):
    _, source, resource = await _seed_source(tmp_path, "semantic-digests")
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    def replace_with(statement: str, call_id: str) -> ToolCall:
        return _semantic_call(
            source.id, resource.id, call_id=call_id, statement=statement
        )

    def view(call_id: str) -> ToolCall:
        return ToolCall(
            id=call_id, name="semantic_view", arguments={"id": "booked-revenue"}
        )

    def delete(call_id: str) -> ToolCall:
        return ToolCall(
            id=call_id,
            name=SEMANTIC_SAVE_TOOL_NAME,
            arguments={"action": "delete", "id": "booked-revenue"},
        )

    # Each run names what it changes; Daita binds the version that run viewed.
    runs = (
        (_semantic_call(source.id, resource.id),),
        (replace_with("Booked revenue uses the paid timestamp.", "blind"),),
        (
            view("view-1"),
            replace_with("Booked revenue uses invoices.paid_at.", "replace-1"),
            replace_with("Booked revenue uses invoices.issued_at.", "stale-1"),
        ),
        (delete("blind-delete"),),
        (
            view("view-2"),
            replace_with("Booked revenue uses invoices.booked_at.", "replace-2"),
            delete("stale-delete"),
        ),
        (view("view-3"), delete("delete")),
    )
    provider = MockModelProvider(
        tuple(
            item
            for calls in runs
            for item in (*(_call(call) for call in calls), _stop("continued"))
        )
    )
    agent = await Agent.open(
        "semantic-digests",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        for message in (
            "Create the definition.",
            "Replace without loading.",
            "Replace twice from one view.",
            "Delete the definition without loading.",
            "Change it, then delete it from the earlier view.",
            "Delete the definition.",
        ):
            await agent.learn(message)
        results = {item.call_id: item for item in _tool_results(provider)}
        assert {
            call_id: _error_code(results[call_id])
            for call_id in ("blind", "stale-1", "blind-delete", "stale-delete")
        } == {
            "blind": "semantic_view_required",
            "stale-1": "semantic_stale_digest",
            "blind-delete": "semantic_view_required",
            "stale-delete": "semantic_stale_digest",
        }
        for call_id in ("semantic-save", "replace-1", "replace-2", "delete"):
            assert results[call_id].is_error is False, results[call_id].output
        # Stale changes fail before their card; each success asked once.
        assert len(approvals) == 4
        assert await agent.read_semantic_annotation("booked-revenue") is None
    finally:
        await agent.close()


async def test_state_change_during_semantic_approval_returns_state_changed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _database, source, resource = await _seed_source(tmp_path, "semantic-state-change")
    provider = MockModelProvider(
        (
            _call(
                _semantic_call(
                    source.id,
                    resource.id,
                )
            ),
            _stop("state changed"),
        )
    )

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        assert request.tool_name == SEMANTIC_SAVE_TOOL_NAME
        return ApprovalDecision.APPROVE

    agent = await Agent.open(
        "semantic-state-change",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        semantic_domain = agent._embedded._semantic_domain
        original_plan = semantic_domain.side_effect_plan
        validations = 0

        async def state_changes(run, call, capability, execution, fingerprint):
            nonlocal validations
            validations += 1
            if validations == 2:
                raise SemanticValidationError("semantic facts changed")
            return await original_plan(
                run,
                call,
                capability,
                execution,
                fingerprint,
            )

        monkeypatch.setattr(semantic_domain, "side_effect_plan", state_changes)
        await agent.learn("Booked revenue means booked_at.")
        result = _tool_results(provider)[0]
        assert _error_code(result) == "state_changed"
        assert await agent.read_semantic_annotation("booked-revenue") is None
        assert await agent.list_semantic_annotations() == ()
    finally:
        await agent.close()


async def test_tool_result_evidence_is_valid_and_save_is_recalled_after_reopen(
    tmp_path,
):
    _, source, resource = await _seed_source(tmp_path, "semantic-tool-evidence")
    query = ToolCall(
        id="query-1",
        name="data_query",
        arguments={
            "source_id": source.id,
            "resource_ids": (resource.id,),
            "sql": "SELECT booked_at, refund_state FROM invoices LIMIT 1",
            "parameters": [],
        },
    )
    semantic = _semantic_call(
        source.id,
        resource.id,
        evidence_kind="tool_result",
        tool_call_id="query-1",
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider((_call(query), _call(semantic), _stop("saved")))
    agent = await Agent.open(
        "semantic-tool-evidence",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.learn(
            "Confirm and save the booked revenue definition from the current data."
        )
        assert result.final_text == "saved"
        assert len(approvals) == 1
        assert approvals[0].tool_name == SEMANTIC_SAVE_TOOL_NAME
        approval_evidence = approvals[0].arguments["evidence"]
        assert isinstance(approval_evidence, tuple)
        assert len(approval_evidence) == 1
        assert isinstance(approval_evidence[0], Mapping)
        assert approval_evidence[0]["run_id"] == result.run_id
        assert approval_evidence[0]["message_position"] == 2
        assert approval_evidence[0]["tool_call_id"] == "query-1"
        saved = await agent.read_semantic_annotation("booked-revenue")
        assert saved is not None
        assert saved.annotation.evidence[0].kind is SemanticEvidenceKind.TOOL_RESULT
        assert saved.annotation.evidence[0].run_id == result.run_id
        assert saved.annotation.evidence[0].message_position == 2
    finally:
        await agent.close()

    reopened_provider = MockModelProvider((_stop("recalled"),))
    reopened = await Agent.open(
        "semantic-tool-evidence",
        root=tmp_path,
        model=reopened_provider,
        model_profile=_profile(reopened_provider),
        id_factory=_ids("recall"),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        await reopened.run("How should invoices booked revenue be interpreted?")
        system_text = "\n".join(
            block.text
            for message in reopened_provider.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "Booked revenue uses invoices.booked_at." in system_text
        assert "current catalog and validated tool results remain authoritative" in (
            system_text
        )
    finally:
        await reopened.close()


async def test_natural_language_and_learn_route_to_semantics_without_new_command(
    tmp_path,
):
    _, source, resource = await _seed_source(tmp_path, "semantic-teaching")
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        (
            _call(
                _semantic_call(
                    source.id,
                    resource.id,
                    statement="Natural language booked revenue uses booked_at.",
                )
            ),
            _stop("learned"),
        )
    )
    agent = await Agent.open(
        "semantic-teaching",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.learn(
            "When we say booked revenue, we mean the invoices booked_at field."
        )
        prompt = "\n".join(
            block.text
            for message in provider.logical_requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "semantic_save=current resource/field meaning" in prompt
        assert "resource/field-scoped semantic annotation" in prompt
        assert "approval card is the only confirmation" in prompt
        assert len(approvals) == 1
    finally:
        await agent.close()

    routed = _learning_invocation_message(
        "/learn  Booked revenue uses invoices.booked_at. "
    )
    assert routed == "/learn Booked revenue uses invoices.booked_at."


async def test_a_note_on_a_view_in_a_source_read_in_full_binds_its_catalog_entry(
    tmp_path,
):
    database = tmp_path / "views.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, status TEXT)")
        connection.execute("CREATE VIEW open_orders AS SELECT id, status FROM orders")

    async def approve(request):
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        (
            _call(
                ToolCall("load", "toolbox_load", {"tool_names": ["semantic_save"]}),
            ),
            _call(
                ToolCall(
                    "save",
                    "semantic_save",
                    {
                        "id": "open-orders-meaning",
                        "kind": "glossary",
                        "statement": "Open orders await shipping.",
                        "subject": {
                            "sources": ["Views"],
                            "tables": ["main.open_orders"],
                            "fields": [
                                {"table": "main.open_orders", "field_name": "status"}
                            ],
                        },
                        "evidence": [{"kind": "user_assertion"}],
                    },
                ),
            ),
            _stop("Saved."),
        ),
        provider_id="mock:semantic-view",
    )
    agent = await Agent.create(
        "semantic-on-view",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        source = await agent.attach_sqlite(database, name="Views")
        view = next(
            item
            for item in await agent.list_catalog_resources(source_id=source.id)
            if item.name == "open_orders"
        )
        result = await agent.learn("Open orders await shipping.")
        assert result.reason == "completed", result
        (saved,) = await agent._embedded._store.list_semantic_annotations(agent.id)
        assert saved.subject.resource_ids == (view.id,)
    finally:
        await agent.close()


async def test_a_note_on_a_table_the_catalog_has_not_seen_is_refused_by_name(
    tmp_path,
):
    database = tmp_path / "fresh.db"
    sqlite3.connect(database).close()
    provider = MockModelProvider(
        (
            _call(ToolCall("load", "toolbox_load", {"tool_names": ["semantic_save"]})),
            _call(
                ToolCall(
                    "save",
                    "semantic_save",
                    {
                        "id": "fresh-meaning",
                        "kind": "glossary",
                        "statement": "Fresh holds new rows.",
                        "subject": {
                            "sources": ["Fresh"],
                            "tables": ["main.fresh"],
                            "fields": [{"table": "main.fresh", "field_name": "id"}],
                        },
                        "evidence": [{"kind": "user_assertion"}],
                    },
                ),
            ),
            _stop("Refused."),
        ),
        provider_id="mock:semantic-uncataloged",
    )
    agent = await Agent.create(
        "semantic-uncataloged",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        await agent.attach_sqlite(database, name="Fresh")
        # Created after attach and never refreshed: the catalog has no entry.
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE fresh(id INTEGER PRIMARY KEY)")
        await agent.learn("Fresh holds new rows.")
        (refusal,) = (item for item in _tool_results(provider) if item.is_error)
        assert _error_code(refusal) == "semantic_unknown_resource"
        assert "catalog-resource:" not in repr(refusal.output)
        assert await agent.list_semantic_annotations() == ()
    finally:
        await agent.close()
