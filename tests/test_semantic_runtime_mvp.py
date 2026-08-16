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
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from daita.semantics import (
    SEMANTIC_DELETE_CAPABILITY_ID,
    SEMANTIC_DELETE_TOOL_NAME,
    SEMANTIC_LIST_CAPABILITY_ID,
    SEMANTIC_LIST_TOOL_NAME,
    SEMANTIC_SAVE_CAPABILITY_ID,
    SEMANTIC_SAVE_TOOL_NAME,
    SEMANTIC_VIEW_CAPABILITY_ID,
    SEMANTIC_VIEW_TOOL_NAME,
    SemanticValidationError,
)
from daita.terminal import _learning_invocation_message

NOW = datetime(2026, 7, 28, 14, tzinfo=UTC)


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
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
    agent = await Agent.create(name, root=tmp_path, clock=lambda: NOW)
    try:
        source = await agent.attach_sqlite(database)
        resource = (await agent.list_catalog_resources())[0]
    finally:
        await agent.close()
    return database, source, resource


def _semantic_call(
    source_id: str,
    resource_id: str,
    revision: str,
    *,
    call_id: str = "semantic-save",
    annotation_id: str = "booked-revenue",
    statement: str = "Booked revenue uses invoices.booked_at.",
    field_name: str = "booked_at",
    evidence_kind: str = "user_assertion",
    tool_call_id: str | None = None,
    expected_sha256: str | None = None,
    supersedes_id: str | None = None,
) -> ToolCall:
    evidence: dict[str, object] = {"kind": evidence_kind}
    if tool_call_id is not None:
        evidence["tool_call_id"] = tool_call_id
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
        "catalog_revisions": [{"resource_id": resource_id, "revision": revision}],
    }
    if expected_sha256 is not None:
        arguments["expected_sha256"] = expected_sha256
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
        id_factory=_ids(),
        clock=lambda: NOW,
    )
    try:
        runtime = agent._embedded._data_tool_runtime
        projection_run = RunInput(
            id="projection-run",
            agent_id=agent.id,
            message="Teach this resource definition.",
            created_at=NOW,
            conversation_id="projection-conversation",
            source_id=source.id,
        )
        runtime.select_explicit_learning_run(projection_run.id)
        definitions = await runtime.definitions(projection_run)
        runtime.clear_explicit_learning_run(projection_run.id)
        names = {item.name for item in definitions}
        assert {
            SEMANTIC_LIST_TOOL_NAME,
            SEMANTIC_VIEW_TOOL_NAME,
            SEMANTIC_SAVE_TOOL_NAME,
            SEMANTIC_DELETE_TOOL_NAME,
        } <= names
        registry = agent._embedded._capabilities
        expected = {
            SEMANTIC_LIST_TOOL_NAME: SEMANTIC_LIST_CAPABILITY_ID,
            SEMANTIC_VIEW_TOOL_NAME: SEMANTIC_VIEW_CAPABILITY_ID,
            SEMANTIC_SAVE_TOOL_NAME: SEMANTIC_SAVE_CAPABILITY_ID,
            SEMANTIC_DELETE_TOOL_NAME: SEMANTIC_DELETE_CAPABILITY_ID,
        }
        for name, capability_id in expected.items():
            view, capability = registry.resolve_tool(name)
            assert view.capability_id == capability_id
            assert capability.id == capability_id
            assert capability.side_effecting is (
                name in {SEMANTIC_SAVE_TOOL_NAME, SEMANTIC_DELETE_TOOL_NAME}
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
        ordinary = await runtime.definitions(
            RunInput(
                id="ordinary-run",
                agent_id=agent.id,
                message="Which invoice has the largest value?",
                created_at=NOW,
                conversation_id="ordinary-conversation",
                source_id=source.id,
            )
        )
        assert not {
            SEMANTIC_LIST_TOOL_NAME,
            SEMANTIC_VIEW_TOOL_NAME,
            SEMANTIC_SAVE_TOOL_NAME,
            SEMANTIC_DELETE_TOOL_NAME,
        }.intersection(item.name for item in ordinary)
        incidental = await runtime.definitions(
            RunInput(
                id="incidental-run",
                agent_id=agent.id,
                message="Remember the definition while answering this semantic query.",
                created_at=NOW,
                conversation_id="incidental-conversation",
                source_id=source.id,
            )
        )
        assert not {
            SEMANTIC_LIST_TOOL_NAME,
            SEMANTIC_VIEW_TOOL_NAME,
            SEMANTIC_SAVE_TOOL_NAME,
            SEMANTIC_DELETE_TOOL_NAME,
        }.intersection(item.name for item in incidental)
    finally:
        await agent.close()


async def test_missing_approval_and_denial_bind_current_evidence_without_state(
    tmp_path,
):
    _, source, resource = await _seed_source(tmp_path, "semantic-denial")
    call = _semantic_call(source.id, resource.id, resource.current_revision)

    provider = MockModelProvider((_call(call), _stop("not saved")))
    agent = await Agent.open(
        "semantic-denial",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        clock=lambda: NOW,
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
        resource.current_revision,
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
        approval_handler=deny,
        id_factory=_ids("denied"),
        clock=lambda: NOW,
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
        _semantic_call(
            source.id,
            "missing-resource",
            "missing-revision",
            annotation_id="missing-resource",
        ),
        _semantic_call(
            "source-other",
            resource.id,
            resource.current_revision,
            annotation_id="source-mismatch",
        ),
        _semantic_call(
            source.id,
            resource.id,
            "stale-revision",
            annotation_id="stale-revision",
        ),
        _semantic_call(
            source.id,
            resource.id,
            resource.current_revision,
            annotation_id="missing-field",
            field_name="missing",
        ),
        _semantic_call(
            source.id,
            resource.id,
            resource.current_revision,
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
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
    )
    try:
        for prompt in (
            "Teach the missing resource definition.",
            "Teach the source-mismatched definition.",
            "Teach stale.",
            "Teach missing.",
            "Teach invalid evidence.",
        ):
            await agent.learn(prompt)
        assert tuple(_error_code(item) for item in _tool_results(provider)) == (
            "resource_read_not_allowed",
            "source_scope_violation",
            "semantic_stale_revision",
            "semantic_unknown_field",
            "semantic_invalid_evidence",
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

    create = _semantic_call(
        source.id,
        resource.id,
        resource.current_revision,
    )
    missing_replacement_digest = _semantic_call(
        source.id,
        resource.id,
        resource.current_revision,
        statement="Booked revenue uses the paid timestamp.",
    )
    stale_replacement_digest = _semantic_call(
        source.id,
        resource.id,
        resource.current_revision,
        statement="Booked revenue uses the paid timestamp.",
        expected_sha256="0" * 64,
    )
    missing_delete_digest = ToolCall(
        id="delete-missing",
        name=SEMANTIC_DELETE_TOOL_NAME,
        arguments={"id": "booked-revenue"},
    )
    stale_delete_digest = ToolCall(
        id="delete-stale",
        name=SEMANTIC_DELETE_TOOL_NAME,
        arguments={"id": "booked-revenue", "expected_sha256": "0" * 64},
    )
    provider = MockModelProvider(
        tuple(
            item
            for call in (
                create,
                missing_replacement_digest,
                stale_replacement_digest,
                missing_delete_digest,
                stale_delete_digest,
            )
            for item in (_call(call), _stop("continued"))
        )
    )
    agent = await Agent.open(
        "semantic-digests",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
    )
    try:
        for message in (
            "Create the definition.",
            "Replace without loading.",
            "Replace from a stale view.",
            "Delete the definition without loading.",
            "Delete the definition from a stale view.",
        ):
            await agent.learn(message)
        results = _tool_results(provider)
        assert results[0].is_error is False
        assert tuple(_error_code(item) for item in results[1:]) == (
            "semantic_expected_sha256_required",
            "semantic_stale_digest",
            "missing_arguments",
            "semantic_stale_digest",
        )
        assert len(approvals) == 1
        current = await agent.read_semantic_annotation("booked-revenue")
        assert current is not None
        assert current.annotation.statement == (
            "Booked revenue uses invoices.booked_at."
        )
        delete_run = RunInput(
            id="delete-approved-run",
            agent_id=agent.id,
            message="forget the definition",
            created_at=NOW,
            conversation_id="delete-approved-conversation",
            source_id=source.id,
        )
        runtime = agent._embedded._data_tool_runtime
        runtime.select_explicit_learning_run(delete_run.id)
        deleted = (
            await runtime.execute_all(
                delete_run,
                (
                    ToolCall(
                        id="delete-approved",
                        name=SEMANTIC_DELETE_TOOL_NAME,
                        arguments={
                            "id": current.annotation.id,
                            "expected_sha256": current.sha256,
                        },
                    ),
                ),
            )
        )[0]
        runtime.clear_explicit_learning_run(delete_run.id)
        assert deleted.is_error is False
        assert len(approvals) == 2
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
                    resource.current_revision,
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
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
    )
    try:
        runtime = agent._embedded._data_tool_runtime
        original_validation = runtime._validate_semantic_preflight
        validations = 0

        async def state_changes(run, capability, fingerprint):
            nonlocal validations
            validations += 1
            if validations == 2:
                raise SemanticValidationError("semantic facts changed")
            return await original_validation(run, capability, fingerprint)

        monkeypatch.setattr(runtime, "_validate_semantic_preflight", state_changes)
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
        name="data_query_sqlite",
        arguments={
            "source_id": source.id,
            "sql": "SELECT booked_at, refund_state FROM invoices LIMIT 1",
            "parameters": [],
        },
    )
    semantic = _semantic_call(
        source.id,
        resource.id,
        resource.current_revision,
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
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
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
                    resource.current_revision,
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
        approval_handler=approve,
        id_factory=_ids(),
        clock=lambda: NOW,
    )
    try:
        await agent.learn(
            "When we say booked revenue, we mean the invoices booked_at field."
        )
        prompt = "\n".join(
            block.text
            for message in provider.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "semantic_save=current resource/field meaning" in prompt
        assert len(approvals) == 1
    finally:
        await agent.close()

    routed = _learning_invocation_message(
        "/learn Booked revenue uses invoices.booked_at."
    )
    assert routed is not None
    assert "resource/field-scoped semantic annotation" in routed
    assert "approval card is the only confirmation" in routed
