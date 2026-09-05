"""Authorized live-model acceptance coverage for the Phase 3 local text edit.

This suite spends at most one live ``Agent.run`` interaction. It proves that a
reviewed real model can carry one exact workspace binding through read, edit
artifact creation, approval, and bound atomic publication. Transformation,
containment, drift, metadata, cancellation, and uncertain-outcome mechanics
remain deterministic contracts in ``tests/test_local_text_edit.py``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    LoopLimits,
    create_llm_provider,
)
from daita.artifacts.models import ArtifactDeliveryMode, ArtifactDeliveryOutcome
from daita.llm.models import ModelProfile, ModelSensitivity, ToolCall, ToolResultBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ManagedModelProvider
from daita.loop.models import (
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_LOCAL_TEXT_EDIT"
_MODEL_ID = "DAITA_LOCAL_TEXT_EDIT_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_LOCAL_TEXT_EDIT_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_LOCAL_TEXT_EDIT_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_KEY_ENVIRONMENT = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_RELATIVE_PATH = "config/service.yaml"
_MARKER = "PHASE3_LIVE_BINDING_7C41D2"
_ORIGINAL_CONTENT = (
    "service: live-edit-check\n"
    "timeout_seconds: 30\n"
    f"verification_marker: {_MARKER}\n"
    "preserve_this: unchanged\n"
)
_EXPECTED_CONTENT = _ORIGINAL_CONTENT.replace(
    "timeout_seconds: 30",
    "timeout_seconds: 45",
)
_PROMPT = (
    f"In the local workspace, update {_RELATIVE_PATH}. Change exactly "
    "`timeout_seconds: 30` to `timeout_seconds: 45`, preserve every other "
    "character, and save the completed change back to that same file. Proceed "
    "with the save when approval is granted. After it succeeds, report the "
    f"relative path, the final timeout value, and verification marker {_MARKER}."
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"source .env and set {_AUTHORIZATION}=1 only after explicitly "
            f"authorizing one live Agent.run capped by {_MAX_COST}"
        ),
    ),
]


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} must be set for the authorized live test")
    return value


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.15")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _live_model() -> tuple[ModelProfile, ManagedModelProvider]:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")
    if not profile.supports_streaming:
        pytest.fail(f"{_MODEL_ID} must name a model with reviewed streaming support")
    provider_name = model_id.partition(":")[0]
    key_environment = _API_KEY_ENVIRONMENT.get(provider_name)
    if key_environment is None:
        pytest.fail(f"{_MODEL_ID} must name one API-backed reviewed model")
    api_key = os.environ.get(_MODEL_KEY) or _required_environment(key_environment)
    provider = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 1_024),
    )
    return profile, provider


def _limits() -> LoopLimits:
    return LoopLimits(
        max_steps=14,
        max_total_tokens=30_000,
        max_wall_time_seconds=120,
        max_estimated_cost_usd=_cost_limit(),
    )


def _successful_exchanges(
    transcript: Transcript,
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    calls = {
        call.id: call for message in transcript.messages for call in message.tool_calls
    }
    return tuple(
        (calls[block.call_id], block)
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.call_id in calls
        and not block.is_error
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        pytest.fail(f"{name} must be a mapping")
    return value


async def test_live_model_reads_edits_approves_and_replaces_exact_bound_file(
    tmp_path: Path,
) -> None:
    """Exercise the complete public Phase 3 model-led edit workflow."""

    profile, provider = _live_model()
    state_root = tmp_path / "live-local-text-edit-state"
    workspace = workspace_for(state_root)
    target = workspace.root / _RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(_ORIGINAL_CONTENT, encoding="utf-8")
    # Temporary directories can inherit a group outside the runner's groups.
    # The successful-publication fixture must satisfy the existing metadata guard.
    os.chown(target, -1, os.getegid())
    approvals: list[ApprovalRequest] = []
    content_at_approval: list[str] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        content_at_approval.append(target.read_text(encoding="utf-8"))
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "live-local-text-edit",
        root=state_root,
        workspace=workspace,
        model=provider,
        model_profile=profile,
        approval_handler=approve,
        limits=_limits(),
    )
    try:
        result = await agent.run(_PROMPT)
        transcript = await agent.transcript(result.run_id)
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()

    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.usage.cost_estimate.code,
    )
    validate_completed_transcript(transcript, result)
    assert result.final_text is not None
    assert _RELATIVE_PATH in result.final_text
    assert "45" in result.final_text
    assert _MARKER in result.final_text

    exchanges = _successful_exchanges(transcript)
    workflow = tuple(
        item
        for item in exchanges
        if item[0].name in {"file_read", "artifact_edit_text", "artifact_save_local"}
    )
    assert tuple(call.name for call, _block in workflow) == (
        "file_read",
        "artifact_edit_text",
        "artifact_save_local",
    )
    (read_call, read_result), (edit_call, edit_result), (save_call, save_result) = (
        workflow
    )

    read_data = _mapping(read_result.output.get("data"), "file_read data")
    binding = read_data.get("binding")
    assert isinstance(binding, str) and binding
    assert read_call.arguments["path"] == _RELATIVE_PATH
    assert set(read_call.arguments) <= {"path", "position"}
    assert read_call.arguments.get("position") in {None, "start"}
    assert set(edit_call.arguments) == {"binding", "replacements"}
    assert edit_call.arguments["binding"] == binding
    assert not {
        "path",
        "revision",
        "physical_revision",
        "content",
        "bytes",
    } & set(edit_call.arguments)

    artifact = _mapping(edit_result.output.get("artifact"), "edit artifact")
    artifact_id = artifact.get("artifact_id")
    assert isinstance(artifact_id, str) and artifact_id
    assert dict(save_call.arguments) == {
        "artifact_id": artifact_id,
        "mode": "replace_bound_file",
    }
    assert edit_result.sensitivity is ModelSensitivity.INTERNAL
    assert edit_result.sensitivity_provenance["authority"] == (
        "local_workspace_binding"
    )

    assert len(approvals) == 1
    assert content_at_approval == [_ORIGINAL_CONTENT]
    approval = approvals[0]
    assert approval.call_id == save_call.id
    assert approval.tool_name == "artifact_save_local"
    assert approval.arguments["artifact_id"] == artifact_id
    assert approval.arguments["mode"] == "replace_bound_file"
    assert approval.arguments["relative_path"] == _RELATIVE_PATH
    assert isinstance(approval.arguments.get("change_summary"), Mapping)

    assert target.read_text(encoding="utf-8") == _EXPECTED_CONTENT
    assert len(result.artifacts) == 1
    committed = result.artifacts[0]
    assert committed.artifact_id == artifact_id
    local_binding = committed.provenance.local_file_binding
    assert local_binding is not None
    assert local_binding.workspace_id.startswith("workspace:sha256:")
    assert local_binding.relative_path == _RELATIVE_PATH
    assert local_binding.change_summary.operation_count == 1

    assert len(result.artifact_deliveries) == 1
    receipt = result.artifact_deliveries[0]
    assert receipt.artifact_id == artifact_id
    assert receipt.mode is ArtifactDeliveryMode.REPLACE_BOUND_FILE
    assert receipt.outcome is ArtifactDeliveryOutcome.SUCCEEDED
    assert receipt.workspace_id == local_binding.workspace_id
    assert receipt.relative_path == local_binding.relative_path
    assert receipt.prior_physical_revision == local_binding.original_physical_revision
    assert receipt.result_physical_revision is not None
    save_data = _mapping(save_result.output.get("data"), "save receipt")
    assert save_data["artifact_id"] == artifact_id
    assert save_data["mode"] == "replace_bound_file"
    assert save_data["outcome"] == "succeeded"

    loaded: set[str] = set()
    for call, _block in exchanges:
        tool_names = call.arguments.get("tool_names")
        if call.name == "toolbox_load" and isinstance(tool_names, (tuple, list)):
            loaded.update(item for item in tool_names if isinstance(item, str))
    assert {"artifact_edit_text", "artifact_save_local"} <= loaded
    assert str(workspace.root) not in repr(transcript)
    assert str(workspace.root) not in approval.reason
    assert str(workspace.root) not in (approval.render_arguments_for_review() or "")
