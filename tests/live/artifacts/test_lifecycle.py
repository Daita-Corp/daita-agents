"""Three real-model runs through artifact creation, reuse, and durable deletion.

Deletion is a typed owner operation, not a model tool. Crash timing, cancellation,
and cleanup failures remain deterministic contracts in tests/artifacts/test_deletion.py.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    LoopLimits,
    create_llm_provider,
)
from daita.artifacts.models import ArtifactError
from daita.llm.models import ModelProfile, ToolCall, ToolResultBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ManagedModelProvider
from daita.loop.models import (
    LoopExit,
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_ARTIFACT_LIFECYCLE"
_MODEL_ID = "DAITA_ARTIFACT_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_ARTIFACT_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_ARTIFACT_LIVE_MAX_COST_USD"
_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_CONTENT = "Artifact lifecycle verification: ARTIFACT_LIVE_8C23A1; amount=73."
_CONTROL_CONTENT = (
    "Retain this independent artifact when lifecycle-proof.txt is deleted."
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 after authorizing three real-model runs; "
            f"each run is capped by {_MAX_COST} (default $0.15)"
        ),
    ),
]


def _cost_limit() -> Decimal:
    try:
        cost = Decimal(os.environ.get(_MAX_COST, "0.15"))
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not cost.is_finite() or cost <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return cost


def _live_model() -> tuple[ModelProfile, ManagedModelProvider]:
    model_id = os.environ.get(_MODEL_ID, "openai:gpt-5.6-terra")
    profile = reviewed_model_profile(model_id)
    key_name = _API_KEYS.get(model_id.partition(":")[0])
    if profile is None or not profile.supports_tools or key_name is None:
        pytest.fail(f"{_MODEL_ID} must name a reviewed API-backed tool-capable model")
    key = os.environ.get(_MODEL_KEY) or os.environ.get(key_name)
    if key is None or not key.strip():
        pytest.fail(f"set {_MODEL_KEY} or {key_name} for the authorized live test")
    return profile, create_llm_provider(
        model_id,
        api_key=key,
        max_output_tokens=min(profile.max_output_tokens, 1_024),
    )


def _exchanges(
    transcript: Transcript, name: str
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    calls = {
        call.id: call
        for message in transcript.messages
        for call in message.tool_calls
        if call.name == name
    }
    return tuple(
        (calls[block.call_id], block)
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id in calls
    )


def _data(block: ToolResultBlock) -> Mapping[str, object]:
    value = block.output["data"]
    assert isinstance(value, Mapping)
    return value


def _completed(
    result: LoopExit,
    transcript: Transcript,
    record_property: Callable[[str, object], None],
    stage: str,
    cost_limit: Decimal,
) -> None:
    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.provider_failure,
    )
    validate_completed_transcript(transcript, result)
    assert result.usage.input_tokens is not None and result.usage.input_tokens > 0
    assert result.usage.output_tokens is not None and result.usage.output_tokens > 0
    cost = result.usage.cost_estimate.amount_usd
    assert cost is not None and cost <= cost_limit
    record_property(f"artifact_{stage}_run_id", result.run_id)
    record_property(f"artifact_{stage}_input_tokens", result.usage.input_tokens)
    record_property(f"artifact_{stage}_output_tokens", result.usage.output_tokens)
    record_property(f"artifact_{stage}_estimated_cost_usd", str(cost))


async def test_live_model_artifact_creation_reuse_and_deletion_survive_reopen(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    profile, provider = _live_model()
    cost_limit = _cost_limit()
    limits = LoopLimits(max_estimated_cost_usd=cost_limit)
    root = tmp_path / "artifact-lifecycle-state"
    downloads = tmp_path / "exports"
    downloads.mkdir()
    workspace = workspace_for(root)
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        # Approve only delivery of this scenario's artifact to its private export.
        assert request.capability_id == "artifact.save_local"
        assert request.arguments["artifact_id"] == ref.artifact_id
        assert request.arguments["mode"] == "create_new"
        assert request.arguments["destination_id"] == destination.destination_id
        approvals.append(request)
        return ApprovalDecision.APPROVE

    async with AsyncExitStack() as resources:
        resources.push_async_callback(provider.close)
        agent = await Agent.create(
            "live-artifact-lifecycle",
            root=root,
            model=provider,
            model_profile=profile,
            limits=limits,
            workspace=workspace,
            downloads_directory=downloads,
            approval_handler=approve,
        )
        resources.push_async_callback(agent.close)
        await agent.set_export_destination(downloads)
        created = await agent.run(
            "Create two plain-text artifacts using artifact_create_document with format txt. "
            "The first is named lifecycle-proof.txt. "
            f"Its entire content must be exactly this string: {_CONTENT!r}. "
            "The second is named retained-control.txt. Its entire content must be "
            f"exactly {_CONTROL_CONTENT!r}. Do not save either locally yet. "
            "Stop after successful creation of both artifacts."
        )
        original = await agent.transcript(created.run_id)
        _completed(created, original, record_property, "creation", cost_limit)
        assert len(created.artifacts) == 2
        by_name = {item.filename: item for item in created.artifacts}
        assert set(by_name) == {"lifecycle-proof.txt", "retained-control.txt"}
        ref = by_name["lifecycle-proof.txt"]
        control = by_name["retained-control.txt"]
        assert (
            len(
                [
                    block
                    for _, block in _exchanges(original, "artifact_create_document")
                    if not block.is_error
                ]
            )
            == 2
        )
        expected = _CONTENT.encode()
        assert (await agent.read_artifact(ref.artifact_id)).content == expected
        assert (
            await agent.read_artifact(control.artifact_id)
        ).content == _CONTROL_CONTENT.encode()
        assert ref.sha256 == "sha256:" + sha256(expected).hexdigest()
        directory = agent.home / "artifacts" / ref.run_id / ref.artifact_id
        assert (directory / "payload").read_bytes() == expected
        assert (directory / "manifest.json").is_file()
        await agent.close()

        agent = await Agent.open(
            "live-artifact-lifecycle",
            root=root,
            model=provider,
            model_profile=profile,
            limits=limits,
            workspace=workspace,
            downloads_directory=downloads,
            approval_handler=approve,
        )
        resources.push_async_callback(agent.close)
        assert (await agent.read_artifact(ref.artifact_id)).content == expected
        destination = await agent.export_destination()
        used = await agent.run(
            f"Read artifact {ref.artifact_id} using artifact_read, then save that "
            "same artifact using artifact_save_local in create_new mode to "
            f"destination {destination.destination_id}. Do not create another artifact. "
            "After saving, report its verification marker and amount from the read result.",
            conversation_id=created.conversation_id,
        )
        used_transcript = await agent.transcript(used.run_id)
        _completed(used, used_transcript, record_property, "reuse", cost_limit)
        reads = _exchanges(used_transcript, "artifact_read")
        assert any(
            call.arguments.get("artifact_id") == ref.artifact_id
            and not block.is_error
            and _data(block)["text"] == _CONTENT
            for call, block in reads
        )
        assert not used.artifacts
        (delivery,) = used.artifact_deliveries
        saved_path = Path(delivery.saved_path)
        assert saved_path.parent == downloads
        assert saved_path.read_bytes() == expected
        assert used.final_text is not None and "ARTIFACT_LIVE_8C23A1" in used.final_text
        assert "73" in used.final_text

        assert await agent.delete_artifact(ref.artifact_id) is True
        assert await agent.delete_artifact(ref.artifact_id) is False
        assert not directory.exists()
        assert await agent.transcript(created.run_id) == original
        assert await agent.transcript(used.run_id) == used_transcript
        assert saved_path.read_bytes() == expected
        await agent.close()

        agent = await Agent.open(
            "live-artifact-lifecycle",
            root=root,
            model=provider,
            model_profile=profile,
            limits=limits,
            workspace=workspace,
            downloads_directory=downloads,
            approval_handler=approve,
        )
        resources.push_async_callback(agent.close)
        assert await agent.transcript(created.run_id) == original
        assert await agent.delete_artifact(ref.artifact_id) is False
        with pytest.raises(ArtifactError) as missing:
            await agent.read_artifact(ref.artifact_id)
        assert missing.value.code == "artifact_missing"
        with pytest.raises(ArtifactError) as missing_save:
            await agent.save_artifact(ref.artifact_id)
        assert missing_save.value.code == "artifact_missing"
        unavailable = await agent.run(
            f"Check whether artifact {ref.artifact_id} is available now. Call "
            "artifact_list, then try artifact_read with that exact ID even if "
            "it is absent from the list. Treat a missing-artifact error as the "
            "answer and clearly report that it is unavailable. Do not recreate "
            "it, create new artifacts, or save anything.",
            conversation_id=created.conversation_id,
        )
        last = await agent.transcript(unavailable.run_id)
        _completed(unavailable, last, record_property, "after_deletion", cost_limit)
        lists = _exchanges(last, "artifact_list")
        assert lists and all(not block.is_error for _, block in lists)
        for _, block in lists:
            summaries = _data(block)["artifacts"]
            assert isinstance(summaries, tuple)
            assert len(summaries) == 1 and isinstance(summaries[0], Mapping)
            assert summaries[0]["artifact_id"] == control.artifact_id
        reads = _exchanges(last, "artifact_read")
        assert reads and all(
            call.arguments.get("artifact_id") == ref.artifact_id
            and block.is_error
            and isinstance(error := block.output.get("error"), Mapping)
            and error.get("code") == "artifact_missing"
            for call, block in reads
        )
        assert unavailable.final_text is not None
        assert "unavailable" in unavailable.final_text.lower()
        assert not unavailable.artifacts and not unavailable.artifact_deliveries
        assert not directory.exists()
        assert saved_path.read_bytes() == expected
        assert (
            await agent.read_artifact(control.artifact_id)
        ).content == _CONTROL_CONTENT.encode()
        with sqlite3.connect(agent.home / "state.db") as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE artifact_id = ?",
                    (ref.artifact_id,),
                ).fetchone()[0]
                == 0
            )
        assert await agent.delete_artifact(control.artifact_id)
        with sqlite3.connect(agent.home / "state.db") as connection:
            assert (
                connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
            )
