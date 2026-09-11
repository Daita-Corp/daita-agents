from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

import daita.artifacts.delivery as delivery_module
from daita import Agent, ArtifactDeliveryReceipt, cli
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactProvenance,
    ArtifactRef,
)
from daita.catalog.models import Sensitivity
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopExit, LoopExitKind, RunInput, Transcript
from daita.tui.projection import artifact_delivery_messages
from tests.support.toolbox_model import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)
from tests.support.workspace import workspace_for


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        return f"{prefix}-{counts[prefix]:032x}"

    return create


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


async def _create_artifact_agent(
    tmp_path: Path,
    name: str,
    downloads: Path,
) -> tuple[Agent, ArtifactRef]:
    provider = MockModelProvider(
        (
            _tool(
                "create",
                "artifact_create_document",
                {
                    "format": "txt",
                    "filename": "result.txt",
                    "content": "surface payload\n",
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="created"),
        ),
        provider_id=f"mock:{name}",
    )
    agent = await Agent.create(
        name,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    result = await agent.run("Create a TXT file.")
    return agent, result.artifacts[0]


def _surface_records() -> tuple[ArtifactRef, ArtifactDeliveryReceipt, LoopExit]:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    ref = ArtifactRef(
        artifact_id="artifact-00000000000000000000000000000001",
        run_id="run-00000000000000000000000000000001",
        conversation_id="conversation-00000000000000000000000000000001",
        call_id="create",
        capability_id="artifact.create_document",
        filename="result.txt",
        media_type="text/plain",
        byte_size=8,
        sha256="sha256:" + "1" * 64,
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        ),
        created_at=now,
    )
    receipt = ArtifactDeliveryReceipt(
        artifact_id=ref.artifact_id,
        destination_id="destination-system-downloads",
        filename="result.txt",
        saved_path="/verified/Downloads/result.txt",
        byte_size=ref.byte_size,
        sha256=ref.sha256,
        renamed_for_collision=False,
        delivered_at=now,
    )
    result = LoopExit(
        run_id=ref.run_id,
        conversation_id=ref.conversation_id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=now,
        final_text="model did not mention the path",
        steps=3,
        artifacts=(ref,),
        artifact_deliveries=(receipt,),
    )
    return ref, receipt, result
