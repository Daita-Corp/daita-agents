"""Authorized live-model acceptance coverage for source-free workspace reads.

This suite intentionally spends at most one live ``Agent.run`` interaction. It
proves that a real model can discover the newest workspace log through
``file_search``, read its tail through ``file_read``, and ground its answer in
the resulting source-free transcript. Descriptor containment, traversal,
revision, cursor, and byte-limit mechanics remain deterministic contracts.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent, LoopLimits, create_llm_provider
from daita._json import canonical_json
from daita.llm.models import (
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import (
    ModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.loop.models import (
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_LOCAL_WORKSPACE"
_MODEL_ID = "DAITA_LOCAL_WORKSPACE_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_LOCAL_WORKSPACE_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_LOCAL_WORKSPACE_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

_LATEST_PATH = "logs/current.log"
_SENTINEL = "LIVE_WORKSPACE_SENTINEL_6F21C9"
_EXPECTED_COUNT = "47"
_PROMPT = (
    "This is an authorized live local-workspace check. Use file_search in paths "
    "mode with order_by=modified_desc to find the most recently modified .log "
    "file. Then use file_read with position=end on that exact relative path. "
    "Return the exact verification token and processed_count found in the file. "
    "You must obtain both values from the file tool results; do not guess them."
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing one live "
            f"Agent.run interaction capped by {_MAX_COST}"
        ),
    ),
]


class _RecordingProvider:
    """Capture canonical requests and responses around one real provider."""

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []
        self.responses: list[ModelResponse] = []

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = await self._delegate.generate(request)
        self.responses.append(response)
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(self._delegate, StreamingModelProvider):
            raise TypeError("the live delegate must support canonical streaming")
        self.requests.append(request)
        async for event in self._delegate.stream(request):
            if isinstance(event, ModelStreamCompleted):
                self.responses.append(event.response)
            yield event


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} must be set for the authorized live test")
    return value


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.10")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _live_provider() -> tuple[ModelProfile, _RecordingProvider]:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(
            f"{_MODEL_ID} must name an API-backed provider: "
            + ", ".join(sorted(_API_PROVIDERS))
        )
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")
    if not profile.supports_streaming:
        pytest.fail(f"{_MODEL_ID} must name a model with reviewed streaming support")
    delegate = create_llm_provider(
        model_id,
        api_key=_required_environment(_MODEL_KEY),
        max_output_tokens=min(profile.max_output_tokens, 1_024),
    )
    return profile, _RecordingProvider(delegate)


def _limits() -> LoopLimits:
    return LoopLimits(
        max_steps=5,
        max_total_tokens=12_000,
        max_wall_time_seconds=90,
        max_estimated_cost_usd=_cost_limit(),
    )


def _successful_exchanges(
    transcript: Transcript,
    tool_name: str,
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    calls = {
        call.id: call
        for message in transcript.messages
        for call in message.tool_calls
        if call.name == tool_name
    }
    return tuple(
        (calls[block.call_id], block)
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.call_id in calls
        and not block.is_error
    )


async def test_live_model_finds_latest_log_and_reads_its_end(
    tmp_path: Path,
) -> None:
    """Exercise the complete source-free file-search/read model loop."""

    profile, provider = _live_provider()
    root = tmp_path / "live-workspace-root"
    workspace = workspace_for(root)
    logs = workspace.root / "logs"
    logs.mkdir()
    older = logs / "older.log"
    latest = logs / "current.log"
    older.write_text("verification_token=OLD\nprocessed_count=2\n", encoding="utf-8")
    latest.write_text(
        "service startup\n"
        "bounded diagnostic context\n"
        f"verification_token={_SENTINEL}\n"
        f"processed_count={_EXPECTED_COUNT}\n",
        encoding="utf-8",
    )
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(latest, (1_800_000_000, 1_800_000_000))

    agent = await Agent.create(
        "live-local-workspace",
        root=root,
        workspace=workspace,
        model=provider,
        model_profile=profile,
        limits=_limits(),
    )
    try:
        result = await agent.run(_PROMPT)
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.usage.cost_estimate.code,
        len(provider.requests),
    )
    assert result.final_text is not None
    assert _SENTINEL in result.final_text
    assert _EXPECTED_COUNT in result.final_text
    validate_completed_transcript(transcript, result)

    searches = _successful_exchanges(transcript, "file_search")
    reads = _successful_exchanges(transcript, "file_read")
    assert searches, "the live model did not complete file_search"
    assert reads, "the live model did not complete file_read"
    search_call, search_result = next(
        item for item in searches if _LATEST_PATH in canonical_json(item[1].output)
    )
    read_call, read_result = next(
        item for item in reads if _SENTINEL in canonical_json(item[1].output)
    )

    assert search_call.arguments["mode"] == "paths"
    assert search_call.arguments["order_by"] == "modified_desc"
    assert read_call.arguments["path"] == _LATEST_PATH
    assert read_call.arguments["position"] == "end"
    assert search_result.sensitivity is ModelSensitivity.INTERNAL
    assert read_result.sensitivity is ModelSensitivity.INTERNAL
    assert search_result.sensitivity_provenance["authority"] == (
        "local_workspace_binding"
    )
    assert read_result.sensitivity_provenance["authority"] == (
        "local_workspace_binding"
    )
    relative_paths = read_result.sensitivity_provenance["relative_paths"]
    assert isinstance(relative_paths, tuple)
    assert _LATEST_PATH in relative_paths

    assert provider.requests
    assert {tool.name for tool in provider.requests[0].tools} >= {
        "file_search",
        "file_read",
    }
    assert all(
        request.sensitivity is ModelSensitivity.INTERNAL
        for request in provider.requests
    )
    assert str(workspace.root) not in repr(provider.requests)
    assert str(workspace.root) not in repr(transcript)
