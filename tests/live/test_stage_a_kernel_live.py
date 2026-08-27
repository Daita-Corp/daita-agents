"""Authorized live-model acceptance coverage for the Stage A kernel.

These tests intentionally cover only boundaries strengthened by a real provider:
canonical finish reasons, model-selected tool progression, stable request context,
provider routing, and durable completion. Exact cancellation, crash, storage-failure,
and uncertain-action timing remain deterministic failure-injection tests.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import AsyncIterator, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent, LoopLimits, SQLiteSource, create_llm_provider
from daita._json import canonical_json
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import (
    ModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.llm.routing import (
    ModelProviderRegistration,
    ModelRouter,
    RetryPolicy,
)
from daita.loop.models import (
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_A_KERNEL"
_MODEL_ID = "DAITA_STAGE_A_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_STAGE_A_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_STAGE_A_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})
_PROBE_TOKEN = "STAGE_A_LIVE_SENTINEL_7C91F2"
_PROBE_PROMPT = (
    "This is an authorized Stage A live-kernel check. Use the attached SQLite "
    "data tools to read the one row in stage_a_probe. Return its exact "
    "verification_token and amount. You must obtain both values from a tool "
    "result; do not guess them and do not answer from catalog schema alone."
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing up to "
            "three live model runs for the Stage A kernel suite"
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


class _UnavailableStreamingProvider:
    """One deterministic pre-I/O transient failure before the live fallback."""

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        self.requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self.profile.id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "injected failure before the live fallback",
            provider_id=self.provider_id,
            usage=ModelUsage(
                cost_estimate=CostEstimate.complete(Decimal("0")),
            ),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "injected failure before the live fallback",
            provider_id=self.provider_id,
            usage=ModelUsage(
                cost_estimate=CostEstimate.complete(Decimal("0")),
            ),
        )
        yield  # pragma: no cover - makes this an async iterator


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


def _live_provider(
    *, max_output_tokens: int
) -> tuple[ModelProfile, _RecordingProvider]:
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
        max_output_tokens=min(profile.max_output_tokens, max_output_tokens),
    )
    return profile, _RecordingProvider(delegate)


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE stage_a_probe (
                   probe_name TEXT PRIMARY KEY,
                   verification_token TEXT NOT NULL,
                   amount INTEGER NOT NULL
               )""")
        connection.execute(
            "INSERT INTO stage_a_probe VALUES (?, ?, ?)",
            ("kernel", _PROBE_TOKEN, 73),
        )


def _successful_query(
    transcript: Transcript,
) -> tuple[ToolCall, ToolResultBlock]:
    calls = {
        call.id: call
        for message in transcript.messages
        for call in message.tool_calls
        if call.name == "data_query_sqlite"
    }
    results = {
        block.call_id: block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }
    for call_id, call in calls.items():
        result = results.get(call_id)
        if result is not None and not result.is_error:
            return call, result
    pytest.fail("the live model did not complete one successful SQLite query")


def _limits() -> LoopLimits:
    return LoopLimits(
        max_steps=6,
        max_total_tokens=12_000,
        max_wall_time_seconds=90,
        max_estimated_cost_usd=_cost_limit(),
    )


async def test_live_tool_round_trip_has_stable_context_and_durable_completion(
    tmp_path: Path,
) -> None:
    """Exercise a real tool turn, then validate the exact reopened run record."""

    profile, provider = _live_provider(max_output_tokens=1_024)
    limits = _limits()
    root = tmp_path / "round-trip-root"
    database = tmp_path / "stage-a-round-trip.sqlite"
    _database(database)

    agent = await Agent.create(
        "live-stage-a-round-trip",
        root=root,
        model=provider,
        model_profile=profile,
        limits=limits,
        workspace=workspace_for(root),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Stage A probe"))
        result = await agent.run(_PROBE_PROMPT, source_id=source.id)
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.usage.cost_estimate.code,
        len(provider.requests),
    )
    assert result.final_text is not None
    assert _PROBE_TOKEN in result.final_text
    assert "73" in result.final_text
    validate_completed_transcript(transcript, result)

    call, query_result = _successful_query(transcript)
    assert _PROBE_TOKEN in canonical_json(query_result.output)
    assert query_result.sensitivity is ModelSensitivity.INTERNAL
    assert query_result.sensitivity_provenance["authority"] == (
        "current_admitted_resource_scope"
    )

    assert len(provider.requests) >= 2
    digests = {
        request.sensitivity_provenance["static_context_sha256"]
        for request in provider.requests
    }
    assert len(digests) == 1
    assert all(
        request.sensitivity is ModelSensitivity.INTERNAL
        for request in provider.requests
    )
    assert all(
        request.messages[0] == provider.requests[0].messages[0]
        for request in provider.requests
    )
    assert all(
        tuple(tool.name for tool in request.tools)
        == tuple(tool.name for tool in provider.requests[0].tools)
        for request in provider.requests
    )
    classified_call_ids: set[object] = set()
    for request in provider.requests:
        classified = request.sensitivity_provenance["classified_results"]
        assert isinstance(classified, tuple)
        classified_call_ids.update(
            item["call_id"] for item in classified if isinstance(item, Mapping)
        )
    assert call.id in classified_call_ids

    reopened = await Agent.open(
        "live-stage-a-round-trip",
        root=root,
        model=provider,
        model_profile=profile,
        limits=limits,
        workspace=workspace_for(root),
    )
    try:
        reopened_transcript = await reopened.transcript(result.run_id)
        conversation = await reopened.conversation_runs(result.conversation_id)
    finally:
        await reopened.close()

    persisted = next(
        item for item in conversation if item.transcript.run.id == result.run_id
    )
    assert persisted.result == result
    assert reopened_transcript == transcript
    assert persisted.result is not None
    validate_completed_transcript(reopened_transcript, persisted.result)


async def test_live_output_exhaustion_never_becomes_normal_completion(
    tmp_path: Path,
) -> None:
    """Force a live provider output limit and require an explicit failed exit."""

    profile, provider = _live_provider(max_output_tokens=64)
    agent = await Agent.create(
        "live-stage-a-output-limit",
        root=tmp_path / "output-limit-root",
        model=provider,
        model_profile=profile,
        limits=_limits(),
        workspace=workspace_for(tmp_path / "output-limit-root"),
    )
    try:
        result = await agent.run(
            "Do not call tools. Output exactly 800 numbered lines. Every line must "
            "contain its number and the phrase STAGE_A_OUTPUT_LIMIT_PROBE. Do not "
            "abbreviate, summarize, omit lines, or stop before line 800."
        )
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert result.kind is LoopExitKind.FAILED, (
        result.kind,
        result.reason,
        result.final_text,
    )
    assert result.reason in {
        "model_output_limit",
        ProviderErrorCode.OUTPUT_LIMIT.value,
    }
    assert result.final_text is None
    assert provider.requests
    assert all(
        response.finish_reason is not FinishReason.STOP
        for response in provider.responses
    )
    assert transcript.messages[0].content


async def test_live_fallback_provider_stays_sticky_through_tool_completion(
    tmp_path: Path,
) -> None:
    """Select a live fallback once and keep it for the later model request."""

    profile, live = _live_provider(max_output_tokens=1_024)
    unavailable_profile = ModelProfile(
        id="mock:stage-a-unavailable",
        context_window_tokens=profile.context_window_tokens,
        max_output_tokens=profile.max_output_tokens,
        supports_tools=True,
        supports_parallel_tools=profile.supports_parallel_tools,
        supports_structured_output=profile.supports_structured_output,
        supports_streaming=True,
    )
    unavailable = _UnavailableStreamingProvider(unavailable_profile)
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=unavailable,
                profile=unavailable_profile,
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
            ModelProviderRegistration(
                provider=live,
                profile=profile,
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
        ),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )
    database = tmp_path / "stage-a-fallback.sqlite"
    _database(database)

    agent = await Agent.create(
        "live-stage-a-fallback",
        root=tmp_path / "fallback-root",
        model=router,
        model_profile=router.model_profile,
        limits=_limits(),
        workspace=workspace_for(tmp_path / "fallback-root"),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Fallback probe"))
        result = await agent.run(_PROBE_PROMPT, source_id=source.id)
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.usage.cost_estimate.code,
        len(live.requests),
    )
    validate_completed_transcript(transcript, result)
    _successful_query(transcript)
    assert len(unavailable.requests) == 1
    assert len(live.requests) >= 2
    assert result.final_text is not None and _PROBE_TOKEN in result.final_text
