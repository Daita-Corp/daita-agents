"""Authorized live-model evaluation for the remaining Phase 5 routing gaps.

The three interactions cover only behavior not already exercised by the core live
baseline: files-only routing after a source is connected, one mixed local-file and
connected-source comparison, and model-led Markdown creation plus local delivery.
The deterministic Phase 5 suite remains authoritative for exact first-tool choices,
working-set replacement, bounds, failures, provenance, and absence of shell fallback.
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
    LoopExit,
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_PHASE5_TOOL_SELECTION"
_MODEL_ID = "DAITA_PHASE5_TOOL_SELECTION_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_PHASE5_TOOL_SELECTION_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_PHASE5_TOOL_SELECTION_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_KEY_ENVIRONMENT = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_FILES_TOKEN = "PHASE5_FILES_ONLY_7B3D19"
_SOURCE_EXCLUSION_TOKEN = "PHASE5_SOURCE_MUST_NOT_APPEAR_2E8A51"
_MIXED_FILE_TOKEN = "PHASE5_MIXED_FILE_4C9F20"
_MIXED_SOURCE_TOKEN = "PHASE5_MIXED_SOURCE_6A1D83"
_REPORT_TOKEN = "PHASE5_REPORT_9E4B27"

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing three live "
            f"Agent.run interactions, each capped by {_MAX_COST}"
        ),
    ),
]


class _RecordingProvider:
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
    raw = os.environ.get(_MAX_COST, "0.15")
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
    key_environment = _API_KEY_ENVIRONMENT.get(provider_name)
    if key_environment is None:
        pytest.fail(f"{_MODEL_ID} must name one API-backed reviewed model")
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")
    if not profile.supports_streaming:
        pytest.fail(f"{_MODEL_ID} must name a model with reviewed streaming support")
    api_key = os.environ.get(_MODEL_KEY) or _required_environment(key_environment)
    delegate = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 1_536),
    )
    return profile, _RecordingProvider(delegate)


def _limits() -> LoopLimits:
    return LoopLimits(
        max_steps=14,
        max_total_tokens=30_000,
        max_wall_time_seconds=120,
        max_estimated_cost_usd=_cost_limit(),
    )


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(f"""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                segment TEXT NOT NULL,
                verification_token TEXT NOT NULL
            );
            INSERT INTO customers VALUES
                (1, 'enterprise', '{_MIXED_SOURCE_TOKEN}'),
                (2, 'self-serve', '{_MIXED_SOURCE_TOKEN}'),
                (3, 'enterprise', '{_MIXED_SOURCE_TOKEN}');
            CREATE TABLE source_only_probe (
                verification_token TEXT NOT NULL
            );
            INSERT INTO source_only_probe VALUES ('{_SOURCE_EXCLUSION_TOKEN}');
            """)


def _exchanges(
    transcript: Transcript,
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    calls = {
        call.id: call for message in transcript.messages for call in message.tool_calls
    }
    return tuple(
        (calls[block.call_id], block)
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id in calls
    )


def _successful(
    transcript: Transcript,
    tool_name: str,
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    return tuple(
        (call, result)
        for call, result in _exchanges(transcript)
        if call.name == tool_name and not result.is_error
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        pytest.fail(f"{name} must be a mapping")
    return value


def _assert_completed(
    result: LoopExit,
    transcript: Transcript,
    provider: _RecordingProvider,
    workspace_root: Path,
) -> None:
    assert result.kind is LoopExitKind.COMPLETED, result
    assert result.final_text is not None
    validate_completed_transcript(transcript, result)
    assert provider.requests
    assert all(
        request.sensitivity is ModelSensitivity.INTERNAL
        for request in provider.requests
    )
    assert str(workspace_root) not in repr(provider.requests)
    assert str(workspace_root) not in repr(transcript)


async def test_live_files_only_ignores_connected_source_and_reads_forecast(
    tmp_path: Path,
) -> None:
    profile, provider = _live_provider()
    state_root = tmp_path / "files-only-state"
    workspace = workspace_for(state_root)
    forecast = workspace.root / "planning" / "Q4-forecast.md"
    forecast.parent.mkdir()
    forecast.write_text(
        f"verification_token={_FILES_TOKEN}\nforecast_total=91\n",
        encoding="utf-8",
    )
    database = tmp_path / "files-only.sqlite"
    _database(database)
    agent = await Agent.create(
        "live-phase5-files-only",
        root=state_root,
        workspace=workspace,
        model=provider,
        model_profile=profile,
        limits=_limits(),
    )
    try:
        await agent.attach(SQLiteSource(database, name="Connected customers"))
        result = await agent.run(
            "This is an authorized Phase 5 files-only check. Use only workspace "
            "files, even though a source is connected. Find the Q4 forecast file, "
            "read it, and report its exact verification_token and forecast_total. "
            "Do not use catalog or source-query tools and do not guess.",
            files_only=True,
        )
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    _assert_completed(result, transcript, provider, workspace.root)
    assert result.final_text is not None
    assert _FILES_TOKEN in result.final_text
    assert "91" in result.final_text
    assert _SOURCE_EXCLUSION_TOKEN not in result.final_text
    names = tuple(call.name for call, _block in _exchanges(transcript))
    assert names and names[0] == "file_search"
    assert "file_read" in names
    assert not any(
        name.startswith(("catalog_", "data_query_", "mcp_")) for name in names
    )
    initial = {tool.name for tool in provider.requests[0].tools}
    assert {"file_search", "file_read"} <= initial
    assert not any(
        name.startswith(("catalog_", "data_query_", "mcp_")) for name in initial
    )


async def test_live_mixed_file_source_comparison_uses_separate_queries(
    tmp_path: Path,
) -> None:
    profile, provider = _live_provider()
    state_root = tmp_path / "mixed-state"
    workspace = workspace_for(state_root)
    (workspace.root / "sales.csv").write_text(
        "region,revenue,verification_token\n"
        f"north,10,{_MIXED_FILE_TOKEN}\n"
        f"south,20,{_MIXED_FILE_TOKEN}\n"
        f"north,5,{_MIXED_FILE_TOKEN}\n",
        encoding="utf-8",
    )
    database = tmp_path / "mixed.sqlite"
    _database(database)
    agent = await Agent.create(
        "live-phase5-mixed",
        root=state_root,
        workspace=workspace,
        model=provider,
        model_profile=profile,
        limits=_limits(),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Connected customers"))
        result = await agent.run(
            "This is an authorized Phase 5 mixed-workspace check. Compare local "
            "sales.csv with the connected customers table using separate validated "
            "queries; never federate them in one SQL statement. Search the Files "
            "toolbox, load file_query, and aggregate local revenue by region with "
            "the local verification token. Separately inspect the connected schema "
            "and query customer counts by segment with its verification token. "
            "Report the exact totals, counts, and both tokens from tool results.",
            source_id=source.id,
        )
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    _assert_completed(result, transcript, provider, workspace.root)
    assert result.final_text is not None
    assert all(
        token in result.final_text
        for token in (_MIXED_FILE_TOKEN, _MIXED_SOURCE_TOKEN, "15", "20", "2", "1")
    )
    exchanges = _exchanges(transcript)
    names = tuple(call.name for call, _block in exchanges)
    assert (
        names.index("toolbox_search")
        < names.index("toolbox_load")
        < names.index("file_query")
    )
    assert names.index("catalog_schema") < names.index("data_query_sqlite")
    file_call, file_result = _successful(transcript, "file_query")[-1]
    source_call, source_result = _successful(transcript, "data_query_sqlite")[-1]
    assert file_call.arguments["path_pattern"] == "sales.csv"
    assert "customers" not in str(file_call.arguments["sql"])
    assert "sales.csv" not in str(source_call.arguments["sql"])
    assert _MIXED_FILE_TOKEN in canonical_json(file_result.output)
    assert _MIXED_SOURCE_TOKEN in canonical_json(source_result.output)
    assert file_result.sensitivity_provenance["authority"] == (
        "local_workspace_binding"
    )
    assert source_result.sensitivity_provenance["authority"] == (
        "current_admitted_resource_scope"
    )


async def test_live_markdown_report_is_created_then_saved_with_receipt(
    tmp_path: Path,
) -> None:
    profile, provider = _live_provider()
    state_root = tmp_path / "report-state"
    workspace = workspace_for(state_root)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    agent = await Agent.create(
        "live-phase5-report",
        root=state_root,
        workspace=workspace,
        downloads_directory=downloads,
        model=provider,
        model_profile=profile,
        limits=_limits(),
    )
    try:
        result = await agent.run(
            "This is an authorized Phase 5 artifact check. Create a Markdown report "
            "named phase5-report.md containing the heading Phase 5 Report, the exact "
            f"verification token {_REPORT_TOKEN}, north total 15, and south total 20. "
            "Use the Artifacts toolbox and save the committed report locally to the "
            "default destination. Claim it was saved only after a successful receipt."
        )
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    _assert_completed(result, transcript, provider, workspace.root)
    assert result.final_text is not None
    exchanges = _exchanges(transcript)
    names = tuple(call.name for call, _block in exchanges)
    assert (
        names.index("toolbox_search")
        < names.index("toolbox_load")
        < names.index("artifact_create_document")
    )
    assert names.index("artifact_create_document") < names.index("artifact_save_local")
    initial = {tool.name for tool in provider.requests[0].tools}
    assert "artifact_create_document" not in initial
    assert "artifact_save_local" not in initial
    assert len(result.artifacts) == 1
    assert len(result.artifact_deliveries) == 1
    artifact = result.artifacts[0]
    receipt = result.artifact_deliveries[0]
    assert receipt.artifact_id == artifact.artifact_id
    saved = Path(receipt.saved_path)
    assert saved.parent == downloads
    assert saved.name == "phase5-report.md"
    content = saved.read_text(encoding="utf-8")
    assert all(token in content for token in (_REPORT_TOKEN, "15", "20"))
    save_result = _successful(transcript, "artifact_save_local")[-1][1]
    save_data = _mapping(save_result.output.get("data"), "artifact save data")
    assert save_data["outcome"] == "succeeded"
    assert save_data["artifact_id"] == artifact.artifact_id
