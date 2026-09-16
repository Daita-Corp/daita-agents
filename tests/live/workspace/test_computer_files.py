"""Authorized live-model acceptance coverage for local-computer files.

The five interactions exercise model behavior that deterministic owner tests cannot:

* glob-only discovery of the newest CSV in a host-known Downloads directory;
* comparison across host-known Downloads and Documents locations;
* truthful reporting when a multi-location search has incomplete coverage;
* one approved exact edit outside the working directory; and
* recovery after an injected malformed first ``file_search`` call.

Every filesystem operand is created under ``tmp_path``. The suite never reads or
writes the runner's real Downloads, Documents, Desktop, or other personal files.
Descriptor containment, protected paths, revisions, limits, cancellation, drift,
approval binding, and atomic publication remain deterministic owner contracts.

Run only after authorizing five paid interactions, for example::

    DAITA_RUN_LIVE_LOCAL_COMPUTER_FILES=1 \
    OPENAI_API_KEY=... \
    pytest tests/live/workspace/test_computer_files.py \
      -o addopts="--tb=short -q --strict-markers" -v
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

import daita.adapters.local_workspace as local_workspace_module
from daita import (
    Agent,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    LocalFileAccess,
    LocalWorkspace,
    LoopLimits,
    create_llm_provider,
)
from daita._json import canonical_json
from daita.artifacts.models import ArtifactDeliveryMode, ArtifactDeliveryOutcome
from daita.llm._lifecycle import closing_stream
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import (
    ManagedModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.loop.models import (
    LoopExit,
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_LOCAL_COMPUTER_FILES"
_MODEL_ID = "DAITA_LOCAL_COMPUTER_FILES_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_LOCAL_COMPUTER_FILES_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_LOCAL_COMPUTER_FILES_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_KEY_ENVIRONMENT = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_LATEST_TOKEN = "LIVE_COMPUTER_LATEST_7D1A93"
_DOWNLOAD_COMPARE_TOKEN = "LIVE_COMPUTER_DOWNLOAD_A4C8E1"
_DOCUMENT_COMPARE_TOKEN = "LIVE_COMPUTER_DOCUMENT_C2F709"
_PARTIAL_TOKEN = "LIVE_COMPUTER_PARTIAL_91B6D3"
_EDIT_TOKEN = "LIVE_COMPUTER_EDIT_53E8A2"
_RECOVERY_TOKEN = "LIVE_COMPUTER_RECOVERY_F6A417"

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing five "
            f"live Agent.run interactions, each capped by {_MAX_COST}"
        ),
    ),
]


class _RecordingProvider:
    """Record real requests and optionally inject one deterministic first call."""

    def __init__(
        self,
        delegate: ManagedModelProvider,
        *,
        first_response: ModelResponse | None = None,
    ) -> None:
        self._delegate = delegate
        self._first_response = first_response
        self._first_response_used = False
        self.requests: list[ModelRequest] = []
        self.responses: list[ModelResponse] = []

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    def _injected(self) -> ModelResponse | None:
        if self._first_response is None or self._first_response_used:
            return None
        self._first_response_used = True
        self.responses.append(self._first_response)
        return self._first_response

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        injected = self._injected()
        if injected is not None:
            return injected
        response = await self._delegate.generate(request)
        self.responses.append(response)
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        injected = self._injected()
        if injected is not None:
            yield ModelStreamCompleted(injected)
            return
        if not isinstance(self._delegate, StreamingModelProvider):
            raise TypeError("the live delegate must support canonical streaming")
        async with closing_stream(self._delegate.stream(request)) as events:
            async for event in events:
                if isinstance(event, ModelStreamCompleted):
                    self.responses.append(event.response)
                yield event

    async def close(self, *, deadline: float | None = None) -> None:
        await self._delegate.close(deadline=deadline)


@dataclass(frozen=True, slots=True)
class _ComputerLocations:
    state_root: Path
    working: Path
    downloads: Path
    documents: Path
    desktop: Path

    @property
    def workspace(self) -> LocalWorkspace:
        return LocalWorkspace(self.working, access=LocalFileAccess.COMPUTER)


@dataclass(frozen=True, slots=True)
class _RunCapture:
    result: LoopExit
    transcript: Transcript
    requests: tuple[ModelRequest, ...]


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


def _live_provider(
    *, first_response: ModelResponse | None = None
) -> tuple[ModelProfile, _RecordingProvider]:
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
    return profile, _RecordingProvider(delegate, first_response=first_response)


def _limits() -> LoopLimits:
    return LoopLimits(max_estimated_cost_usd=_cost_limit())


def _computer_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _ComputerLocations:
    locations = _ComputerLocations(
        state_root=tmp_path / "state",
        working=tmp_path / "unrelated-project",
        downloads=tmp_path / "Synthetic Downloads",
        documents=tmp_path / "Synthetic Documents",
        desktop=tmp_path / "Synthetic Desktop",
    )
    for directory in (
        locations.working,
        locations.downloads,
        locations.documents,
        locations.desktop,
    ):
        directory.mkdir(parents=True)
    known = {
        "Downloads": locations.downloads,
        "Documents": locations.documents,
        "Desktop": locations.desktop,
    }

    def resolve_known_directory(
        name: str,
        *,
        user_home: Path | None = None,
    ) -> Path:
        del user_home
        return known[name]

    monkeypatch.setattr(
        local_workspace_module,
        "resolve_os_known_directory",
        resolve_known_directory,
    )
    return locations


async def _run_live(
    *,
    name: str,
    locations: _ComputerLocations,
    prompt: str,
    first_response: ModelResponse | None = None,
    approval_handler: ApprovalHandler | None = None,
) -> _RunCapture:
    profile, provider = _live_provider(first_response=first_response)
    workspace = locations.workspace
    agent = await Agent.create(
        name,
        root=locations.state_root,
        workspace=workspace,
        model=provider,
        model_profile=profile,
        approval_handler=approval_handler,
        downloads_directory=locations.downloads,
        limits=_limits(),
    )
    try:
        result = await agent.run(prompt)
        transcript = await agent.transcript(result.run_id)
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()

    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.usage.cost_estimate.code,
        len(provider.requests),
    )
    assert result.final_text is not None
    validate_completed_transcript(transcript, result)
    assert provider.requests
    assert all(
        request.sensitivity is ModelSensitivity.INTERNAL
        for request in provider.requests
    )
    initial_context = "\n".join(
        block.text
        for message in provider.requests[0].messages
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert str(locations.working) in initial_context
    assert str(locations.downloads) in initial_context
    assert str(locations.documents) in initial_context
    return _RunCapture(result, transcript, tuple(provider.requests))


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


def _data(result: ToolResultBlock, name: str) -> Mapping[str, object]:
    assert not result.is_error
    return _mapping(result.output.get("data"), name)


def _result_with_text(
    exchanges: tuple[tuple[ToolCall, ToolResultBlock], ...],
    tool_name: str,
    text: str,
) -> tuple[ToolCall, ToolResultBlock]:
    return next(
        (call, result)
        for call, result in exchanges
        if call.name == tool_name and text in canonical_json(result.output)
    )


async def test_live_model_finds_latest_download_with_glob_only_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _computer_locations(tmp_path, monkeypatch)
    older = locations.downloads / "older.csv"
    latest = locations.downloads / "latest.csv"
    older.write_text("name,token\nold,IGNORE_OLD\n", encoding="utf-8")
    latest.write_text(
        f"name,token\nlatest,{_LATEST_TOKEN}\n",
        encoding="utf-8",
    )
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(latest, (1_800_000_000, 1_800_000_000))

    capture = await _run_live(
        name="live-computer-latest",
        locations=locations,
        prompt=(
            "Find the most recently modified CSV in Downloads. Use file_search "
            "in paths mode with path set to the host-known Downloads directory, "
            "glob='*.csv', no query, and order_by=modified_desc. Read the selected "
            "file and return its exact token. Base the choice only on tool evidence."
        ),
    )

    assert _LATEST_TOKEN in (capture.result.final_text or "")
    searches = _successful(capture.transcript, "file_search")
    reads = _successful(capture.transcript, "file_read")
    search_call, search_result = next(
        item for item in searches if str(latest) in canonical_json(item[1].output)
    )
    read_call, _read_result = next(
        item for item in reads if _LATEST_TOKEN in canonical_json(item[1].output)
    )
    assert search_call.arguments["path"] == str(locations.downloads)
    assert search_call.arguments["glob"] == "*.csv"
    assert search_call.arguments["mode"] == "paths"
    assert search_call.arguments["order_by"] == "modified_desc"
    assert "query" not in search_call.arguments
    assert read_call.arguments["path"] == str(latest)
    search_data = _data(search_result, "file_search data")
    assert search_data["scan_complete"] is True


async def test_live_model_compares_downloads_and_documents_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _computer_locations(tmp_path, monkeypatch)
    downloaded = locations.downloads / "download_snapshot.csv"
    documented = locations.documents / "document_snapshot.csv"
    downloaded.write_text(
        f"amount,token\n31,{_DOWNLOAD_COMPARE_TOKEN}\n",
        encoding="utf-8",
    )
    documented.write_text(
        f"amount,token\n12,{_DOCUMENT_COMPARE_TOKEN}\n",
        encoding="utf-8",
    )

    capture = await _run_live(
        name="live-computer-compare",
        locations=locations,
        prompt=(
            "Compare download_snapshot.csv in Downloads with document_snapshot.csv "
            "in Documents. Use file_read on both exact files. Return both exact "
            "tokens, both amounts, and the Downloads amount minus the Documents "
            "amount. Do not guess paths or values."
        ),
    )

    final = capture.result.final_text or ""
    assert _DOWNLOAD_COMPARE_TOKEN in final
    assert _DOCUMENT_COMPARE_TOKEN in final
    assert "19" in final
    reads = _successful(capture.transcript, "file_read")
    read_paths = {
        call.arguments.get("path")
        for call, _result in reads
        if isinstance(call.arguments.get("path"), str)
    }
    assert {str(downloaded), str(documented)} <= read_paths
    assert any(
        _DOWNLOAD_COMPARE_TOKEN in canonical_json(result.output)
        for _call, result in reads
    )
    assert any(
        _DOCUMENT_COMPARE_TOKEN in canonical_json(result.output)
        for _call, result in reads
    )


async def test_live_model_reports_incomplete_multi_location_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _computer_locations(tmp_path, monkeypatch)
    available = locations.downloads / "available.csv"
    missing = tmp_path / "missing-location"
    available.write_text(
        f"name,token\navailable,{_PARTIAL_TOKEN}\n",
        encoding="utf-8",
    )

    capture = await _run_live(
        name="live-computer-partial",
        locations=locations,
        prompt=(
            "Search exactly these two locations for CSV files with one file_search "
            f"call: {locations.downloads} and {missing}. Set the file_search "
            f'"paths" array to both values; do not use the singular "path" '
            "argument. Use glob='*.csv', paths mode, and path ordering. Read the "
            "useful match, return its exact token, and state plainly whether both "
            "requested locations were completely searched. Do not describe a "
            "failed location as empty."
        ),
    )

    searches = _successful(capture.transcript, "file_search")
    assert len(searches) == 1
    search_call, search_result = searches[0]
    requested_paths = search_call.arguments.get("paths")
    assert isinstance(requested_paths, tuple), (
        "the live model did not use the multi-location paths argument",
        dict(search_call.arguments),
    )
    assert set(requested_paths) == {str(locations.downloads), str(missing)}
    search_data = _data(search_result, "partial file_search data")
    assert search_data["scan_complete"] is False
    coverage = search_data.get("coverage")
    assert isinstance(coverage, tuple)
    by_path = {
        str(item["path"]): item for item in coverage if isinstance(item, Mapping)
    }
    assert by_path[str(locations.downloads)]["status"] == "complete"
    assert by_path[str(missing)]["status"] == "failed"
    _result_with_text(_exchanges(capture.transcript), "file_read", _PARTIAL_TOKEN)
    final = (capture.result.final_text or "").casefold().replace("*", "")
    assert _PARTIAL_TOKEN.casefold() in final
    assert any(
        phrase in final
        for phrase in (
            "incomplete",
            "not complete",
            "not fully",
            "could not",
            "failed",
            "unavailable",
        )
    )


async def test_live_model_approved_edit_replaces_exact_documents_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _computer_locations(tmp_path, monkeypatch)
    target = locations.documents / "external_config.txt"
    collision = locations.working / "external_config.txt"
    original = f"timeout_seconds=30\ntoken={_EDIT_TOKEN}\n"
    expected = f"timeout_seconds=45\ntoken={_EDIT_TOKEN}\n"
    target.write_text(original, encoding="utf-8")
    collision.write_text("timeout_seconds=10\ntoken=DO_NOT_CHANGE\n", encoding="utf-8")
    if hasattr(os, "chown"):
        os.chown(target, -1, os.getegid())
    approvals: list[ApprovalRequest] = []
    content_at_approval: list[str] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        content_at_approval.append(target.read_text(encoding="utf-8"))
        return ApprovalDecision.APPROVE

    capture = await _run_live(
        name="live-computer-external-edit",
        locations=locations,
        approval_handler=approve,
        prompt=(
            "In Documents, edit external_config.txt. Change exactly "
            "timeout_seconds=30 to timeout_seconds=45, preserve every other "
            "character, and save the approved change back to that same file. "
            f"After success report the final value and exact token {_EDIT_TOKEN}."
        ),
    )

    assert target.read_text(encoding="utf-8") == expected
    assert collision.read_text(encoding="utf-8") == (
        "timeout_seconds=10\ntoken=DO_NOT_CHANGE\n"
    )
    assert _EDIT_TOKEN in (capture.result.final_text or "")
    assert "45" in (capture.result.final_text or "")
    workflow = tuple(
        item
        for item in _exchanges(capture.transcript)
        if item[0].name in {"file_read", "artifact_edit_text", "artifact_save_local"}
        and not item[1].is_error
    )
    assert tuple(call.name for call, _result in workflow) == (
        "file_read",
        "artifact_edit_text",
        "artifact_save_local",
    )
    (read_call, read_result), (edit_call, edit_result), (save_call, _save_result) = (
        workflow
    )
    assert read_call.arguments["path"] == str(target)
    read_data = _data(read_result, "external file_read data")
    binding = read_data.get("binding")
    assert isinstance(binding, str) and binding
    assert edit_call.arguments["binding"] == binding
    artifact = _mapping(edit_result.output.get("artifact"), "edit artifact")
    artifact_id = artifact.get("artifact_id")
    assert isinstance(artifact_id, str) and artifact_id
    assert dict(save_call.arguments) == {
        "artifact_id": artifact_id,
        "mode": "replace_bound_file",
    }
    assert len(approvals) == 1
    assert content_at_approval == [original]
    assert str(target) in approvals[0].reason
    assert approvals[0].arguments["target_path"] == str(target)
    assert len(capture.result.artifact_deliveries) == 1
    receipt = capture.result.artifact_deliveries[0]
    assert receipt.mode is ArtifactDeliveryMode.REPLACE_BOUND_FILE
    assert receipt.outcome is ArtifactDeliveryOutcome.SUCCEEDED
    assert receipt.relative_path is not None
    assert receipt.relative_path.endswith("Synthetic Documents/external_config.txt")


async def test_live_model_recovers_from_invalid_glob_without_claiming_no_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _computer_locations(tmp_path, monkeypatch)
    target = locations.downloads / "recover.csv"
    target.write_text(
        f"name,token\nrecovered,{_RECOVERY_TOKEN}\n",
        encoding="utf-8",
    )
    injected = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="injected-invalid-computer-search",
                name="file_search",
                arguments={
                    "path": str(locations.downloads),
                    "glob": "**/*.csv",
                    "mode": "paths",
                    "order_by": "path",
                },
            ),
        ),
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
    )

    capture = await _run_live(
        name="live-computer-recovery",
        locations=locations,
        first_response=injected,
        prompt=(
            "Find recover.csv in Downloads, read it, and return its exact token. "
            "If a file_search call fails, use the returned error guidance to "
            "correct the call. Do not infer that local access is unavailable from "
            "one invalid argument."
        ),
    )

    exchanges = _exchanges(capture.transcript)
    invalid_call, invalid_result = exchanges[0]
    assert invalid_call.id == "injected-invalid-computer-search"
    assert invalid_result.is_error
    invalid_error = _mapping(invalid_result.output.get("error"), "search error")
    assert invalid_error["code"] == "search_invalid"
    assert "glob='*.csv'" in str(invalid_error["message"])
    corrected = tuple(
        (call, result)
        for call, result in exchanges[1:]
        if call.name == "file_search" and not result.is_error
    )
    assert corrected
    corrected_call, corrected_result = corrected[0]
    assert corrected_call.arguments["path"] == str(locations.downloads)
    assert corrected_call.arguments["glob"] == "*.csv"
    assert str(target) in canonical_json(corrected_result.output)
    _result_with_text(exchanges, "file_read", _RECOVERY_TOKEN)
    final = (capture.result.final_text or "").casefold()
    assert _RECOVERY_TOKEN.casefold() in final
    assert not any(
        phrase in final
        for phrase in (
            "do not have access",
            "don't have access",
            "cannot access local files",
            "can't access local files",
            "upload the file",
        )
    )
