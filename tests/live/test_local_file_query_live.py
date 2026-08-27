"""Authorized live-model acceptance coverage for Phase 4 ``file_query``.

This suite spends at most five live ``Agent.run`` interactions. It evaluates
real-model tool discovery, on-demand loading, structured-file aggregation,
partition handling, recovery from an incompatible pattern, unsafe-SQL handling,
and truthful bounded-result reporting. Descriptor containment, SQL admission,
memory/RSS/spill limits, cancellation, cleanup, and isolation remain deterministic
contracts in ``tests/test_local_file_query.py``.
"""

from __future__ import annotations

from _workspace_support import workspace_for

import multiprocessing
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import Agent, LocalWorkspace, LoopLimits, create_llm_provider
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

_AUTHORIZATION = "DAITA_RUN_LIVE_LOCAL_FILE_QUERY"
_MODEL_ID = "DAITA_LOCAL_FILE_QUERY_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_LOCAL_FILE_QUERY_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_LOCAL_FILE_QUERY_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_KEY_ENVIRONMENT = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_SINGLE_TOKEN = "PHASE4_SINGLE_6A21F9"
_PARTITION_TOKEN = "PHASE4_PARTITIONS_90C4B7"
_RECOVERY_TOKEN = "PHASE4_RECOVERY_57D2A1"
_SECURITY_TOKEN = "PHASE4_SECURITY_83E5C2"
_TRUNCATION_TOKEN = "PHASE4_TRUNCATION_41B6D8"

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing up to "
            f"five live Agent.run interactions, each capped by {_MAX_COST}"
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


def _api_key(provider_name: str) -> str:
    override = os.environ.get(_MODEL_KEY)
    if override is not None and override.strip():
        return override
    environment = _API_KEY_ENVIRONMENT.get(provider_name)
    if environment is None:
        pytest.fail(f"{_MODEL_ID} must name one API-backed reviewed model")
    return _required_environment(environment)


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
    if provider_name not in _API_KEY_ENVIRONMENT:
        pytest.fail(
            f"{_MODEL_ID} must name an API-backed provider: "
            + ", ".join(sorted(_API_KEY_ENVIRONMENT))
        )
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")
    if not profile.supports_streaming:
        pytest.fail(f"{_MODEL_ID} must name a model with reviewed streaming support")
    delegate = create_llm_provider(
        model_id,
        api_key=_api_key(provider_name),
        max_output_tokens=min(profile.max_output_tokens, 1_536),
    )
    return profile, _RecordingProvider(delegate)


def _limits() -> LoopLimits:
    return LoopLimits(
        max_steps=12,
        max_total_tokens=30_000,
        max_wall_time_seconds=120,
        max_estimated_cost_usd=_cost_limit(),
    )


async def _run_live(
    *,
    name: str,
    state_root: Path,
    workspace: LocalWorkspace,
    prompt: str,
) -> _RunCapture:
    profile, provider = _live_provider()
    before_children = {child.pid for child in multiprocessing.active_children()}
    agent = await Agent.create(
        name,
        root=state_root,
        workspace=workspace,
        model=provider,
        model_profile=profile,
        limits=_limits(),
    )
    try:
        result = await agent.run(prompt)
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    scratch = state_root / "agents" / name / "file-query-scratch"
    assert not scratch.exists() or not tuple(scratch.iterdir())
    assert {child.pid for child in multiprocessing.active_children()} <= before_children
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
    assert str(workspace.root) not in repr(provider.requests)
    assert str(workspace.root) not in repr(transcript)
    return _RunCapture(result, transcript, tuple(provider.requests))


def _exchanges(
    transcript: Transcript,
    tool_name: str | None = None,
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    calls = {
        call.id: call
        for message in transcript.messages
        for call in message.tool_calls
        if tool_name is None or call.name == tool_name
    }
    return tuple(
        (calls[block.call_id], block)
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id in calls
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        pytest.fail(f"{name} must be a mapping")
    return value


def _query_data(block: ToolResultBlock) -> Mapping[str, object]:
    assert not block.is_error
    return _mapping(block.output.get("data"), "file_query data")


def _query_error(block: ToolResultBlock) -> Mapping[str, object]:
    assert block.is_error
    return _mapping(block.output.get("error"), "file_query error")


def _rows(data: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    value = data.get("rows")
    if not isinstance(value, tuple) or any(
        not isinstance(item, Mapping) for item in value
    ):
        pytest.fail("file_query rows must be a tuple of mappings")
    return value


async def test_live_model_discovers_loads_and_aggregates_one_csv(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "single-state"
    workspace = workspace_for(state_root)
    (workspace.root / "sales.csv").write_text(
        "region,amount,verification_token\n"
        f"north,11,{_SINGLE_TOKEN}\n"
        f"south,5,{_SINGLE_TOKEN}\n"
        f"north,7,{_SINGLE_TOKEN}\n",
        encoding="utf-8",
    )
    prompt = (
        "This is an authorized live Phase 4 check. Search the Files toolbox for "
        "the structured-file aggregation tool, load file_query, and use it on "
        "sales.csv. Return region, SUM(amount) AS total_amount, and "
        "MAX(verification_token) AS verification_token, ordered by region. Report "
        "the exact totals and token from the tool result; do not use file_read and "
        "do not guess."
    )

    capture = await _run_live(
        name="live-phase4-single",
        state_root=state_root,
        workspace=workspace,
        prompt=prompt,
    )

    initial_tools = {tool.name for tool in capture.requests[0].tools}
    assert {"toolbox_search", "toolbox_load"} <= initial_tools
    assert "file_query" not in initial_tools
    assert any(
        "file_query" in {tool.name for tool in request.tools}
        for request in capture.requests[1:]
    )

    exchanges = _exchanges(capture.transcript)
    names = tuple(call.name for call, _block in exchanges)
    search_index = names.index("toolbox_search")
    load_index = names.index("toolbox_load")
    query_index = names.index("file_query")
    assert search_index < load_index < query_index
    load_call, load_result = exchanges[load_index]
    assert not load_result.is_error
    loaded_names = load_call.arguments["tool_names"]
    assert isinstance(loaded_names, tuple)
    assert "file_query" in loaded_names

    query_call, query_result = exchanges[query_index]
    assert query_call.arguments["path_pattern"] == "sales.csv"
    data = _query_data(query_result)
    rows = {str(row["region"]): row for row in _rows(data)}
    assert rows["north"]["total_amount"] == 18
    assert rows["south"]["total_amount"] == 5
    assert rows["north"]["verification_token"] == _SINGLE_TOKEN
    assert data["input_file_count"] == 1
    assert query_result.sensitivity_provenance["authority"] == (
        "local_workspace_binding"
    )
    assert capture.result.final_text is not None
    assert _SINGLE_TOKEN in capture.result.final_text
    assert "18" in capture.result.final_text
    assert "5" in capture.result.final_text


async def test_live_model_aggregates_one_hundred_twenty_partitions(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "partitions-state"
    workspace = workspace_for(state_root)
    partitions = workspace.root / "sales"
    partitions.mkdir()
    for index in range(120):
        region = "north" if index % 2 == 0 else "south"
        amount = 2 if region == "north" else 3
        (partitions / f"part-{index:03d}.csv").write_text(
            "region,amount,verification_token\n"
            f"{region},{amount},{_PARTITION_TOKEN}\n",
            encoding="utf-8",
        )
    prompt = (
        "Use the on-demand file_query tool to aggregate every compatible CSV "
        "partition matching sales/part-*.csv. Return region, SUM(amount) AS "
        "total_amount, and MAX(verification_token) AS verification_token, ordered "
        "by region. Report both exact totals, the verification token, and the input "
        "file count from the tool result."
    )

    capture = await _run_live(
        name="live-phase4-partitions",
        state_root=state_root,
        workspace=workspace,
        prompt=prompt,
    )

    successful = tuple(
        (call, block)
        for call, block in _exchanges(capture.transcript, "file_query")
        if not block.is_error
    )
    assert successful, "the live model did not complete file_query"
    call, block = successful[-1]
    assert call.arguments["path_pattern"] == "sales/part-*.csv"
    data = _query_data(block)
    rows = {str(row["region"]): row for row in _rows(data)}
    assert rows["north"]["total_amount"] == 120
    assert rows["south"]["total_amount"] == 180
    assert rows["north"]["verification_token"] == _PARTITION_TOKEN
    assert data["input_file_count"] == 120
    bindings = block.sensitivity_provenance["bindings"]
    assert isinstance(bindings, tuple)
    assert len(bindings) == 120
    assert capture.result.final_text is not None
    assert _PARTITION_TOKEN in capture.result.final_text
    assert "120" in capture.result.final_text
    assert "180" in capture.result.final_text


async def test_live_model_recovers_from_incompatible_pattern(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "recovery-state"
    workspace = workspace_for(state_root)
    mixed = workspace.root / "mixed"
    mixed.mkdir()
    (mixed / "part-a.csv").write_text(
        "region,amount,verification_token\n" f"north,19,{_RECOVERY_TOKEN}\n",
        encoding="utf-8",
    )
    (mixed / "part-b.csv").write_text(
        "region,amount,verification_token\n" f"north,23,{_RECOVERY_TOKEN}\n",
        encoding="utf-8",
    )
    (mixed / "metadata.csv").write_text(
        "key,value\nowner,operations\n",
        encoding="utf-8",
    )
    prompt = (
        "This is a live recovery check. Load file_query and first attempt to query "
        "mixed/*.csv with SQL that returns SUM(amount) AS total_amount and "
        "MAX(verification_token) AS verification_token from data. The broad pattern "
        "contains an incompatible metadata file. After the expected structured "
        "error, retry with mixed/part-*.csv and report the exact total and token from "
        "the successful tool result."
    )

    capture = await _run_live(
        name="live-phase4-recovery",
        state_root=state_root,
        workspace=workspace,
        prompt=prompt,
    )

    queries = _exchanges(capture.transcript, "file_query")
    broad = tuple(
        (call, block)
        for call, block in queries
        if call.arguments.get("path_pattern") == "mixed/*.csv"
    )
    assert broad, "the live model skipped the requested incompatible-pattern check"
    assert broad[0][1].is_error
    assert _query_error(broad[0][1])["code"] == "file_query_invalid"

    recovered = tuple(
        (call, block)
        for call, block in queries
        if call.arguments.get("path_pattern") == "mixed/part-*.csv"
        and not block.is_error
    )
    assert recovered, "the live model did not recover with the compatible pattern"
    _call, result = recovered[-1]
    data = _query_data(result)
    rows = _rows(data)
    assert len(rows) == 1
    assert rows[0]["total_amount"] == 42
    assert rows[0]["verification_token"] == _RECOVERY_TOKEN
    assert data["input_file_count"] == 2
    assert capture.result.final_text is not None
    assert _RECOVERY_TOKEN in capture.result.final_text
    assert "42" in capture.result.final_text


async def test_live_model_handles_unsafe_sql_without_host_data(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "security-state"
    workspace = workspace_for(state_root)
    (workspace.root / "safe.csv").write_text(
        "amount,verification_token\n"
        f"17,{_SECURITY_TOKEN}\n"
        f"8,{_SECURITY_TOKEN}\n",
        encoding="utf-8",
    )
    prompt = (
        "This is an authorized security-recovery check. Load file_query. First, "
        "attempt the deliberately invalid SQL SELECT * FROM "
        "read_csv('/etc/hosts') with path_pattern safe.csv so the admission boundary "
        "can reject it. Whether you attempt it or decline it as unsafe, then run a "
        "safe query against only data that returns SUM(amount) AS total_amount and "
        "MAX(verification_token) AS verification_token. Report only the safe total, "
        "token, and whether the unsafe form was rejected or avoided; never claim any "
        "host-file contents."
    )

    capture = await _run_live(
        name="live-phase4-security",
        state_root=state_root,
        workspace=workspace,
        prompt=prompt,
    )

    queries = _exchanges(capture.transcript, "file_query")
    unsafe = tuple(
        (call, block)
        for call, block in queries
        if "read_csv" in str(call.arguments.get("sql", "")).lower()
    )
    assert all(block.is_error for _call, block in unsafe)
    assert all(
        _query_error(block)["code"] == "file_query_invalid" for _call, block in unsafe
    )

    safe = tuple(
        (call, block)
        for call, block in queries
        if "from data" in str(call.arguments.get("sql", "")).lower()
        and not block.is_error
    )
    assert safe, "the live model did not complete the safe relation-only retry"
    _call, result = safe[-1]
    rows = _rows(_query_data(result))
    assert len(rows) == 1
    assert rows[0]["total_amount"] == 25
    assert rows[0]["verification_token"] == _SECURITY_TOKEN
    assert capture.result.final_text is not None
    assert _SECURITY_TOKEN in capture.result.final_text
    assert "25" in capture.result.final_text
    assert "localhost" not in capture.result.final_text.lower()


async def test_live_model_reports_bounded_result_truncation_truthfully(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "truncation-state"
    workspace = workspace_for(state_root)
    rows = "id,verification_token\n" + "".join(
        f"{index},{_TRUNCATION_TOKEN}\n" for index in range(150)
    )
    (workspace.root / "rows.csv").write_text(rows, encoding="utf-8")
    prompt = (
        "Use file_query on rows.csv with SELECT id, verification_token FROM data "
        "ORDER BY id and do not add a SQL LIMIT. Report the exact returned_rows value, "
        "whether the result is truncated, the truncation reason, and the verification "
        "token. Do not claim that every source row was returned."
    )

    capture = await _run_live(
        name="live-phase4-truncation",
        state_root=state_root,
        workspace=workspace,
        prompt=prompt,
    )

    successful = tuple(
        (call, block)
        for call, block in _exchanges(capture.transcript, "file_query")
        if not block.is_error
    )
    assert successful, "the live model did not complete the bounded query"
    call, result = successful[-1]
    assert "limit" not in str(call.arguments["sql"]).lower()
    data = _query_data(result)
    assert data["returned_rows"] == 100
    assert data["truncated"] is True
    reasons = data["truncation_reasons"]
    assert isinstance(reasons, tuple)
    assert "row_limit" in reasons
    returned = _rows(data)
    assert len(returned) == 100
    assert all(row["verification_token"] == _TRUNCATION_TOKEN for row in returned)
    assert capture.result.final_text is not None
    final_text = capture.result.final_text.lower()
    assert _TRUNCATION_TOKEN.lower() in final_text
    assert "100" in final_text
    assert "truncat" in final_text
