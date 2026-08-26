"""Shared, bounded fixtures for the opt-in Stage B live benchmarks."""

from __future__ import annotations

from _workspace_support import workspace_for

import asyncio
import json
import os
import sqlite3
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import Agent, JobStatus, LoopLimits, SQLiteSource, create_llm_provider
from daita._json import canonical_json
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import (
    ModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import (
    LoopExit,
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)

DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
MODEL_IDS_ENV = "DAITA_STAGE_B_BENCHMARK_MODEL_IDS"
GENERIC_KEY_ENV = "DAITA_STAGE_B_BENCHMARK_LLM_API_KEY"
MAX_COST_ENV = "DAITA_STAGE_B_BENCHMARK_MAX_COST_USD"
API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

IMMEDIATE_TOKEN = "STAGE_B_BENCHMARK_SENTINEL_91C7E4"
IMMEDIATE_AMOUNT = 41
PROFILE_SAMPLE_ROWS = 5
PROFILE_NULL_VALUES = 2
TARGET_IMMEDIATE_TABLE = "stage_b_immediate_probe"
TARGET_PROFILE_TABLE = "stage_b_profile_probe"
LIFECYCLE_TOOLS = frozenset(
    {"job_list", "job_inspect", "job_read_results", "job_cancel"}
)
TERMINAL_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.NEEDS_ATTENTION,
    }
)

OFFLINE_EAGER_LIMITS = LoopLimits()


@dataclass(frozen=True, slots=True)
class ProbeHome:
    name: str
    root: Path
    database: Path
    source_id: str
    resource_ids: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class LiveAgentFixture:
    agent: Agent
    provider: RecordingProvider
    source_id: str
    resource_ids: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RunCapture:
    result: LoopExit
    transcript: Transcript
    requests: tuple[ModelRequest, ...]


class RecordingProvider:
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
            raise TypeError("the live benchmark provider must support streaming")
        self.requests.append(request)
        async for event in self._delegate.stream(request):
            if isinstance(event, ModelStreamCompleted):
                self.responses.append(event.response)
            yield event


def benchmark_marks(
    authorization_env: str,
    *,
    maximum_interactions: int,
    per_model: bool = False,
) -> list[object]:
    """Return explicit paid-test markers with a visible interaction ceiling."""

    scope = "per configured model" if per_model else "for this module"
    return [
        pytest.mark.acceptance,
        pytest.mark.integration,
        pytest.mark.requires_llm,
        pytest.mark.skipif(
            os.environ.get(authorization_env) != "1",
            reason=(
                f"set {authorization_env}=1 only after authorizing at most "
                f"{maximum_interactions} live Agent.run interactions {scope}; "
                f"each interaction is capped by {MAX_COST_ENV}"
            ),
        ),
    ]


def configured_model_ids() -> tuple[str, ...]:
    raw = os.environ.get(MODEL_IDS_ENV, DEFAULT_MODEL_ID)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"{MODEL_IDS_ENV} must contain at least one model ID")
    if len(values) > 6:
        raise ValueError(f"{MODEL_IDS_ENV} cannot contain more than six model IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{MODEL_IDS_ENV} cannot contain duplicate model IDs")
    return values


def benchmark_limits() -> LoopLimits:
    return LoopLimits(
        max_steps=12,
        max_total_tokens=30_000,
        max_wall_time_seconds=180,
        max_estimated_cost_usd=_cost_limit(),
    )


def live_provider(model_id: str) -> tuple[ModelProfile, RecordingProvider]:
    provider_name = model_id.partition(":")[0]
    if provider_name not in API_PROVIDERS:
        pytest.fail(
            f"{MODEL_IDS_ENV} must contain only API-backed providers: "
            + ", ".join(sorted(API_PROVIDERS))
        )
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{model_id} is not a release-reviewed tool-capable model")
    if not profile.supports_streaming:
        pytest.fail(f"{model_id} does not have reviewed streaming support")
    delegate = create_llm_provider(
        model_id,
        api_key=_api_key(provider_name),
        max_output_tokens=min(profile.max_output_tokens, 1_536),
    )
    return profile, RecordingProvider(delegate)


def create_probe_database(path: Path, *, distractor_tables: int = 0) -> None:
    if distractor_tables < 0 or distractor_tables > 512:
        raise ValueError("distractor_tables must be between zero and 512")
    with sqlite3.connect(path) as connection:
        connection.executescript(f"""
            CREATE TABLE {TARGET_IMMEDIATE_TABLE} (
                probe_name TEXT PRIMARY KEY,
                verification_token TEXT NOT NULL,
                amount INTEGER NOT NULL
            );
            CREATE TABLE {TARGET_PROFILE_TABLE} (
                id INTEGER PRIMARY KEY,
                nullable_code TEXT,
                measure REAL NOT NULL
            );
            """)
        connection.execute(
            f"INSERT INTO {TARGET_IMMEDIATE_TABLE} VALUES (?, ?, ?)",
            ("immediate", IMMEDIATE_TOKEN, IMMEDIATE_AMOUNT),
        )
        connection.executemany(
            f"INSERT INTO {TARGET_PROFILE_TABLE} VALUES (?, ?, ?)",
            (
                (1, "alpha", 10.5),
                (2, None, 11.5),
                (3, "beta", 12.5),
                (4, None, 13.5),
                (5, "alpha", 14.5),
            ),
        )
        for index in range(distractor_tables):
            stem = (
                "stage_b_profile_probe_archive"
                if index % 2 == 0
                else "stage_b_immediate_probe_backup"
            )
            table_name = f"{stem}_{index:03d}"
            connection.execute(
                f"CREATE TABLE {table_name} "
                "(id INTEGER PRIMARY KEY, distractor_value TEXT NOT NULL)"
            )
            connection.execute(
                f"INSERT INTO {table_name}(distractor_value) VALUES (?)",
                (f"distractor-{index}",),
            )


async def create_probe_home(
    tmp_path: Path,
    name: str,
    *,
    distractor_tables: int = 0,
) -> ProbeHome:
    root = tmp_path / f"{name}-root"
    database = tmp_path / f"{name}.sqlite"
    create_probe_database(database, distractor_tables=distractor_tables)
    agent = await Agent.create(name, root=root, workspace=workspace_for(root))
    try:
        source = await agent.attach(SQLiteSource(database, name="Stage B benchmarks"))
        resources = await agent.list_catalog_resources(source_id=source.id)
        resource_ids = {item.name: item.id for item in resources}
        assert TARGET_IMMEDIATE_TABLE in resource_ids
        assert TARGET_PROFILE_TABLE in resource_ids
        assert len(resource_ids) == distractor_tables + 2
        return ProbeHome(
            name=name,
            root=root,
            database=database,
            source_id=source.id,
            resource_ids=resource_ids,
        )
    finally:
        await agent.close()


async def create_live_agent(
    tmp_path: Path,
    name: str,
    model_id: str,
    *,
    distractor_tables: int = 0,
) -> LiveAgentFixture:
    profile, provider = live_provider(model_id)
    database = tmp_path / f"{name}.sqlite"
    create_probe_database(database, distractor_tables=distractor_tables)
    agent = await Agent.create(
        name,
        root=tmp_path / f"{name}-root",
        model=provider,
        model_profile=profile,
        limits=benchmark_limits(),
        workspace=workspace_for(tmp_path / f"{name}-root"),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Stage B benchmarks"))
        resources = await agent.list_catalog_resources(source_id=source.id)
        resource_ids = {item.name: item.id for item in resources}
        assert len(resource_ids) == distractor_tables + 2
        return LiveAgentFixture(
            agent=agent,
            provider=provider,
            source_id=source.id,
            resource_ids=resource_ids,
        )
    except BaseException:
        await agent.close()
        raise


async def open_live_home(home: ProbeHome, model_id: str) -> LiveAgentFixture:
    profile, provider = live_provider(model_id)
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=profile,
        limits=benchmark_limits(),
        workspace=workspace_for(home.root),
    )
    return LiveAgentFixture(
        agent=agent,
        provider=provider,
        source_id=home.source_id,
        resource_ids=home.resource_ids,
    )


async def seed_completed_profile_agent(
    tmp_path: Path,
    name: str,
    model_id: str,
    *,
    distractor_tables: int = 0,
) -> tuple[LiveAgentFixture, str]:
    """Create one real completed job without spending a live model interaction."""

    home = await create_probe_home(
        tmp_path,
        name,
        distractor_tables=distractor_tables,
    )
    provider = MockModelProvider(
        (
            start_profile_response(
                home.resource_ids[TARGET_PROFILE_TABLE],
                call_id="seed-profile",
            ),
            stop_response("Seed job admitted."),
        )
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        limits=OFFLINE_EAGER_LIMITS,
        workspace=workspace_for(home.root),
    )
    try:
        result = await agent.run(
            "Seed one deterministic completed profile job.",
            source_id=home.source_id,
        )
        assert result.kind is LoopExitKind.COMPLETED
        jobs = await agent.list_jobs()
        assert len(jobs) == 1
        job_id = jobs[0].job_id
        terminal = await wait_for_terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED
        await assert_profile_result(agent, job_id)
    finally:
        await agent.close()
    return await open_live_home(home, model_id), job_id


async def capture_run(
    fixture: LiveAgentFixture,
    message: str,
    *,
    source_id: str | None = None,
    conversation_id: str | None = None,
) -> RunCapture:
    start = len(fixture.provider.requests)
    result = await fixture.agent.run(
        message,
        source_id=source_id,
        conversation_id=conversation_id,
    )
    transcript = await fixture.agent.transcript(result.run_id)
    return RunCapture(
        result=result,
        transcript=transcript,
        requests=tuple(fixture.provider.requests[start:]),
    )


def assert_completed(capture: RunCapture) -> None:
    result = capture.result
    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.usage.cost_estimate.code,
        len(capture.requests),
    )
    assert result.reason == "completed"
    assert result.final_text is not None
    assert 1 <= len(capture.requests) <= 12
    assert result.usage.total_tokens > 0
    validate_completed_transcript(capture.transcript, result)


def assert_on_demand_invocation(capture: RunCapture, tool_name: str) -> None:
    assert capture.requests
    visible_names = {tool.name for tool in capture.requests[0].tools}
    assert {"toolbox_search", "toolbox_load"} <= visible_names
    assert tool_name not in visible_names
    names = logical_names(capture.transcript)
    assert "toolbox_search" in names
    assert "toolbox_load" in names
    assert tool_name in names
    assert results_for(capture.transcript, tool_name)


def logical_names(transcript: Transcript) -> tuple[str, ...]:
    outer_names = {
        call.id: call.name
        for message in transcript.messages
        for call in message.tool_calls
    }
    names: list[str] = []
    for message in transcript.messages:
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            name = outer_names.get(block.call_id)
            if name is not None:
                names.append(name)
    return tuple(names)


def results_for(
    transcript: Transcript,
    tool_name: str,
) -> tuple[ToolResultBlock, ...]:
    outer_names = {
        call.id: call.name
        for message in transcript.messages
        for call in message.tool_calls
    }
    selected: list[ToolResultBlock] = []
    for message in transcript.messages:
        for block in message.content:
            if not isinstance(block, ToolResultBlock) or block.is_error:
                continue
            name = outer_names.get(block.call_id)
            if name == tool_name:
                selected.append(block)
    return tuple(selected)


def job_id_from_start(transcript: Transcript) -> str:
    receipts = results_for(transcript, "start_data_profile")
    assert len(receipts) == 1
    data = receipts[0].output.get("data")
    assert isinstance(data, Mapping)
    assert data.get("status") == JobStatus.QUEUED.value
    job_id = data.get("job_id")
    assert isinstance(job_id, str) and job_id
    return job_id


async def wait_for_terminal(agent: Agent, job_id: str, *, timeout: float = 30.0):
    deadline = asyncio.get_running_loop().time() + timeout
    latest = None
    while asyncio.get_running_loop().time() < deadline:
        latest = await agent.inspect_job(job_id)
        assert latest is not None
        if latest.summary.status in TERMINAL_STATUSES:
            return latest
        await asyncio.sleep(0.02)
    pytest.fail(f"job did not reach a terminal state: {latest!r}")


async def wait_for_running(agent: Agent, job_id: str, *, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    latest = None
    while asyncio.get_running_loop().time() < deadline:
        latest = await agent.inspect_job(job_id)
        assert latest is not None
        if latest.summary.status is JobStatus.RUNNING:
            return latest
        await asyncio.sleep(0.01)
    pytest.fail(f"job did not enter the running state: {latest!r}")


async def assert_profile_result(agent: Agent, job_id: str) -> Mapping[str, object]:
    result = await agent.read_job_result(job_id)
    assert result is not None
    assert result.summary["sampled_rows"] == PROFILE_SAMPLE_ROWS
    assert len(result.artifact_refs) == 1
    payload = await agent.read_artifact(result.artifact_refs[0].artifact_id)
    document = json.loads(payload.content)
    assert isinstance(document, Mapping)
    resources = document["resources"]
    assert isinstance(resources, Sequence)
    assert len(resources) == 1
    resource = resources[0]
    assert isinstance(resource, Mapping)
    assert resource["sampled_rows"] == PROFILE_SAMPLE_ROWS
    profiles = resource["column_profiles"]
    assert isinstance(profiles, Sequence)
    nullable = next(
        item
        for item in profiles
        if isinstance(item, Mapping) and item.get("column") == "nullable_code"
    )
    assert nullable["null_values"] == PROFILE_NULL_VALUES
    return document


def start_profile_response(
    resource_id: str,
    *,
    call_id: str = "start-profile",
    sample_rows: int = PROFILE_SAMPLE_ROWS,
) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id=call_id,
                name="start_data_profile",
                arguments={
                    "resource_ids": [resource_id],
                    "sample_rows": sample_rows,
                },
            ),
        ),
    )


def toolbox_load_response(*tool_names: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="benchmark-toolbox-load",
                name="toolbox_load",
                arguments={"tool_names": list(tool_names)},
            ),
        ),
    )


def stop_response(text: str = "Done.") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def transcript_text(transcript: Transcript) -> str:
    return canonical_json(
        [
            {
                "role": message.role.value,
                "content": [
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                ],
            }
            for message in transcript.messages
        ]
    )


def record_metrics(
    record_property: Callable[[str, object], None],
    model_id: str,
    capture: RunCapture,
) -> None:
    estimate = capture.result.usage.cost_estimate
    record_property("model_id", model_id)
    record_property("model_requests", len(capture.requests))
    record_property("steps", capture.result.steps)
    record_property("total_tokens", capture.result.usage.total_tokens)
    record_property(
        "estimated_cost_usd",
        None if estimate.amount_usd is None else str(estimate.amount_usd),
    )


def _cost_limit() -> Decimal:
    raw = os.environ.get(MAX_COST_ENV, "0.15")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{MAX_COST_ENV} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{MAX_COST_ENV} must be a finite positive decimal")
    return value


def _api_key(provider_name: str) -> str:
    specific_name = (
        "DAITA_STAGE_B_BENCHMARK_"
        + provider_name.upper().replace("-", "_").replace(".", "_")
        + "_API_KEY"
    )
    value = os.environ.get(specific_name) or os.environ.get(GENERIC_KEY_ENV)
    if value is None or not value.strip():
        pytest.fail(
            f"set {specific_name} or {GENERIC_KEY_ENV} for the authorized benchmark"
        )
    return value


__all__ = [
    "DEFAULT_MODEL_ID",
    "IMMEDIATE_AMOUNT",
    "IMMEDIATE_TOKEN",
    "LIFECYCLE_TOOLS",
    "LiveAgentFixture",
    "MAX_COST_ENV",
    "MODEL_IDS_ENV",
    "OFFLINE_EAGER_LIMITS",
    "PROFILE_NULL_VALUES",
    "PROFILE_SAMPLE_ROWS",
    "ProbeHome",
    "RecordingProvider",
    "RunCapture",
    "TARGET_IMMEDIATE_TABLE",
    "TARGET_PROFILE_TABLE",
    "assert_completed",
    "assert_on_demand_invocation",
    "assert_profile_result",
    "benchmark_limits",
    "benchmark_marks",
    "capture_run",
    "configured_model_ids",
    "create_live_agent",
    "create_probe_database",
    "create_probe_home",
    "job_id_from_start",
    "live_provider",
    "logical_names",
    "open_live_home",
    "record_metrics",
    "results_for",
    "seed_completed_profile_agent",
    "start_profile_response",
    "toolbox_load_response",
    "stop_response",
    "transcript_text",
    "wait_for_running",
    "wait_for_terminal",
]
