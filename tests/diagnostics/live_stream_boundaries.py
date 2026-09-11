"""Frozen, bounded live qualification across HTTP, SDK, adapter and configured router.

Use --prepare --output PATH for an offline manifest, or --live --output PATH
only after live authorization. Each output path is single-use. Missing keys are
reported as missing coverage. The matrix stops at its first failed/uncertain case.
No captured prompts, existing agent homes, or external data sources are used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import time
from contextlib import AsyncExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import openai
from _stream_boundary_support import live_probe_fixture
from dotenv import dotenv_values
from google import genai
from google.genai import types
from live.benchmarks._support import RecordingProvider

from daita import Agent, AgentConfig, LocalWorkspace, LoopLimits, SQLiteSource
from daita._json import canonical_json, thaw_json
from daita.llm._lifecycle import (
    AttemptLifecycle,
    await_cleanup,
    closing_stream,
    native_events,
    transport_timeout,
)
from daita.llm.errors import ModelProviderError, interrupted_model_usage
from daita.llm.factory import create_model_route_provider
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelCallPolicy,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.gemini import GeminiProvider, _argument_snapshot_grew
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from daita.loop.models import LoopExitKind, validate_completed_transcript
from daita.security import SecretReference

MODELS = {"openai": "gpt-5.6-terra", "gemini": "gemini-3.5-flash"}
ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com",
}
KEY_NAMES = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
LAYERS = ("http", "sdk", "adapter", "router")
POLICY = ModelCallPolicy(max_request_seconds=90, max_attempt_seconds=90)
ISOLATED_RETRY = RetryPolicy(
    max_attempts_per_candidate=1, max_total_attempts=1, backoff_seconds=0
)
WORKFLOW_RETRY = RetryPolicy()
TOTAL_ALLOWANCE = Decimal("1.25")
TOTAL_SECONDS = 35 * 60
OUTPUT_TOKENS = 512
DB_TOKEN = "SYNTHETIC_CATALOG_7C91F2"
FILE_TOKEN = "SYNTHETIC_WORKSPACE_4B82D1"
READ_TOOLS = frozenset(
    {
        "toolbox_search",
        "toolbox_load",
        "toolbox_inspect",
        "catalog_search",
        "catalog_inspect",
        "data_query",
        "file_search",
        "file_read",
    }
)


class DiagnosticFailure(ValueError):
    """A fixed, non-secret harness failure reason suitable for evidence."""


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    layer: str
    scale: str
    allowance: Decimal
    seconds: int
    generations: int


def cases() -> tuple[Case, ...]:
    isolated = tuple(
        Case(f"{family}-{scale}-{layer}", family, layer, scale, Decimal("0.05"), 90, 1)
        for scale in ("small", "scaled")
        for family in MODELS
        for layer in LAYERS
    )
    workflows = tuple(
        Case(
            f"workflow-{name}",
            "openai",
            "workflow",
            name,
            Decimal("0.10"),
            120,
            4
            * min(
                WORKFLOW_RETRY.max_attempts_per_candidate,
                WORKFLOW_RETRY.max_total_attempts,
            ),
        )
        for name in ("answer", "catalog_correction", "workspace")
    )
    return isolated + workflows


class Ledger:
    """Diagnostic-only admission and evidence for the entire fixed matrix."""

    def __init__(self, planned: tuple[Case, ...], output: Path):
        self.planned = planned
        self.output = output
        self.started = time.monotonic()
        self.deadline = self.started + TOTAL_SECONDS
        self.current: Case | None = None
        self.rows: list[dict] = []
        self.launch: dict = {}
        self.stopped = False
        self.allowance = sum((case.allowance for case in planned), Decimal(0))
        if (
            self.allowance > TOTAL_ALLOWANCE
            or sum(c.generations for c in planned if c.layer != "workflow") > 16
        ):
            raise DiagnosticFailure("planned matrix exceeds the frozen allowance")
        self.claimed: set[str] = set()

    def save(self) -> None:
        known = sum(
            (
                Decimal(row["usage"]["known_estimated_cost_usd"])
                for row in self.rows
                if row.get("usage", {}).get("known_estimated_cost_usd") is not None
            ),
            Decimal(0),
        )
        unknown = [
            row["case"]
            for row in self.rows
            if row.get("physical_generations", 0)
            and row.get("usage", {}).get("status") != "complete"
        ]
        document = {
            "launch": self.launch,
            "rows": self.rows,
            "stopped": self.stopped,
            "accounting": {
                "admitted_planned_allowance_usd": str(self.allowance),
                "known_estimated_cost_usd": str(known),
                "unknown_usage_cases": unknown,
                "unstarted_cases": [
                    case.name for case in self.planned if case.name not in self.claimed
                ],
            },
        }
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, default=str) + "\n")
        temporary.replace(self.output)

    def begin(self, case: Case) -> dict:
        if self.stopped or case.name in self.claimed or case not in self.planned:
            raise RuntimeError("matrix cannot restart or replace a case")
        if time.monotonic() >= self.deadline:
            self.stopped = True
            raise TimeoutError("matrix execution deadline expired")
        self.claimed.add(case.name)
        self.current = case
        row = {
            "case": case.name,
            "family": case.family,
            "layer": case.layer,
            "scale": case.scale,
            "outcome": "started",
            "admitted_allowance_usd": str(case.allowance),
            "physical_generations": 0,
            "requests": [],
            "observations": [],
        }
        self.rows.append(row)
        self.save()
        return row

    def generation(self, row: dict) -> None:
        case = self.current
        if (
            self.stopped
            or case is None
            or row is not self.rows[-1]
            or time.monotonic() >= self.deadline
        ):
            raise RuntimeError("generation after matrix stop/deadline")
        if row["physical_generations"] >= case.generations:
            raise RuntimeError("physical generation ceiling exceeded")
        row["physical_generations"] += 1
        # Persist dispatch intent before transport I/O. Unknown usage never refunds it.
        self.save()

    def finish(self, row: dict) -> None:
        if row["outcome"] != "completed":
            self.stopped = True
        self.save()


def request(scale: str) -> ModelRequest:
    prompt, tools = live_probe_fixture(scale)
    return ModelRequest(
        messages=(CanonicalMessage(MessageRole.USER, content=(TextBlock(prompt),)),),
        tools=tuple(ToolDefinition(**tool) for tool in tools),
        max_total_tokens=20_000,
        max_estimated_cost_usd=Decimal("0.05"),
        call_policy=POLICY,
    )


def build_native(family: str, http: httpx.AsyncClient, key: str):
    sdk: Any
    provider: Any
    if family == "openai":
        sdk = openai.AsyncOpenAI(api_key=key, http_client=http, max_retries=0)
        provider = OpenAIResponsesProvider(
            MODELS[family], client=cast(Any, sdk), max_output_tokens=OUTPUT_TOKENS
        )
    else:
        sdk = genai.Client(
            api_key=key, http_options=types.HttpOptions(httpx_async_client=http)
        )
        provider = GeminiProvider(
            MODELS[family], client=cast(Any, sdk), max_output_tokens=OUTPUT_TOKENS
        )
    return sdk, provider


async def close_sdk(family: str, sdk: Any) -> None:
    if family == "gemini":
        await sdk.aio.aclose()
        sdk.close()
    else:
        await sdk.close()


class Keys:
    def __init__(self, values: dict[str, str]):
        self.values = values

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]


def route(family: str, retry: RetryPolicy) -> ModelRoute:
    profile = reviewed_model_profile(f"{family}:{MODELS[family]}")
    if profile is None:
        raise DiagnosticFailure("the frozen model has no reviewed profile")
    profile = replace(profile, max_output_tokens=OUTPUT_TOKENS)
    return ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=profile.id,
                profile=profile,
                secret_reference=SecretReference.environment(KEY_NAMES[family]),
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
        ),
        retry,
    )


@contextmanager
def sdk_constructor(family: str, sdk: Any):
    """Instrument only SDK construction; retain the production configured factory."""
    used = []

    def construct(**kwargs):
        used.append(True)
        return sdk

    module, name = (openai, "AsyncOpenAI") if family == "openai" else (genai, "Client")
    with patch.object(module, name, side_effect=construct):
        yield used


async def prepared_payload(family: str, scale: str) -> dict:
    """Obtain exact SDK wire JSON offline; the transport cannot access a network."""
    captured = []

    class Captured(Exception):
        pass

    def capture(req):
        captured.append(
            {
                "path": req.url.path,
                "query": req.url.query.decode(),
                "body": json.loads(req.content),
            }
        )
        raise Captured

    async with httpx.AsyncClient(transport=httpx.MockTransport(capture)) as http:
        sdk, provider = build_native(family, http, "offline-serialization-only")
        arguments = provider._request_arguments(request(scale))
        try:
            if family == "openai":
                arguments.update(max_output_tokens=OUTPUT_TOKENS, stream=True)
                await sdk.responses.create(**arguments)
            else:
                stream = await sdk.aio.models.generate_content_stream(**arguments)
                async with closing_stream(stream):
                    await anext(stream)
        except Exception:
            # OpenAI wraps transport exceptions; capture itself performs no I/O.
            if not captured:
                raise
        finally:
            await provider.close()
            await close_sdk(family, sdk)
    if len(captured) != 1:
        raise DiagnosticFailure(
            "offline serialization did not produce exactly one request"
        )
    value = captured[0]
    value["payload_sha256"] = digest(value["body"])
    return value


class Capture:
    """The same request, byte and response-release observations for all four arms."""

    def __init__(self, ledger: Ledger, row: dict, expected: dict | None):
        self.ledger, self.row, self.expected = ledger, row, expected
        self.started = time.monotonic()

    def seconds(self) -> float:
        return round(time.monotonic() - self.started, 6)

    async def sent(self, req: httpx.Request) -> None:
        case = self.ledger.current
        assert case is not None
        if req.url.host != httpx.URL(ENDPOINTS[case.family]).host:
            raise DiagnosticFailure("unexpected request origin")
        counting = req.url.path.endswith(("/input_tokens", ":countTokens"))
        generation = req.url.path.endswith(("/responses", ":streamGenerateContent"))
        if not (counting or generation):
            raise DiagnosticFailure("unexpected SDK request path")
        payload_hash = digest(json.loads(req.content))
        if generation:
            if self.expected is not None and (
                payload_hash != self.expected["payload_sha256"]
                or req.url.path != self.expected["path"]
                or req.url.query.decode() != self.expected["query"]
            ):
                raise DiagnosticFailure("prepared generation payload drift")
            self.ledger.generation(self.row)
        self.row["requests"].append(
            {
                "phase": "count" if counting else "generation",
                "path": req.url.path,
                "payload_sha256": payload_hash,
                "sent_seconds": self.seconds(),
            }
        )
        self.ledger.save()

    async def received(self, response: httpx.Response) -> None:
        observation: dict[str, Any] = {
            "path": response.request.url.path,
            "status": response.status_code,
            "headers_seconds": self.seconds(),
            "bytes": 0,
            "chunks": 0,
            "first_body_seconds": None,
            "last_body_seconds": None,
            "close_started_seconds": None,
            "close_finished_seconds": None,
            "close_outcome": "pending",
        }
        self.row["observations"].append(observation)
        if not isinstance(response.stream, httpx.AsyncByteStream):
            raise TypeError("expected asynchronous response stream")
        source: httpx.AsyncByteStream = response.stream
        capture = self

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                async for chunk in source:
                    observation["chunks"] += 1
                    observation["bytes"] += len(chunk)
                    observation["first_body_seconds"] = (
                        observation["first_body_seconds"] or capture.seconds()
                    )
                    observation["last_body_seconds"] = capture.seconds()
                    yield chunk

            async def aclose(self):
                observation["close_started_seconds"] = capture.seconds()
                try:
                    await source.aclose()
                except BaseException:
                    observation["close_outcome"] = "uncertain"
                    raise
                else:
                    observation["close_outcome"] = "released"
                finally:
                    observation["close_finished_seconds"] = capture.seconds()

        response.stream = Body()


async def raw_events(
    http: httpx.AsyncClient,
    family: str,
    key: str,
    payload: dict,
    attempt: AttemptLifecycle,
):
    headers = (
        {"Authorization": f"Bearer {key}"}
        if family == "openai"
        else {"x-goog-api-key": key}
    )
    origin = str(httpx.URL(ENDPOINTS[family]).copy_with(path="")).rstrip("/")
    url = (
        origin + payload["path"] + ("?" + payload["query"] if payload["query"] else "")
    )
    async with http.stream(
        "POST",
        url,
        headers=headers,
        json=payload["body"],
        timeout=transport_timeout(attempt.request),
    ) as response:
        attempt.track_response_release(response)
        attempt.observation.headers(
            "generation", response.status_code, response.headers.get("x-request-id")
        )
        response.raise_for_status()
        data = []
        size = 0
        async for line in response.aiter_lines():
            if len(line) > 1_000_000:
                raise DiagnosticFailure("oversized SSE frame")
            if line.startswith("data:"):
                size += len(line)
                if size > 1_000_000:
                    raise DiagnosticFailure("oversized SSE frame")
                data.append(line[5:].lstrip())
            elif not line and data:
                encoded = "\n".join(data)
                data.clear()
                size = 0
                if encoded != "[DONE]":
                    yield json.loads(encoded)
        if data:
            raise DiagnosticFailure("unterminated SSE frame")


async def direct(
    case: Case,
    req: ModelRequest,
    sdk: Any,
    provider: Any,
    http: httpx.AsyncClient,
    key: str,
    expected: dict,
) -> ModelResponse:
    attempt = AttemptLifecycle(
        provider._native_owner, req, headers_supported=True, observable=True
    )
    req = attempt.request
    failure = None
    response = None
    try:
        async with attempt:
            arguments = provider._request_arguments(req)
            counted = await provider._admit_request(
                req, arguments, attempt, requested_at=datetime.now(UTC)
            )
            cap = (
                arguments.get("max_output_tokens")
                if case.family == "openai"
                else arguments["config"]["max_output_tokens"]
            )
            if cap != OUTPUT_TOKENS:
                raise DiagnosticFailure("frozen output allowance is inadmissible")
            attempt.values["output_cap"] = cap
            attempt.values["counted_input_tokens"] = counted
            attempt.dispatch()
            if case.layer == "http":
                source = raw_events(http, case.family, key, expected, attempt)
            elif case.family == "openai":
                source = native_events(
                    lambda: sdk.responses.create(
                        **arguments, stream=True, timeout=transport_timeout(req)
                    )
                )
            else:
                arguments["config"]["http_options"]["timeout"] = int(
                    min(
                        POLICY.read_timeout_seconds,
                        attempt.execution_deadline - time.monotonic(),
                    )
                    * 1000
                )
                source = native_events(
                    lambda: sdk.aio.models.generate_content_stream(**arguments)
                )
            terminal = None
            parts: list[Any] = []
            call_ids: dict[str, Any] = {}
            usage, finish, model_version, response_id = None, None, None, None
            async with attempt.stream(source) as events:
                async for native in events:
                    if case.family == "openai":
                        event = (
                            native if isinstance(native, dict) else native.model_dump()
                        )
                        kind = event.get("type")
                        attempt.native(kind, (event.get("response") or {}).get("id"))
                        if kind in {
                            "response.output_text.delta",
                            "response.function_call_arguments.delta",
                            "response.reasoning_summary_text.delta",
                            "response.reasoning_text.delta",
                        }:
                            attempt.progress(event.get("delta", ""))
                        if kind in {
                            "response.completed",
                            "response.incomplete",
                            "response.failed",
                        }:
                            if terminal is not None:
                                raise DiagnosticFailure("duplicate terminal response")
                            terminal = event["response"]
                            response = provider._decode_response(terminal)
                            attempt.response(response)
                    else:
                        chunk = (
                            types.GenerateContentResponse.model_validate(native)
                            if isinstance(native, dict)
                            else native
                        )
                        attempt.native("generate_content_chunk", chunk.response_id)
                        usage = chunk.usage_metadata or usage
                        model_version = chunk.model_version or model_version
                        response_id = chunk.response_id or response_id
                        if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                            raise DiagnosticFailure("provider blocked the probe")
                        for candidate in chunk.candidates or []:
                            if candidate.finish_reason:
                                if (
                                    finish is not None
                                    and finish != candidate.finish_reason
                                ):
                                    raise DiagnosticFailure("finish reason changed")
                                finish = candidate.finish_reason
                            for part in (
                                candidate.content.parts if candidate.content else []
                            ) or []:
                                if part.text:
                                    attempt.progress(part.text)
                                call = part.function_call
                                if call is not None:
                                    if (
                                        call.partial_args is not None
                                        or call.will_continue
                                    ):
                                        raise DiagnosticFailure(
                                            "partial function protocol unsupported"
                                        )
                                    if call.id and call.id in call_ids:
                                        previous = call_ids[call.id]
                                        if previous.function_call.name != call.name:
                                            raise DiagnosticFailure(
                                                "function identity changed"
                                            )
                                        if _argument_snapshot_grew(
                                            previous.function_call.args, call.args
                                        ):
                                            previous.function_call.args = call.args
                                            attempt.progress(canonical_json(call.args))
                                        continue
                                    if call.id:
                                        call_ids[call.id] = part
                                    attempt.progress(canonical_json(call.args))
                                parts.append(part)
                if case.family == "gemini":
                    response = provider._decode_response(
                        {
                            "response_id": response_id,
                            "model_version": model_version,
                            "usage_metadata": usage,
                            "candidates": [
                                {"finish_reason": finish, "content": {"parts": parts}}
                            ],
                        }
                    )
                    attempt.response(response)
                if response is None:
                    raise DiagnosticFailure("EOF without terminal response")
    except BaseException as error:
        failure = error
        raise
    finally:
        attempt.finish(failure)
    return attempt.observation.response(response)


class ReadOnlyRecorder(RecordingProvider):
    def __init__(self, delegate, allowed: frozenset[str]):
        super().__init__(delegate)
        self.allowed = allowed

    def check(self, response: ModelResponse) -> None:
        if any(call.name not in self.allowed for call in response.tool_calls):
            raise DiagnosticFailure(
                "workflow requested a tool outside its read-only fixture"
            )

    async def close(self, *, deadline: float | None = None) -> None:
        await self._delegate.close(deadline=deadline)

    async def generate(self, request):
        response = await super().generate(request)
        self.check(response)
        return response

    async def stream(self, request):
        async with closing_stream(super().stream(request)) as events:
            async for event in events:
                if isinstance(event, ModelStreamCompleted):
                    self.check(event.response)
                yield event


async def prepare_workflow(case: Case, root: Path, keys: Keys) -> dict:
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "probe.txt").write_text(FILE_TOKEN + "\n")
    model_route = route("openai", WORKFLOW_RETRY)
    recorder = ReadOnlyRecorder(
        create_model_route_provider(model_route, secret_provider=keys),
        frozenset() if case.scale == "answer" else READ_TOOLS,
    )
    agent = await Agent.create(
        "reliability-probe",
        root=root / "agents",
        workspace=LocalWorkspace(workspace),
        model=recorder,
        model_profile=model_route.model_profile,
        config=AgentConfig(model_call_policy=POLICY),
        limits=LoopLimits(
            max_steps=4,
            max_wall_time_seconds=120,
            max_total_tokens=32_000,
            max_estimated_cost_usd=case.allowance,
        ),
    )
    source_ids: tuple[str, ...] = ()
    bindings = {}
    try:
        if case.scale == "answer":
            prompt = "Reply exactly RELIABILITY_OK. Do not call tools."
        elif case.scale == "workspace":
            prompt = "Read probe.txt in the admitted workspace using file_read. Return its exact verification token. Do not guess or use other tools."
        else:
            database = root / "probe.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE reliability_probe (verification_token TEXT NOT NULL, amount INTEGER NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO reliability_probe VALUES (?, ?)", (DB_TOKEN, 73)
                )
            source = await agent.attach(
                SQLiteSource(database, name="Synthetic reliability probe")
            )
            (resource,) = await agent.catalog_preview()
            source_ids = (source.id,)
            bindings = {"source_id": source.id, "resource_id": resource.id}
            prompt = (
                "Exercise ordinary read-tool correction on this synthetic fixture. "
                f"Use source_id {source.id} and resource_ids [{resource.id}]. "
                "Load data_query if necessary. First call data_query with exactly "
                "SELECT correction_probe FROM reliability_probe. The missing column is intentional: "
                "submit that one invalid read to observe the normal tool error. Then correct the SQL "
                "to SELECT verification_token, amount FROM reliability_probe and return the values "
                "from the successful tool result. Do not repeat the invalid call or use other data tools."
            )
        return {
            "agent": agent,
            "recorder": recorder,
            "prompt": prompt,
            "source_ids": source_ids,
            "bindings": bindings,
            "root": root,
            "fixture_sha256": digest(
                {
                    "schema": "reliability_probe(verification_token TEXT, amount INTEGER)",
                    "row": [DB_TOKEN, 73],
                    "file": FILE_TOKEN,
                }
            ),
        }
    except BaseException:
        await agent.close()
        await recorder.close()
        raise


async def workflow(case: Case, fixture: dict, row: dict) -> Any:
    agent = fixture["agent"]
    result = await agent.run(
        fixture["prompt"],
        source_scope_ids=fixture["source_ids"],
        files_only=case.scale == "workspace",
    )
    transcript = await agent.transcript(result.run_id)
    row["run"] = {
        "kind": result.kind.value,
        "reason": result.reason,
        "steps": result.steps,
        "transcript_sha256": digest(
            [
                message.to_dict() if hasattr(message, "to_dict") else repr(message)
                for message in transcript.messages
            ]
        ),
    }
    row["usage"] = usage_record(result.usage)
    if result.kind is not LoopExitKind.COMPLETED:
        raise DiagnosticFailure("public workflow did not complete")
    validate_completed_transcript(transcript, result)
    calls = {
        call.id: call for message in transcript.messages for call in message.tool_calls
    }
    results = [
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    row["tool_results"] = [
        {"name": calls[block.call_id].name, "is_error": block.is_error}
        for block in results
    ]
    if case.scale == "answer":
        if calls or (result.final_text or "").strip() != "RELIABILITY_OK":
            raise DiagnosticFailure("tool-free outcome mismatch")
    elif case.scale == "workspace":
        if FILE_TOKEN not in (result.final_text or "") or not any(
            calls[r.call_id].name == "file_read"
            and not r.is_error
            and FILE_TOKEN in canonical_json(r.output)
            for r in results
        ):
            raise DiagnosticFailure("workspace evidence missing")
    else:
        queries = [r for r in results if calls[r.call_id].name == "data_query"]
        if len(queries) != 2 or not queries[0].is_error or queries[1].is_error:
            raise DiagnosticFailure("ordinary tool correction was not demonstrated")
        if "correction_probe" not in calls[queries[0].call_id].arguments.get("sql", ""):
            raise DiagnosticFailure("unexpected initial query failure")
        if (
            DB_TOKEN not in canonical_json(queries[1].output)
            or DB_TOKEN not in (result.final_text or "")
            or "73" not in (result.final_text or "")
        ):
            raise DiagnosticFailure("catalog read evidence missing")
    return result.usage


def usage_record(usage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "status": usage.cost_estimate.status.value,
        "known_estimated_cost_usd": (
            str(usage.cost_estimate.amount_usd)
            if usage.cost_estimate.amount_usd is not None
            else None
        ),
    }


async def one(
    case: Case,
    ledger: Ledger,
    keys: dict[str, str],
    expected: dict | None,
    fixture: dict | None = None,
    transport=None,
) -> dict:
    row = ledger.begin(case)
    capture = Capture(ledger, row, expected)
    deadline = min(ledger.deadline, time.monotonic() + case.seconds)
    row["emergency_guard_fired"] = False
    row["native_observation"] = "attempt_diagnostic"
    row["canonical_observation"] = (
        "not_applicable" if case.layer in {"http", "sdk"} else "recorded"
    )
    http = httpx.AsyncClient(
        timeout=transport_timeout(),
        transport=transport,
        event_hooks={"request": [capture.sent], "response": [capture.received]},
    )
    sdk, provider = build_native(case.family, http, keys[KEY_NAMES[case.family]])
    recorder = None
    owned_by_route = False
    try:
        # Later than production execution plus its single cleanup grace.
        async with asyncio.timeout_at(deadline + POLICY.cleanup_timeout_seconds + 2):
            if case.layer == "workflow":
                assert fixture is not None
                recorder = fixture["recorder"]
                with sdk_constructor(case.family, sdk) as constructed:
                    try:
                        usage = await workflow(case, fixture, row)
                    finally:
                        owned_by_route = bool(constructed)
            elif case.layer in {"adapter", "router"}:
                source = (
                    provider
                    if case.layer == "adapter"
                    else create_model_route_provider(
                        route(case.family, ISOLATED_RETRY), secret_provider=Keys(keys)
                    )
                )
                recorder = ReadOnlyRecorder(source, frozenset({"record_probe"}))
                response = None
                with sdk_constructor(case.family, sdk) as constructed:
                    try:
                        async with closing_stream(
                            recorder.stream(
                                replace(request(case.scale), deadline=deadline)
                            )
                        ) as events:
                            async for event in events:
                                if isinstance(event, ModelStreamCompleted):
                                    response = event.response
                    finally:
                        owned_by_route = bool(constructed)
                if response is None:
                    raise DiagnosticFailure("no terminal canonical response")
                usage = response.usage
                row["usage"] = usage_record(usage)
                validate_probe(response)
            else:
                response = await direct(
                    case,
                    replace(request(case.scale), deadline=deadline),
                    sdk,
                    provider,
                    http,
                    keys[KEY_NAMES[case.family]],
                    cast(dict, expected),
                )
                usage = response.usage
                row["usage"] = usage_record(usage)
                validate_probe(response)
                row["attempt_diagnostic"] = thaw_json(
                    cast(Any, response.provider_metadata.get("attempt_diagnostic", {}))
                )
            row["usage"] = usage_record(usage)
            if usage.cost_estimate.status.value != "complete":
                raise DiagnosticFailure("unknown usage stops qualification")
            if (
                usage.cost_estimate.amount_usd is None
                or usage.cost_estimate.amount_usd > case.allowance
            ):
                raise DiagnosticFailure("case cost allowance exceeded")
            if case.layer != "workflow" and row["physical_generations"] != 1:
                raise DiagnosticFailure("isolated generation count mismatch")
            row["outcome"] = "completed"
    except BaseException as error:
        row["outcome"] = "failed"
        row["error_type"] = type(error).__name__
        if isinstance(error, DiagnosticFailure):
            row["failure_reason"] = str(error)
        if isinstance(error, ModelProviderError):
            row["error_code"] = error.code.value
            row["usage"] = usage_record(error.usage)
            row["diagnostic_code"] = error.diagnostic.code if error.diagnostic else None
        elif isinstance(error, (asyncio.CancelledError, TimeoutError)):
            row.setdefault("usage", usage_record(interrupted_model_usage(error)))
            row["emergency_guard_fired"] = isinstance(error, TimeoutError)
        if row["physical_generations"] and "usage" not in row:
            row["usage"] = {"status": "unknown", "known_estimated_cost_usd": None}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            ledger.stopped = True
            ledger.save()
            raise
    finally:
        cleanup_started = time.monotonic()
        # Reuse native generation retirement's grace when it has already begun.
        starts = [
            o["close_started_seconds"]
            for o in row["observations"]
            if o["path"].endswith(("/responses", ":streamGenerateContent"))
            and o["close_started_seconds"] is not None
        ]
        cleanup_deadline = min(
            deadline + POLICY.cleanup_timeout_seconds,
            (capture.started + max(starts) if starts else cleanup_started)
            + POLICY.cleanup_timeout_seconds,
        )

        # Drain public state ownership in this task; never retain SQLite work
        # in the lifecycle's native-only task owner.
        state_error = None
        if fixture is not None:
            try:
                await fixture["agent"].close()
            except BaseException as error:
                state_error = error
            fixture["closed"] = True

        async def cleanup_native():
            closures = [provider.close(deadline=cleanup_deadline), http.aclose()]
            if recorder is not None:
                closures.append(recorder.close(deadline=cleanup_deadline))
            if not owned_by_route:
                closures.append(close_sdk(case.family, sdk))
            outcomes = await asyncio.gather(*closures, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

        try:
            await await_cleanup(
                asyncio.create_task(cleanup_native()),
                deadline=cleanup_deadline,
                owner=provider._native_owner,
            )
            if state_error is not None:
                raise state_error
            if any(o["close_outcome"] != "released" for o in row["observations"]):
                raise DiagnosticFailure("response release unconfirmed")
            row["cleanup"] = "released"
        except BaseException as error:
            row["cleanup"] = "uncertain"
            row["cleanup_error_type"] = type(error).__name__
            row["outcome"] = "failed"
        row["cleanup_seconds"] = round(time.monotonic() - cleanup_started, 6)
        row["seconds"] = capture.seconds()
        if recorder is not None:
            row["attempts"] = recorder.timings
        ledger.finish(row)
    return row


async def close_fixture(fixture: dict, ledger: Ledger) -> None:
    if fixture.get("closed"):
        return
    fixture["closed"] = True
    try:
        try:
            await fixture["agent"].close()
        finally:
            await fixture["recorder"].close(
                deadline=time.monotonic() + POLICY.cleanup_timeout_seconds
            )
    except BaseException as error:
        ledger.stopped = True
        ledger.launch["result"] = "fixture_cleanup_failed"
        ledger.launch["cleanup_error_type"] = type(error).__name__
        ledger.save()


def validate_probe(response: ModelResponse) -> None:
    calls = response.tool_calls
    if (
        len(calls) != 1
        or calls[0].name != "record_probe"
        or dict(calls[0].arguments) != {"value": 7}
    ):
        raise DiagnosticFailure("terminal inert tool call mismatch")


def code_manifest() -> dict:
    paths = sorted(
        {
            Path("pyproject.toml"),
            Path(__file__),
            Path("tests/_stream_boundary_support.py"),
            Path("tests/live/benchmarks/_support.py"),
            *Path("src/daita").rglob("*.py"),
            *Path("src/daita/llm").glob("*.json"),
        }
    )
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


async def main(
    output: Path, *, live: bool = False, key_values: dict[str, str] | None = None
) -> dict:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite/restart a prior frozen run.
    with output.open("x") as handle:
        handle.write("{}\n")
    values = key_values
    if values is None:
        file_values = dotenv_values(Path.cwd() / ".env") if live else {}
        values = (
            {
                name: str(os.environ.get(name) or file_values.get(name) or "")
                for name in KEY_NAMES.values()
            }
            if live
            else {}
        )
    ledger = Ledger(cases(), output)
    prepared = {
        f"{family}-{scale}": await prepared_payload(family, scale)
        for family in MODELS
        for scale in ("small", "scaled")
    }
    with tempfile.TemporaryDirectory(
        prefix="daita-live-qualification-", dir=output.parent
    ) as temporary:
        async with AsyncExitStack() as stack:
            fixtures = {}
            for case in ledger.planned:
                if case.layer == "workflow":
                    fixture = await prepare_workflow(
                        case, Path(temporary) / case.scale, Keys(values)
                    )
                    fixtures[case.name] = fixture
                    stack.push_async_callback(close_fixture, fixture, ledger)
            ledger.launch = {
                "started_utc": datetime.now(UTC).isoformat(),
                "mode": "live" if live else "prepare_only",
                "head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "worktree_status": subprocess.check_output(
                    ["git", "status", "--short"], text=True
                ),
                "code_sha256": code_manifest(),
                "dependencies": {
                    name: version(name)
                    for name in ("openai", "google-genai", "httpx", "anyio")
                },
                "models": MODELS,
                "routes": {
                    family: {
                        "profile": asdict(route(family, ISOLATED_RETRY).model_profile),
                        "secret_reference_name": KEY_NAMES[family],
                        "allowed_sensitivities": sorted(
                            s.value for s in ModelSensitivity
                        ),
                    }
                    for family in MODELS
                },
                "expected_outcomes": {
                    "isolated": {
                        "tool": "record_probe",
                        "arguments": {"value": 7},
                        "calls": 1,
                    },
                    "answer": "RELIABILITY_OK, no tools",
                    "catalog_correction": "one invalid data_query, one successful corrected read, token and amount in final answer",
                    "workspace": "successful file_read and exact file token in final answer",
                },
                "workflow_limits": {
                    "max_steps": 4,
                    "max_wall_time_seconds": 120,
                    "max_total_tokens": 32_000,
                    "max_estimated_cost_usd": "0.10",
                },
                "endpoints": ENDPOINTS,
                "policy": asdict(POLICY),
                "isolated_retry": asdict(ISOLATED_RETRY),
                "workflow_retry": asdict(WORKFLOW_RETRY),
                "cases": [asdict(case) for case in ledger.planned],
                "prepared": prepared,
                "workflows": {
                    name: {
                        "prompt_sha256": digest(value["prompt"]),
                        "fixture_sha256": value["fixture_sha256"],
                        "bindings": value["bindings"],
                    }
                    for name, value in fixtures.items()
                },
                "admitted_planned_allowance_usd": str(ledger.allowance),
                "whole_allowance_usd": str(TOTAL_ALLOWANCE),
                "whole_seconds": TOTAL_SECONDS,
                "isolated_output_tokens": OUTPUT_TOKENS,
                "scaled_reference_input_tokens": 6479,
                "missing_credentials": (
                    [name for name in KEY_NAMES.values() if not values.get(name)]
                    if live
                    else "not_read"
                ),
                "result": "prepared" if not live else "running",
            }
            ledger.save()
            if not live:
                return ledger.launch
            for case in ledger.planned:
                if not values.get(KEY_NAMES[case.family]):
                    ledger.rows.append(
                        {
                            "case": case.name,
                            "outcome": "missing_credentials",
                            "physical_generations": 0,
                        }
                    )
                    ledger.save()
                    continue
                if code_manifest() != ledger.launch["code_sha256"]:
                    ledger.stopped = True
                    ledger.launch["result"] = "code_drift"
                    break
                try:
                    row = await one(
                        case,
                        ledger,
                        values,
                        prepared.get(f"{case.family}-{case.scale}"),
                        fixtures.get(case.name),
                    )
                except BaseException as error:
                    ledger.stopped = True
                    ledger.launch["result"] = "failed_before_case"
                    ledger.launch["error_type"] = type(error).__name__
                    break
                print(
                    json.dumps(
                        {key: row[key] for key in ("case", "outcome", "seconds")}
                    ),
                    flush=True,
                )
                if ledger.stopped:
                    break
            if ledger.launch["result"] == "running":
                ledger.launch["result"] = (
                    "failed"
                    if ledger.stopped
                    else (
                        "missing_coverage"
                        if ledger.launch["missing_credentials"]
                        else "passed"
                    )
                )
            ledger.save()
            return ledger.launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = asyncio.run(main(arguments.output, live=arguments.live))
    raise SystemExit(0 if result["result"] in {"prepared", "passed"} else 1)
