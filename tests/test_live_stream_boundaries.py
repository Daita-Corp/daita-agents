"""Offline qualification of the bounded live diagnostic; no network or API keys."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from typing import Any

import httpx
import pytest
from _stream_boundary_support import COMPLETED, sse
from diagnostics import live_stream_boundaries as runner

from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelStreamCompleted,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockStreamingModelProvider


class Body(httpx.AsyncByteStream):
    def __init__(self, data: bytes, *, fail_close: bool = False):
        self.data = data
        self.fail_close = fail_close

    async def __aiter__(self):
        yield self.data

    async def aclose(self):
        if self.fail_close:
            raise RuntimeError("synthetic close failure")


def transport_for(family, *, unknown_usage=False, fail_close=False):
    """Exercise the installed SDKs with real wire events and no socket access."""

    def handle(request):
        event: dict[str, Any]
        counting = request.url.path.endswith(("/input_tokens", ":countTokens"))
        if counting:
            payload = (
                {"object": "response.input_tokens", "input_tokens": 6479}
                if family == "openai"
                else {"totalTokens": 6479}
            )
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=Body(json.dumps(payload).encode()),
            )
        if family == "openai":
            event = deepcopy(COMPLETED)
            event["response"]["output"] = [
                {
                    "id": "fc_probe",
                    "type": "function_call",
                    "call_id": "probe",
                    "name": "record_probe",
                    "arguments": '{"value":7}',
                    "status": "completed",
                }
            ]
            if unknown_usage:
                event["response"].pop("usage")
        else:
            event = {
                "responseId": "gemini_probe",
                "modelVersion": runner.MODELS[family],
                "candidates": [
                    {
                        "index": 0,
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "record_probe",
                                        "args": {"value": 7},
                                    }
                                }
                            ],
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 11,
                },
            }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Body(sse(event), fail_close=fail_close),
        )

    return httpx.MockTransport(handle)


@pytest.mark.parametrize("case", runner.cases()[:16], ids=lambda case: case.name)
async def test_installed_sdk_matrix_uses_frozen_payload_and_one_generation(
    tmp_path, case
):
    expected = await runner.prepared_payload(case.family, case.scale)
    ledger = runner.Ledger((case,), tmp_path / "evidence.json")
    row = await runner.one(
        case,
        ledger,
        {runner.KEY_NAMES[case.family]: "offline-only"},
        expected,
        transport=transport_for(case.family),
    )
    assert row["outcome"] == "completed", row
    assert row["physical_generations"] == 1
    assert row["usage"]["status"] == "complete"
    assert row["cleanup"] == "released"
    assert not row["emergency_guard_fired"]
    assert [r["phase"] for r in row["requests"]] == ["count", "generation"]
    assert row["requests"][-1]["payload_sha256"] == expected["payload_sha256"]
    assert all(
        o["bytes"] > 0 and o["close_outcome"] == "released" for o in row["observations"]
    )
    assert not ledger.stopped


@pytest.mark.parametrize("fault", ("unknown_usage", "fail_close"))
async def test_uncertainty_stops_matrix_without_replay(tmp_path, fault):
    case = next(
        c for c in runner.cases() if c.family == "openai" and c.layer == "adapter"
    )
    ledger = runner.Ledger(runner.cases(), tmp_path / "evidence.json")
    row = await runner.one(
        case,
        ledger,
        {"OPENAI_API_KEY": "offline-only"},
        await runner.prepared_payload("openai", "small"),
        transport=transport_for("openai", **{fault: True}),
    )
    assert row["outcome"] == "failed"
    assert row["physical_generations"] == 1
    assert ledger.stopped
    if fault == "fail_close":
        assert row["cleanup"] == "uncertain"
        assert row["usage"]["status"] == "complete"
        assert row["usage"]["total_tokens"] == 11
    with pytest.raises(RuntimeError, match="cannot restart"):
        ledger.begin(runner.cases()[1])
    document = json.loads(ledger.output.read_text())
    assert document["accounting"]["admitted_planned_allowance_usd"] == "1.10"


async def test_transport_rejects_payload_drift_and_extra_generation_before_io(tmp_path):
    case = runner.cases()[0]
    expected = await runner.prepared_payload(case.family, case.scale)
    ledger = runner.Ledger((case,), tmp_path / "evidence.json")
    row = ledger.begin(case)
    capture = runner.Capture(ledger, row, expected)
    url = "https://api.openai.com" + expected["path"]
    with pytest.raises(runner.DiagnosticFailure, match="payload drift"):
        await capture.sent(httpx.Request("POST", url, json={"changed": True}))
    assert row["physical_generations"] == 0
    await capture.sent(httpx.Request("POST", url, json=expected["body"]))
    with pytest.raises(RuntimeError, match="generation ceiling"):
        await capture.sent(httpx.Request("POST", url, json=expected["body"]))
    assert row["physical_generations"] == 1
    ledger.deadline = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="deadline"):
        ledger.generation(row)
    with pytest.raises(runner.DiagnosticFailure, match="allowance"):
        runner.Ledger(
            (replace(case, allowance=Decimal("1.26")),), tmp_path / "bad.json"
        )


async def test_prepare_freezes_entire_plan_offline_without_reading_keys(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "prepare must not read credentials or send network traffic"
        )

    monkeypatch.setattr(runner, "dotenv_values", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    output = tmp_path / "manifest.json"
    launch = await runner.main(output)
    assert launch["result"] == "prepared"
    assert len(launch["cases"]) == 19
    assert len(launch["prepared"]) == 4
    assert len(launch["workflows"]) == 3
    assert launch["missing_credentials"] == "not_read"
    assert launch["admitted_planned_allowance_usd"] == "1.10"
    assert launch["whole_allowance_usd"] == "1.25"
    assert launch["isolated_retry"]["max_total_attempts"] == 1
    assert launch["isolated_retry"]["max_attempts_per_candidate"] == 1
    assert launch["workflow_limits"]["max_steps"] == 4
    assert launch["code_sha256"] and launch["dependencies"] and launch["head"]
    for family in runner.MODELS:
        small = launch["prepared"][f"{family}-small"]
        scaled = launch["prepared"][f"{family}-scaled"]
        assert len(json.dumps(scaled["body"])) > 20 * len(json.dumps(small["body"]))
        assert small["payload_sha256"] != scaled["payload_sha256"]
    assert json.loads(output.read_text())["rows"] == []
    with pytest.raises(FileExistsError):
        await runner.main(output)


@pytest.mark.parametrize("credentials", (False, True))
async def test_matrix_missing_coverage_or_first_failure_is_terminal(
    tmp_path, monkeypatch, credentials
):
    seen = []

    async def fail(case, ledger, *args):
        seen.append(case.name)
        row = ledger.begin(case)
        ledger.generation(row)
        row.update(outcome="failed", seconds=0, usage={"status": "unknown"})
        ledger.finish(row)
        return row

    monkeypatch.setattr(runner, "one", fail)
    output = tmp_path / "matrix.json"
    launch = await runner.main(
        output,
        live=True,
        key_values=(
            {name: "offline-only" for name in runner.KEY_NAMES.values()}
            if credentials
            else {}
        ),
    )
    document = json.loads(output.read_text())
    if credentials:
        assert launch["result"] == "failed"
        assert len(seen) == len(document["rows"]) == 1
        assert document["accounting"]["unknown_usage_cases"] == seen
    else:
        assert launch["result"] == "missing_coverage"
        assert seen == []
        assert len(document["rows"]) == 19
    assert document["accounting"]["admitted_planned_allowance_usd"] == "1.10"


@pytest.mark.parametrize("case", runner.cases()[16:], ids=lambda case: case.name)
async def test_public_workflow_fixtures_use_actual_agent_and_tool_results(
    tmp_path, case
):
    ledger = runner.Ledger((case,), tmp_path / "workflow.json")
    fixture = await runner.prepare_workflow(case, tmp_path / "fixture", runner.Keys({}))
    usage = ModelUsage(
        input_tokens=1,
        output_tokens=1,
        cost_estimate=CostEstimate.complete(Decimal("0.00001")),
    )

    def answer(text):
        return ModelResponse(FinishReason.STOP, text=text, usage=usage)

    def call(identity, name, arguments):
        return ModelResponse(
            FinishReason.TOOL_CALLS,
            tool_calls=(ToolCall(identity, name, arguments),),
            usage=usage,
        )

    if case.scale == "answer":
        responses = [answer("RELIABILITY_OK")]
    elif case.scale == "workspace":
        responses = [
            call("read", "file_read", {"path": "probe.txt"}),
            answer(runner.FILE_TOKEN),
        ]
    else:
        binding = fixture["bindings"]
        arguments = {
            "source_id": binding["source_id"],
            "resource_ids": [binding["resource_id"]],
        }
        responses = [
            call("load", "toolbox_load", {"tool_names": ["data_query"]}),
            call(
                "bad",
                "data_query",
                {**arguments, "sql": "SELECT correction_probe FROM reliability_probe"},
            ),
            call(
                "good",
                "data_query",
                {
                    **arguments,
                    "sql": "SELECT verification_token, amount FROM reliability_probe",
                },
            ),
            answer(f"{runner.DB_TOKEN} 73"),
        ]
    fixture["recorder"]._delegate = MockStreamingModelProvider(
        [(ModelStreamCompleted(response),) for response in responses],
        provider_id="openai:" + runner.MODELS["openai"],
        complete_pricing=True,
    )
    row: dict[str, Any] = {}
    try:
        result_usage = await runner.workflow(case, fixture, row)
        assert result_usage.cost_estimate.status.value == "complete"
        assert row["run"]["kind"] == "completed"
        assert row["run"]["steps"] == len(responses)
        if case.scale == "catalog_correction":
            queries = [r for r in row["tool_results"] if r["name"] == "data_query"]
            assert [r["is_error"] for r in queries] == [True, False]
        with pytest.raises(runner.DiagnosticFailure, match="read-only fixture"):
            fixture["recorder"].check(call("effect", "data_update_rows", {}))
    finally:
        await runner.close_fixture(fixture, ledger)
    assert not ledger.stopped
