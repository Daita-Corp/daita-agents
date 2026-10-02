"""Offline qualification of live read evidence, budgets and transport ownership."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import httpx2 as httpx
import pytest

from daita import MCPAuthentication, MCPToolSelection
from daita.adapters.mcp import MCPTransportError, SDKMCPClientFactory
from daita.llm.models import FinishReason, ModelResponse, ModelUsage, ToolCall
from daita.llm.pricing import CostEstimate
from daita.loop.models import LoopExit, LoopExitKind
from daita.security import EmptySecretProvider
from tests.support import mcp_read_harness as harness
from tests.support.job_benchmarks import RecordingProvider
from tests.support.mcp import MCPConformanceTransport
from tests.support.paths import REPO_ROOT
from tests.support.toolbox_model import ToolboxAwareMockModelProvider


def test_read_budget_preserves_missing_usage_and_never_clips_known_consumption():
    budget = harness.ReadBudget(Decimal("1.00"), Decimal("0.50"))
    budget.reserve()
    budget.settle(None)
    assert budget.charged == Decimal("0.50") and budget.incomplete_runs == 1
    budget.reserve()
    with pytest.raises(RuntimeError, match="before dispatch"):
        budget.reserve()
    result = LoopExit(
        "run",
        "conversation",
        LoopExitKind.COMPLETED,
        "completed",
        datetime.now(UTC),
        final_text="done",
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0.70"))),
    )
    budget.settle(result)
    assert budget.charged == Decimal("1.20")
    with pytest.raises(RuntimeError, match="before dispatch"):
        budget.reserve()
    incomplete = harness.ReadBudget(Decimal("1.00"), Decimal("0.50"))
    incomplete.reserve()
    incomplete.settle(
        replace(
            result,
            usage=ModelUsage(
                cost_estimate=CostEstimate.partial(
                    Decimal("0.70"), code="missing_usage"
                )
            ),
        )
    )
    assert incomplete.charged == Decimal("0.70") and incomplete.incomplete_runs == 1


@pytest.mark.parametrize(
    "total,per_run", [("NaN", "0.5"), ("1", "0"), ("1", "Infinity"), ("0.25", "0.5")]
)
def test_read_budget_rejects_unbounded_or_unfunded_configuration(total, per_run):
    with pytest.raises(ValueError):
        harness.ReadBudget(Decimal(total), Decimal(per_run))


@pytest.mark.parametrize("failure_limit", [0, 6, True])
def test_read_budget_rejects_unbounded_provider_failure_limits(failure_limit):
    with pytest.raises(ValueError, match="between one and five"):
        harness.ReadBudget(
            Decimal("5"), Decimal("0.50"), provider_failure_limit=failure_limit
        )


@pytest.mark.parametrize("reason", ["timeout", "provider_unavailable"])
def test_read_budget_stops_consecutive_provider_failures_and_resets_on_completion(
    reason,
):
    budget = harness.ReadBudget(Decimal("5"), Decimal("0.50"))
    failure = LoopExit(
        "run", "conversation", LoopExitKind.FAILED, reason, datetime.now(UTC)
    )
    success = replace(
        failure,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        final_text="The lookup timed out; no balance is available.",
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0.01"))),
    )
    budget.reserve()
    budget.settle(failure)
    assert budget.consecutive_provider_failures == 1 and budget.stop_reason is None
    budget.reserve()
    budget.settle(success)
    assert budget.consecutive_provider_failures == 0 and budget.stop_reason is None
    for _ in range(2):
        budget.reserve()
        budget.settle(failure)
    assert budget.stop_reason == "provider_failures" and budget.incomplete_runs == 3
    assert budget.charged == Decimal("1.51")
    with pytest.raises(RuntimeError, match="provider failures before dispatch"):
        budget.reserve()
    assert budget.charged == Decimal("1.51")


@pytest.mark.parametrize("stop", ["provider_failures", "budget_exhausted"])
def test_live_read_controller_stops_pytest_after_teardown_and_retains_failed_cases(
    tmp_path, stop
):
    # Run the actual narrow pytest controller. No provider or credential is used.
    (tmp_path / "conftest.py").write_text(
        (REPO_ROOT / "tests/live/mcp/conftest.py").read_text()
    )
    (tmp_path / "test_guard.py").write_text("""
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
import pytest
from daita.llm.models import ModelUsage
from daita.llm.pricing import CostEstimate
from daita.loop.models import LoopExit, LoopExitKind

@pytest.fixture(autouse=True)
def retain_teardown(read_budget):
    yield
    Path(os.environ["READ_GUARD_JOURNAL"]).write_text(json.dumps({
        "charged": str(read_budget.charged),
        "stop_reason": read_budget.stop_reason,
        "incomplete": read_budget.incomplete_runs,
    }))

@pytest.mark.parametrize("index", range(6))
def test_bounded_sample(read_budget, index):
    read_budget.reserve()
    marker = Path(os.environ["READ_GUARD_MARKER"])
    with marker.open("a") as stream:
        stream.write(str(index) + "\\n")
    success = index == 0
    result = LoopExit("run", "conversation", LoopExitKind.FAILED, "provider_unavailable", datetime.now(UTC))
    if success:
        result = replace(result, kind=LoopExitKind.COMPLETED, reason="completed", final_text="done", usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(os.environ["READ_GUARD_COST"])) ))
    read_budget.settle(result)
    assert success, "offline provider failure"
""")
    marker, journal, junit = (
        tmp_path / name for name in ("dispatches.txt", "teardown.json", "results.xml")
    )
    settings = os.environ.copy()
    settings.update(
        {
            harness.TOTAL_COST_ENV: "5" if stop == "provider_failures" else "0.70",
            harness.COST_ENV: "0.50",
            harness.FAILURE_LIMIT_ENV: "2",
            "READ_GUARD_COST": "0" if stop == "provider_failures" else "0.25",
            "READ_GUARD_MARKER": str(marker),
            "READ_GUARD_JOURNAL": str(journal),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path / "test_guard.py"),
            "-o",
            "addopts=--tb=short -q --strict-markers",
            f"--junitxml={junit}",
        ],
        cwd=REPO_ROOT,
        env=settings,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"stopped: {stop}" in result.stdout
    dispatched = marker.read_text().splitlines()
    assert dispatched == (["0", "1", "2"] if stop == "provider_failures" else ["0"])
    cases = list(ET.parse(junit).iter("testcase"))
    assert len(cases) == len(dispatched)
    assert sum(case.find("failure") is not None for case in cases) == (
        2 if stop == "provider_failures" else 0
    )
    assert json.loads(journal.read_text()) == {
        "charged": "1.00" if stop == "provider_failures" else "0.25",
        "stop_reason": stop,
        "incomplete": 2 if stop == "provider_failures" else 0,
    }


async def test_measured_transport_includes_streamed_bytes_and_closes_borrowed_owner():
    release = asyncio.Event()
    first = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"one"
            first.set()
            await release.wait()
            yield b"two"

        async def aclose(self):
            self.closed = True

    stream = Stream()

    class Transport(httpx.AsyncBaseTransport):
        closed = False

        async def handle_async_request(self, request):
            return httpx.Response(200, stream=stream, request=request)

        async def aclose(self):
            self.closed = True

    delegate = Transport()
    measured = harness.MeasuredMCPTransport(delegate)
    async with httpx.AsyncClient(transport=measured) as client:
        async with client.stream(
            "POST",
            "https://fixture.test/mcp",
            headers={"Authorization": "Bearer PRIVATE_CREDENTIAL"},
            json={"method": "tools/call", "params": {"secret": "PRIVATE_ARGUMENT"}},
        ) as response:
            reading = asyncio.create_task(response.aread())
            await asyncio.wait_for(first.wait(), 1)
            assert measured.events[0]["response_bytes"] == 3
            release.set()
            assert await asyncio.wait_for(reading, 1) == b"onetwo"
    assert stream.closed and not delegate.closed
    assert measured.events[0]["closed"] and measured.events[0]["response_bytes"] == 6
    assert cast(float, measured.events[0]["seconds"]) > 0
    evidence = json.dumps(measured.events)
    assert "PRIVATE_CREDENTIAL" not in evidence and "PRIVATE_ARGUMENT" not in evidence
    await measured.close_owned()
    assert delegate.closed


async def test_measured_transport_does_not_save_raw_failure_text():
    async def fail(request):
        raise httpx.ReadError("SECRET_RESPONSE_TOKEN", request=request)

    measured = harness.MeasuredMCPTransport(httpx.MockTransport(fail))
    async with httpx.AsyncClient(transport=measured) as client:
        with pytest.raises(httpx.ReadError):
            await client.post("https://fixture.test/mcp", json={"method": "tools/list"})
    assert measured.events[0]["failure_type"] == "ReadError"
    assert "SECRET_RESPONSE_TOKEN" not in json.dumps(measured.events)
    await measured.close_owned()


async def test_read_report_retains_completed_run_when_grounding_assertion_fails(
    tmp_path, monkeypatch
):
    server = harness.account_identity("Europe", "2026-07-28")
    model = ToolboxAwareMockModelProvider([], complete_pricing=True)
    provider = RecordingProvider(model)
    monkeypatch.setattr(
        harness, "read_provider", lambda: (model.model_profile, provider)
    )
    monkeypatch.setenv("MCP_TOKEN", "MCP_ECHO_PRIVATE")
    transport = harness.MeasuredMCPTransport(
        httpx.MockTransport(MCPConformanceTransport(server))
    )
    budget = harness.ReadBudget(Decimal("1"), Decimal("0.50"))
    path = tmp_path / "evidence.json"
    with pytest.raises(AssertionError, match="independent oracle"):
        async with harness.evaluate_reads(
            tmp_path / "home",
            transport=transport,
            budget=budget,
            report=path,
            case_id="failed grounding",
        ) as scenario:
            status = await scenario.agent.attach_mcp_server(
                endpoint=server.endpoint, selections=(MCPToolSelection("find_account"),)
            )
            model.replace_script(
                [
                    ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=(
                            ToolCall(
                                "lookup",
                                status.binding.tools[0].local_name,
                                {"query": "Cedar Works"},
                            ),
                        ),
                        usage=ModelUsage(
                            cost_estimate=CostEstimate.complete(Decimal("0"))
                        ),
                    ),
                    ModelResponse(
                        finish_reason=FinishReason.STOP,
                        text="MCP_ECHO_PRIVATE",
                        usage=ModelUsage(
                            input_tokens=19,
                            output_tokens=3,
                            cost_estimate=CostEstimate.complete(Decimal("0")),
                        ),
                    ),
                ]
            )
            capture = await scenario.run("Read the fixture", phase="failed_oracle")
            assert capture.result.kind is LoopExitKind.COMPLETED, capture.result.reason
            raise AssertionError("independent oracle")
    payload = path.read_text()
    evidence = json.loads(payload)
    assert (
        evidence["status"] == "failed" and evidence["failure_type"] == "AssertionError"
    )
    assert "MCP_ECHO_PRIVATE" not in payload
    run = evidence["runs"][0]
    assert run["model_requests"] == len(provider.requests) == 3
    assert len(run["model_request_messages"]) == run["model_requests"]
    assert run["mcp_methods"]["tools/call"] == 1
    assert run["mcp_methods"]["tools/list"] >= 1
    assert run["mcp_response_bytes"] > 0
    assert run["tool_errors"] == 0
    assert run["model_timing_complete"] is True
    assert run["messages"] and run["result_kind"] == "completed"
    assert run["total_tokens"] == 22 and run["cost_complete"] is True
    assert evidence["budget"]["charged_usd"] == "0.00"
    assert all(item["closed"] for item in transport.events)


async def test_read_harness_exhaustion_prevents_model_and_mcp_dispatch(
    tmp_path, monkeypatch
):
    model = ToolboxAwareMockModelProvider([], complete_pricing=True)
    provider = RecordingProvider(model)
    monkeypatch.setattr(
        harness, "read_provider", lambda: (model.model_profile, provider)
    )
    transport = harness.MeasuredMCPTransport(
        httpx.MockTransport(MCPConformanceTransport())
    )
    budget = harness.ReadBudget(Decimal("0.50"), Decimal("0.50"))
    budget.reserve()
    budget.settle(None)
    path = tmp_path / "evidence.json"
    with pytest.raises(RuntimeError, match="before dispatch"):
        async with harness.evaluate_reads(
            tmp_path / "home",
            transport=transport,
            budget=budget,
            report=path,
            case_id="exhausted",
        ) as scenario:
            await scenario.run("Do not spend", phase="blocked")
    assert provider.requests == [] and transport.events == []
    assert json.loads(path.read_text())["status"] == "failed"


@pytest.mark.parametrize("protocol", ["2026-07-28", "2025-11-25", "2025-06-18"])
async def test_read_delay_fixture_uses_sdk_deadline_and_drains_partial_body(protocol):
    server = harness.account_identity("Europe", protocol)
    service = harness.DelayedMCPReadTransport(server, body_seconds=0.5)
    measured = harness.MeasuredMCPTransport(httpx.MockTransport(service))
    client = SDKMCPClientFactory(http_transport=measured, timeout_seconds=0.1).create(
        endpoint=server.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=datetime.now(UTC))
        with pytest.raises(MCPTransportError) as failure:
            await client.call_tool("read_balance", {"account_id": "fixture-account"})
        assert failure.value.code == "mcp_timeout"
        assert server.calls == [("read_balance", {"account_id": "fixture-account"})]
    finally:
        await asyncio.wait_for(client.close(), 2)
        await measured.close_owned()
    assert len(service.streams) == 1 and service.streams[0].closed
    assert service.streams[0].cancelled
    calls = [event for event in measured.events if event["method"] == "tools/call"]
    assert len(calls) == 1 and calls[0]["response_bytes"] == 64
    assert calls[0]["failure_type"] == "CancelledError" and calls[0]["closed"]


async def test_read_delay_fixture_preserves_large_streamed_payload_and_tail_marker():
    server = harness.account_identity("Europe", "2026-07-28", appendix_bytes=64 * 1024)
    service = harness.DelayedMCPReadTransport(
        server, headers_seconds=0.01, body_seconds=0.01
    )
    measured = harness.MeasuredMCPTransport(httpx.MockTransport(service))
    client = SDKMCPClientFactory(http_transport=measured).create(
        endpoint=server.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=datetime.now(UTC))
        result = await client.call_tool(
            "read_balance", {"account_id": "fixture-account"}
        )
        assert len(result.text[0].encode()) >= 64 * 1024
        assert len(re.findall(r"APPENDIX_[0-9a-f]{32}", result.text[0])) == 1
        assert (
            result.text[0].splitlines()[-1].startswith("Appendix verification marker: ")
        )
    finally:
        await client.close()
        await measured.close_owned()
    call = next(event for event in measured.events if event["method"] == "tools/call")
    assert cast(int, call["response_bytes"]) >= 64 * 1024 and call["closed"]
    assert (
        0
        < cast(float, call["headers_seconds"])
        <= cast(float, call["first_response_byte_seconds"])
        < cast(float, call["seconds"])
    )
    assert all(stream.closed for stream in service.streams)


def _write_latency_report(
    path: Path,
    *,
    elapsed: float,
    workload: str = "normal",
    status: str = "passed",
    timeout: bool = False,
    service_kind: str = "fixture",
    phase: str = "dependent_reads",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "setup": {
                    "service_kind": service_kind,
                    "workload": workload,
                    "protocol": "2026-07-28",
                },
                "runs": [
                    {
                        "phase": phase,
                        "elapsed_seconds": elapsed,
                        "model_seconds": elapsed / 2,
                        "mcp_seconds": elapsed / 4,
                        "model_requests": 6,
                        "model_usage_complete": True,
                        "tool_errors": int(timeout),
                        "tool_error_codes": {"mcp_timeout": 1} if timeout else {},
                        "mcp_timeout_errors": int(timeout),
                        "provider_timeout": False,
                        "run_deadline_exhausted": False,
                        "result_kind": "completed",
                        "cost_complete": True,
                        "known_estimated_cost_usd": "0.03",
                        "max_mcp_response_bytes": 65536,
                    }
                ],
            }
        )
    )
    return path


@pytest.mark.parametrize("service_kind", ["fixture", "real_remote"])
def test_latency_summary_reports_nearest_rank_p95_and_keeps_forced_timeouts_separate(
    tmp_path,
    service_kind,
):
    phase = "cold" if service_kind == "real_remote" else "dependent_reads"
    paths = [
        _write_latency_report(
            tmp_path / f"normal-{index}.json",
            elapsed=float(index),
            service_kind=service_kind,
            phase=phase,
        )
        for index in range(1, 21)
    ]
    paths += [
        _write_latency_report(
            tmp_path / f"timeout-{index}.json",
            elapsed=15.0,
            workload="timeout",
            timeout=True,
        )
        for index in range(5)
    ]
    summary = harness.summarize_read_reports(paths)
    normal = summary["cohorts"][f"{service_kind}/normal/{phase}"]
    assert normal["all_attempt_latency"] == {
        "samples": 20,
        "p50_seconds": 10.5,
        "p95_seconds": 19.0,
        "max_seconds": 20.0,
        "p95_is_maximum": False,
    }
    assert normal["timeout_run_rate"] == 0 and normal["samples"] == 20
    if service_kind == "real_remote":
        assert summary["cohorts"]["real_remote/normal/all_reads"] == normal
    forced = summary["cohorts"]["fixture/timeout/dependent_reads"]
    assert forced["mcp_timeout_runs"] == 5 and forced["timeout_run_rate"] == 1
    assert forced["successful_no_tool_error_latency"]["p95_seconds"] is None
    assert summary["known_estimated_model_cost_usd"] == "0.75"


def test_latency_summary_includes_failed_attempts_and_rejects_duplicate_evidence(
    tmp_path,
):
    paths = [
        _write_latency_report(tmp_path / f"passed-{index}.json", elapsed=float(index))
        for index in range(1, 10)
    ]
    paths += [
        _write_latency_report(tmp_path / "failed.json", elapsed=100, status="failed")
    ]
    summary = harness.summarize_read_reports(paths)
    group = summary["cohorts"]["fixture/normal/dependent_reads"]
    assert group["all_attempt_latency"]["p95_seconds"] == 100
    assert group["successful_no_tool_error_latency"]["p95_seconds"] == 9
    assert summary["case_statuses"] == {"passed": 9, "failed": 1}
    with pytest.raises(ValueError, match="inflate"):
        harness.summarize_read_reports([paths[0], paths[0]])
