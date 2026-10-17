"""One comparable, offline campaign; scripted models do not measure reasoning."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import platform
import resource
import sqlite3
import statistics
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import pytest

from daita import Agent, AgentConfig, GraphState, HostCapacityLimits
from daita._json import canonical_json
from daita.adapters.sqlite_query import SQLiteQueryBackend
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopExitKind
from tests.support.job_benchmarks import admit_profile_job, p95_bootstrap_interval
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.slow

ROWS = tuple((index, "x" * 96) for index in range(500))
FIXTURE_DIGEST = sha256(json.dumps(ROWS).encode()).hexdigest()


class _Provider:
    provider_id = "mock:throughput-component"
    model_profile = ModelProfile(
        id=provider_id,
        context_window_tokens=128_000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_parallel_tools=True,
    )

    def __init__(self):
        self.calls: tuple[ToolCall, ...] = ()
        self.request_count = 0

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.request_count += 1
        await asyncio.sleep(0.002)  # Fixed transport fixture, common to campaigns.
        start = max(
            index
            for index, message in enumerate(request.messages)
            if message.role is MessageRole.USER
        )
        results = {
            block.call_id: block
            for message in request.messages[start:]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        if not self.calls or all(call.id in results for call in self.calls):
            assert all(not results[call.id].is_error for call in self.calls), results
            return ModelResponse(finish_reason=FinishReason.STOP, text="verified")
        missing = {call.name for call in self.calls} - {
            tool.name for tool in request.tools
        }
        calls = (
            (
                ToolCall(
                    id="load",
                    name="toolbox_load",
                    arguments={"tool_names": sorted(missing)},
                ),
            )
            if missing
            else self.calls
        )
        return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


def _p95(samples: list[float]) -> float:
    return sorted(samples)[math.ceil(0.95 * len(samples)) - 1]


async def _run(
    agent: Agent,
    *,
    conversation_id: str | None = None,
    message: str = "Execute the controlled component fixture.",
):
    result = await agent.run(message, conversation_id=conversation_id)
    assert result.kind is LoopExitKind.COMPLETED, result
    transcript = await agent.transcript(result.run_id)
    assert all(
        output is not None and not output.is_error
        for _, output in transcript.tool_pairs
    )
    return result


async def test_public_composition_throughput_campaign(tmp_path, monkeypatch):
    if os.environ.get("DAITA_RUN_THROUGHPUT_COMPONENT") != "1":
        pytest.skip(
            "choose the bounded component campaign with DAITA_RUN_THROUGHPUT_COMPONENT=1"
        )
    count = int(os.environ.get("DAITA_THROUGHPUT_SAMPLES", "30"))
    assert 30 <= count <= 100
    campaign = os.environ.get("DAITA_THROUGHPUT_CAMPAIGN", "baseline")
    assert campaign in {"baseline", "final", "additional"}
    report_dir = Path(
        os.environ.get("DAITA_THROUGHPUT_REPORT_DIR")
        or tempfile.mkdtemp(prefix="daita-throughput-")
    )
    assert report_dir.is_absolute() and not report_dir.is_relative_to(Path.cwd())
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    provider = _Provider()
    config = AgentConfig()
    workspace = workspace_for(tmp_path)
    (workspace.root / "one.txt").write_text("42")
    (workspace.root / "two.txt").write_text("42")
    agent = await Agent.create(
        "throughput",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace,
        config=config,
    )
    cases: dict[str, object] = {}
    campaign_deadline = asyncio.get_running_loop().time() + 600
    native_peak = native_sum_peaks = 0
    original_read = SQLiteQueryBackend.execute_read
    hold_graph = asyncio.Event()
    graph_started = asyncio.Event()
    blocked = False
    slow_source: str | None = None

    async def fixture_read(self, **kwargs):
        if blocked:
            graph_started.set()
            await hold_graph.wait()
        if kwargs["source_id"] == slow_source:
            await asyncio.sleep(0.02)
        return await original_read(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", fixture_read)

    async def measure(
        name: str, operation: Callable[[], Awaitable[None]], *, tasks: int = 1
    ) -> None:
        samples: list[float] = []
        requests_before = provider.request_count
        started = perf_counter()
        for _ in range(count):
            at = perf_counter()
            async with asyncio.timeout_at(campaign_deadline):
                await operation()
            samples.append(perf_counter() - at)
        elapsed = perf_counter() - started
        cases[name] = {
            "samples": count,
            "configuration": asdict(config),
            "latency_seconds": samples,
            "p50_seconds": statistics.median(samples),
            "p95_seconds": _p95(samples),
            "p95_bootstrap_95_percent": p95_bootstrap_interval(samples),
            "completed_tasks_per_minute": count * tasks * 60 / elapsed,
            "model_requests": provider.request_count - requests_before,
            "tokens": None,
            "cost_usd": None,
            "usage_status": "unknown",
            "correctness": "passed independent fixture oracles",
            "admission_cumulative": asdict(agent.admission_diagnostics()),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "peak_individual_native_rss_bytes": native_peak or None,
            "sum_native_generation_peak_rss_upper_bound": native_sum_peaks or None,
        }

    try:
        bindings = []
        for index in range(3):
            database = tmp_path / f"source-{index}.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE records (id INTEGER, label TEXT)")
                connection.executemany("INSERT INTO records VALUES (?, ?)", ROWS)
            source = await agent.attach_sqlite(database)
            table = (await agent.list_catalog_resources(source_id=source.id))[0]
            bindings.append((source.id, table.id))

        async def distinct():
            await asyncio.gather(*(_run(agent) for _ in range(5)))

        await measure("distinct_conversations", distinct, tasks=5)

        async def ordered():
            seed = await _run(agent)
            turns = await asyncio.gather(
                *(_run(agent, conversation_id=seed.conversation_id) for _ in range(3))
            )
            retained = await agent.conversation_runs(seed.conversation_id)
            assert [turn.result.run_id for turn in retained if turn.result] == [
                seed.run_id,
                *(turn.run_id for turn in turns),
            ]

        await measure("one_conversation", ordered, tasks=4)

        def query(index, call_index):
            source_id, resource_id = bindings[index]
            return ToolCall(
                id=f"read-{call_index}",
                name="data_query",
                arguments={
                    "source_id": source_id,
                    "resource_ids": [resource_id],
                    "sql": "SELECT COUNT(*) AS n, SUM(id) AS total FROM records",
                },
            )

        async def checked_reads():
            result = await _run(agent)
            transcript = await agent.transcript(result.run_id)
            for call, output in transcript.tool_pairs:
                if call.name == "data_query":
                    assert output is not None
                    assert json.loads(canonical_json(output.output))["data"][
                        "rows"
                    ] == [{"n": 500, "total": 124750}]

        provider.calls = tuple(query(index, index) for index in range(3))
        await measure("independent_sources", checked_reads, tasks=3)
        provider.calls = tuple(query(0, index) for index in range(3))
        await measure("one_source", checked_reads, tasks=3)
        provider.calls = tuple(query(index, index) for index in range(3))
        slow_source = bindings[0][0]
        await measure("slow_read_beside_fast_reads", checked_reads, tasks=3)
        slow_source = None

        source_id, resource_id = bindings[0]
        provider.calls = (
            ToolCall(
                id="export",
                name="data_query",
                arguments={
                    "source_id": source_id,
                    "resource_ids": [resource_id],
                    "sql": "SELECT id, label FROM records ORDER BY id",
                    "output": "csv",
                },
            ),
        )

        async def export():
            result = await _run(agent)
            assert len(result.artifacts) == 1
            ref = result.artifacts[0]
            content = (await agent.read_artifact(ref.artifact_id)).content
            assert len(content) > 16 * 1024
            assert (
                tuple(
                    (int(row[0]), row[1])
                    for row in list(csv.reader(io.StringIO(content.decode())))[1:]
                )
                == ROWS
            )
            assert ref.sha256 == "sha256:" + sha256(content).hexdigest()
            await agent.delete_artifact(ref.artifact_id)

        await measure("exact_dataset_csv", export)

        async def saturated_graph():
            nonlocal blocked
            hold_graph.clear()
            graph_started.clear()
            blocked = True
            provider.calls = ()
            origin = await _run(agent)
            for _ in range(4):
                await admit_profile_job(agent, resource_id, origin, sample_rows=5)
            await asyncio.wait_for(graph_started.wait(), timeout=5)
            await _run(agent)
            blocked = False
            hold_graph.set()
            async with asyncio.timeout(15):
                while any(
                    job.state
                    not in {
                        GraphState.SUCCEEDED,
                        GraphState.FAILED,
                        GraphState.NEEDS_ATTENTION,
                        GraphState.CANCELLED,
                    }
                    for job in await agent.list_jobs()
                ):
                    await asyncio.sleep(0.01)
            assert all(
                job.state is GraphState.SUCCEEDED for job in await agent.list_jobs()
            )

        await measure("queued_graphs_with_foreground", saturated_graph, tasks=5)

        if agent.analysis_runtime()["available"]:
            await agent.close()
            native = _NativeProvider()
            agent = await Agent.open(
                "throughput",
                root=tmp_path,
                model=native,
                model_profile=native.model_profile,
                workspace=workspace,
                config=config,
            )
            provider = native

            async def python_reads():
                nonlocal native_peak, native_sum_peaks
                result = await _run(
                    agent,
                    message="Use Python to read the local files one.txt and two.txt and sum their integers.",
                )
                transcript = await agent.transcript(result.run_id)
                output = next(
                    output
                    for call, output in transcript.tool_pairs
                    if call.name == "python_execute"
                )
                assert output is not None
                assert (
                    json.loads(canonical_json(output.output))["data"]["stdout"]
                    == "84\n"
                )
                evidence = await agent.analysis_evidence(result.run_id)
                generation = next(
                    record for record in evidence if record.kind == "generation"
                )
                usages = [
                    json.loads(canonical_json(record.facts))["usage"]
                    for record in evidence
                    if record.kind == "generation"
                ]
                native_peak = max(
                    native_peak,
                    *(int(usage["peak_rss_bytes"] or 0) for usage in usages),
                )
                native_sum_peaks = max(
                    native_sum_peaks,
                    sum(int(usage["peak_rss_bytes"] or 0) for usage in usages),
                )
                cleanup = json.loads(canonical_json(generation.facts))["cleanup"]
                assert cleanup["process_reaped"] is True
                assert cleanup["scratch_deleted"] is True

            await measure("python_sequential_reads", python_reads, tasks=2)
            if campaign != "baseline":
                native.mode = "batch"
                await measure("python_batched_reads", python_reads, tasks=2)
                native.export_arguments = {
                    "source_id": bindings[0][0],
                    "resource_ids": [bindings[0][1]],
                    "sql": "SELECT id, label FROM records ORDER BY id",
                    "output": "arrow",
                }

                async def python_dataset():
                    nonlocal native_peak, native_sum_peaks
                    result = await _run(
                        agent,
                        message="Use Python to compute statistics over all records in the connected table from a complete exact dataset.",
                    )
                    transcript = await agent.transcript(result.run_id)
                    cell = next(
                        output
                        for call, output in transcript.tool_pairs
                        if call.name == "python_execute"
                    )
                    assert cell is not None
                    data = json.loads(canonical_json(cell.output))["data"]
                    assert data["stdout"] == "500 124750\n", data
                    artifact_id = (
                        result.artifacts[0].artifact_id
                        if native.mode == "export_bind"
                        else data["captured_inputs"][0]["artifact"]["artifact_id"]
                    )
                    payload = await agent.read_artifact(artifact_id)
                    assert (
                        payload.ref.provenance.row_count == 500
                        and len(payload.content) > 16384
                    )
                    assert (
                        payload.ref.sha256
                        == "sha256:" + sha256(payload.content).hexdigest()
                    )
                    from importlib import import_module

                    pa = import_module("pyarrow")

                    with pa.ipc.open_file(pa.BufferReader(payload.content)) as reader:
                        table = reader.read_all()
                        assert (
                            tuple(
                                zip(
                                    table.column("id").to_pylist(),
                                    table.column("label").to_pylist(),
                                    strict=True,
                                )
                            )
                            == ROWS
                        )
                    evidence = await agent.analysis_evidence(result.run_id)
                    generations = [
                        json.loads(canonical_json(record.facts))
                        for record in evidence
                        if record.kind == "generation"
                    ]
                    assert all(
                        g["cleanup"]["process_reaped"]
                        and g["cleanup"]["scratch_deleted"]
                        for g in generations
                    )
                    native_peak = max(
                        native_peak,
                        *(int(g["usage"]["peak_rss_bytes"] or 0) for g in generations),
                    )
                    native_sum_peaks = max(
                        native_sum_peaks,
                        sum(
                            int(g["usage"]["peak_rss_bytes"] or 0) for g in generations
                        ),
                    )
                    await agent.delete_artifact(artifact_id)

                native.mode = "export_bind"
                await measure("python_arrow_export_bind", python_dataset)
                native.mode = "capture"
                await measure("python_arrow_capture", python_dataset)
                await agent.close()
                provider = _Provider()
                config = AgentConfig(
                    host_capacity=HostCapacityLimits(
                        execution_capacity=8,
                        provider_capacity=4,
                        graph_parallelism=4,
                        postgresql_read_capacity=4,
                    )
                )
                agent = await Agent.open(
                    "throughput",
                    root=tmp_path,
                    model=provider,
                    model_profile=provider.model_profile,
                    workspace=workspace,
                    config=config,
                )
                await measure(
                    "distinct_conversations_capacity8_provider4", distinct, tasks=5
                )

        else:
            cases["python_sequential_reads"] = {
                "status": "unperformed",
                "reason": "native runtime unavailable",
            }
    finally:
        blocked = False
        hold_graph.set()
        await agent.close()
    diagnostics = agent.admission_diagnostics()
    assert diagnostics.active_leases == diagnostics.active_permits == 0
    report = {
        "campaign": campaign,
        "evidence_kind": "offline components; scripted model",
        "fixture_sha256": FIXTURE_DIGEST,
        "configuration": asdict(AgentConfig()),
        "host": {
            "platform": sys.platform,
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "cases": cases,
        "admission": asdict(diagnostics),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "cleanup": {"active_leases": 0, "active_permits": 0, "agent_closed": True},
        "unperformed": [
            "real PostgreSQL cold/reused connections: explicit live gate required",
            "live model quality: explicit live gate required",
            *(
                ("proposed Python batch/Arrow/pool counterparts until implemented",)
                if campaign == "baseline"
                else ()
            ),
        ],
    }
    path = report_dir / f"{campaign}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Component report: {path}")


class _NativeProvider(_Provider):
    def __init__(self):
        super().__init__()
        self.mode = "sequential"
        self.export_arguments: dict[str, object] = {}

    async def generate(self, request):
        self.request_count += 1
        names = {tool.name for tool in request.tools}
        start = max(
            index
            for index, message in enumerate(request.messages)
            if message.role is MessageRole.USER
        )
        results = {
            block.call_id: block
            for message in request.messages[start:]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        if "python" in results:
            assert not results["python"].is_error, results["python"]
            return ModelResponse(finish_reason=FinishReason.STOP, text="verified")
        required = {
            "python_execute",
            *({"data_query"} if self.mode in {"capture", "export_bind"} else set()),
        }
        if required - names:
            calls = (
                ToolCall(
                    id="load",
                    name="toolbox_load",
                    arguments={"tool_names": sorted(required)},
                ),
            )
        elif self.mode == "export_bind" and "export" not in results:
            calls = (
                ToolCall(
                    id="export",
                    name="data_query",
                    arguments=self.export_arguments,
                ),
            )
        else:
            arguments: dict[str, object] = {
                "expected_state": {"generation": 0, "revision": 0}
            }
            if self.mode in {"capture", "export_bind"}:
                if self.mode == "capture":
                    prefix = f"r = tools.capture('data_query', {self.export_arguments!r}, name='rows')\nassert not r['is_error'], r\n"
                else:
                    exported = results["export"]
                    assert not exported.is_error
                    reference = json.loads(canonical_json(exported.output))["artifact"]
                    arguments["inputs"] = {
                        "rows": {
                            "kind": "artifact",
                            "artifact_id": reference["artifact_id"],
                        }
                    }
                    prefix = ""
                code = (
                    prefix
                    + "import pyarrow.ipc as ipc\nwith ipc.open_file(inputs.path('rows')) as reader:\n    table = reader.read_all()\nassert table.column_names == ['id','label']\nprint(table.num_rows, sum(table.column('id').to_pylist()))"
                )
            else:
                prefix = (
                    "rs = tools.batch([{'name':'file_read', 'arguments':{'path':p}} for p in ['one.txt','two.txt']])"
                    if self.mode == "batch"
                    else "rs = [tools.call('file_read', {'path':p}) for p in ['one.txt','two.txt']]"
                )
                code = (
                    prefix
                    + "\nassert all(not r['is_error'] for r in rs), rs\nprint(sum(int(r['output']['data']['content']) for r in rs))"
                )
            arguments["code"] = code
            calls = (ToolCall(id="python", name="python_execute", arguments=arguments),)
        return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)
