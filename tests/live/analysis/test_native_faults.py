"""Real production workers and public hosts; only model turns are scripted here."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx2 as httpx
import pytest

from daita import Agent, AgentConfig, MCPToolSelection
from daita._json import canonical_json
from daita.adapters.analytical_workspace import NativePythonWorker
from daita.adapters.mcp import SDKMCPClientFactory
from daita.config import AnalysisLimits
from daita.llm.models import FinishReason, ModelResponse
from daita.loop.models import LoopLimits
from daita.observation import AgentEvent, AgentEventKind
from tests.artifacts._public_surface_support import _profile, _tool
from tests.support.mcp import MCPConformanceTransport, conformance_identities
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for


async def capture_run(agent, run_id, report):
    records = await agent.analysis_evidence(run_id)
    report["run_id"] = run_id
    report["measurements"] = [
        {"id": item.evidence_id, "kind": item.kind, "facts": item.facts}
        for item in records
    ]
    generations = [
        {**item.facts, **item.facts.get("recovery", {})}
        for item in records
        if item.kind == "generation"
    ]
    report["cleanup"] = [item["cleanup"] for item in generations]
    assert generations
    for item in generations:
        assert item["cleanup"]["process_reaped"] is True
        assert item["cleanup"]["scratch_deleted"] is True
        assert (
            item["cleanup"]["remaining_bytes"]
            == item["cleanup"]["remaining_files"]
            == 0
        )
        assert not Path(item["scratch"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(item["pid"], 0)
    return records


@pytest.mark.parametrize("trace_bound", ["bytes", "records"])
async def test_trace_exhaustion_stops_before_another_model_request(
    tmp_path, analysis_report, trace_bound
):
    code = "print(42)\n#" + "x" * 63_000
    model = ToolboxAwareMockModelProvider(
        tuple(
            _tool(
                f"trace-{index}",
                "analysis_execute",
                {
                    "code": code,
                    "expected_state": {
                        "generation": 0 if index == 0 else 1,
                        "revision": index,
                    },
                },
            )
            for index in range(80)
        )
    )
    agent = await Agent.create(
        "trace",
        root=tmp_path,
        model=model,
        model_profile=replace(_profile(model), context_window_tokens=2_000_000),
        config=AgentConfig(
            analysis_limits=(
                AnalysisLimits(trace_bytes=1024 * 1024)
                if trace_bound == "bytes"
                else AnalysisLimits(trace_records=6)
            ),
            limits=LoopLimits(
                max_steps=100, max_tool_calls_per_run=128, max_total_tokens=2_000_000
            ),
        ),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Retain bounded computation evidence.")
        records = await capture_run(agent, result.run_id, analysis_report)
        transcript = await agent.transcript(result.run_id)
        cells = [
            (call, block)
            for call, block in transcript.tool_pairs
            if call.name == "analysis_execute"
        ]
        assert result.reason == "analysis_trace_budget_exhausted"
        assert result.kind.value == "failed" and result.final_text is None
        assert 1 < len(cells) < 80
        assert len(model.logical_requests) == len(cells)
        assert cells[-1][1] is not None and cells[-1][1].is_error
        assert all(
            record.facts["status"] not in {"admitted", "running"} for record in records
        )
        summary = await agent.analysis_usage(result.run_id)
        assert summary is not None and summary["reservations_released"] is True
        analysis_report["accepted_cells_before_trace_limit"] = len(cells) - 1
    finally:
        await agent.close()


async def test_artifact_quota_failure_preserves_execution_and_existing_candidate(
    tmp_path, analysis_report
):
    script = []
    for index in range(10):
        script.append(
            _tool(
                f"save-{index}",
                "analysis_execute",
                {
                    "code": (
                        "n=0\nopen('value.txt','w').write('42')\noutputs.add('value','value.txt')\n"
                        if index == 0
                        else ""
                    )
                    + "n+=1; print(n)",
                    "expected_state": {
                        "generation": 0 if index == 0 else 1,
                        "revision": index,
                    },
                    **({"save_output": "value"} if index < 9 else {}),
                },
            )
        )
    agent = await make_agent(tmp_path, tuple(script), context_window_tokens=200_000)
    try:
        result = await agent.run(
            "Save this result and exercise the finite artifact quota."
        )
        await capture_run(agent, result.run_id, analysis_report)
        cells = [
            block
            for call, block in (await agent.transcript(result.run_id)).tool_pairs
            if call.name == "analysis_execute"
        ]
        assert len(result.artifacts) == 8
        assert cells[8].output["data"]["save_status"] == "failed"
        assert cells[8].output["data"]["save_error"] == "artifact_quota_exceeded"
        assert dict(cells[8].output["data"]["expected_state"]) == {
            "generation": 1,
            "revision": 9,
        }
        assert cells[9].output["data"]["stdout"] == "10\n"
        for item in result.artifacts:
            assert (await agent.read_artifact(item.artifact_id)).content == b"42"
        assert (await agent.analysis_usage(result.run_id))[
            "artifact_bytes_committed"
        ] == 16
    finally:
        await agent.close()


async def test_stale_raw_rpc_token_is_denied_before_read_dispatch(
    tmp_path, analysis_report
):
    workspace = workspace_for(tmp_path)
    (workspace.root / "secret.txt").write_text("must never be read")
    code = "import json, sys\nsys.__stdout__.write(json.dumps({'kind':'child','token':'forged','sequence':1,'name':'file_read','arguments':{'path':'secret.txt'}})+'\\n'); sys.__stdout__.flush()\nwhile True: pass"
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "forged",
                "analysis_execute",
                {"code": code, "expected_state": {"generation": 0, "revision": 0}},
            ),
        ),
    )
    try:
        result = await agent.run("Reject a forged broker frame.")
        records = await capture_run(agent, result.run_id, analysis_report)
        children = [record for record in records if record.kind == "child"]
        assert len(children) == 1
        assert children[0].facts["tool_name"] == "<invalid-analysis-frame>"
        assert children[0].facts["dispatched"] is False
        assert children[0].facts["status"] == "failed"
        assert "must never be read" not in canonical_json(analysis_report)
    finally:
        await agent.close()


async def test_cell_deadline_includes_waiting_for_worker_admission(
    tmp_path, analysis_report
):
    events: list[AgentEvent] = []
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "occupy",
                "analysis_execute",
                {
                    "code": "while True: pass",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            _tool(
                "queue-timeout",
                "analysis_execute",
                {
                    "code": "print(42)",
                    "timeout_seconds": 0.15,
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
        ),
        observer=events.append,
    )
    task = asyncio.create_task(agent.run("Occupy one real worker."))
    try:
        busy_id, _ = await wait_for_record(agent, events, "generation", "running")
        started = asyncio.get_running_loop().time()
        queued = await asyncio.wait_for(agent.run("Bound this queued cell."), 5)
        elapsed = asyncio.get_running_loop().time() - started
        records = await agent.analysis_evidence(queued.run_id)
        assert elapsed < 5 and not task.done()
        assert not any(record.kind == "generation" for record in records)
        cell = next(record for record in records if record.kind == "cell")
        assert cell.facts["status"] == "interrupted"
        assert cell.facts["completion_observed"] is False
        summary = next(record for record in records if record.kind == "run")
        assert summary.facts["worker_queue_wait_seconds"] >= 0.1
        assert summary.facts["reservations_released"] is True
        analysis_report["queued_measurements"] = [
            dict(record.facts) for record in records
        ]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await capture_run(agent, busy_id, analysis_report)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def make_agent(tmp_path, script, **kwargs):
    model = kwargs.pop("model_override", None)
    if model is None:
        model = ToolboxAwareMockModelProvider(
            (*script, ModelResponse(finish_reason=FinishReason.STOP, text="done"))
        )
    holder = kwargs.pop("model_holder", None)
    if holder is not None:
        holder.append(model)
    context_window_tokens = kwargs.pop("context_window_tokens", 32_000)
    return await Agent.create(
        "faults",
        root=tmp_path,
        model=model,
        model_profile=replace(
            _profile(model), context_window_tokens=context_window_tokens
        ),
        workspace=workspace_for(tmp_path),
        **kwargs,
    )


async def test_revoked_candidate_cannot_be_saved_after_worker_replacement(
    tmp_path, analysis_report
):
    events: list[AgentEvent] = []
    models: list[ToolboxAwareMockModelProvider] = []
    identity, _ = conformance_identities()
    agent = await make_agent(
        tmp_path,
        (),
        model_holder=models,
        observer=events.append,
        mcp_client_factory=SDKMCPClientFactory(
            http_transport=httpx.MockTransport(MCPConformanceTransport(identity))
        ),
    )
    task = None
    try:
        attached = await agent.attach_mcp_server(
            endpoint=identity.endpoint,
            selections=(
                MCPToolSelection(
                    remote_name="lookup",
                    local_alias="lookup",
                    description="Read isolated data.",
                ),
            ),
        )
        name = attached.binding.tools[0].local_name
        models[0].replace_script(
            (
                _tool(
                    "load", "toolbox_load", {"tool_names": ["analysis_execute", name]}
                ),
                _tool(
                    "candidate",
                    "analysis_execute",
                    {
                        "code": f"import json\nr=tools.call({name!r}, {{'query':'fixture'}})\nassert not r['is_error']\nopen('derived.json','w').write(json.dumps(r['output']))\noutputs.add('derived','derived.json')",
                        "expected_state": {"generation": 0, "revision": 0},
                    },
                ),
                _tool(
                    "wait-revocation",
                    "analysis_execute",
                    {
                        "code": "while True: pass",
                        "expected_state": {"generation": 1, "revision": 1},
                    },
                ),
                _tool(
                    "try-save-old",
                    "analysis_execute",
                    {
                        "code": "print('fresh')",
                        "save_output": "derived",
                        "expected_state": {"generation": 1, "revision": 0},
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="revoked"),
            )
        )
        task = asyncio.create_task(
            agent.run("Keep a candidate private and enforce revocation before saving.")
        )
        run_id, _ = await wait_for_record(agent, events, "generation", "running")
        async with asyncio.timeout(10):
            while True:
                records = await agent.analysis_evidence(run_id)
                if any(
                    record.evidence_id == "wait-revocation"
                    and record.facts["status"] == "admitted"
                    for record in records
                ):
                    break
                await asyncio.sleep(0.02)
        await agent.revoke_mcp_server(attached.binding.binding_id)
        result = await asyncio.wait_for(task, 10)
        records = await capture_run(agent, run_id, analysis_report)
        assert not result.artifacts and await agent.list_artifacts() == ()
        assert len(identity.calls) == 1
        last = next(
            record for record in records if record.evidence_id == "try-save-old"
        )
        assert last.facts["status"] == "interrupted"
        assert (
            len(
                [
                    record
                    for record in records
                    if record.kind == "generation" and record.facts.get("role") is None
                ]
            )
            == 2
        )
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_python_threads_serialize_broker_channel_and_keep_response_identity(
    tmp_path, analysis_report
):
    workspace = workspace_for(tmp_path)
    (workspace.root / "one.txt").write_text("one")
    (workspace.root / "two.txt").write_text("two")
    code = "from concurrent.futures import ThreadPoolExecutor\nwith ThreadPoolExecutor(max_workers=2) as pool:\n for i in range(4):\n  results=list(pool.map(lambda p: tools.call('file_read', {'path':p}), ['one.txt','two.txt']))\n  assert all(not r['is_error'] for r in results)\n  print([r['output']['data']['content'] for r in results])"
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "thread-reads",
                "analysis_execute",
                {"code": code, "expected_state": {"generation": 0, "revision": 0}},
            ),
        ),
    )
    try:
        result = await agent.run(
            "Keep synchronous broker calls serialized across Python threads."
        )
        records = await capture_run(agent, result.run_id, analysis_report)
        cell = next(record for record in records if record.kind == "cell")
        assert cell.facts["result"]["status"] == "success"
        assert cell.facts["result"]["stdout"] == "['one', 'two']\n" * 4
        children = [record for record in records if record.kind == "child"]
        assert len(children) == 8
        for child in children:
            assert child.facts["status"] == "succeeded"
            assert child.facts["result"]["data"]["content"] == child.facts["arguments"][
                "path"
            ].removesuffix(".txt")
    finally:
        await agent.close()


async def test_unreadable_scratch_is_unknown_then_closed_and_deleted(analysis_report):
    worker = NativePythonWorker()
    try:
        await worker.start()
        with pytest.raises(RuntimeError, match="Scratch measurement is unavailable"):
            await worker.execute(
                "import os\nos.mkdir('hidden')\nopen('hidden/payload','w').write('x'*1024)\nos.chmod('hidden',0)\nwhile True: pass",
                deadline=asyncio.get_running_loop().time() + 5,
            )
        assert worker.usage["scratch_samples_incomplete"] is True
        assert worker.usage["current_scratch_bytes"] is None
        assert worker.usage["current_scratch_files"] is None
    finally:
        analysis_report["cleanup"] = await worker.close()
        analysis_report["measurements"] = dict(worker.usage)
    assert analysis_report["cleanup"]["process_reaped"] is True
    assert analysis_report["cleanup"]["scratch_deleted"] is True
    assert (
        analysis_report["cleanup"]["remaining_bytes"]
        == analysis_report["cleanup"]["remaining_files"]
        == 0
    )


async def test_directory_entries_consume_the_scratch_file_allowance(analysis_report):
    worker = NativePythonWorker(limits=AnalysisLimits(scratch_files=2))
    try:
        await worker.start()
        with pytest.raises(RuntimeError, match="allowance exhausted"):
            await worker.execute(
                "import os\nfor i in range(10): os.mkdir('directory-'+str(i))",
                deadline=asyncio.get_running_loop().time() + 5,
            )
        assert worker.usage["observed_peak_scratch_files"] > worker.limits.scratch_files
        assert worker.usage["scratch_samples_incomplete"] is True
        assert worker.usage["peak_scratch_files"] is None
    finally:
        analysis_report["cleanup"] = await worker.close()
        analysis_report["measurements"] = dict(worker.usage)
    assert analysis_report["cleanup"]["process_reaped"] is True
    assert analysis_report["cleanup"]["scratch_deleted"] is True


async def test_broken_worker_input_counts_actual_os_transfers(
    tmp_path, monkeypatch, analysis_report
):
    from daita.adapters.analytical_workspace import native

    workspace = workspace_for(tmp_path)
    (workspace.root / "value.txt").write_text("source fixture")
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "close-input",
                "analysis_execute",
                {
                    "code": "import sys\nsys.stdin.close()\ntry: tools.call('file_read', {'path':'value.txt'})\nexcept ValueError:\n while True: pass",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
        ),
    )
    observed = 0
    actual_write = os.write

    def observe_write(fd: int, data: bytes) -> int:
        nonlocal observed
        written = actual_write(fd, data)
        domain = agent._embedded._capability_runtime._domains["analysis"]
        for state in domain._runs.values():
            worker = state.worker
            if (
                worker is not None
                and worker.process is not None
                and worker.process.stdin is not None
                and not worker.process.stdin.closed
                and worker.process.stdin.fileno() == fd
            ):
                observed += written
        return written

    # Observe successful syscall return values; no execution or I/O is replaced.
    monkeypatch.setattr(native.os, "write", observe_write)
    try:
        result = await agent.run(
            "Retain exact transfer evidence on a broken input channel."
        )
        records = await capture_run(agent, result.run_id, analysis_report)
        generation = next(record for record in records if record.kind == "generation")
        usage = generation.facts["usage"]
        assert usage["protocol_input_bytes"] == observed > 0
        assert usage["protocol_output_bytes"] > 0
        assert (
            next(record for record in records if record.kind == "cell").facts["status"]
            == "interrupted"
        )
        assert (
            next(record for record in records if record.kind == "child").facts[
                "dispatched"
            ]
            is True
        )
        analysis_report["os_write_bytes_observed"] = observed
    finally:
        await agent.close()


async def wait_for_record(agent, events, kind, status):
    async with asyncio.timeout(15):
        while True:
            started = next(
                (event for event in events if event.kind is AgentEventKind.RUN_STARTED),
                None,
            )
            if started is not None:
                records = await agent.analysis_evidence(started.run_id)
                match = next(
                    (
                        item
                        for item in records
                        if item.kind == kind and item.facts.get("status") == status
                    ),
                    None,
                )
                if match is not None:
                    return started.run_id, match
            await asyncio.sleep(0.02)


@pytest.mark.parametrize("fault", ["cpu", "memory", "scratch_files"])
async def test_real_resource_limits_preserve_partial_usage_and_cleanup(
    fault, analysis_report
):
    limits = AnalysisLimits()
    code = "while True: pass"
    deadline = 5.0
    if fault == "cpu":
        limits = replace(limits, cpu_seconds=0.25)
    elif fault == "memory":
        limits = replace(limits, memory_bytes=64 * 1024 * 1024)
        code = "x = bytearray(128 * 1024 * 1024)\nwhile True: pass"
    else:
        limits = replace(limits, scratch_files=3)
        code = (
            "[open(str(i), 'w').write('fixture') for i in range(8)]\nwhile True: pass"
        )
    worker = NativePythonWorker(limits=limits)
    try:
        await worker.start()
        with pytest.raises((RuntimeError, TimeoutError)):
            await worker.execute(
                code, deadline=asyncio.get_running_loop().time() + deadline
            )
    finally:
        cleanup = await worker.close()
        analysis_report.update(measurements=dict(worker.usage), cleanup=cleanup)
    assert cleanup["process_reaped"] is True
    assert cleanup["scratch_deleted"] is True
    assert worker.usage["cpu_complete"] is True
    assert worker.usage["user_cpu_seconds"] > 0
    assert worker.usage["wall_seconds"] > 0
    if fault == "cpu":
        assert (
            worker.usage["user_cpu_seconds"] + worker.usage["system_cpu_seconds"]
            >= limits.cpu_seconds
        )
    if fault == "memory":
        assert worker.usage["memory_overshoot_bytes"] > 0


@pytest.mark.parametrize("where", ["computation", "broker"])
async def test_public_cancellation_settles_real_worker_and_child_io(
    tmp_path, where, analysis_report
):
    events: list[AgentEvent] = []
    models: list[ToolboxAwareMockModelProvider] = []
    code = "while True: pass"
    script = []
    kwargs = {"observer": events.append, "model_holder": models}
    identity = None
    if where == "broker":
        identity, _ = conformance_identities()
        identity.block_calls = asyncio.Event()
        kwargs["mcp_client_factory"] = SDKMCPClientFactory(
            http_transport=httpx.MockTransport(MCPConformanceTransport(identity))
        )
        script.append(
            _tool(
                "load", "toolbox_load", {"tool_names": ["analysis_execute", "lookup"]}
            )
        )
        code = "r = tools.call('lookup', {'query':'isolated fixture'}); print(r)"
    script.append(
        _tool(
            "cancel-cell",
            "analysis_execute",
            {"code": code, "expected_state": {"generation": 0, "revision": 0}},
        )
    )
    agent = await make_agent(tmp_path, script, **kwargs)
    task = None
    try:
        if identity is not None:
            attached = await agent.attach_mcp_server(
                endpoint=identity.endpoint,
                selections=(
                    MCPToolSelection(
                        remote_name="lookup",
                        local_alias="lookup",
                        description="Read the isolated fixture.",
                    ),
                ),
            )
            local_name = attached.binding.tools[0].local_name
            models[0].replace_script(
                (
                    _tool(
                        "load",
                        "toolbox_load",
                        {"tool_names": ["analysis_execute", local_name]},
                    ),
                    _tool(
                        "cancel-cell",
                        "analysis_execute",
                        {
                            "code": f"r = tools.call({local_name!r}, {{'query':'isolated fixture'}}); print(r)",
                            "expected_state": {"generation": 0, "revision": 0},
                        },
                    ),
                    ModelResponse(finish_reason=FinishReason.STOP, text="done"),
                )
            )
        task = asyncio.create_task(agent.run("Exercise real cancellation."))
        run_id, _ = await wait_for_record(
            agent, events, "child" if where == "broker" else "generation", "running"
        )
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        records = await capture_run(agent, run_id, analysis_report)
        transcript = await agent.transcript(run_id)
        assert transcript.run.conversation_id is not None
        terminal = (await agent.conversation_runs(transcript.run.conversation_id))[
            0
        ].result
        assert terminal is not None
        analysis_report["terminal"] = terminal.kind.value
        assert terminal.kind.value == "interrupted"
        assert terminal.reason == "cancelled"
        if where == "broker":
            assert identity is not None
            child = next(item for item in records if item.kind == "child")
            assert child.facts["status"] == "interrupted"
            assert child.facts["dispatched"] is True
            assert not identity.calls
            assert (
                next(item for item in records if item.kind == "generation").facts[
                    "usage"
                ]["broker_wait_seconds"]
                >= 0.15
            )
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_public_shared_child_budget_and_forbidden_targets(
    tmp_path, analysis_report
):
    code = "for name in ['analysis_execute','toolbox_load','artifact_save_local','data_update_rows','unloaded_fixture']:\n r=tools.call(name, {})\n assert r['is_error']"
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "denials",
                "analysis_execute",
                {"code": code, "expected_state": {"generation": 0, "revision": 0}},
            ),
        ),
        config=AgentConfig(
            limits=LoopLimits(max_tool_calls_per_run=5, max_tool_calls_per_response=5)
        ),
    )
    try:
        result = await agent.run(
            "Exercise denied attempts against the shared allowance."
        )
        records = await capture_run(agent, result.run_id, analysis_report)
        children = [item for item in records if item.kind == "child"]
        assert len(children) == 5
        assert all(item.facts["dispatched"] is False for item in children)
        summary = await agent.analysis_usage(result.run_id)
        assert summary is not None
        assert summary["child_attempted"] == summary["child_denied"] == 5
        assert summary["outer_tool_calls"]["attempted"] == 2
        assert result.kind.value == "failed"
        assert result.reason == "tool_calls_per_run_exceeded"
    finally:
        await agent.close()


async def test_real_host_crash_public_reopen_authenticates_closure_without_replay(
    tmp_path, analysis_report
):
    code = """
import asyncio, json, sys
from daita import Agent
from tests.artifacts._public_surface_support import _profile, _tool
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for
from pathlib import Path
async def main():
 root=Path(sys.argv[1])
 model=ToolboxAwareMockModelProvider((_tool('abandoned','analysis_execute',{'code':'while True: pass','expected_state':{'generation':0,'revision':0}}),))
 agent=await Agent.create('crashed', root=root, model=model, model_profile=_profile(model), workspace=workspace_for(root))
 task=asyncio.create_task(agent.run('Crash this disposable host.'))
 while True:
  states=agent._embedded._capability_runtime._domains['analysis']._runs
  if states:
   run_id=next(iter(states))
   records=await agent.analysis_evidence(run_id)
   record=next((r for r in records if r.kind=='generation' and r.facts.get('status')=='running'),None)
   if record:
    print(json.dumps({'run_id':run_id,'pid':record.facts['pid'],'scratch':record.facts['scratch']}), flush=True)
    break
  await asyncio.sleep(.02)
 await task
asyncio.run(main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdout is not None and process.stderr is not None
        async with asyncio.timeout(20):
            raw = await asyncio.to_thread(process.stdout.readline)
        assert raw, await asyncio.to_thread(process.stderr.read)
        facts = json.loads(raw)
        process.kill()
        await asyncio.to_thread(process.wait, 5)
        agent = await Agent.open(
            "crashed", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        try:
            records = await capture_run(agent, facts["run_id"], analysis_report)
            assert (
                next(item for item in records if item.kind == "generation").facts[
                    "recovery"
                ]["status"]
                == "recovered_closed"
            )
            assert (
                next(item for item in records if item.kind == "cell").facts["status"]
                == "interrupted"
            )
            transcript = await agent.transcript(facts["run_id"])
            assert transcript.run.conversation_id is not None
            terminal = (await agent.conversation_runs(transcript.run.conversation_id))[
                0
            ].result
            assert terminal is not None and terminal.kind.value == "interrupted"
            assert not Path(facts["scratch"]).exists()
        finally:
            await agent.close()
    finally:
        if process.poll() is None:
            process.kill()
        await asyncio.to_thread(process.wait, 5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


@pytest.mark.parametrize(
    "extension,source",
    [
        (".csv", "open('output.csv','w').write('n,v\\n1,2\\n')"),
        (".json", "open('output.json','w').write('{\"value\":42}')"),
        (
            ".parquet",
            "import pyarrow as pa, pyarrow.parquet as pq\npq.write_table(pa.table({'n':[1,None,1]}),'output.parquet')",
        ),
        (
            ".png",
            "import matplotlib.pyplot as p\np.plot([1,2]); p.savefig('output.png')",
        ),
        (".md", "open('output.md','w').write('# Fixture\\n')"),
        (".txt", "open('output.txt','w').write('42')"),
        (".py", "open('output.py','w').write('print(42)')"),
        (".sql", "open('output.sql','w').write('SELECT 42;')"),
    ],
)
async def test_every_output_format_is_isolated_validated_saved_and_deleted(
    tmp_path, extension, source, analysis_report
):
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "produce",
                "analysis_execute",
                {
                    "code": source + f"\noutputs.add('fixture','output{extension}')",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            _tool(
                "save",
                "analysis_execute",
                {
                    "code": "",
                    "expected_state": {"generation": 1, "revision": 1},
                    "save_output": "fixture",
                },
            ),
        ),
    )
    try:
        result = await agent.run("Explicitly retain the fixture output.")
        await capture_run(agent, result.run_id, analysis_report)
        assert len(result.artifacts) == 1
        ref = result.artifacts[0]
        payload = await agent.read_artifact(ref.artifact_id)
        assert payload.content
        assert ref.filename == "output" + extension
        await agent.clear_conversations()
        assert (await agent.read_artifact(ref.artifact_id)).content == payload.content
        assert await agent.delete_artifact(ref.artifact_id)
        assert await agent.list_artifacts() == ()
        assert not await agent.delete_artifact(ref.artifact_id)
    finally:
        await agent.close()


async def test_broker_data_fidelity_null_decimal_large_integer_timezone_and_duplicates(
    tmp_path, analysis_report
):
    fixture = [
        {"id": 9007199254740993, "amount": "0.10", "at": "2026-10-08T10:00:00-05:00"},
        {"id": 9007199254740993, "amount": "0.20", "at": None},
        {"id": 2, "amount": None, "at": "2026-10-08T15:00:00+00:00"},
    ]
    workspace = workspace_for(tmp_path)
    (workspace.root / "exact.json").write_text(json.dumps(fixture))
    code = """
import json, pyarrow as pa, pyarrow.parquet as pq
from decimal import Decimal
from datetime import datetime
r=tools.call('file_read', {'path':'exact.json'})
assert not r['is_error']
assert r['output']['data']['complete']
rows=json.loads(r['output']['data']['content'])
table=pa.table({
 'id':pa.array([row['id'] for row in rows],type=pa.int64()),
 'amount':pa.array([None if row['amount'] is None else Decimal(row['amount']) for row in rows],type=pa.decimal128(38,2)),
 'at':pa.array([None if row['at'] is None else datetime.fromisoformat(row['at']) for row in rows],type=pa.timestamp('us',tz='UTC'))})
pq.write_table(table,'exact.parquet')
outputs.add('exact','exact.parquet')
print(sum(Decimal(row['amount']) for row in rows if row['amount'] is not None))
"""
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "exact",
                "analysis_execute",
                {
                    "code": code,
                    "expected_state": {"generation": 0, "revision": 0},
                    "save_output": "exact",
                },
            ),
        ),
    )
    try:
        result = await agent.run("Retain the exact analytical dataset.")
        records = await capture_run(agent, result.run_id, analysis_report)
        assert len(result.artifacts) == 1
        import io
        from datetime import datetime, timezone
        from decimal import Decimal
        from importlib import import_module

        pq = import_module("pyarrow.parquet")
        table = pq.read_table(
            io.BytesIO(
                (await agent.read_artifact(result.artifacts[0].artifact_id)).content
            )
        )
        rows = table.to_pylist()
        assert [row["id"] for row in rows] == [9007199254740993, 9007199254740993, 2]
        assert [row["amount"] for row in rows] == [
            Decimal("0.10"),
            Decimal("0.20"),
            None,
        ]
        assert (
            rows[0]["at"]
            == rows[2]["at"]
            == datetime(2026, 10, 8, 15, tzinfo=timezone.utc)
        )
        assert rows[1]["at"] is None
        child = next(record for record in records if record.kind == "child")
        assert child.facts["result_sensitivity"] == "internal"
        record = await agent._embedded._store.get_artifact_record(
            result.artifacts[0].artifact_id
        )
        assert (
            record.computation_evidence["children"][0]["evidence_id"]
            == child.evidence_id
        )
        analysis_report["data_oracle"] = {
            "rows": 3,
            "duplicate_ids": 2,
            "decimal_sum": "0.30",
            "timezone": "UTC",
            "nulls": 2,
        }
    finally:
        await agent.close()


async def test_active_binding_revocation_closes_contaminated_interpreter(
    tmp_path, analysis_report
):
    events: list[AgentEvent] = []
    models: list[ToolboxAwareMockModelProvider] = []
    identity, _ = conformance_identities()
    agent = await make_agent(
        tmp_path,
        (),
        model_holder=models,
        observer=events.append,
        mcp_client_factory=SDKMCPClientFactory(
            http_transport=httpx.MockTransport(MCPConformanceTransport(identity))
        ),
    )
    task = None
    try:
        attached = await agent.attach_mcp_server(
            endpoint=identity.endpoint,
            selections=(
                MCPToolSelection(
                    remote_name="lookup",
                    local_alias="lookup",
                    description="Read isolated data.",
                ),
            ),
        )
        name = attached.binding.tools[0].local_name
        models[0].replace_script(
            (
                _tool(
                    "load", "toolbox_load", {"tool_names": ["analysis_execute", name]}
                ),
                _tool(
                    "tainted",
                    "analysis_execute",
                    {
                        "code": f"r=tools.call({name!r}, {{'query':'fixture'}})\nassert not r['is_error']\nwhile True: pass",
                        "expected_state": {"generation": 0, "revision": 0},
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="revoked"),
            )
        )
        task = asyncio.create_task(agent.run("Exercise active revocation."))
        run_id, _ = await wait_for_record(agent, events, "child", "succeeded")
        started = asyncio.get_running_loop().time()
        await agent.revoke_mcp_server(attached.binding.binding_id)
        result = await asyncio.wait_for(task, 5)
        records = await capture_run(agent, run_id, analysis_report)
        cell = next(item for item in records if item.kind == "cell")
        assert cell.facts["status"] == "interrupted"
        assert len(identity.calls) == 1
        assert not result.artifacts
        analysis_report["revocation_seconds"] = (
            asyncio.get_running_loop().time() - started
        )
        assert analysis_report["revocation_seconds"] < 5
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_unrelated_foreground_read_progresses_with_running_and_queued_analysis(
    tmp_path, analysis_report
):
    events: list[AgentEvent] = []
    workspace = workspace_for(tmp_path)
    (workspace.root / "quick.txt").write_text("independent fixture")
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "busy",
                "analysis_execute",
                {
                    "code": "while True: pass",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            _tool(
                "queued",
                "analysis_execute",
                {
                    "code": "print(9)",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            _tool("quick", "file_read", {"path": "quick.txt"}),
        ),
        observer=events.append,
    )
    tasks = []
    try:
        tasks.append(asyncio.create_task(agent.run("Run actual computation.")))
        busy_id, generation = await wait_for_record(
            agent, events, "generation", "running"
        )
        tasks.append(asyncio.create_task(agent.run("Queue another analytical cell.")))
        async with asyncio.timeout(10):
            while True:
                started = [
                    event
                    for event in events
                    if event.kind is AgentEventKind.RUN_STARTED
                ]
                if len(started) >= 2:
                    queued_id = started[1].run_id
                    records = await agent.analysis_evidence(queued_id)
                    if any(record.kind == "cell" for record in records):
                        break
                await asyncio.sleep(0.02)
        started_at = asyncio.get_running_loop().time()
        quick = await asyncio.wait_for(agent.run("Read the independent fixture."), 5)
        assert quick.kind.value == "completed"
        assert not tasks[0].done() and not tasks[1].done()
        os.kill(generation.facts["pid"], 0)
        pairs = (await agent.transcript(quick.run_id)).tool_pairs
        assert any(
            call.name == "file_read" and block is not None and not block.is_error
            for call, block in pairs
        )
        analysis_report["unrelated_read_seconds"] = (
            asyncio.get_running_loop().time() - started_at
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await capture_run(agent, busy_id, analysis_report)
        queued = await agent.analysis_evidence(queued_id)
        assert not any(record.kind == "generation" for record in queued)
        analysis_report["queued_admission"] = [
            {"kind": record.kind, "facts": record.facts} for record in queued
        ]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await agent.close()


async def test_completed_cpu_and_broker_wait_have_distinct_host_measurements(
    tmp_path, analysis_report
):
    events: list[AgentEvent] = []
    models: list[ToolboxAwareMockModelProvider] = []
    identity, _ = conformance_identities()
    identity.block_calls = asyncio.Event()
    agent = await make_agent(
        tmp_path,
        (),
        model_holder=models,
        observer=events.append,
        mcp_client_factory=SDKMCPClientFactory(
            http_transport=httpx.MockTransport(MCPConformanceTransport(identity))
        ),
    )
    task = None
    try:
        attached = await agent.attach_mcp_server(
            endpoint=identity.endpoint,
            selections=(
                MCPToolSelection(
                    remote_name="lookup",
                    local_alias="lookup",
                    description="Read isolated data.",
                ),
            ),
        )
        name = attached.binding.tools[0].local_name
        models[0].replace_script(
            (
                _tool(
                    "load", "toolbox_load", {"tool_names": ["analysis_execute", name]}
                ),
                _tool(
                    "measure",
                    "analysis_execute",
                    {
                        "code": f"import time\nstarted=time.process_time()\nwhile time.process_time()-started < .15: total=sum(range(10000))\nr=tools.call({name!r}, {{'query':'fixture'}})\nassert not r['is_error']\nprint(total)",
                        "expected_state": {"generation": 0, "revision": 0},
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="measured"),
            )
        )
        task = asyncio.create_task(agent.run("Measure actual CPU and a broker wait."))
        run_id, _ = await wait_for_record(agent, events, "child", "running")
        await asyncio.sleep(0.5)
        identity.block_calls.set()
        result = await asyncio.wait_for(task, 10)
        assert result.kind.value == "completed"
        records = await capture_run(agent, run_id, analysis_report)
        usage = next(record for record in records if record.kind == "generation").facts[
            "usage"
        ]
        assert usage["user_cpu_seconds"] + usage["system_cpu_seconds"] >= 0.15
        assert usage["broker_wait_seconds"] >= 0.5
        assert usage["wall_seconds"] >= usage["broker_wait_seconds"]
        child = next(record for record in records if record.kind == "child")
        assert (
            child.facts["status"] == "succeeded" and child.facts["dispatched"] is True
        )
        summary = await agent.analysis_usage(run_id)
        assert summary is not None
        assert summary["outer_tool_calls"] == {
            "attempted": 2,
            "dispatched": 2,
            "failed": 0,
            "denied": 0,
            "unsettled": 0,
        }
        assert summary["child_attempted"] == summary["child_dispatched"] == 1
        assert summary["child_denied"] == summary["child_failed"] == 0
    finally:
        identity.block_calls.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()
