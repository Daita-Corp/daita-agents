"""Real native computation through the shared public foreground path."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    AgentEvent,
    AgentEventKind,
    IntervalSchedule,
    RoutineIntent,
)
from daita._json import canonical_json
from daita.adapters.analytical_workspace.native import NativePythonWorker
from daita.adapters.local_workspace import LocalWorkspaceBackend
from daita.adapters.models import SourceRegistration
from daita.adapters.sqlite_query import SQLiteQueryBackend
from daita.artifacts.models import ArtifactError
from daita.config import AgentConfig, AnalysisLimits
from daita.distribution import OutcomeState
from daita.llm.models import FinishReason, ModelResponse, ModelUsage
from daita.llm.pricing import CostEstimate
from daita.loop.models import LoopExitKind, LoopLimits, RunOrigin
from tests.artifacts._public_surface_support import _profile, _tool
from tests.support.analysis import facts as evidence_facts
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for


async def test_public_agent_runs_python_and_dependent_cell(tmp_path):
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "cell-one",
                "python_execute",
                {
                    "code": "answer = 6 * 7; print(answer)",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            _tool(
                "cell-two",
                "python_execute",
                {
                    "code": "print(answer + 1)",
                    "expected_state": {"generation": 1, "revision": 1},
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="42 and 43"),
        )
    )
    agent = await Agent.create(
        "native-slice",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Calculate in Python.")
        assert result.final_text == "42 and 43"
        transcript = await agent.transcript(result.run_id)
        cells = [
            output
            for call, output in transcript.tool_pairs
            if call.name == "python_execute"
        ]
        assert len(cells) == 2
        assert all(output is not None and output.is_error is False for output in cells)
        assert cells[0] is not None
        assert cells[1] is not None
        assert json.loads(canonical_json(cells[0].output))["data"]["stdout"] == "42\n"
        assert json.loads(canonical_json(cells[1].output))["data"]["stdout"] == "43\n"
        evidence = await agent.analysis_evidence(result.run_id)
        generations = [record for record in evidence if record.kind == "generation"]
        assert len(generations) == 1
        generation = evidence_facts(generations[0])
        assert generation["usage"]["peak_rss_bytes"] > 0
        assert generation["usage"]["user_cpu_seconds"] is not None
        assert generation["usage"]["wall_seconds"] > 0
        assert generation["cleanup"]["process_reaped"] is True
        assert generation["cleanup"]["scratch_deleted"] is True
        assert not Path(generation["scratch"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(generation["pid"], 0)
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "batch,captured", ((False, False), (True, False), (False, True))
)
async def test_scheduled_analysis_reads_exact_scope_and_delivers_artifact(
    tmp_path, batch, captured
):
    database = tmp_path / "values.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE numbers (n INTEGER)")
        connection.execute("INSERT INTO numbers VALUES (42)")
    outside = tmp_path / "outside.sqlite"
    with sqlite3.connect(outside) as connection:
        connection.execute("CREATE TABLE secret (n INTEGER)")
        connection.execute("INSERT INTO secret VALUES (999)")
    usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
    model = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Authorized", usage=usage
            ),
        ),
        complete_pricing=True,
    )
    agent = await Agent.create(
        "scheduled-analysis",
        root=tmp_path,
        model=model,
        # A scheduled run now sees its whole run-start scope, not three tools.
        model_profile=replace(_profile(model), context_window_tokens=64_000),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_sqlite(database)
        excluded = await agent.attach_sqlite(outside)
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        excluded_resource = (await agent.list_catalog_resources(source_id=excluded.id))[
            0
        ]
        origin = await agent.run("Authorize a scoped numerical report.")
        model.replace_script(
            (
                replace(
                    _tool(
                        "load-reads",
                        "toolbox_load",
                        {
                            "tool_names": [
                                "python_execute",
                                *(["data_query"] if captured else []),
                            ]
                        },
                    ),
                    usage=usage,
                ),
                replace(
                    _tool(
                        "retained-input",
                        "python_execute",
                        {
                            "code": "raise AssertionError('must not execute')",
                            "expected_state": {"generation": 0, "revision": 0},
                            "inputs": {
                                "old": {
                                    "kind": "artifact",
                                    "artifact_id": "ungranted-artifact",
                                }
                            },
                        },
                    ),
                    usage=usage,
                ),
                replace(
                    _tool(
                        "scheduled-cell",
                        "python_execute",
                        {
                            "code": "import json\n"
                            "assert tools.call('file_read', {'source_id':'s1', 'path':'values.sqlite'})['is_error']\n"
                            f"assert tools.call('data_query', {{'source_id':{excluded.id!r}, 'resource_ids':[{excluded_resource.id!r}], 'sql':'SELECT n FROM secret'}})['is_error']\n"
                            f"r = tools.call('data_query', {{'source_id':{source.id!r}, 'resource_ids':[{resource.id!r}], 'sql':'SELECT n FROM numbers'}})\n"
                            "assert not r['is_error'], r\n"
                            "assert r['output']['data']['rows'] == [{'n':42}], r\n"
                            + (
                                f"c = tools.capture('data_query', {{'source_id':{source.id!r}, 'resource_ids':[{resource.id!r}], 'sql':'SELECT n FROM numbers', 'output':'arrow'}}, name='current')\nassert not c['is_error'], c\nimport pyarrow.ipc as ipc\nwith ipc.open_file(inputs.path('current')) as reader:\n    assert reader.read_all().to_pydict() == {{'n':[42]}}\n"
                                if captured
                                else ""
                            )
                            + "open('answer.json','w').write(json.dumps({'n':42}))\n"
                            "outputs.add('answer','answer.json')"
                            + (
                                "\nopen('recipe.py','w').write('print(42)')\noutputs.add('recipe','recipe.py')"
                                if batch
                                else ""
                            ),
                            "expected_state": {"generation": 0, "revision": 0},
                            **(
                                {"save_outputs": ["answer", "recipe"]}
                                if batch
                                else {"save_output": "answer"}
                            ),
                        },
                    ),
                    usage=usage,
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="42", usage=usage),
            )
        )
        # The routine names its one source; its runs never reach the other.
        stored = await agent.create_routine(
            RoutineIntent(
                title="Scoped Python report",
                prompt="Read only the granted numbers and produce a JSON artifact using Python.",
                schedule=IntervalSchedule(3600, datetime.now(UTC) + timedelta(hours=1)),
                sources=(source.id,),
            ),
            conversation_id=origin.conversation_id,
        )
        await agent.run_routine_now(
            stored.routine_id, expected_revision=stored.revision
        )
        async with asyncio.timeout(30):
            while not (
                deliveries := await agent.list_deliveries(
                    conversation_id=origin.conversation_id
                )
            ):
                await asyncio.sleep(0.01)
        assert len(deliveries) == 1
        assert deliveries[0].conclusion_state is OutcomeState.SUCCEEDED, (
            deliveries[0].failure_code,
            [
                (run.result, run.transcript.tool_pairs)
                for run in await agent.conversation_runs(origin.conversation_id)
            ],
        )
        run_id = deliveries[0].resulting_run_id
        assert run_id is not None
        transcript = await agent.transcript(run_id)
        assert transcript.run.origin is RunOrigin.SCHEDULED_ROUTINE
        assert transcript.run.execution_scope is not None
        assert transcript.run.execution_scope.allowed_source_ids == (source.id,)
        denied = next(
            output
            for call, output in transcript.tool_pairs
            if call.id == "retained-input"
        )
        assert denied is not None and denied.is_error
        assert (
            json.loads(canonical_json(denied.output))["error"]["code"]
            == "analysis_input_not_admitted"
        )
        records = await agent.analysis_evidence(run_id)
        children = sorted(
            (evidence_facts(r) for r in records if r.kind == "child"),
            key=lambda record: record["sequence"],
        )
        assert len(children) == (4 if captured else 3)
        assert [r["dispatched"] for r in children] == [
            False,
            False,
            True,
            *([True] if captured else []),
        ]
        ref = next(
            ref
            for ref in deliveries[0].artifact_references
            if ref.media_type == "application/json"
        )
        assert len(deliveries[0].artifact_references) == (2 if batch else 1)
        assert json.loads((await agent.read_artifact(ref.artifact_id)).content) == {
            "n": 42
        }
        computation = await agent.artifact_computation_evidence(ref.artifact_id)
        assert computation["output_sha256"] == ref.sha256
        generation = next(evidence_facts(r) for r in records if r.kind == "generation")
        assert generation["cleanup"]["process_reaped"]
        assert generation["cleanup"]["scratch_deleted"]
    finally:
        await agent.close()


async def test_explicit_analysis_save_commits_one_artifact_and_persists_provenance(
    tmp_path,
):
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "cell-save",
                "python_execute",
                {
                    "code": "open('answer.txt', 'w').write('42\\n')\noutputs.add('answer','answer.txt')",
                    "expected_state": {"generation": 0, "revision": 0},
                    "save_output": "answer",
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Saved"),
        )
    )
    agent = await Agent.create(
        "save-slice",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Save the answer as a text artifact.")
        assert len(result.artifacts) == 1, [
            (call.name, None if output is None else output.output)
            for call, output in (await agent.transcript(result.run_id)).tool_pairs
        ]
        artifact = result.artifacts[0]
        assert (await agent.read_artifact(artifact.artifact_id)).content == b"42\n"
        record = await agent._embedded._store.get_artifact_record(artifact.artifact_id)
        assert record is not None
        assert record.computation_evidence["output_sha256"] == artifact.sha256
        assert json.loads(canonical_json(record.computation_evidence))["cells"][0][
            "code"
        ]
        await agent.clear_conversations()
        record = await agent._embedded._store.get_artifact_record(artifact.artifact_id)
        assert record is not None
        assert record.computation_evidence["output_sha256"] == artifact.sha256
    finally:
        await agent.close()


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
async def test_nonstandard_json_constants_never_publish_artifacts(tmp_path, constant):
    payload = '{"value":' + constant + "}"
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "nonstandard-json",
                "python_execute",
                {
                    "code": f"open('dataset.json','w').write({payload!r})\n"
                    "outputs.add('dataset','dataset.json')",
                    "expected_state": {"generation": 0, "revision": 0},
                    "save_output": "dataset",
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Validation reviewed"),
        )
    )
    agent = await Agent.create(
        "json-publication",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Create a JSON dataset.")
        assert result.artifacts == ()
        assert await agent.list_artifacts() == ()
        output = next(
            output
            for call, output in (await agent.transcript(result.run_id)).tool_pairs
            if call.id == "nonstandard-json"
        )
        assert output is not None and not output.is_error
        data = output.output["data"]
        assert isinstance(data, Mapping)
        assert data["status"] == "state_lost"
    finally:
        await agent.close()


@pytest.mark.parametrize("failure", (None, "missing_candidate", "second_commit"))
async def test_batch_publication_retains_exact_outputs_and_reports_partial_failure(
    tmp_path, monkeypatch, failure
):
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "cell-save-batch",
                "python_execute",
                {
                    "code": "open('answer.json','w').write('{\"n\":42}')\n"
                    "open('recipe.py','w').write('print(42)')\n"
                    "outputs.add('answer','answer.json')\noutputs.add('recipe','recipe.py')",
                    "expected_state": {"generation": 0, "revision": 0},
                    "save_outputs": [
                        "answer",
                        "absent" if failure == "missing_candidate" else "recipe",
                    ],
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Publication result reviewed"
            ),
        )
    )
    agent = await Agent.create(
        "batch-publication",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace_for(tmp_path),
    )
    try:
        if failure == "second_commit":
            store = agent._embedded._artifact_store
            original = store.commit
            calls = 0

            async def commit(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ArtifactError(
                        "artifact_quota_exceeded", "Injected quota failure", {}
                    )
                return await original(*args, **kwargs)

            monkeypatch.setattr(store, "commit", commit)
        result = await agent.run("Retain the dataset and its computation.")
        count = {None: 2, "missing_candidate": 0, "second_commit": 1}[failure]
        assert len(result.artifacts) == count
        for ref in result.artifacts:
            payload = await agent.read_artifact(ref.artifact_id)
            assert payload.content in (b'{"n":42}', b"print(42)")
            evidence = await agent.artifact_computation_evidence(ref.artifact_id)
            assert evidence["output_sha256"] == ref.sha256
        output = next(
            output
            for call, output in (await agent.transcript(result.run_id)).tool_pairs
            if call.id == "cell-save-batch"
        )
        assert output is not None and not output.is_error
        expected = {
            None: "requested",
            "missing_candidate": "candidate_not_found",
            "second_commit": "failed",
        }[failure]
        data = output.output["data"]
        assert isinstance(data, Mapping)
        assert data["save_status"] == expected
        if failure == "second_commit":
            artifact = output.output["artifact"]
            assert isinstance(artifact, Mapping)
            assert artifact["artifact_id"] == result.artifacts[0].artifact_id
    finally:
        await agent.close()


async def test_batch_reads_overlap_finish_out_of_order_and_retain_individual_errors(
    tmp_path, monkeypatch
):
    workspace = workspace_for(tmp_path)
    (workspace.root / "one.txt").write_text("17")
    (workspace.root / "two.txt").write_text("25")
    first = asyncio.Event()
    second_finished = asyncio.Event()
    release = asyncio.Event()
    original = LocalWorkspaceBackend.read

    async def read(self, *, path, **kwargs):
        if path == "one.txt":
            first.set()
            await release.wait()
        elif path == "two.txt":
            await first.wait()
        result = await original(self, path=path, **kwargs)
        if path == "two.txt":
            second_finished.set()
        return result

    monkeypatch.setattr(LocalWorkspaceBackend, "read", read)
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "batch",
                "python_execute",
                {
                    "code": "rs = tools.batch([{'name':'file_read','arguments':{'source_id':'s1','path':p}} for p in ['one.txt','missing.txt','two.txt']])\nassert [r['is_error'] for r in rs] == [False,True,False], rs\nassert [rs[i]['output']['data']['content'] for i in [0,2]] == ['17','25'], rs\nprint('ordered')",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="42"),
        )
    )
    agent = await Agent.create(
        "batch-overlap",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace,
    )
    task = asyncio.create_task(
        agent.run("Use Python to read local one.txt and two.txt.")
    )
    try:
        await asyncio.wait_for(second_finished.wait(), timeout=5)
        assert first.is_set() and not release.is_set()
        compute = [
            metric
            for metric in agent.admission_diagnostics().metrics
            if metric.kind == "analysis_compute"
        ]
        assert compute and all(metric.active == 0 for metric in compute)
        release.set()
        result = await asyncio.wait_for(task, timeout=5)
        assert result.kind is LoopExitKind.COMPLETED
        children = sorted(
            (
                record
                for record in await agent.analysis_evidence(result.run_id)
                if record.kind == "child"
            ),
            key=lambda record: evidence_facts(record)["sequence"],
        )
        assert len(children) == 3
        assert [evidence_facts(child)["arguments"]["path"] for child in children] == [
            "one.txt",
            "missing.txt",
            "two.txt",
        ]
        assert [evidence_facts(child)["status"] for child in children] == [
            "succeeded",
            "failed",
            "succeeded",
        ]
        assert len({child.evidence_id for child in children}) == 3
        transcript = await agent.transcript(result.run_id)
        assert all(
            not call.id.startswith("analysis-child-")
            for call, output in transcript.tool_pairs
        )
        cell = next(
            output for call, output in transcript.tool_pairs if call.id == "batch"
        )
        assert (
            cell is not None
            and json.loads(canonical_json(cell.output))["data"]["stdout"] == "ordered\n"
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()
    assert agent.admission_diagnostics().active_permits == 0


@pytest.mark.parametrize("failure", ("cell_limit", "run_limit", "malformed"))
async def test_batch_rejects_whole_envelopes_before_io_and_counts_attempts(
    tmp_path, monkeypatch, failure
):
    workspace = workspace_for(tmp_path)
    (workspace.root / "one.txt").write_text("42")
    dispatched = []
    original = LocalWorkspaceBackend.read

    async def read(self, *, path, **kwargs):
        dispatched.append(path)
        return await original(self, path=path, **kwargs)

    monkeypatch.setattr(LocalWorkspaceBackend, "read", read)
    calls = [
        {"name": "file_read", "arguments": {"source_id": "s1", "path": "one.txt"}}
        for _ in range(3)
    ]
    if failure == "malformed":
        calls[-1] = {"name": "file_read"}
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "batch",
                "python_execute",
                {
                    "code": f"rs = tools.batch({calls!r})\nassert len(rs) == 3 and all(r['is_error'] for r in rs), rs\nprint('denied')",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="denied"),
        )
    )
    config = AgentConfig(
        analysis_limits=AnalysisLimits(child_calls=2 if failure == "cell_limit" else 16)
    )
    agent = await Agent.create(
        "batch-denied",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace,
        config=config,
        limits=(
            LoopLimits(max_tool_calls_per_run=4, max_tool_calls_per_response=4)
            if failure == "run_limit"
            else None
        ),
    )
    try:
        result = await agent.run("Use Python to read local one.txt.")
        assert dispatched == []
        assert (result.kind is LoopExitKind.FAILED) == (failure == "run_limit")
        evidence = await agent.analysis_evidence(result.run_id)
        children = [record for record in evidence if record.kind == "child"]
        assert len(children) == 3
        assert all(
            evidence_facts(child)["status"] == "failed"
            and evidence_facts(child)["dispatched"] is False
            for child in children
        )
        summary = next(record for record in evidence if record.kind == "run")
        assert evidence_facts(summary)["child_attempted"] == 3
    finally:
        await agent.close()


@pytest.mark.parametrize("interruption", ("cancel", "timeout"))
async def test_batch_interruption_settles_dispatched_and_queued_children(
    tmp_path, monkeypatch, interruption
):
    entered = asyncio.Event()
    settled = asyncio.Event()
    calls = []
    events: list[AgentEvent] = []

    async def read(self, *, path, **kwargs):
        calls.append(path)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    monkeypatch.setattr(LocalWorkspaceBackend, "read", read)
    code = "tools.batch([{'name':'file_read','arguments':{'source_id':'s1','path':p}} for p in ['one.txt','two.txt','three.txt']])"
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "batch",
                "python_execute",
                {
                    "code": code,
                    "expected_state": {"generation": 0, "revision": 0},
                    # The cell budget also covers worker start and first dispatch,
                    # so a tighter one can expire before the blocked read begins.
                    "timeout_seconds": 2 if interruption == "timeout" else 30,
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="The cell stopped."),
        )
    )
    agent = await Agent.create(
        "batch-interruption",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace_for(tmp_path),
        observer=events.append,
        limits=LoopLimits(max_parallel_reads=1, max_parallel_reads_per_source=1),
    )
    task = asyncio.create_task(agent.run("Use Python to read the local text files."))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        run_id = next(
            event.run_id for event in events if event.kind is AgentEventKind.RUN_STARTED
        )
        if interruption == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, timeout=5)
        assert settled.is_set()
        assert calls == ["one.txt"]
        evidence = await agent.analysis_evidence(run_id)
        children = sorted(
            (record for record in evidence if record.kind == "child"),
            key=lambda record: evidence_facts(record)["sequence"],
        )
        assert len(children) == 3
        assert all(
            evidence_facts(child)["status"] == "interrupted" for child in children
        )
        assert [evidence_facts(child)["dispatched"] for child in children] == [
            True,
            False,
            False,
        ]
        generation = next(record for record in evidence if record.kind == "generation")
        cleanup = evidence_facts(generation)["cleanup"]
        assert (
            cleanup["child_io_settled"]
            and cleanup["process_reaped"]
            and cleanup["scratch_deleted"]
        )
        summary = next(record for record in evidence if record.kind == "run")
        assert evidence_facts(summary)["child_attempted"] == 3
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()
    assert agent.admission_diagnostics().active_permits == 0


async def test_batch_trace_exhaustion_dispatches_nothing_and_retains_requested_usage(
    tmp_path, monkeypatch
):
    dispatched = []

    async def read(self, **kwargs):
        dispatched.append(kwargs)
        raise AssertionError("trace exhaustion must precede I/O")

    monkeypatch.setattr(LocalWorkspaceBackend, "read", read)
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "batch",
                "python_execute",
                {
                    "code": "tools.batch([{'name':'file_read','arguments':{'source_id':'s1','path':'one.txt'}}]*3)",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Must not request completion after exhaustion.",
            ),
        )
    )
    agent = await Agent.create(
        "batch-trace",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace_for(tmp_path),
        config=AgentConfig(analysis_limits=AnalysisLimits(trace_bytes=1024 * 1024)),
    )
    try:
        result = await agent.run("Read local one.txt in a bounded Python batch.")
        assert result.kind is LoopExitKind.FAILED
        assert dispatched == []
        assert len(model.logical_requests) == 1
        evidence = await agent.analysis_evidence(result.run_id)
        summary = next(record for record in evidence if record.kind == "run")
        assert evidence_facts(summary)["child_attempted"] == 3
        assert all(
            not evidence_facts(record)["dispatched"]
            for record in evidence
            if record.kind == "child"
        )
    finally:
        await agent.close()


async def test_python_threads_serialize_batches_and_keep_each_response_identity(
    tmp_path,
):
    workspace = workspace_for(tmp_path)
    for name in ("one", "two"):
        (workspace.root / f"{name}.txt").write_text(name)
    code = "from concurrent.futures import ThreadPoolExecutor\ndef fetch(name):\n rs = tools.batch([{'name':'file_read','arguments':{'source_id':'s1','path':name+'.txt'}}]*2)\n assert all(not r['is_error'] and r['output']['data']['content'] == name for r in rs), rs\n return name\nwith ThreadPoolExecutor(max_workers=2) as pool:\n print(list(pool.map(fetch, ['one','two'])))"
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "batch",
                "python_execute",
                {"code": code, "expected_state": {"generation": 0, "revision": 0}},
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Read both."),
        )
    )
    agent = await Agent.create(
        "thread-batches",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace,
    )
    try:
        result = await agent.run("Use Python to read local one.txt and two.txt.")
        children = sorted(
            (
                record
                for record in await agent.analysis_evidence(result.run_id)
                if record.kind == "child"
            ),
            key=lambda record: evidence_facts(record)["sequence"],
        )
        assert len(children) == 4
        assert [evidence_facts(record)["sequence"] for record in children] == [
            1,
            2,
            3,
            4,
        ]
        paths = [evidence_facts(record)["arguments"]["path"] for record in children]
        assert paths in [
            ["one.txt", "one.txt", "two.txt", "two.txt"],
            ["two.txt", "two.txt", "one.txt", "one.txt"],
        ]
        cell = next(
            output
            for call, output in (await agent.transcript(result.run_id)).tool_pairs
            if call.id == "batch"
        )
        assert (
            cell is not None
            and json.loads(canonical_json(cell.output))["data"]["stdout"]
            == "['one', 'two']\n"
        )
    finally:
        await agent.close()


async def test_batch_uses_distinct_source_lanes_and_rechecks_revoked_authority(
    tmp_path, monkeypatch
):
    first = asyncio.Event()
    second = asyncio.Event()
    release = asyncio.Event()
    calls = []
    sources: list[SourceRegistration] = []
    original = SQLiteQueryBackend.execute_read

    async def read(self, **kwargs):
        source_id = kwargs["source_id"]
        calls.append(source_id)
        if source_id == sources[0].id and not first.is_set():
            first.set()
            await release.wait()
        result = await original(self, **kwargs)
        if source_id == sources[1].id:
            second.set()
        return result

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
    model = ToolboxAwareMockModelProvider(())
    agent = await Agent.create(
        "batch-authority",
        root=tmp_path,
        model=model,
        # Two tool round trips with python_execute and the data tools loaded
        # outgrow the shared 32k profile under the fail-closed char estimate.
        model_profile=replace(_profile(model), context_window_tokens=40_000),
        workspace=workspace_for(tmp_path),
    )
    for index in range(2):
        path = tmp_path / f"source-{index}.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE numbers (n INTEGER)")
            connection.execute("INSERT INTO numbers VALUES (42)")
        sources.append(await agent.attach_sqlite(path))
    resources = [
        (await agent.list_catalog_resources(source_id=source.id))[0]
        for source in sources
    ]
    children = [
        {
            "name": "data_query",
            "arguments": {
                "source_id": sources[index].id,
                "resource_ids": [resources[index].id],
                "sql": "SELECT n FROM numbers",
            },
        }
        for index in (0, 0, 1)
    ]
    model.replace_script(
        (
            _tool(
                "batch",
                "python_execute",
                {
                    "code": f"rs = tools.batch({children!r})\nassert [r['is_error'] for r in rs] == [True,True,False], rs\nassert rs[2]['output']['data']['rows'] == [{{'n':42}}]\nprint('revalidated')",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP, text="The admitted source returned 42."
            ),
        )
    )
    task = asyncio.create_task(
        agent.run("Use Python to read the connected numbers tables.")
    )
    try:
        await asyncio.wait_for(second.wait(), timeout=5)
        assert calls.count(sources[0].id) == calls.count(sources[1].id) == 1
        preview = await agent.preview_source_permissions(
            source_id=sources[0].id,
            read_mode="none",
            read_resource_ids=(),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=sources[0].id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )
        release.set()
        result = await asyncio.wait_for(task, timeout=5)
        assert result.kind is LoopExitKind.COMPLETED
        transcript = await agent.transcript(result.run_id)
        cell = next(
            output for call, output in transcript.tool_pairs if call.id == "batch"
        )
        assert (
            cell is not None
            and json.loads(canonical_json(cell.output))["data"]["stdout"]
            == "revalidated\n"
        )
        evidence = sorted(
            (
                record
                for record in await agent.analysis_evidence(result.run_id)
                if record.kind == "child"
            ),
            key=lambda record: evidence_facts(record)["sequence"],
        )
        assert [evidence_facts(record)["status"] for record in evidence] == [
            "failed",
            "failed",
            "succeeded",
        ]
        assert calls.count(sources[1].id) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize("field", ("token", "sequence", "kind"))
async def test_worker_rejects_stale_batch_response_identity_without_replay(
    tmp_path, monkeypatch, field
):
    original = NativePythonWorker._send
    response_count = 0

    async def send(self, value, **kwargs):
        nonlocal response_count
        if value.get("kind") == "child_result":
            response_count += 1
            value = {**value, field: 99 if field == "sequence" else "stale"}
        return await original(self, value, **kwargs)

    monkeypatch.setattr(NativePythonWorker, "_send", send)
    workspace = workspace_for(tmp_path)
    (workspace.root / "one.txt").write_text("42")
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "batch",
                "python_execute",
                {
                    "code": "tools.batch([{'name':'file_read','arguments':{'source_id':'s1','path':'one.txt'}}])",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP, text="The response was rejected."
            ),
        )
    )
    agent = await Agent.create(
        "batch-identity",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace,
    )
    try:
        result = await agent.run("Use Python to read local one.txt.")
        assert response_count == 1
        cell = next(
            output
            for call, output in (await agent.transcript(result.run_id)).tool_pairs
            if call.id == "batch"
        )
        assert cell is not None
        data = json.loads(canonical_json(cell.output))["data"]
        assert data["status"] == "python_error"
        assert "Invalid or stale broker response identity" in data["traceback"]
        children = [
            record
            for record in await agent.analysis_evidence(result.run_id)
            if record.kind == "child"
        ]
        assert (
            len(children) == 1 and evidence_facts(children[0])["status"] == "succeeded"
        )
    finally:
        await agent.close()


@pytest.mark.parametrize("with_catalog", (False, True))
async def test_real_python_broker_reads_and_denies_recursive_calls(
    tmp_path, with_catalog
):
    workspace = workspace_for(tmp_path)
    (workspace.root / "values.txt").write_text("17,25")
    model = ToolboxAwareMockModelProvider(())
    agent = await Agent.create(
        "broker-slice",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace,
    )
    # Attached sources take the first handles, so name the session folder exactly.
    model.replace_script(
        (
            _tool(
                "cell-read",
                "python_execute",
                {
                    "code": "r = tools.call('file_read', "
                    f"{{'source_id':'{agent.session_source_id}', 'path': 'values.txt'}})"
                    "\nprint(r)\n"
                    "assert not r['is_error']\n"
                    "denied = tools.call('python_execute', {})\nassert denied['is_error']",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Read complete"),
        )
    )
    try:
        if with_catalog:
            for index in range(3):
                database = tmp_path / f"source-{index}.sqlite"
                with sqlite3.connect(database) as connection:
                    connection.execute("CREATE TABLE unrelated (n INTEGER)")
                    connection.execute("INSERT INTO unrelated VALUES (999)")
                await agent.attach_sqlite(database)
        result = await agent.run("Read values with Python.")
        evidence = await agent.analysis_evidence(result.run_id)
        children = sorted(
            [record for record in evidence if record.kind == "child"],
            key=lambda record: json.loads(canonical_json(record.facts))["sequence"],
        )
        assert len(children) == 2, [record.facts["result"] for record in children]
        assert children[0].facts["status"] == "succeeded"
        assert children[0].facts["dispatched"] is True
        assert children[1].facts["status"] == "failed"
        assert children[1].facts["dispatched"] is False
        transcript = await agent.transcript(result.run_id)
        assert not any(
            call.id.startswith("analysis-child-") for call, _ in transcript.tool_pairs
        )
        cell = [
            output
            for call, output in transcript.tool_pairs
            if call.name == "python_execute"
        ][0]
        assert cell is not None
        assert json.loads(canonical_json(cell.output))["data"]["status"] == "success"
        assert (
            json.loads(canonical_json(cell.output))["data"]["usage"][
                "broker_wait_seconds"
            ]
            > 0
        )
    finally:
        await agent.close()


@pytest.mark.parametrize("handoff", ("export_then_bind", "capture"))
async def test_exact_arrow_export_bind_compute_and_reopen(tmp_path, handoff):
    from hashlib import sha256

    from daita.llm.models import ToolResultBlock

    database = tmp_path / "dataset.sqlite"
    expected = [
        (2**60 + index, ("x" * 200 + str(index)) if index % 3 else None, b"\x00\xff")
        for index in range(150)
    ]
    expected.append(expected[0])
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE values_data (seq INTEGER, n INTEGER, label TEXT, payload BLOB)"
        )
        connection.executemany(
            "INSERT INTO values_data VALUES (?, ?, ?, ?)",
            [(i, *row) for i, row in enumerate(expected)],
        )

    class BindModel(ToolboxAwareMockModelProvider):
        async def generate(self, request):
            response = await super().generate(request)
            return compute(request) if response.text == "bind-arrow" else response

    model = BindModel(())
    agent = await Agent.create(
        "arrow-input",
        root=tmp_path,
        model=model,
        model_profile=replace(_profile(model), context_window_tokens=64000),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_sqlite(database)
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]

        export_arguments: dict[str, object] = {
            "source_id": source.id,
            "resource_ids": [resource.id],
            "sql": "SELECT n AS alias, label, payload FROM values_data ORDER BY seq",
            "output": "arrow",
        }

        def compute(request):
            inputs = {}
            if handoff == "capture":
                prefix = f"captured = tools.capture('data_query', {export_arguments!r}, name='rows')\nassert not captured['is_error'], captured\nreference = captured['output']['artifact']\n"
                digest_expression = "reference['sha256'].removeprefix('sha256:')"
            else:
                exported = next(
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "export-arrow"
                )
                assert not exported.is_error, exported.output
                data = exported.output["data"]
                assert isinstance(data, Mapping)
                assert (
                    data["complete"] is True
                    and data["coverage"] == "complete_query_result"
                )
                assert data["columns"] == ("alias", "label", "payload")
                assert data["column_types"] == ("int64", "string", "binary")
                reference = exported.output["artifact"]
                assert isinstance(reference, Mapping)
                assert int(str(reference["byte_size"])) > 16 * 1024
                inputs = {
                    "rows": {
                        "kind": "artifact",
                        "artifact_id": reference["artifact_id"],
                    }
                }
                prefix = ""
                digest_expression = repr(
                    str(reference["sha256"]).removeprefix("sha256:")
                )
            return _tool(
                "compute-arrow",
                "python_execute",
                {
                    "code": prefix
                    + "import pyarrow.ipc as ipc\nfrom hashlib import sha256\n"
                    "with ipc.open_file(inputs.path('rows')) as reader:\n    table = reader.read_all()\n"
                    "assert table.column_names == ['alias', 'label', 'payload']\n"
                    f"assert table.column('label').null_count == {sum(row[1] is None for row in expected)}\n"
                    f"assert sha256(open(inputs.path('rows'), 'rb').read()).hexdigest() == {digest_expression}\n"
                    f"assert table.num_rows == {len(expected)}\nprint(sum(table.column('alias').to_pylist()))",
                    "inputs": inputs,
                    "expected_state": {"generation": 0, "revision": 0},
                },
            )

        model.replace_script(
            (
                _tool(
                    "load-arrow",
                    "toolbox_load",
                    {"tool_names": ["data_query", "python_execute"]},
                ),
                *(
                    (
                        _tool("export-arrow", "data_query", export_arguments),
                        ModelResponse(
                            finish_reason=FinishReason.STOP, text="bind-arrow"
                        ),
                    )
                    if handoff == "export_then_bind"
                    else (compute(None),)
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="Exact result"),
            )
        )
        result = await agent.run("Compute from the complete query in Python.")
        transcript = await agent.transcript(result.run_id)
        outputs = {call.id: output for call, output in transcript.tool_pairs}
        assert "compute-arrow" in outputs, (
            result.kind,
            result.reason,
            [(call.name, output) for call, output in transcript.tool_pairs],
        )
        assert (
            outputs["compute-arrow"] is not None
            and not outputs["compute-arrow"].is_error
        ), outputs["compute-arrow"]
        cell_data = outputs["compute-arrow"].output["data"]
        assert isinstance(cell_data, Mapping)
        assert cell_data["stdout"] == str(sum(row[0] for row in expected)) + "\n"
        reference = (await agent.list_artifacts())[0]
        assert (
            reference.byte_size > 16 * 1024
            and reference.provenance.row_count == len(expected)
        )
        if handoff == "capture":
            captured = json.loads(canonical_json(cell_data["captured_inputs"]))
            assert len(captured) == 1 and captured[0]["installed"] is True
            assert captured[0]["artifact"]["artifact_id"] == reference.artifact_id
            evidence = [
                r
                for r in await agent.analysis_evidence(result.run_id)
                if r.kind == "child"
            ]
            assert len(evidence) == 1 and evidence[0].facts["status"] == "succeeded"
            assert all(call.name != "data_query" for call, _ in transcript.tool_pairs)
        content = (await agent.read_artifact(reference.artifact_id)).content
        assert reference.sha256 == "sha256:" + sha256(content).hexdigest()
        from importlib import import_module

        pa = import_module("pyarrow")
        with pa.ipc.open_file(pa.BufferReader(content)) as reader:
            assert reader.read_all().to_pydict() == {
                "alias": [row[0] for row in expected],
                "label": [row[1] for row in expected],
                "payload": [row[2] for row in expected],
            }

        await agent.close()
        agent = await Agent.open(
            "arrow-input", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        assert (await agent.read_artifact(reference.artifact_id)).content == content
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "failure",
    (
        "name",
        "statements",
        "format",
        "publication",
        "parser",
        "parser_bound",
        "decoded_bound",
        "drift",
        "revocation",
        "deletion",
        "cancel",
        "commit_cancel",
        "commit_cancel_race",
    ),
)
async def test_capture_failure_retains_publication_without_query_replay(
    tmp_path, monkeypatch, failure
):
    from daita.artifacts.store import AgentHomeArtifactStore
    from daita.domains.analysis.domain import AnalysisCapabilityDomain

    database = tmp_path / "capture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE numbers (n INTEGER)")
        connection.executemany(
            "INSERT INTO numbers VALUES (?)", [(n,) for n in range(200)]
        )
    events: list[AgentEvent] = []
    model = ToolboxAwareMockModelProvider(())
    agent = await Agent.create(
        "capture-failure",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        observer=events.append,
        workspace=workspace_for(tmp_path),
        config=AgentConfig(
            analysis_limits=AnalysisLimits(
                table_rows=10 if failure == "parser_bound" else 1_000_000,
                decoded_bytes=16 if failure == "decoded_bound" else 128 * 1024 * 1024,
            )
        ),
    )
    calls = 0
    entered, release = asyncio.Event(), asyncio.Event()
    execute = SQLiteQueryBackend.execute_exact_tabular
    validate = AnalysisCapabilityDomain._validate_candidate

    async def export(self, **kwargs):
        nonlocal calls
        calls += 1
        return await execute(self, **kwargs)

    async def parser(self, state, candidate, compute, deadline):
        assert state.worker is not None and state.worker.suspended
        assert all(
            metric.active == 0
            for metric in agent.admission_diagnostics().metrics
            if metric.kind == "analysis_compute"
        )
        entered.set()
        if failure == "parser":
            raise ValueError("Injected input validation failure")
        if failure in {
            "revocation",
            "deletion",
            "cancel",
            "commit_cancel",
            "commit_cancel_race",
        }:
            await release.wait()
        await validate(self, state, candidate, compute, deadline)

    commit = AgentHomeArtifactStore.commit

    async def published(self, *args, **kwargs):
        ref = await commit(self, *args, **kwargs)
        entered.set()
        await release.wait()
        return ref

    async def publication(self, **kwargs):
        raise ArtifactError("artifact_storage_failed", "Injected publication failure")

    monkeypatch.setattr(SQLiteQueryBackend, "execute_exact_tabular", export)
    monkeypatch.setattr(AnalysisCapabilityDomain, "_validate_candidate", parser)
    if failure == "publication":
        monkeypatch.setattr(AgentHomeArtifactStore, "commit", publication)
    elif failure in {"commit_cancel", "commit_cancel_race"}:
        monkeypatch.setattr(AgentHomeArtifactStore, "commit", published)
    source = await agent.attach_sqlite(database)
    resource = (await agent.list_catalog_resources(source_id=source.id))[0]
    arguments = {
        "source_id": source.id,
        "resource_ids": [resource.id],
        **(
            {"statements": [{"sql": "SELECT n FROM numbers"}]}
            if failure == "statements"
            else {"sql": "SELECT n FROM numbers"}
        ),
        "output": "csv" if failure == "format" else "arrow",
    }
    name = "../outside" if failure == "name" else "numbers"
    model.replace_script(
        (
            _tool(
                "load",
                "toolbox_load",
                {
                    "tool_names": [
                        "python_execute",
                        "data_query",
                    ]
                },
            ),
            _tool(
                "capture",
                "python_execute",
                {
                    "code": f"r = tools.capture('data_query', {arguments!r}, name={name!r})\nassert r['is_error'], r\n"
                    + (
                        "assert 'artifact' in r['output']\n"
                        if failure == "parser"
                        else ""
                    )
                    + "print('bounded failure')",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Failed capture reported"
            ),
        )
    )
    if failure == "drift":
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE numbers ADD COLUMN extra TEXT")
    task = asyncio.create_task(
        agent.run("Use Python with the connected numbers table.")
    )
    try:
        if failure in {
            "revocation",
            "deletion",
            "cancel",
            "commit_cancel",
            "commit_cancel_race",
        }:
            await asyncio.wait_for(entered.wait(), timeout=5)
            references = await agent.list_artifacts()
            assert len(references) == 1
            if failure == "revocation":
                preview = await agent.preview_source_permissions(
                    source_id=source.id,
                    read_mode="none",
                    read_resource_ids=(),
                    relational_write_scopes={},
                )
                await agent.apply_source_permissions(
                    source_id=source.id,
                    confirmation_fingerprint=preview.confirmation_fingerprint,
                )
            elif failure == "deletion":
                await agent.delete_artifact(references[0].artifact_id)
            else:
                task.cancel()
            if failure != "commit_cancel":
                release.set()
        if failure in {"cancel", "commit_cancel", "commit_cancel_race"}:
            with pytest.raises(asyncio.CancelledError):
                await task
            from tests.support.analysis import terminal

            conversation_id = next(
                event.conversation_id
                for event in events
                if event.kind is AgentEventKind.RUN_STARTED
            )
            result = await terminal(agent, conversation_id)
        else:
            result = await asyncio.wait_for(task, timeout=10)
        evidence = await agent.analysis_evidence(result.run_id)
        children = [r for r in evidence if r.kind == "child"]
        assert len(children) == 1
        assert calls == (0 if failure in {"name", "statements", "format"} else 1)
        committed = failure in {
            "parser",
            "revocation",
            "deletion",
            "cancel",
            "commit_cancel",
            "commit_cancel_race",
        }
        assert children[0].facts["status"] == (
            "interrupted"
            if failure == "commit_cancel"
            else "succeeded" if committed else "failed"
        )
        cells = [evidence_facts(r) for r in evidence if r.kind == "cell"]
        if committed:
            child_result = children[0].facts["result"]
            assert isinstance(child_result, Mapping) and "artifact" in child_result
            if failure not in {"commit_cancel", "commit_cancel_race"}:
                assert len(cells[0]["result"]["captured_inputs"]) == 1
                assert cells[0]["result"]["captured_inputs"][0]["installed"] is False
        assert len(await agent.list_artifacts()) == (
            1 if committed and failure != "deletion" else 0
        )
        if failure in {"cancel", "commit_cancel", "commit_cancel_race"}:
            assert result.kind is LoopExitKind.INTERRUPTED
        generations = [evidence_facts(r) for r in evidence if r.kind == "generation"]
        assert all(
            r["cleanup"]["child_io_settled"]
            and r["cleanup"]["process_reaped"]
            and r["cleanup"]["scratch_deleted"]
            for r in generations
        )
        await agent.close()
        agent = await Agent.open(
            "capture-failure", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        reopened = await agent.analysis_evidence(result.run_id)
        assert [r.facts for r in reopened if r.kind == "child"] == [
            r.facts for r in children
        ]
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_capture_metadata_delivery_bound_keeps_committed_reference(
    tmp_path, monkeypatch
):
    names = [f"c{i:03}_" + "x" * 190 for i in range(256)]
    database = tmp_path / "wide.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE wide ("
            + ",".join(f'"{name}" INTEGER' for name in names)
            + ")"
        )
        connection.execute(
            "INSERT INTO wide VALUES (" + ",".join("1" for _ in names) + ")"
        )
    calls = 0
    original = SQLiteQueryBackend.execute_exact_tabular

    async def exported(self, **kwargs):
        nonlocal calls
        calls += 1
        return await original(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_exact_tabular", exported)
    model = ToolboxAwareMockModelProvider(())
    agent = await Agent.create(
        "wide-capture",
        root=tmp_path,
        model=model,
        model_profile=replace(_profile(model), context_window_tokens=128000),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_sqlite(database)
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        arguments = {
            "source_id": source.id,
            "resource_ids": [resource.id],
            "sql": "SELECT * FROM wide",
            "output": "arrow",
        }
        model.replace_script(
            (
                _tool(
                    "load",
                    "toolbox_load",
                    {"tool_names": ["python_execute", "data_query"]},
                ),
                _tool(
                    "capture",
                    "python_execute",
                    {
                        "expected_state": {"generation": 0, "revision": 0},
                        "code": f"r = tools.capture('data_query', {arguments!r}, name='wide')\nassert r['is_error'], r\nassert r['output']['artifact']['artifact_id']\nprint('reference retained')",
                    },
                ),
                ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="The export reference was retained.",
                ),
            )
        )
        result = await agent.run("Capture the complete wide query for Python.")
        transcript = await agent.transcript(result.run_id)
        cell = next(
            output for call, output in transcript.tool_pairs if call.id == "capture"
        )
        assert cell is not None
        data = json.loads(canonical_json(cell.output))["data"]
        assert data["stdout"] == "reference retained\n", data
        assert calls == 1
        reference = (await agent.list_artifacts())[0]
        assert (
            reference.provenance.columns == tuple(names)
            and reference.provenance.row_count == 1
        )
        assert (
            data["captured_inputs"][0]["artifact"]["artifact_id"]
            == reference.artifact_id
        )
        child = next(
            r for r in await agent.analysis_evidence(result.run_id) if r.kind == "child"
        )
        assert child.facts["status"] == "failed" and child.facts["dispatched"]
        assert (
            json.loads(canonical_json(child.facts["result"]))["artifact"]["sha256"]
            == reference.sha256
        )
    finally:
        await agent.close()
