"""Native closure and cumulative admission through the public lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.adapters.analytical_workspace.native import NativePythonWorker
from daita.artifacts.models import ArtifactAuthorship, ArtifactDraft, ArtifactProvenance
from daita.capabilities import ArtifactPolicy
from daita.catalog.models import Sensitivity
from daita.config import AnalysisLimits
from daita.llm.models import FinishReason, ModelResponse
from daita.loop.analysis import AnalysisEvidence
from daita.loop.models import LoopExitKind, RunInput
from daita.loop.transcripts import RunSessionWriter
from daita.storage.sql import SQLStateStore
from tests.artifacts._public_surface_support import _profile, _tool
from tests.support.analysis import facts as evidence_facts, terminal, usage
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for


def cell(name, code, **extra):
    return _tool(
        name,
        "analysis_execute",
        {
            "code": code,
            "expected_state": {"generation": 0, "revision": 0},
            **extra,
        },
    )


async def create(root, *turns):
    model = ToolboxAwareMockModelProvider(
        (*turns, ModelResponse(finish_reason=FinishReason.STOP, text="done"))
    )
    return await Agent.create(
        "recovery",
        root=root,
        workspace=workspace_for(root),
        model=model,
        model_profile=_profile(model),
    )


async def test_parser_cleanup_failure_blocks_success_and_recovers_terminal_run(
    tmp_path, monkeypatch
):
    original = NativePythonWorker.start
    parsers = []

    async def fault(worker):
        candidate = worker.scratch / "candidate.bin"
        if candidate.exists():
            os.chflags(candidate, stat.UF_IMMUTABLE)
            parsers.append(worker)
        await original(worker)

    monkeypatch.setattr(NativePythonWorker, "start", fault)
    agent = await create(
        tmp_path,
        cell(
            "output",
            "open('answer.txt','w').write('42')\noutputs.add('answer','answer.txt')",
        ),
    )
    try:
        result = await agent.run("Create a validated text output.")
        before = await agent.analysis_evidence(result.run_id)
        assert len(parsers) == 1
        parser = next(r for r in before if r.evidence_id == "parser-1")
        assert evidence_facts(parser)["status"] == "cleanup_failed"
        assert evidence_facts(parser)["cleanup"]["scratch_deleted"] is False
        assert result.kind is LoopExitKind.FAILED
        await agent.clear_conversations()
        assert parser in await agent.analysis_evidence(result.run_id)
        await agent.close()
        with pytest.raises((PermissionError, RuntimeError)):
            await Agent.open(
                "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
            )
        os.chflags(parsers[0].scratch / "candidate.bin", 0)
        agent = await Agent.open(
            "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        after = await agent.analysis_evidence(result.run_id)
        recovered = next(r for r in after if r.evidence_id == "parser-1")
        assert evidence_facts(recovered)["status"] == "cleanup_failed"
        assert evidence_facts(recovered)["cleanup"] == evidence_facts(parser)["cleanup"]
        assert (
            evidence_facts(recovered)["recovery"]["cleanup"]["scratch_deleted"] is True
        )
        assert (await terminal(agent, result.conversation_id)) == result
        assert not parsers[0].scratch.exists()
        assert not Path(str(evidence_facts(parser)["closure_path"])).exists()
    finally:
        await agent.close()
        for worker in parsers:
            if worker.scratch.exists():
                os.chflags(worker.scratch / "candidate.bin", 0)
                from daita.adapters.analytical_workspace.cleanup import (
                    remove_owned_scratch,
                )

                remove_owned_scratch(
                    worker.scratch,
                    worker.identity["scratch_device"],
                    worker.identity["scratch_inode"],
                )
            worker.dispose_closure_evidence()


async def test_recovery_traverses_closed_pages_before_interrupting_runs(tmp_path):
    agent = await create(tmp_path)
    store = agent._embedded._store
    workers = []
    try:
        for i in range(2):
            run = RunInput(
                id=f"run-page-{i}",
                agent_id=agent.id,
                message="fixture",
                created_at=datetime.now(UTC),
                conversation_id=f"page-{i}",
            )
            writer = RunSessionWriter(store, run)
            await writer.start()
            for j in range(256 if i == 0 else 1):
                await writer.record_analysis(
                    AnalysisEvidence(
                        run.id, f"cell-{j:03}", "cell", {"status": "completed"}
                    )
                )
        run = RunInput(
            id="run-page-z",
            agent_id=agent.id,
            message="pending",
            created_at=datetime.now(UTC),
            conversation_id="page-z",
        )
        writer = RunSessionWriter(store, run)
        await writer.start()
        worker = NativePythonWorker(durable_closure=True)
        workers.append(worker)
        await writer.record_analysis(
            AnalysisEvidence(
                run.id,
                "generation-1",
                "generation",
                {"status": "admitted", **worker.identity},
            )
        )
        await worker.start()
        await worker.close()
        await agent.close()
        agent = await Agent.open(
            "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        evidence = await agent.analysis_evidence(run.id)
        assert (
            evidence_facts(evidence[0]).get("recovery", {}).get("status")
            == "recovered_closed"
        )
        assert (
            await terminal(agent, run.conversation_id)
        ).kind is LoopExitKind.INTERRUPTED
        assert not Path(str(worker.identity["closure_path"])).exists()
    finally:
        await agent.close()
        for worker in workers:
            await worker.close()
            worker.dispose_closure_evidence()


@pytest.mark.parametrize("extension", ["csv", "arrow", "parquet"])
async def test_input_parser_cpu_reduces_main_resume_allowance(
    tmp_path, monkeypatch, extension
):
    code = {
        "csv": "open('input.csv','w').write('n,v\\n'+'1,2\\n'*10000)",
        "arrow": "import pyarrow as pa, pyarrow.ipc as ipc\nt=pa.table({'n':[1]*10000,'v':[2]*10000})\nwith ipc.new_file('input.arrow',t.schema) as writer: writer.write_table(t)",
        "parquet": "import pyarrow as pa, pyarrow.parquet as pq\npq.write_table(pa.table({'n':[1]*10000,'v':[2]*10000}),'input.parquet')",
    }[extension]
    agent = await create(
        tmp_path,
        cell(
            "seed",
            (
                code + f"\noutputs.add('input','input.{extension}')"
                if extension != "arrow"
                else "print('seed')"
            ),
            save_output="input",
        ),
    )
    try:
        seed = await agent.run("Save an input fixture.")
        if extension == "arrow":
            pa = import_module("pyarrow")
            ipc = import_module("pyarrow.ipc")

            table = pa.table({"n": [1] * 10000, "v": [2] * 10000})
            stream = pa.BufferOutputStream()
            with ipc.new_file(stream, table.schema) as writer:
                writer.write_table(table)
            media = "application/vnd.apache.arrow.file"
            ref = await agent._embedded._artifact_store.commit(
                ArtifactDraft(
                    content=stream.getvalue().to_pybytes(),
                    suggested_filename="input.arrow",
                    media_type=media,
                    sensitivity=Sensitivity.RESTRICTED,
                    provenance=ArtifactProvenance(
                        authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
                    ),
                ),
                ArtifactPolicy(
                    allowed_media_types=frozenset({media}),
                    allowed_authorships=frozenset(
                        {ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS}
                    ),
                    allowed_extensions=((media, (".arrow",)),),
                    artifact_required=True,
                    max_artifact_count=1,
                    max_bytes_per_artifact=16 * 1024 * 1024,
                    max_total_bytes_per_call=16 * 1024 * 1024,
                ),
                run_id=seed.run_id,
                conversation_id=seed.conversation_id,
                call_id="typed-fixture",
                capability_id="analysis-test-fixture",
            )
        else:
            ref = seed.artifacts[0]
        await agent.close()
        caps = []
        original = NativePythonWorker.resume

        def observe(worker):
            if not (worker.scratch / "candidate.bin").exists():
                caps.append(worker.limits.cpu_seconds)
            original(worker)

        monkeypatch.setattr(NativePythonWorker, "resume", observe)
        model = ToolboxAwareMockModelProvider(
            (
                cell(
                    "import",
                    "print('executed')",
                    inputs={
                        "fixture": {"kind": "artifact", "artifact_id": ref.artifact_id}
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            )
        )
        agent = await Agent.open(
            "recovery",
            root=tmp_path,
            workspace=workspace_for(tmp_path),
            model=model,
            model_profile=_profile(model),
        )
        result = await agent.run("Use the validated input.")
        evidence = await agent.analysis_evidence(result.run_id)
        parser = next(r for r in evidence if r.evidence_id == "parser-1")
        cpu = (
            evidence_facts(parser)["usage"]["user_cpu_seconds"]
            + evidence_facts(parser)["usage"]["system_cpu_seconds"]
        )
        assert cpu > 0
        assert caps[-1] == pytest.approx(AnalysisLimits().cpu_seconds - cpu)
        input_cap = caps[-1]
        model.replace_script(
            (
                cell("control", "print('control')"),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            )
        )
        control = await agent.run("Comparable execution without input parsing.")
        assert control.kind is LoopExitKind.COMPLETED
        assert caps[-1] == pytest.approx(AnalysisLimits().cpu_seconds)
        assert caps[-1] > input_cap
    finally:
        await agent.close()


@pytest.mark.parametrize("interruption", ["page", "before_disposal", "after_disposal"])
async def test_recovery_of_more_than_one_page_is_idempotent(
    tmp_path, monkeypatch, interruption
):
    agent = await create(tmp_path)
    store = agent._embedded._store
    workers, runs = [], []
    try:
        for i in range(23):
            run = RunInput(
                id=f"run-many-{i:03}",
                agent_id=agent.id,
                message="pending generations",
                created_at=datetime.now(UTC),
                conversation_id=f"many-{i}",
            )
            runs.append(run)
            writer = RunSessionWriter(store, run)
            await writer.start()
            for j in range(12):
                worker = NativePythonWorker(durable_closure=True)
                workers.append(worker)
                await writer.record_analysis(
                    AnalysisEvidence(
                        run.id,
                        f"generation-{j:02}",
                        "generation",
                        {"status": "admitted", **worker.identity},
                    )
                )
                await worker.close()
        await agent.close()
        import daita.adapters.analytical_workspace.recovery as native_recovery

        original_page, original_dispose, original_recover = (
            SQLStateStore.pending_analysis_evidence,
            native_recovery.dispose_generation_proof,
            SQLStateStore.recover_analysis_evidence,
        )
        injected = False

        async def page(owner, agent_id, *, after=("", "")):
            nonlocal injected
            if interruption == "page" and after != ("", "") and not injected:
                injected = True
                raise OSError("page checkpoint")
            return await original_page(owner, agent_id, after=after)

        def dispose(facts):
            nonlocal injected
            if interruption == "before_disposal" and not injected:
                injected = True
                raise OSError("disposal checkpoint")
            original_dispose(facts)

        async def recover(
            owner, agent_id, expected, recovery=None, *, proof_disposed=False
        ):
            nonlocal injected
            if interruption == "after_disposal" and proof_disposed and not injected:
                injected = True
                raise OSError("disposed checkpoint")
            return await original_recover(
                owner, agent_id, expected, recovery, proof_disposed=proof_disposed
            )

        monkeypatch.setattr(SQLStateStore, "pending_analysis_evidence", page)
        monkeypatch.setattr(SQLStateStore, "recover_analysis_evidence", recover)
        monkeypatch.setattr(native_recovery, "dispose_generation_proof", dispose)
        with pytest.raises(OSError, match="checkpoint"):
            await Agent.open(
                "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
            )
        assert injected
        for _ in range(2):
            agent = await Agent.open(
                "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
            )
            for run in runs:
                records = await agent.analysis_evidence(run.id)
                assert len(records) == 12
                assert all(
                    evidence_facts(r)["proof_disposed"] is True
                    and evidence_facts(r)["recovery"]["usage"]["cpu_complete"] is True
                    for r in records
                )
                assert (
                    await terminal(agent, run.conversation_id)
                ).kind is LoopExitKind.INTERRUPTED
            await agent.close()
        assert all(
            not w.scratch.exists()
            and not Path(str(w.identity["closure_path"])).exists()
            for w in workers
        )
    finally:
        await agent.close()
        for worker in workers:
            await worker.close()
            worker.dispose_closure_evidence()


@pytest.mark.parametrize(
    "failure",
    [
        "input_validation",
        "output_validation",
        "cancel_validation",
        "evidence_write",
        "unknown_cpu",
    ],
)
async def test_parser_failures_keep_ownership_and_conserve_final_cpu(
    tmp_path, monkeypatch, failure
):
    agent = await create(
        tmp_path,
        cell(
            "seed",
            "open('input.json','w').write('{\"n\":42}')\noutputs.add('input','input.json')",
            save_output="input",
        ),
    )
    seed = await agent.run("Retain a valid input.")
    await agent.close()
    inputs = (
        {"fixture": {"kind": "artifact", "artifact_id": seed.artifacts[0].artifact_id}}
        if failure == "input_validation"
        else {}
    )
    model = ToolboxAwareMockModelProvider(
        (
            cell(
                "fault",
                "open('answer.json','w').write('{broken' if "
                + repr(failure == "output_validation")
                + " else '{\"n\":42}')\noutputs.add('answer','answer.json')",
                inputs=inputs,
            ),
            *(
                (
                    _tool(
                        "blocked-replacement",
                        "analysis_execute",
                        {
                            "code": "print('must not dispatch')",
                            "expected_state": {"generation": 1, "revision": 0},
                        },
                    ),
                )
                if failure in {"evidence_write", "unknown_cpu"}
                else ()
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    agent = await Agent.open(
        "recovery",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=model,
        model_profile=_profile(model),
    )
    workers, run_ids = [], []
    original_start, original_record, original_execute, original_receive = (
        NativePythonWorker.start,
        SQLStateStore.record_analysis_evidence,
        NativePythonWorker.execute,
        NativePythonWorker.receive,
    )
    entered = asyncio.Event()
    injected = False

    async def start(worker):
        workers.append(worker)
        if (
            failure == "input_validation"
            and (worker.scratch / "candidate.bin").exists()
        ):
            (worker.scratch / "candidate.bin").write_text("{broken")
        await original_start(worker)

    async def execute(worker, *args, **kwargs):
        if (worker.scratch / "candidate.bin").exists():
            worker._testing_validating = True
        return await original_execute(worker, *args, **kwargs)

    async def receive(worker, deadline):
        if failure == "cancel_validation" and getattr(
            worker, "_testing_validating", False
        ):
            entered.set()
            await asyncio.Event().wait()
        return await original_receive(worker, deadline)

    async def record(store, agent_id, evidence):
        nonlocal injected
        run_ids.append(evidence.run_id)
        if (
            failure == "evidence_write"
            and evidence.evidence_id == "parser-1"
            and evidence.facts["status"] == "closed"
            and not injected
        ):
            injected = True
            raise OSError("terminal parser evidence write failed")
        await original_record(store, agent_id, evidence)

    monkeypatch.setattr(NativePythonWorker, "start", start)
    monkeypatch.setattr(NativePythonWorker, "execute", execute)
    monkeypatch.setattr(NativePythonWorker, "receive", receive)
    monkeypatch.setattr(SQLStateStore, "record_analysis_evidence", record)
    if failure == "unknown_cpu":
        original_close = NativePythonWorker.close

        async def unknown(worker):
            cleanup = await original_close(worker)
            if worker.identity["scratch"] in [str(w.scratch) for w in workers[1:]]:
                worker.usage.update(
                    cpu_complete=False, user_cpu_seconds=None, system_cpu_seconds=None
                )
            return cleanup

        monkeypatch.setattr(NativePythonWorker, "close", unknown)
    task = asyncio.create_task(agent.run("Exercise parser resource settlement."))
    try:
        if failure == "cancel_validation":
            await asyncio.wait_for(entered.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.kind is (
                LoopExitKind.FAILED
                if failure in {"evidence_write", "unknown_cpu"}
                else LoopExitKind.COMPLETED
            )
        records = await agent.analysis_evidence(run_ids[-1])
        generations = [r for r in records if r.kind == "generation"]
        assert len(generations) >= 2
        if failure in {"evidence_write", "unknown_cpu"}:
            assert len(workers) == 2  # No replacement allocation or user dispatch.
        for item in generations:
            assert evidence_facts(item)["cleanup"]["process_reaped"] is True
            assert evidence_facts(item)["cleanup"]["descriptors_closed"] is True
            assert evidence_facts(item)["cleanup"]["scratch_deleted"] is True
            with pytest.raises(ProcessLookupError):
                os.kill(evidence_facts(item)["pid"], 0)
        summary = await usage(agent, run_ids[-1])
        if failure == "unknown_cpu":
            assert summary["user_cpu_seconds"] is None
            assert (
                next(r for r in generations if r.evidence_id == "parser-1").facts[
                    "status"
                ]
                == "cleanup_failed"
            )
        else:
            assert summary["user_cpu_seconds"] == pytest.approx(
                sum(w.usage["user_cpu_seconds"] for w in workers)
            )
        await agent.close()
        agent = await Agent.open(
            "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        assert all(
            not w.scratch.exists()
            and not Path(str(w.identity["closure_path"])).exists()
            for w in workers
        )
        if failure == "unknown_cpu":
            recovered = await usage(agent, run_ids[-1])
            assert recovered["user_cpu_seconds"] == pytest.approx(
                sum(
                    evidence_facts(r).get("recovery", evidence_facts(r))["usage"][
                        "user_cpu_seconds"
                    ]
                    for r in await agent.analysis_evidence(run_ids[-1])
                    if r.kind == "generation"
                )
            )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()
        for worker in workers:
            worker.dispose_closure_evidence()


async def test_multiple_parsers_and_replacement_preserve_cumulative_cpu(
    tmp_path, monkeypatch
):
    workers, caps = [], []
    original_start, original_resume = (
        NativePythonWorker.start,
        NativePythonWorker.resume,
    )

    async def start(worker):
        workers.append(worker)
        await original_start(worker)

    def resume(worker):
        caps.append((worker, worker.limits.cpu_seconds))
        original_resume(worker)

    monkeypatch.setattr(NativePythonWorker, "start", start)
    monkeypatch.setattr(NativePythonWorker, "resume", resume)
    agent = await create(
        tmp_path,
        cell(
            "outputs",
            "open('first.json','w').write('{\"n\":42}')\nopen('second.csv','w').write('n\\n42\\n')\noutputs.add('first','first.json')\noutputs.add('second','second.csv')",
        ),
        _tool(
            "timeout",
            "analysis_execute",
            {
                "code": "lost = 19\nwhile True: pass",
                "expected_state": {"generation": 1, "revision": 1},
                "timeout_seconds": 0.15,
            },
        ),
        _tool(
            "replacement",
            "analysis_execute",
            {
                "code": "assert 'lost' not in globals()\nprint(42)",
                "expected_state": {"generation": 1, "revision": 0},
            },
        ),
    )
    try:
        result = await agent.run(
            "Validate outputs, interrupt and explicitly replace the interpreter."
        )
        assert result.kind is LoopExitKind.COMPLETED
        transcript = await agent.transcript(result.run_id)
        cells = [
            block
            for call, block in transcript.tool_pairs
            if call.name == "analysis_execute"
        ]
        assert len(cells) == 3
        assert cells[1] is not None and cells[2] is not None
        assert json.loads(canonical_json(cells[1].output))["data"]["state_lost"] is True
        assert json.loads(canonical_json(cells[2].output))["data"]["stdout"] == "42\n"
        records = await agent.analysis_evidence(result.run_id)
        generations = [r for r in records if r.kind == "generation"]
        assert {r.evidence_id for r in generations} == {
            "generation-1",
            "generation-2",
            "parser-1",
            "parser-2",
        }
        main = next(
            w
            for w in workers
            if str(w.scratch)
            == next(r for r in generations if r.evidence_id == "generation-2").facts[
                "scratch"
            ]
        )
        previous = sum(
            w.usage["user_cpu_seconds"] + w.usage["system_cpu_seconds"]
            for w in workers
            if w is not main
        )
        assert next(cap for worker, cap in caps if worker is main) == pytest.approx(
            AnalysisLimits().cpu_seconds - previous
        )
        summary = await usage(agent, result.run_id)
        assert summary["user_cpu_seconds"] + summary[
            "system_cpu_seconds"
        ] == pytest.approx(
            previous + main.usage["user_cpu_seconds"] + main.usage["system_cpu_seconds"]
        )
        assert all(evidence_facts(r)["proof_disposed"] is True for r in generations)
        for record in generations:
            generation = evidence_facts(record)
            cleanup = generation["cleanup"]
            assert cleanup["process_reaped"] is True
            assert cleanup["scratch_deleted"] is True
            assert cleanup["remaining_bytes"] == cleanup["remaining_files"] == 0
            with pytest.raises(ProcessLookupError):
                os.kill(generation["pid"], 0)
        assert all(
            not w.scratch.exists()
            and not Path(str(w.identity["closure_path"])).exists()
            for w in workers
        )
    finally:
        await agent.close()


async def test_parser_exhaustion_denies_user_code_dispatch(tmp_path):
    from daita import AgentConfig

    agent = await create(
        tmp_path,
        cell(
            "seed",
            "open('input.csv','w').write('n,v\\n'+'1,2\\n'*750000)\noutputs.add('input','input.csv')",
            save_output="input",
        ),
    )
    seed = await agent.run("Retain a real large CSV fixture.")
    await agent.close()
    model = ToolboxAwareMockModelProvider(
        (
            cell(
                "input",
                "print('must not dispatch')",
                inputs={
                    "fixture": {
                        "kind": "artifact",
                        "artifact_id": seed.artifacts[0].artifact_id,
                    }
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    agent = await Agent.open(
        "recovery",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=model,
        model_profile=_profile(model),
        config=AgentConfig(analysis_limits=AnalysisLimits(cpu_seconds=0.07)),
    )
    try:
        result = await agent.run("Parsing consumes the finite cumulative allowance.")
        records = await agent.analysis_evidence(result.run_id)
        assert any(r.evidence_id == "parser-1" for r in records), [
            evidence_facts(r) for r in records
        ]
        execution = next(r for r in records if r.kind == "cell")
        assert evidence_facts(execution)["status"] == "interrupted"
        assert result.kind is LoopExitKind.FAILED
        measured = await usage(agent, result.run_id)
        assert measured["user_cpu_seconds"] + measured["system_cpu_seconds"] >= 0.07
        assert all(
            evidence_facts(r)["cleanup"]["process_reaped"] is True
            and evidence_facts(r)["cleanup"]["scratch_deleted"] is True
            for r in records
            if r.kind == "generation"
        )
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "boundary", ["main_admission", "parser_admission", "main_start", "parser_start"]
)
async def test_failed_generation_admission_and_start_settle_native_ownership(
    tmp_path, monkeypatch, boundary
):
    original_record, original_spawn = (
        SQLStateStore.record_analysis_evidence,
        subprocess.Popen,
    )
    rejected, allocations = False, []

    async def record(store, agent_id, evidence):
        nonlocal rejected
        if evidence.kind == "generation" and evidence.facts["status"] == "admitted":
            allocations.append(evidence)
            parser = evidence.evidence_id.startswith("parser-")
            if (
                boundary == ("parser_admission" if parser else "main_admission")
                and not rejected
            ):
                rejected = True
                raise OSError("generation admission failed")
        await original_record(store, agent_id, evidence)

    def spawn(*args, **kwargs):
        nonlocal rejected
        parser = (Path(kwargs.get("cwd", "/")) / "candidate.bin").exists()
        if boundary == ("parser_start" if parser else "main_start") and not rejected:
            rejected = True
            raise OSError("guardian launch failed")
        return original_spawn(*args, **kwargs)

    monkeypatch.setattr(SQLStateStore, "record_analysis_evidence", record)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    agent = await create(
        tmp_path,
        cell(
            "fault",
            "open('answer.txt','w').write('42')\noutputs.add('answer','answer.txt')",
        ),
    )
    try:
        result = await agent.run("Exercise pre-dispatch settlement failures.")
        assert rejected
        records = await agent.analysis_evidence(result.run_id)
        generations = [r for r in records if r.kind == "generation"]
        assert generations and all(
            evidence_facts(r)["proof_disposed"] is True for r in generations
        )
        target = next(
            r
            for r in generations
            if r.evidence_id.startswith("parser-") == boundary.startswith("parser")
        )
        assert evidence_facts(target)["cleanup"]["process_spawned"] is False
        assert evidence_facts(target)["usage"]["cpu_complete"] is True
        assert evidence_facts(target)["usage"]["user_cpu_seconds"] == 0
        assert all(
            not Path(str(evidence_facts(r)["scratch"])).exists()
            and not Path(str(evidence_facts(r)["closure_path"])).exists()
            for r in allocations
        )
    finally:
        await agent.close()


async def test_uncertain_generation_blocks_open_before_run_finalization(tmp_path):
    import sqlite3

    agent = await create(tmp_path)
    run = RunInput(
        id="run-uncertain",
        agent_id=agent.id,
        message="fixture",
        created_at=datetime.now(UTC),
        conversation_id="uncertain",
    )
    writer = RunSessionWriter(agent._embedded._store, run)
    await writer.start()
    worker = NativePythonWorker(durable_closure=True)
    path = Path(str(worker.identity["closure_path"]))
    try:
        await writer.record_analysis(
            AnalysisEvidence(
                run.id,
                "generation-1",
                "generation",
                {"status": "admitted", **worker.identity},
            )
        )
        await worker.start()
        await worker.close()
        raw = path.read_bytes()
        path.write_text(
            json.dumps(
                {
                    "kind": "launch_committed",
                    "closure_nonce": worker.identity["closure_nonce"],
                }
            )
        )
        home = agent.home
        await agent.close()
        with pytest.raises(ValueError, match="closure evidence"):
            await Agent.open(
                "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
            )
        with sqlite3.connect(home / "state.db") as connection:
            assert (
                connection.execute(
                    "SELECT result FROM runs WHERE id = ?", (run.id,)
                ).fetchone()[0]
                is None
            )
        assert path.exists()
        path.write_bytes(raw)
        agent = await Agent.open(
            "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        assert (
            await terminal(agent, run.conversation_id)
        ).kind is LoopExitKind.INTERRUPTED
    finally:
        await agent.close()
        await worker.close()
        worker.dispose_closure_evidence()
