"""Authenticated snapshots, candidate rejection and native boundary fixtures."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from importlib import import_module
from pathlib import Path

import pytest

from daita._json import canonical_json
from daita.adapters.analytical_workspace import NativePythonWorker
from daita.artifacts.models import ArtifactAuthorship, ArtifactDraft, ArtifactProvenance
from daita.capabilities import ArtifactPolicy
from daita.catalog.models import Sensitivity
from daita.llm.models import FinishReason, ModelRequest, ModelResponse, ToolResultBlock
from daita.observation import AgentEvent
from tests.artifacts._public_surface_support import _tool
from tests.live.analysis.test_native_faults import (
    capture_run,
    make_agent,
    wait_for_record,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider


async def test_native_runtime_cannot_be_mutated_through_a_scratch_hardlink(
    analysis_report,
):
    canary = Path(sys.prefix) / f"analysis-readonly-canary-{uuid.uuid4().hex}"
    canary.write_text("immutable runtime fixture")
    worker = NativePythonWorker()
    try:
        await worker.start()
        result = await worker.execute(
            f"import os\nassert open({str(canary)!r}).read()=='immutable runtime fixture'\ntry: os.link({str(canary)!r}, 'alias.txt')\nexcept PermissionError: print('hardlink denied')\nelse:\n open('alias.txt','w').write('mutated')\n raise AssertionError('runtime hardlink escaped')",
            deadline=asyncio.get_running_loop().time() + 5,
        )
        assert result["status"] == "success" and result["stdout"] == "hardlink denied\n"
        assert canary.read_text() == "immutable runtime fixture"
    finally:
        analysis_report["cleanup"] = await worker.close()
        analysis_report["measurements"] = dict(worker.usage)
        canary.unlink()
    assert analysis_report["cleanup"]["process_reaped"] is True
    assert analysis_report["cleanup"]["scratch_deleted"] is True


async def test_partial_broker_read_retains_coverage_and_contract_after_history_clear(
    tmp_path, analysis_report
):
    from tests.support.workspace import workspace_for

    workspace = workspace_for(tmp_path)
    (workspace.root / "partial.txt").write_bytes(b"x" * 81920)
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "partial",
                "analysis_execute",
                {
                    "code": "import json\nr=tools.call('file_read', {'path':'partial.txt'})\nassert not r['is_error']\nd=r['output']['data']\nsummary={'captured_bytes':len(d['content'].encode()), 'complete':d['complete']}\nprint(summary)\nopen('coverage.json','w').write(json.dumps(summary))\noutputs.add('coverage','coverage.json')",
                    "expected_state": {"generation": 0, "revision": 0},
                    "save_output": "coverage",
                },
            ),
        ),
    )
    try:
        result = await agent.run("Retain a bounded read's actual coverage.")
        records = await capture_run(agent, result.run_id, analysis_report)
        assert len(result.artifacts) == 1
        artifact = result.artifacts[0]

        parsers = [
            record
            for record in records
            if record.kind == "generation"
            and record.facts.get("role") == "output_parser"
        ]
        assert (
            sum(record.facts["usage"]["validation_input_bytes"] for record in parsers)
            == artifact.byte_size
        )
        summary = await agent.analysis_usage(result.run_id)
        assert (
            summary is not None
            and summary["validation_input_bytes"] == artifact.byte_size
        )
        assert json.loads(
            (await agent.read_artifact(artifact.artifact_id)).content
        ) == {"captured_bytes": 49152, "complete": False}
        evidence = await agent.artifact_computation_evidence(artifact.artifact_id)
        child = evidence["children"][0]
        assert child["coverage"]["complete"] is False
        assert child["capability_id"] == "data.local_file.read"
        assert child["contract_digest"].startswith("sha256:")
        assert child["input_schema_digest"].startswith("sha256:")
        issued = next(record for record in records if record.kind == "child")
        assert child["evidence_id"] == issued.evidence_id
        assert child["result_sha256"] == issued.facts["result_digest"]
        await agent.clear_conversations()
        retained = await agent.artifact_computation_evidence(artifact.artifact_id)
        assert retained == evidence
        analysis_report["retained_coverage"] = dict(child)
    finally:
        await agent.close()


async def test_broker_artifact_reads_and_authenticated_child_input_use_existing_owner(
    tmp_path, analysis_report
):
    class ChildInputProvider(ToolboxAwareMockModelProvider):
        bound_child = False

        async def generate(self, request: ModelRequest) -> ModelResponse:
            for message in reversed(request.messages):
                for block in message.content:
                    if (
                        isinstance(block, ToolResultBlock)
                        and block.call_id == "read-preview"
                        and not self.bound_child
                    ):
                        self.bound_child = True
                        self.replace_script(
                            (
                                consume(block),
                                ModelResponse(
                                    finish_reason=FinishReason.STOP, text="read"
                                ),
                            )
                        )
            return await super().generate(request)

    models: list[ToolboxAwareMockModelProvider] = []
    producer = _tool(
        "produce",
        "analysis_execute",
        {
            "code": "open('answer.txt','w').write('42')\noutputs.add('answer','answer.txt')",
            "expected_state": {"generation": 0, "revision": 0},
            "save_output": "answer",
        },
    )
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "produce",
                "analysis_execute",
                {
                    "code": "open('answer.txt','w').write('42')\noutputs.add('answer','answer.txt')",
                    "expected_state": {"generation": 0, "revision": 0},
                    "save_output": "answer",
                },
            ),
        ),
        model_holder=models,
        model_override=ChildInputProvider(
            (producer, ModelResponse(finish_reason=FinishReason.STOP, text="saved"))
        ),
    )
    try:
        seed = await agent.run("Retain a text result.")
        artifact = seed.artifacts[0]

        def consume(result: ToolResultBlock) -> ModelResponse:
            visible = json.loads(canonical_json(result.output))
            evidence_id = visible["data"]["child_calls"][0]["evidence_id"]
            return _tool(
                "use-preview",
                "analysis_execute",
                {
                    "code": "print(inputs.get('preview')['data']['text'])",
                    "inputs": {
                        "preview": {"kind": "tool_result", "call_id": evidence_id}
                    },
                    "expected_state": {"generation": 1, "revision": 1},
                },
            )

        models[0].replace_script(
            (
                _tool(
                    "read-preview",
                    "analysis_execute",
                    {
                        "code": f"r=tools.call('artifact_read', {{'artifact_id':{artifact.artifact_id!r}}})\nassert not r['is_error']\nprint(r['output']['data']['text'])\nlisting=tools.call('artifact_list', {{}})\nassert not listing['is_error']\ndenied=tools.call('artifact_create_document', {{}})\nassert denied['is_error']",
                        "expected_state": {"generation": 0, "revision": 0},
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="read"),
            )
        )
        result = await agent.run(
            "Read the known artifact through the broker and reuse its exact child evidence."
        )
        records = await capture_run(agent, result.run_id, analysis_report)
        cells = [
            block
            for call, block in (await agent.transcript(result.run_id)).tool_pairs
            if call.name == "analysis_execute"
        ]
        assert len(cells) == 2
        assert all(block.output["data"]["stdout"] == "42\n" for block in cells)
        children = [record for record in records if record.kind == "child"]
        assert len(children) == 3
        assert [
            record.facts["status"]
            for record in sorted(children, key=lambda record: record.facts["sequence"])
        ] == ["succeeded", "succeeded", "failed"]
        assert sum(record.facts["dispatched"] is True for record in children) == 2
        assert not result.artifacts
    finally:
        await agent.close()


async def test_arrow_input_preserves_schema_decimal_null_integer_timezone_duplicates(
    tmp_path, analysis_report
):
    pa = import_module("pyarrow")
    ipc = import_module("pyarrow.ipc")

    fixture = pa.table(
        {
            "identifier": pa.array([2**53 + 1, 2**53 + 1, None], type=pa.int64()),
            "amount": pa.array(
                [Decimal("1.25"), Decimal("1.25"), None], type=pa.decimal128(18, 2)
            ),
            "time": pa.array(
                [datetime(2026, 1, 1, tzinfo=timezone.utc), None, None],
                type=pa.timestamp("us", "UTC"),
            ),
        }
    )
    stream = pa.BufferOutputStream()
    with ipc.new_file(stream, fixture.schema) as writer:
        writer.write_table(fixture)
    models: list[ToolboxAwareMockModelProvider] = []
    agent = await make_agent(tmp_path, (), model_holder=models)
    try:
        seed = await agent.run("Admit a disposable fixture home.")
        media = "application/vnd.apache.arrow.file"
        artifact = await agent._embedded._artifact_store.commit(
            ArtifactDraft(
                content=stream.getvalue().to_pybytes(),
                suggested_filename="fixture.arrow",
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
        models[0].replace_script(
            (
                _tool(
                    "arrow-input",
                    "analysis_execute",
                    {
                        "code": "import pyarrow.ipc as ipc\nwith ipc.open_file(inputs.path('fixture')) as reader: t=reader.read_all()\nprint(t.schema)\nprint(t.to_pylist())",
                        "inputs": {
                            "fixture": {
                                "kind": "artifact",
                                "artifact_id": artifact.artifact_id,
                            }
                        },
                        "expected_state": {"generation": 0, "revision": 0},
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="read"),
            )
        )
        result = await agent.run("Read the exact Arrow snapshot.")
        records = await capture_run(agent, result.run_id, analysis_report)
        cell = next(record for record in records if record.kind == "cell")
        output = cell.facts["result"]
        assert output["status"] == "success"
        assert str(2**53 + 1) in output["stdout"]
        assert output["stdout"].count("Decimal('1.25')") == 2
        assert "decimal128(18, 2)" in output["stdout"]
        assert "timestamp[us, tz=UTC]" in output["stdout"]
        assert "None" in output["stdout"]
        assert result.sensitivity.value == "restricted"
        assert not result.artifacts
        analysis_report["input_oracle"] = {
            "schema": str(fixture.schema),
            "retained_sha256": artifact.sha256,
            "rows": 3,
        }
        assert artifact.sha256 in canonical_json(cell.facts)
        assert output["usage"]["input_bytes"] == artifact.byte_size
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "failure", ["symlink", "traversal", "malformed_json", "forged_input", "stale_state"]
)
async def test_invalid_inputs_and_candidates_never_publish(
    tmp_path, failure, analysis_report
):
    arguments: dict[str, object] = {
        "code": "print(42)",
        "expected_state": {"generation": 0, "revision": 0},
        "save_output": "bad",
    }
    if failure == "symlink":
        arguments["code"] = (
            "import os\nos.symlink('/etc/passwd','bad.json')\noutputs.add('bad','bad.json')"
        )
    elif failure == "traversal":
        arguments["code"] = "outputs.add('bad','../bad.json')"
    elif failure == "malformed_json":
        arguments["code"] = (
            "open('bad.json','w').write('{broken')\noutputs.add('bad','bad.json')"
        )
    elif failure == "forged_input":
        arguments["inputs"] = {
            "bad": {"kind": "tool_result", "call_id": "unissued-result"}
        }
    else:
        arguments["expected_state"] = {"generation": 9, "revision": 99}
    agent = await make_agent(
        tmp_path, (_tool("invalid", "analysis_execute", arguments),)
    )
    try:
        result = await agent.run("Reject invalid candidate and input references.")
        records = await agent.analysis_evidence(result.run_id)
        analysis_report["measurements"] = [
            {"kind": record.kind, "facts": record.facts} for record in records
        ]
        assert not result.artifacts and await agent.list_artifacts() == ()
        pair = next(
            (call, block)
            for call, block in (await agent.transcript(result.run_id)).tool_pairs
            if call.name == "analysis_execute"
        )
        if failure == "stale_state":
            assert (
                pair[1].is_error
                and pair[1].output["error"]["code"] == "analysis_stale_state"
            )
            assert not any(record.kind == "generation" for record in records)
        else:
            await capture_run(agent, result.run_id, analysis_report)
            assert pair[1].output["data"]["state_lost"] is True
    finally:
        await agent.close()


async def test_saved_input_new_run_and_active_artifact_deletion(
    tmp_path, analysis_report
):
    events: list[AgentEvent] = []
    models: list[ToolboxAwareMockModelProvider] = []
    agent = await make_agent(
        tmp_path,
        (
            _tool(
                "produce",
                "analysis_execute",
                {
                    "code": "open('retained.json','w').write('{\"value\":42}')\noutputs.add('retained','retained.json')",
                    "expected_state": {"generation": 0, "revision": 0},
                    "save_output": "retained",
                },
            ),
            _tool(
                "produce-other",
                "analysis_execute",
                {
                    "code": "open('other.json','w').write('{\"value\":43}')\noutputs.add('other','other.json')",
                    "expected_state": {"generation": 1, "revision": 1},
                    "save_output": "other",
                },
            ),
        ),
        model_holder=models,
        observer=events.append,
    )
    task = None
    try:
        saved = await agent.run("Retain the exact result.")
        artifact, other = saved.artifacts
        evidence = await agent.artifact_computation_evidence(artifact.artifact_id)
        assert evidence["output_sha256"] == artifact.sha256
        await agent.clear_conversations()
        assert (
            await agent.artifact_computation_evidence(artifact.artifact_id) == evidence
        )
        models[0].replace_script(
            (
                _tool(
                    "consume",
                    "analysis_execute",
                    {
                        "code": "old = inputs.get('retained')['value']\nprint(old)",
                        "inputs": {
                            "retained": {
                                "kind": "artifact",
                                "artifact_id": artifact.artifact_id,
                            }
                        },
                        "expected_state": {"generation": 0, "revision": 0},
                    },
                ),
                _tool(
                    "replace-input-name",
                    "analysis_execute",
                    {
                        "code": "assert inputs.get('retained')['value']==43\nprint(old)\nwhile True: pass",
                        "inputs": {
                            "retained": {
                                "kind": "artifact",
                                "artifact_id": other.artifact_id,
                            }
                        },
                        "expected_state": {"generation": 1, "revision": 1},
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="input deleted"),
            )
        )
        events.clear()
        task = asyncio.create_task(
            agent.run("Import the saved input in a fresh interpreter.")
        )
        run_id, _ = await wait_for_record(agent, events, "generation", "running")
        async with asyncio.timeout(10):
            while True:
                records = await agent.analysis_evidence(run_id)
                if (
                    sum(
                        record.kind == "generation"
                        and record.facts.get("role") == "output_parser"
                        and record.facts.get("status") == "closed"
                        for record in records
                    )
                    >= 2
                ):
                    break
                await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)
        started = asyncio.get_running_loop().time()
        assert await agent.delete_artifact(artifact.artifact_id)
        result = await asyncio.wait_for(task, 5)
        records = await capture_run(agent, run_id, analysis_report)
        assert not result.artifacts
        assert (
            next(
                record
                for record in records
                if record.evidence_id == "replace-input-name"
            ).facts["status"]
            == "interrupted"
        )
        analysis_report["deletion_revocation_seconds"] = (
            asyncio.get_running_loop().time() - started
        )
        assert analysis_report["deletion_revocation_seconds"] < 5
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_native_runtime_sibling_metadata_signal_ipc_and_environment_denials(
    tmp_path, analysis_report
):
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("private fixture")
    worker = NativePythonWorker()
    host_pid = os.getpid()
    code = f"""
import os, socket, sys, ctypes
assert 'OPENAI_API_KEY' not in os.environ
assert 'ANTHROPIC_API_KEY' not in os.environ
actions=[lambda: os.stat({str(sibling)!r}), lambda: open(sys.prefix+'/mutated-fixture','w'),
 lambda: os.kill({host_pid},0), lambda: socket.socket(socket.AF_UNIX).connect({str(tmp_path / 'ipc')!r})]
for action in actions:
 try: action()
 except OSError: print('denied')
 else: raise AssertionError('native boundary failed')
library=ctypes.CDLL('/usr/lib/libproc.dylib')
buffer=ctypes.create_string_buffer(4096)
library.proc_pidinfo.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int]
assert library.proc_pidinfo({host_pid},4,0,buffer,len(buffer))==0
"""
    try:
        await worker.start()
        result = await worker.execute(
            code, deadline=asyncio.get_running_loop().time() + 10
        )
        analysis_report["cell_result"] = result
        assert result["status"] == "success", result
        assert result["stdout"] == "denied\ndenied\ndenied\ndenied\n"
    finally:
        cleanup = await worker.close()
        analysis_report.update(measurements=dict(worker.usage), cleanup=cleanup)
    assert cleanup["process_reaped"] and cleanup["scratch_deleted"]
