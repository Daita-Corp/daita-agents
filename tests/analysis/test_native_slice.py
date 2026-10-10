"""Real native computation through the shared public foreground path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.llm.models import FinishReason, ModelResponse
from tests.artifacts._public_surface_support import _profile, _tool
from tests.support.analysis import facts as evidence_facts
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for


async def test_public_agent_runs_python_and_dependent_cell(tmp_path):
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "cell-one",
                "analysis_execute",
                {
                    "code": "answer = 6 * 7; print(answer)",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            _tool(
                "cell-two",
                "analysis_execute",
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
            if call.name == "analysis_execute"
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


async def test_explicit_analysis_save_commits_one_artifact_and_persists_provenance(
    tmp_path,
):
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "cell-save",
                "analysis_execute",
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


async def test_real_python_broker_reads_and_denies_recursive_calls(tmp_path):
    workspace = workspace_for(tmp_path)
    (workspace.root / "values.txt").write_text("17,25")
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "cell-read",
                "analysis_execute",
                {
                    "code": "r = tools.call('file_read', {'path': 'values.txt'})\nprint(r)\n"
                    "assert not r['is_error']\n"
                    "denied = tools.call('analysis_execute', {})\nassert denied['is_error']",
                    "expected_state": {"generation": 0, "revision": 0},
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Read complete"),
        )
    )
    agent = await Agent.create(
        "broker-slice",
        root=tmp_path,
        model=model,
        model_profile=_profile(model),
        workspace=workspace,
    )
    try:
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
            if call.name == "analysis_execute"
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
