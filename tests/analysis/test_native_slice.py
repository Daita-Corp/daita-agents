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
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    ScheduledRoutineDraft,
)
from daita._json import canonical_json
from daita.artifacts.models import ArtifactAuthorship, ArtifactError
from daita.distribution import ArtifactRequirement, OutcomeState
from daita.llm.models import FinishReason, ModelResponse, ModelSensitivity, ModelUsage
from daita.llm.pricing import CostEstimate
from daita.loop.models import RunOrigin
from tests.artifacts._public_surface_support import _profile, _tool
from tests.support.analysis import facts as evidence_facts
from tests.support.distribution import no_artifact_outcome_contract
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


@pytest.mark.parametrize("batch", (False, True))
async def test_scheduled_analysis_reads_exact_scope_and_delivers_artifact(
    tmp_path, batch
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
        model_profile=_profile(model),
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
                        {"tool_names": ["data_query", "analysis_execute"]},
                    ),
                    usage=usage,
                ),
                replace(
                    _tool(
                        "retained-input",
                        "analysis_execute",
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
                        "analysis_execute",
                        {
                            "code": "import json\n"
                            "assert tools.call('file_read', {'path':'values.sqlite'})['is_error']\n"
                            f"assert tools.call('data_query', {{'source_id':{excluded.id!r}, 'resource_ids':[{excluded_resource.id!r}], 'sql':'SELECT n FROM secret'}})['is_error']\n"
                            f"r = tools.call('data_query', {{'source_id':{source.id!r}, 'resource_ids':[{resource.id!r}], 'sql':'SELECT n FROM numbers'}})\n"
                            "assert not r['is_error'], r\n"
                            "assert r['output']['data']['rows'] == [{'n':42}], r\n"
                            "open('answer.json','w').write(json.dumps({'n':42}))\n"
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
        now = datetime.now(UTC)
        outcome = replace(
            no_artifact_outcome_contract(),
            maximum_total_artifact_bytes=4096,
            artifact_requirements=(
                ArtifactRequirement(
                    required=True,
                    minimum_count=2 if batch else 1,
                    maximum_count=2 if batch else 1,
                    allowed_media_types=("application/json", "text/x-python"),
                    allowed_authorships=(ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,),
                    allowed_producer_capability_ids=("analysis.execute",),
                    maximum_artifact_bytes=4096,
                    maximum_total_bytes=4096,
                    maximum_sensitivity=ModelSensitivity.INTERNAL,
                ),
            ),
        )
        destinations = await agent.distribution_destinations(origin.conversation_id)
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Scoped Python report",
            authorized_instruction="Read only the granted numbers and produce a JSON artifact using Python.",
            schedule=IntervalSchedule(3600, now + timedelta(hours=1)),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(source.id,),
            allowed_resource_ids=(resource.id,),
            allowed_connector_binding_ids=(),
            allowed_capability_ids=("data.query", "analysis.execute"),
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
            outcome_contract=outcome,
            distribution_destination_id=destinations[0].destination_id,
            eligible_model_routes=(model.provider_id,),
            per_run_max_tokens=50000,
            per_run_max_cost_usd=Decimal("0.50"),
            cumulative_max_tokens=50000,
            cumulative_max_cost_usd=Decimal("0.50"),
            cumulative_max_attempts=1,
            cumulative_max_occurrences=1,
            maximum_consecutive_failures=1,
            expires_at=now + timedelta(days=1),
        )
        stored = await agent.create_routine(await agent.propose_routine(draft))
        await agent.run_routine_now(
            stored.routine_id, expected_revision=stored.revision
        )
        async with asyncio.timeout(30):
            while not (
                inbox := await agent.inbox(conversation_id=origin.conversation_id)
            ):
                await asyncio.sleep(0.01)
        assert len(inbox) == 1
        assert inbox[0].conclusion_state is OutcomeState.SUCCEEDED, (
            inbox[0].failure_code,
            [
                (run.result, run.transcript.tool_pairs)
                for run in await agent.conversation_runs(origin.conversation_id)
            ],
        )
        run_id = inbox[0].resulting_run_id
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
        assert len(children) == 3
        assert [r["dispatched"] for r in children] == [False, False, True]
        ref = next(
            ref
            for ref in inbox[0].artifact_references
            if ref.media_type == "application/json"
        )
        assert len(inbox[0].artifact_references) == (2 if batch else 1)
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


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
async def test_nonstandard_json_constants_never_publish_artifacts(tmp_path, constant):
    payload = '{"value":' + constant + "}"
    model = ToolboxAwareMockModelProvider(
        (
            _tool(
                "nonstandard-json",
                "analysis_execute",
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
                "analysis_execute",
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
