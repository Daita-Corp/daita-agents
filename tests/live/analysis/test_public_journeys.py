"""Paid-model journeys through production public foreground owners."""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from daita import Agent, AgentConfig, LocalWorkspace
from daita._json import canonical_json
from daita.loop.models import LoopLimits
from daita.tui.app import DaitaApp

pytestmark = pytest.mark.requires_llm


def computed_sum(stdout):
    for line in reversed(str(stdout).strip().splitlines()):
        try:
            value = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, dict) and "sum" in value:
            return value["sum"]
    return None


async def retain_evidence(agent, result, report):
    transcript = await agent.transcript(result.run_id)
    records = await agent.analysis_evidence(result.run_id)
    report.update(
        {
            "run_id": result.run_id,
            "terminal_kind": result.kind.value,
            "terminal_reason": result.reason,
            "final_text": result.final_text,
            "model_usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cost_usd": (
                    None
                    if result.usage.cost_estimate.amount_usd is None
                    else str(result.usage.cost_estimate.amount_usd)
                ),
            },
            "calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "output": None if output is None else output.output,
                }
                for call, output in transcript.tool_pairs
            ],
            "measurements": [
                {"id": record.evidence_id, "kind": record.kind, "facts": record.facts}
                for record in records
            ],
        }
    )
    generations = [record for record in records if record.kind == "generation"]
    report["cleanup"] = [record.facts.get("cleanup") for record in generations]
    assert result.kind.value == "completed", report
    assert result.usage.cost_estimate.amount_usd is not None
    assert generations
    for record in generations:
        assert record.facts["cleanup"]["process_reaped"] is True
        assert record.facts["cleanup"]["scratch_deleted"] is True
        if "scratch" in record.facts:
            assert not Path(record.facts["scratch"]).exists()
    return transcript, records


async def test_real_model_python_broker_duckdb_chart_dataset_and_artifact_lifecycle(
    tmp_path, paid_case, analysis_report
):
    profile, provider, cost = paid_case
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sales.csv").write_text(
        "region,units,unit_price,currency\nA,2,10,USD\nB,3,4.5,EUR\nA,5,7,USD\n"
    )
    (workspace / "rates.csv").write_text("currency,usd_rate\nUSD,1\nEUR,1.2\n")
    local = LocalWorkspace(workspace)
    agent = await Agent.create(
        "live-api-analysis",
        root=tmp_path / "homes",
        model=provider,
        model_profile=profile,
        config=AgentConfig(
            limits=LoopLimits(
                max_estimated_cost_usd=cost,
                max_steps=24,
                max_total_tokens=100000,
                max_wall_time_seconds=300,
            )
        ),
        workspace=local,
    )
    try:
        result = await agent.run(
            "Analyze sales.csv and rates.csv with the local Python tool. Read both files programmatically using tools.call('file_read', {'path': ...}) inside Python. "
            "Combine their rows using DuckDB SQL and compute total USD sales (units * unit_price * usd_rate). "
            "Create a Matplotlib PNG chart and a JSON dataset containing total_usd and by_region, register both named output candidates, "
            "and explicitly save both as two artifacts through separate analysis calls. Do not deliver them to external directories. Report the total."
        )
        transcript, records = await retain_evidence(agent, result, analysis_report)
        children = [
            record
            for record in records
            if record.kind == "child" and record.facts["status"] == "succeeded"
        ]
        assert sum(record.facts["tool_name"] == "file_read" for record in children) >= 2
        assert len(result.artifacts) == 2
        assert any(
            "duckdb" in str(call.arguments.get("code", "")).lower()
            for call, _ in transcript.tool_pairs
        )
        dataset = next(
            ref for ref in result.artifacts if ref.media_type == "application/json"
        )
        chart = next(ref for ref in result.artifacts if ref.media_type == "image/png")
        data = json.loads((await agent.read_artifact(dataset.artifact_id)).content)
        analysis_report.update(
            {
                "expected_total_usd": 71.2,
                "actual_dataset": data,
                "artifacts": [dataset.artifact_id, chart.artifact_id],
            }
        )
        assert abs(float(data["total_usd"]) - 71.2) < 1e-9
        assert (await agent.read_artifact(chart.artifact_id)).content.startswith(
            b"\x89PNG\r\n\x1a\n"
        )
        await agent.clear_conversations()
        await agent.close()
        agent = await Agent.open(
            "live-api-analysis", root=tmp_path / "homes", workspace=local
        )
        assert (
            json.loads((await agent.read_artifact(dataset.artifact_id)).content) == data
        )
        assert await agent.delete_artifact(dataset.artifact_id) is True
        assert await agent.delete_artifact(dataset.artifact_id) is False
        assert (await agent.read_artifact(chart.artifact_id)).content.startswith(
            b"\x89PNG"
        )
        analysis_report["artifact_deletion"] = {
            "deleted": dataset.artifact_id,
            "retained": chart.artifact_id,
            "registry_absent": await agent._embedded._store.get_artifact_record(
                dataset.artifact_id
            )
            is None,
        }
        assert analysis_report["artifact_deletion"]["registry_absent"]
    finally:
        await agent.close()
        await provider.close()


async def test_real_model_dependent_cells_and_error_correction(
    tmp_path, paid_case, analysis_report
):
    profile, provider, cost = paid_case
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = await Agent.create(
        "live-cells",
        root=tmp_path / "homes",
        model=provider,
        model_profile=profile,
        config=AgentConfig(
            limits=LoopLimits(max_estimated_cost_usd=cost, max_steps=24)
        ),
        workspace=LocalWorkspace(workspace),
    )
    try:
        result = await agent.run(
            "Exercise retained Python state in three separate analysis cells. First set retained_value=40 and print it. "
            "In the second cell increment retained_value by one and deliberately raise ValueError('intentional correction fixture'). "
            "After inspecting that Python error, run a third cell that prints retained_value+1 without resetting the variable. Report its value. "
            "Do not catch the intentional exception or replay either earlier cell."
        )
        transcript, _ = await retain_evidence(agent, result, analysis_report)
        cells = [
            output.output["data"]
            for call, output in transcript.tool_pairs
            if call.name == "analysis_execute"
            and output is not None
            and not output.is_error
        ]
        analysis_report["expected"] = {
            "cell_statuses": ["success", "python_error", "success"],
            "final_stdout": "42\n",
        }
        analysis_report["actual"] = cells
        assert len(cells) >= 3
        assert cells[0]["status"] == "success"
        assert cells[1]["status"] == "python_error"
        assert cells[1]["state_may_have_changed"] is True
        assert cells[2]["status"] == "success"
        assert cells[2]["stdout"].strip() == "42"
    finally:
        await agent.close()
        await provider.close()


async def test_real_model_cli_python_and_broker(tmp_path, paid_case, analysis_report):
    profile, provider, cost = paid_case
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "numbers.txt").write_text("1\n2\n3\n")
    root = tmp_path / "homes"
    bootstrap = await Agent.create(
        "live-cli", root=root, workspace=LocalWorkspace(workspace)
    )
    await bootstrap.close()
    command = [
        str(Path(sys.executable).with_name("daita")),
        "--root",
        str(root),
        "--workspace",
        str(workspace),
        "run",
        "live-cli",
        "Use analysis_execute. Inside Python, call tools.call('file_read', {'path': 'numbers.txt'}), inspect r['is_error'] and r['output']['data']['content'], then compute and print {'sum': the_sum}. Make the file_read call inside Python even if you already read it elsewhere. Report the sum.",
        "--model",
        profile.id,
        "--max-cost-usd",
        str(cost),
    ]
    analysis_report["command"] = command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.to_thread(process.communicate), 300
        )
        analysis_report["exit_code"] = process.returncode
        analysis_report["stderr"] = stderr.decode()[-8192:]
        assert process.returncode == 0, stderr.decode()
        payload = json.loads(stdout)
        analysis_report.update(
            {
                "cli_output": payload,
                "model_usage": payload["model_usage"],
                "expected_sum": 6,
            }
        )
        assert payload["status"] == "completed"
        reopened = await Agent.open(
            "live-cli", root=root, workspace=LocalWorkspace(workspace)
        )
        try:
            records = await reopened.analysis_evidence(payload["run_id"])
            analysis_report["measurements"] = [
                {"kind": record.kind, "facts": record.facts} for record in records
            ]
            assert any(
                record.kind == "child" and record.facts["status"] == "succeeded"
                for record in records
            )
            cells = [
                json.loads(canonical_json(record.facts))
                for record in records
                if record.kind == "cell"
            ]
            values = [
                computed_sum(cell.get("result", {}).get("stdout", "")) for cell in cells
            ]
            analysis_report["actual_sums"] = values
            assert 6 in values
            assert payload["analysis_usage"]["processes_reaped"] is True
            assert payload["analysis_usage"]["scratch_deleted"] is True
            analysis_report["cleanup"] = payload["analysis_usage"]
        finally:
            await reopened.close()
    finally:
        if process.poll() is None:
            process.kill()
        await asyncio.to_thread(process.wait, 5)
        await provider.close()


async def test_real_model_tui_python_and_broker(tmp_path, paid_case, analysis_report):
    profile, provider, cost = paid_case
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "numbers.txt").write_text("10\n20\n12\n")
    local = LocalWorkspace(workspace)
    app = DaitaApp(
        root=tmp_path / "homes",
        start_bootstrap=False,
        workspace=local,
        model=provider,
        model_profile=profile,
    )
    agent = await Agent.create(
        "live-tui",
        root=tmp_path / "homes",
        workspace=local,
        model=provider,
        model_profile=profile,
        observer=app._observer,
        config=AgentConfig(
            limits=LoopLimits(max_estimated_cost_usd=cost, max_steps=24)
        ),
    )
    app.controller.agent = agent
    try:
        async with app.run_test(size=(110, 36)) as pilot:
            try:
                await app._show_chat()
                await app.submit_composer(
                    "Use analysis_execute. Inside Python, call tools.call('file_read', {'path': 'numbers.txt'}), inspect r['is_error'] and r['output']['data']['content'], then compute and print {'sum': the_sum}. Make the file_read call inside Python even if you already read the file elsewhere. Report the sum."
                )
                assert app._run_task is not None
                await asyncio.wait_for(app._run_task, 300)
                await pilot.pause()
                assert app.controller.conversation_id is not None
                runs = await agent.conversation_runs(app.controller.conversation_id)
                result = runs[-1].result
                assert result is not None
                _, records = await retain_evidence(agent, result, analysis_report)
                analysis_report["expected_sum"] = 42
                assert any(
                    record.kind == "child" and record.facts["status"] == "succeeded"
                    for record in records
                )
                cells = [record.facts for record in records if record.kind == "cell"]
                values = [
                    computed_sum(cell.get("result", {}).get("stdout", ""))
                    for cell in cells
                ]
                analysis_report["actual_sums"] = values
                assert 42 in values
            finally:
                app.exit(0)
    finally:
        await agent.close()
        await provider.close()
