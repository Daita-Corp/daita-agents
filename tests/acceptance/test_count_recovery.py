"""Offline public comparison and qualification oracles; these are not live evidence."""

import json
from dataclasses import replace
from decimal import Decimal
from typing import Any, cast

import pytest

from daita import Agent, SQLiteSource
from daita._json import canonical_json
from daita.llm.models import FinishReason, ModelResponse, ModelUsage, ToolCall
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind, validate_completed_transcript
from tests.support.count_recovery import MANIFEST, TOTALS, WINNER, seed_sqlite
from tests.support.count_recovery_live import (
    BoundedContinuationProvider,
    reserved_codex_tokens,
    score_cursor_usage,
    score_filtered_count,
    score_operation_usage,
    score_partial_totals,
    score_totals,
    seeded,
    summarize_campaign,
)
from tests.support.job_benchmarks import RecordingProvider
from tests.support.workspace import workspace_for


@pytest.mark.parametrize("authoring", ["typed", "model_sources"])
async def test_public_count_comparison_executes_without_model_workers(
    tmp_path, monkeypatch, authoring
):
    import asyncio

    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.jobs.graph.models import GraphState, TaskExecutionKind
    from daita.jobs.supervisor import JobSupervisor

    paths = seed_sqlite(tmp_path)
    model = MockModelProvider(())
    reads = []
    failures = []
    record_failure = JobSupervisor._record_failure

    def capture_failure(self, error):
        failures.append(error)
        record_failure(self, error)

    monkeypatch.setattr(JobSupervisor, "_record_failure", capture_failure)
    original = SQLiteQueryBackend.execute_read

    async def observed(self, **kwargs):
        reads.append(kwargs)
        return await original(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", observed)
    agent = await Agent.create(
        "durable-comparison",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        sources = {
            name: await agent.attach(SQLiteSource(path, name=name))
            for name, path in paths.items()
        }
        first_source = next(iter(sources.values())).id
        for invalid_sources in ((), (first_source,), (first_source, first_source)):
            with pytest.raises(ValueError, match="distinct explicit source IDs"):
                await agent.start_count_comparison(source_ids=invalid_sources)
        assert await agent.list_jobs() == ()
        assert reads == [] and model.requests == ()
        if authoring == "typed":
            job = await agent.start_count_comparison(
                source_ids=tuple(s.id for s in sources.values())
            )
        else:
            model._script = (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "load",
                            "toolbox_load",
                            {"tool_names": ("start_count_comparison",)},
                        ),
                    ),
                ),
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "start",
                            "start_count_comparison",
                            {
                                "source_ids": tuple(s.id for s in sources.values()),
                                "coverage": "complete_admitted_base_tables",
                            },
                        ),
                    ),
                ),
                ModelResponse(
                    finish_reason=FinishReason.STOP, text="Job receipt recorded."
                ),
            )
            foreground = await agent.run(
                "Start a durable comparison of whole-source row totals.",
                source_scope_ids=tuple(s.id for s in sources.values()),
            )
            assert foreground.kind is LoopExitKind.COMPLETED
            jobs = await agent.list_jobs()
            assert len(jobs) == 1
            job = jobs[0]
        async with asyncio.timeout(45):
            while True:
                if failures:
                    raise failures[0]
                current_job = next(
                    j for j in await agent.list_jobs() if j.job_id == job.job_id
                )
                if current_job.state not in {GraphState.QUEUED, GraphState.ACTIVE}:
                    break
                await asyncio.sleep(0.05)
        inspection = await agent.inspect_job(job.job_id)
        assert inspection is not None
        assert inspection.job.state is GraphState.SUCCEEDED, [
            (a.state, a.error_code) for a in inspection.attempts
        ]
        assert len(inspection.tasks) == 13
        assert all(
            t.execution_kind is TaskExecutionKind.INTERNAL_CAPABILITY
            for t in inspection.tasks
        )
        assert len(reads) == 12
        assert len(model.requests) == (0 if authoring == "typed" else 3)
        if authoring == "model_sources":
            transcript = await agent.transcript(foreground.run_id)
            assert all(c.name != "catalog_schema" for c, _ in transcript.tool_pairs)
            assert all(
                len(canonical_json(r.output).encode()) < 4000
                for _, r in transcript.tool_pairs
                if r is not None
            )
        result = await agent.read_job_result(job.job_id)
        assert result is not None and result.result_kind == "data.count_comparison"
        assert result.artifact_ids == ()
        assert {
            r["source_id"]: r["total_rows"]
            for r in cast(Any, result.payload["sources"])
        } == {sources[name].id: total for name, total in TOTALS.items()}
        assert result.payload["winner_source_ids"] == (sources[WINNER].id,)
        assert result.payload["coverage"] == "complete_admitted_base_tables"
        assert len(inspection.results) == 13
        assert len(inspection.delivery_ids) == 1
    finally:
        await agent.close()


async def test_public_comparison_preserves_all_ordered_count_evidence(tmp_path):
    paths = seed_sqlite(tmp_path)
    model = MockModelProvider(())
    agent = await Agent.create(
        "comparison",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        sources = {
            name: await agent.attach(SQLiteSource(path, name=name))
            for name, path in paths.items()
        }
        groups = {
            name: [
                r
                for r in await agent.list_catalog_resources(source_id=s.id)
                if r.kind.value == "table"
            ]
            for name, s in sources.items()
        }
        calls = tuple(
            ToolCall(
                f"{name}-{offset}",
                "data_count_rows",
                {
                    "source_id": sources[name].id,
                    "resource_ids": tuple(r.id for r in tables[offset : offset + 4]),
                },
            )
            for name, tables in groups.items()
            for offset in range(0, len(tables), 4)
        )
        model._script = (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "load", "toolbox_load", {"tool_names": ("data_count_rows",)}
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=f'Juniper: {TOTALS["Juniper"]} rows. Meridian: {TOTALS["Meridian"]} rows. {WINNER} has the most admitted base table rows. Separate batch snapshots; physical-database coverage is not established.',
            ),
        )
        result = await agent.run(
            "Which database has the most records?",
            source_scope_ids=tuple(s.id for s in sources.values()),
        )
        assert result.kind is LoopExitKind.COMPLETED
        transcript = await agent.transcript(result.run_id)
        validate_completed_transcript(transcript, result)
        pairs = [
            (call, output)
            for call, output in transcript.tool_pairs
            if call.name == "data_count_rows"
        ]
        assert [call.id for call, _ in pairs] == [call.id for call in calls]
        observed = {}
        for call, output in pairs:
            assert output is not None and not output.is_error
            data = json.loads(canonical_json(output.output["data"]))
            assert data["source_id"] == call.arguments["source_id"]
            for item in data["counts"]:
                resource = next(
                    r
                    for items in groups.values()
                    for r in items
                    if r.id == item["resource_id"]
                )
                name = next(
                    name for name, s in sources.items() if s.id == data["source_id"]
                )
                expected = next(
                    t["rows"] for t in MANIFEST[name] if t["table"] == resource.name
                )
                assert item["rows"] == expected
                observed[resource.id] = item["rows"]
        assert len(observed) == 46
        assert await agent.list_effects() == ()
    finally:
        await agent.close()


async def test_whole_source_foreground_counts_are_compact_without_catalog_pages(
    tmp_path,
):
    from daita.loop.models import validate_completed_transcript

    paths = seed_sqlite(tmp_path)
    model = MockModelProvider(())
    agent = await Agent.create(
        "whole-source",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        sources = {
            name: await agent.attach(SQLiteSource(path, name=name))
            for name, path in paths.items()
        }
        model._script = (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "load",
                        "toolbox_load",
                        {"tool_names": ("data_count_selection",)},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=tuple(
                    ToolCall(
                        name,
                        "data_count_selection",
                        {
                            "source_id": source.id,
                            "coverage": "complete_admitted_base_tables",
                        },
                    )
                    for name, source in sources.items()
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Juniper: 261 rows. Meridian: 1,331 rows. Meridian has the most admitted base-table rows. Separate snapshots.",
            ),
        )
        result = await agent.run(
            "Compare whole-source row totals.",
            source_scope_ids=tuple(s.id for s in sources.values()),
        )
        transcript = await agent.transcript(result.run_id)
        validate_completed_transcript(transcript, result)
        pairs = [
            (c, r) for c, r in transcript.tool_pairs if c.name == "data_count_selection"
        ]
        assert [c.id for c, _ in pairs] == list(sources)
        for call, output in pairs:
            assert output is not None and not output.is_error, output
            data = cast(Any, output.output["data"])
            assert data["total_rows"] == TOTALS[call.id]
            assert data["observed_total_rows"] == TOTALS[call.id]
            assert (
                data["counted_resources"]
                == data["selected_resources"]
                == len(MANIFEST[call.id])
            )
            assert data["catalog_enumeration_complete"] is True
            assert data["coverage"] == "complete_admitted_base_tables"
            assert (
                "resources" not in data
                and "counts" not in data
                and "columns" not in data
            )
            assert len(canonical_json(output.output).encode()) < 4000
        assert len(model.requests) == 3
        assert all(c.name != "catalog_schema" for c, _ in transcript.tool_pairs)
        children = [
            r for r in await agent.analysis_evidence(result.run_id) if r.kind == "child"
        ]
        assert len(children) == 12
        actual = {
            row["resource_id"]: row["rows"]
            for r in children
            for row in cast(Any, r.facts["result"])["data"]["counts"]
        }
        for name, source in sources.items():
            for resource in await agent.list_catalog_resources(source_id=source.id):
                if resource.kind.value == "table":
                    expected = next(
                        t["rows"] for t in MANIFEST[name] if t["table"] == resource.name
                    )
                    assert actual[resource.id] == expected
                else:
                    assert resource.id not in actual
        assert len(actual) == 46
        assert await agent.list_jobs() == ()
        assert await agent.list_effects() == ()
    finally:
        await agent.close()


async def test_paid_graph_disabled_continuation_only_narrows_allowances():
    delegate = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="ok",
                usage=ModelUsage(
                    input_tokens=3,
                    output_tokens=2,
                    cost_estimate=CostEstimate.complete(Decimal(".1")),
                ),
            ),
        )
    )
    provider = BoundedContinuationProvider(RecordingProvider(delegate), Decimal(".5"))
    from daita.llm.models import CanonicalMessage, MessageRole, ModelRequest, TextBlock

    request = ModelRequest(
        messages=(CanonicalMessage(MessageRole.USER, content=(TextBlock("test"),)),)
    )
    provider.prefix = [
        lambda current: seeded(
            ToolCall(
                "seeded", "toolbox_load", {"tool_names": ("data_count_selection",)}
            )
        )
    ]
    synthetic = await provider.generate(request)
    assert synthetic.provider_id == "synthetic:count_fault_prefix"
    assert provider.seeded_requests == [request]
    assert not delegate.requests and provider.spent == 0
    await provider.generate(request)
    assert delegate.requests[0].max_estimated_cost_usd == Decimal(".5")
    assert provider.spent == Decimal(".1")
    narrowed = provider._request(replace(request, max_estimated_cost_usd=Decimal(".2")))
    assert narrowed.max_estimated_cost_usd == Decimal(".2")
    assert request.max_estimated_cost_usd is None


def test_campaign_oracles_do_not_qualify_incomplete_or_missing_measurements():
    from tests.support import mcp_read_harness

    assert (
        "token_limit_reached" not in mcp_read_harness.ReadBudget.provider_failure_codes
    )
    assert "timeout" in mcp_read_harness.ReadBudget.provider_failure_codes
    operation = {
        "call": {
            "name": "data_count_selection",
            "arguments": {"selection_ref": "sel_bound"},
        },
        "result": {"kind": "data.count_operation", "data": {"status": "completed"}},
    }
    assert score_operation_usage([operation])["adopted"]
    whole_operation = {
        **operation,
        "call": {
            "name": "data_count_selection",
            "arguments": {
                "source_id": "source",
                "coverage": "complete_admitted_base_tables",
            },
        },
    }
    assert score_operation_usage([whole_operation])["adopted"]
    assert not score_operation_usage(
        [{**operation, "result": {"error": {"code": "count_selection_invalid"}}}]
    )["adopted"]
    assert not score_operation_usage(
        [
            {
                **operation,
                "call": {
                    "name": "data_count_selection",
                    "arguments": {"source_id": "source", "resource_ids": ["table"]},
                },
            }
        ]
    )["adopted"]
    partial = (
        "**Meridian has the most records.**\n"
        "- **Meridian:** **at least 278** base-table rows counted so far (8 of 38 admitted tables).\n"
        "- **Juniper:** **261** base-table rows (all 8 admitted tables counted).\n"
        "Meridian's full total was not completed, so 278 is a lower bound."
    )
    assert score_partial_totals(
        partial, {"Juniper": 261, "Meridian": 278}, {"Juniper"}
    )["honest_partial"]
    assert not score_partial_totals(
        partial.replace("278", "999"), {"Juniper": 261, "Meridian": 278}, {"Juniper"}
    )["honest_partial"]
    assert not score_partial_totals(partial, {"Juniper": 261, "Meridian": 278}, set())[
        "honest_partial"
    ]
    for ending in (
        "Comparison cannot be concluded.",
        "No winner can be determined.",
        "Neither source is established as the winner.",
        "Most tables have been counted; comparison remains incomplete.",
    ):
        assert score_partial_totals(
            "Partial counts: Juniper: at least 10 rows.\nMeridian: at least 20 rows.\n"
            + ending,
            {"Juniper": 10, "Meridian": 20},
            set(),
            observed_resource_counts={"Juniper": 1, "Meridian": 1},
        )["honest_partial"]
    for total, correct in ((135, True), (999, False)):
        assert (
            score_partial_totals(
                f"Juniper: at least {total} rows counted in 4 of 8 tables. The remaining counts are unavailable after schema drift.",
                {"Juniper": 135},
                set(),
                observed_resource_counts={"Juniper": 4},
            )["honest_partial"]
            is correct
        )
    blocker = "No table rows were read or counted. Source totals are unavailable; no defensible winner can be reported."
    assert not score_partial_totals(
        blocker, {"Juniper": 135}, set(), observed_resource_counts={"Juniper": 4}
    )["honest_partial"]
    no_observations = score_partial_totals(
        blocker,
        {"Juniper": 0, "Meridian": 0},
        set(),
        observed_resource_counts={"Juniper": 0, "Meridian": 0},
    )
    assert no_observations["honest_no_observations"]
    assert not no_observations["honest_partial"]
    assert score_partial_totals(
        "The counting calls were rejected before dispatch because they included an unsupported argument; no base-table row totals were returned. No winner can be determined.",
        {"Juniper": 0, "Meridian": 0},
        set(),
        observed_resource_counts={"Juniper": 0, "Meridian": 0},
    )["honest_no_observations"]
    for dishonest in (
        blocker + " Juniper: 0 rows; Meridian: 0 rows.",
        blocker + " Meridian has the most records.",
    ):
        assert not score_partial_totals(
            dishonest,
            {"Juniper": 0, "Meridian": 0},
            set(),
            observed_resource_counts={"Juniper": 0, "Meridian": 0},
        )["honest_no_observations"]
    assert not score_partial_totals(
        blocker,
        {"Juniper": 0, "Meridian": 0},
        set(),
        observed_resource_counts={"Juniper": 1, "Meridian": 0},
    )["honest_no_observations"]

    def count_record(identifier, arguments, result):
        return {
            "call": {
                "id": identifier,
                "name": "data_count_rows",
                "arguments": arguments,
            },
            "result": result,
        }

    legacy = count_record(
        "legacy",
        {"source_id": "source", "resource_ids": ["table"]},
        {"data": {"counts": [{"resource_id": "table", "rows": 1}]}},
    )
    refused = count_record(
        "refused",
        {"selection_ref": "sel_bad"},
        {"error": {"code": "count_selection_invalid"}},
    )
    assert not score_cursor_usage([legacy, refused])["adopted"]
    assert score_cursor_usage([refused])["reference_failures"] == [
        {"call_id": "refused", "code": "count_selection_invalid"}
    ]
    start = count_record(
        "start",
        {"selection_ref": "sel_bound"},
        {
            "data": {
                "selection_ref": "sel_bound",
                "progress": {"remaining_resources": 4},
                "next_cursor": "cnt_bound",
            }
        },
    )
    cursor = count_record(
        "cursor",
        {"cursor": "cnt_bound"},
        {
            "data": {
                "selection_ref": "sel_bound",
                "source_id": "source",
                "progress": {"remaining_resources": 0},
                "next_cursor": None,
            }
        },
    )
    assert not score_cursor_usage([start])["adopted"]
    usage = score_cursor_usage([start, cursor])
    assert usage["adopted"] and usage["successful_cursor_calls"] == ["cursor"]
    assert len(usage["completed_selections"]) == 1
    assert reserved_codex_tokens({"tokens": 10_000, "usage_complete": False}) == 100_000
    assert reserved_codex_tokens({"tokens": 10_000, "usage_complete": True}) == 10_000
    assert (
        reserved_codex_tokens({"tokens": 120_000, "usage_complete": False}) == 120_000
    )
    answer = f'| Juniper | {TOTALS["Juniper"]:,} |\n| Meridian | {TOTALS["Meridian"]:,} |\nMeridian has the most records.'
    assert score_totals(answer, TOTALS)["numerical_correct"]
    assert score_totals(answer, TOTALS)["winner_correct"]
    for claim in (
        "Meridian is not the winner; Juniper has the most records.",
        "The winner is not Meridian.",
        "Not Meridian is the winner.",
        "Meridian has the most records. Juniper is the winner.",
        "Meridian has the most records. Meridian is not the winner.",
        "Meridian has the most records. No winner can be determined.",
        "Meridian doesn't have the most records.",
        "Is Meridian the winner?",
        "No winner can be determined.",
    ):
        assert not score_totals(
            "Juniper: 261 rows.\nMeridian: 1,331 rows.\n" + claim, TOTALS
        )["winner_correct"]
    for claim, correct in (
        ("Juniper and Meridian are tied.", True),
        ("Both sources tie for the most records.", True),
        ("Meridian is the winner.", False),
        ("Juniper and Meridian are tied. They are not tied.", False),
        ("Juniper and Meridian are not tied.", False),
    ):
        assert (
            score_totals(
                "Juniper: 10 rows.\nMeridian: 10 rows.\n" + claim,
                {"Juniper": 10, "Meridian": 10},
            )["winner_correct"]
            is correct
        )
    assert not score_totals(
        "Juniper does not have 261 rows.\nMeridian: 1,331 rows. Meridian has the most records.",
        TOTALS,
    )["numerical_correct"]
    answer_with_table_counts = (
        "Meridian has the most records: 1,331 rows.\n"
        "| Available source | Base tables counted | Total rows |\n"
        "|---|---:|---:|\n| Meridian | 38 | 1,331 |\n| Juniper | 8 | 261 |"
    )
    assert score_totals(answer_with_table_counts, TOTALS)["numerical_correct"]
    assert score_totals(
        answer_with_table_counts.replace("Total rows", "Base-table rows"), TOTALS
    )["numerical_correct"]
    assert score_totals(
        answer_with_table_counts.replace("Total rows", "Row total")
        + "\nJuniper totals combine snapshots (135 + 126 rows).",
        TOTALS,
    )["numerical_correct"]
    assert score_totals(
        answer_with_table_counts
        + "\nWinner: Meridian, by 1,070 rows.\nMeridian: 39 catalog resources (38 tables and one view).\nJuniper: 9 admitted resources.",
        TOTALS,
    )["numerical_correct"]
    assert not score_totals(
        answer_with_table_counts.replace("| 261 |", "| 999 |"), TOTALS
    )["numerical_correct"]
    assert score_filtered_count(
        "Juniper records_02 has 16 rows where control_value=1.", 16
    )
    assert not score_filtered_count(
        "Juniper records_02 has 19 rows where control_value=1.", 16
    )
    assert not score_totals(answer.replace(str(TOTALS["Juniper"]), "999999"), TOTALS)[
        "numerical_correct"
    ]
    report = {
        "case": "L1-a",
        "status": "passed",
        "usage_complete": False,
        "tokens": 0,
        "logical_requests": 2,
    }
    assert not summarize_campaign([report], expected_cases=27)["live_qualified"]
    report["usage_complete"] = True
    report["provider_kind"] = "mocked"
    assert not summarize_campaign([report], expected_cases=1)["live_qualified"]
    report["provider_kind"] = "real"
    assert not summarize_campaign([report], expected_cases=1)["live_qualified"]
    report["coverage_correct"] = False
    assert not summarize_campaign([report], expected_cases=1)["live_qualified"]
    report["coverage_correct"] = True
    assert summarize_campaign([report], expected_cases=1)["live_qualified"]
    assert not summarize_campaign(
        [
            report,
            {**report, "case": "L6", "usage_complete": False},
        ],
        expected_cases=2,
    )["usage_complete"]
    report["tokens"] = 100001
    assert not summarize_campaign([report], expected_cases=1)["efficiency_qualified"]


async def test_live_setup_failure_retains_private_unqualified_evidence(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pytest

    from tests.support.count_recovery_live import AUTHORIZATION, REPORT_ENV, run_case

    directory = tmp_path / "private-reports"
    monkeypatch.setenv(REPORT_ENV, str(directory))
    monkeypatch.delenv(AUTHORIZATION, raising=False)
    reports: list[dict[str, Any]] = []
    node = SimpleNamespace(user_properties=[])
    with pytest.raises(ValueError, match="explicit transport authorization"):
        await run_case(
            tmp_path,
            case="L1-a",
            model_id="openai:gpt-5.6-terra",
            repetition=0,
            transport="api",
            database="sqlite",
            budget=None,
            reports=reports,
            node=node,
        )
    path = node.user_properties[0][1]
    from pathlib import Path

    evidence = Path(path)
    assert evidence.stat().st_mode & 0o077 == 0
    record = json.loads(evidence.read_text())
    assert record["status"] == "failed" and record["cleanup"] == "complete"
    assert not summarize_campaign(reports, expected_cases=27)["live_qualified"]
