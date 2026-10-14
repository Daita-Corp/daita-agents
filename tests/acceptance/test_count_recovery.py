"""Offline public comparison and qualification oracles; these are not live evidence."""

import json
from dataclasses import replace
from decimal import Decimal
from typing import Any

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
    blocker = "No table rows were read or counted. Source totals are unavailable; no defensible winner can be reported."
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
