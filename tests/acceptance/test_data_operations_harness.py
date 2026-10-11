"""Qualify live-test oracles and the reused runner without paid model calls."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal

import httpx2 as httpx
import pytest

from daita.llm.models import FinishReason, ModelResponse, ModelUsage, ToolCall
from daita.llm.pricing import CostEstimate
from tests.support.data_operations import (
    AUTHORIZATION,
    CASES,
    METRIC_DEFINITION,
    create_database,
    expected_rows,
    fixture_digest,
    parse_answer,
    require_authorization,
    score_answer,
)
from tests.support.job_benchmarks import RecordingProvider
from tests.support.mcp import MCPConformanceTransport
from tests.support.mcp_read_harness import (
    MeasuredMCPTransport,
    ReadBudget,
    evaluate_reads,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_oracle_matches_independent_relational_calculation(tmp_path, case):
    path = tmp_path / "sales.sqlite"
    create_database(path, case.variant)
    expressions = {
        "paid_totals": ("SUM(amount_minor-refund_minor)", "status='paid'"),
        "refund_totals": ("SUM(refund_minor)", "1=1"),
        "join_grain": (
            "SUM(amount_minor)",
            "status='paid' AND EXISTS(SELECT 1 FROM order_items i WHERE i.order_id=orders.order_id)",
        ),
        "local_day": (
            "SUM(amount_minor)",
            "status='paid' AND created_at>='2026-09-06T05:00:00+00:00' "
            "AND created_at<'2026-09-07T05:00:00+00:00'",
        ),
        "paid_count": ("COUNT(DISTINCT order_id)", "status='paid'"),
    }
    expression, predicate = expressions[case.kind]
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            f"SELECT currency,{expression} FROM orders WHERE {predicate} "
            "GROUP BY currency ORDER BY currency"
        ).fetchall()
    actual = [
        {"currency": currency, "amount_minor": amount} for currency, amount in rows
    ]
    assert expected_rows(case.kind, case.variant) == actual
    assert score_answer(json.dumps({"rows": actual}), actual)["correct"]


def test_matrix_is_fifty_distinct_frozen_cases():
    assert len(CASES) == len({case.case_id for case in CASES}) == 50
    assert len({fixture_digest(variant) for variant in range(10)}) == 10
    assert expected_rows("paid_totals") == [
        {"currency": "EUR", "amount_minor": 17833},
        {"currency": "USD", "amount_minor": 37605},
    ]
    assert expected_rows("settlement_contribution") == [
        {"currency": "EUR", "amount_minor": 17296},
        {"currency": "USD", "amount_minor": 36472},
    ]


@pytest.mark.parametrize(
    "answer",
    [
        None,
        '{"rows":[{"currency":"USD","amount_minor":true}]}',
        '{"rows":[{"currency":"USD","amount_minor":"10"}]}',
        '{"rows":[{"currency":"USD","amount_minor":NaN}]}',
        '{"rows":[{"currency":"USD","amount_minor":10.5}]}',
        '{"rows":[{"currency":"EUR","amount_minor":10}]}',
        '{"rows":[{"currency":"USD","amount_minor":10},{"currency":"USD","amount_minor":10}]}',
        '{"rows":[],"rows":[{"currency":"USD","amount_minor":10}]}',
        '{"rows":[{"currency":"USD","amount_minor":10}],"unsupported_claim":true}',
    ],
)
def test_score_rejects_wrong_or_unusable_results(answer):
    assert not score_answer(answer, [{"currency": "USD", "amount_minor": 10}])[
        "correct"
    ]


def test_explicit_authorization_guard_rejects_a_missing_gate(monkeypatch):
    monkeypatch.delenv(AUTHORIZATION, raising=False)
    with pytest.raises(ValueError, match="required before live construction"):
        require_authorization()
    monkeypatch.setenv(AUTHORIZATION, "1")
    require_authorization()


async def test_existing_evaluation_runner_supports_scoped_local_reads_and_learning(
    tmp_path,
):
    database = tmp_path / "sales.sqlite"
    create_database(database)
    expected = expected_rows("paid_totals")
    delegate = ToolboxAwareMockModelProvider(
        (),
        complete_pricing=True,
    )
    provider = RecordingProvider(delegate)
    service = MCPConformanceTransport()
    transport = MeasuredMCPTransport(httpx.MockTransport(service))
    budget = ReadBudget(total=Decimal("1"), per_run=Decimal("0.15"))
    report = tmp_path / "evidence.json"
    async with evaluate_reads(
        tmp_path / "homes",
        transport=transport,
        budget=budget,
        report=report,
        case_id="offline-dataops-qualification",
        workspace=workspace_for(tmp_path),
        provider_pair=(delegate.model_profile, provider),
    ) as evaluation:
        source = await evaluation.agent.attach_sqlite(database)
        resources = await evaluation.agent.list_catalog_resources(source_id=source.id)
        order = next(resource for resource in resources if resource.name == "orders")
        delegate.replace_script(
            [
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
                    tool_calls=(
                        ToolCall(
                            "query",
                            "data_query",
                            {
                                "source_id": source.id,
                                "resource_ids": [order.id],
                                "sql": "SELECT currency,SUM(amount_minor-refund_minor) AS amount_minor "
                                "FROM orders WHERE status='paid' GROUP BY currency ORDER BY currency",
                                "parameters": [],
                            },
                        ),
                    ),
                ),
                ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text=json.dumps({"rows": expected}),
                    usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
                ),
            ]
        )
        capture = await evaluation.run(
            CASES[0].prompt,
            phase="offline_current_read",
            source_scope_ids=(source.id,),
        )
        assert score_answer(capture.result.final_text, expected)["correct"], (
            capture.result.kind.value,
            capture.result.reason,
            capture.result.final_text,
        )
        assert capture.transcript.run.source_scope_ids == (source.id,)
        assert any(
            call.name == "data_query" and output and not output.is_error
            for call, output in capture.transcript.tool_pairs
        )
        delegate.replace_script(
            [
                ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="No durable change requested.",
                    usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
                )
            ]
        )
        taught = await evaluation.run(
            METRIC_DEFINITION,
            phase="offline_learning_entry",
            learning=True,
            source_scope_ids=(source.id,),
        )
        assert taught.result.kind.value == "completed"
        assert await evaluation.agent.list_semantic_annotations() == ()
    payload = json.loads(report.read_text())
    assert payload["status"] == "passed"
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["model_requests"] >= 2
    assert payload["runs"][0]["model_usage_complete"]
    assert payload["budget"]["incomplete_runs"] == 0
    assert service.identities == {}
    assert parse_answer(json.dumps({"rows": expected}))["rows"] == expected
