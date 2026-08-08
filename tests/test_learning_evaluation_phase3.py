from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Mapping

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest
from daita._json import FrozenJsonObject
from daita.catalog import CatalogResource
from daita.evaluation import (
    EVALUATION_MAX_CASES,
    EVALUATION_MAX_EVENTS_PER_OUTCOME,
    BenchmarkJudgment,
    BenchmarkOutcome,
    BenchmarkVariant,
    RunMeasurement,
    build_learning_effectiveness_report,
    measure_observer_events,
)
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.observation import AgentEvent, AgentEventKind

NOW = datetime(2026, 7, 28, 12, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class _HumanLabeledCase:
    case_id: str
    expectation: str
    phases: tuple[str, ...]


_LIFECYCLE_PHASES = (
    "fresh_catalog_baseline",
    "explicit_teaching_or_correction",
    "exact_approved_durable_knowledge",
    "new_conversation_related_task",
    "expected_recall",
)

_HUMAN_LABELED_CASES = (
    _HumanLabeledCase(
        "paid-contribution-margin",
        "exact_learned_formula_and_paid_order_filter",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "customer-health",
        "learned_bounded_formula",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "paid-invoiced-revenue",
        "top_region_within_each_currency_after_rejected_sql",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "daily-gross-refunded-net-revenue",
        "daily_grain_preserved_after_result_truncation",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "currency-narrative-isolation",
        "no_cross_currency_comparison_without_conversion",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "overlapping-current-archived-order-ids",
        "current_and_archived_orders_counted_independently",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "denied-learning",
        "denial_has_no_future_effect",
        ("fresh_catalog_baseline", "explicit_teaching_or_correction", "denial"),
    ),
    _HumanLabeledCase(
        "catalog-change-stale-exclusion",
        "catalog_change_excludes_stale_meaning",
        (*_LIFECYCLE_PHASES, "catalog_revision_change"),
    ),
    _HumanLabeledCase(
        "conflicting-definitions",
        "conflicting_claims_are_withheld",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "exact-duplicates",
        "one_deterministic_recall_slot_per_duplicate_group",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "supersession",
        "new_exact_claim_replaces_old_claim",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "cross-source-isolation",
        "foreign_source_meaning_is_never_recalled",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "irrelevant-recall",
        "unrelated_meaning_is_not_recalled",
        _LIFECYCLE_PHASES,
    ),
    _HumanLabeledCase(
        "skill-create-invoke",
        "created_skill_is_selected_in_later_conversation",
        (*_LIFECYCLE_PHASES, "skill_creation", "skill_invocation"),
    ),
    _HumanLabeledCase(
        "loaded-skill-repair",
        "validated_failure_then_digest_protected_skill_repair",
        (*_LIFECYCLE_PHASES, "validated_failure", "skill_repair"),
    ),
)


def _event(
    kind: AgentEventKind,
    data: Mapping[str, object],
    *,
    run_id: str = "run-1",
) -> AgentEvent:
    return AgentEvent(
        kind=kind,
        occurred_at=NOW,
        run_id=run_id,
        conversation_id="conversation-1",
        data=FrozenJsonObject.from_mapping(data),
    )


def _measurement(
    *,
    model_calls: int,
    tool_calls: int,
    failed_sql_calls: int,
    duration_ms: int,
    total_tokens: int,
    approval_outcome: str | None = None,
) -> RunMeasurement:
    return RunMeasurement(
        model_calls=model_calls,
        tool_calls=tool_calls,
        catalog_discovery_calls=min(tool_calls, 2),
        failed_sql_calls=failed_sql_calls,
        corrected_sql_calls=failed_sql_calls,
        duration_ms=duration_ms,
        input_tokens=total_tokens * 3 // 4,
        output_tokens=total_tokens // 4,
        total_tokens=total_tokens,
        estimated_cost_usd=Decimal("0"),
        cost_complete=True,
        learning_proposals=1 if approval_outcome is not None else 0,
        proposal_succeeded=1 if approval_outcome == "approved" else 0,
        proposal_failed=1 if approval_outcome == "denied" else 0,
        approvals_requested=1 if approval_outcome is not None else 0,
        approvals=1 if approval_outcome == "approved" else 0,
        denials=1 if approval_outcome == "denied" else 0,
    )


def _usage(input_tokens: int, output_tokens: int) -> ModelUsage:
    return ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _semantic_save_call(
    *,
    call_id: str,
    annotation_id: str,
    source_id: str,
    resources: Mapping[str, CatalogResource],
    statement: str,
) -> ToolCall:
    selected = tuple(
        resources[name] for name in ("orders", "order_items", "products", "regions")
    )
    resource_ids = sorted(resource.id for resource in selected)
    revisions = sorted(
        (
            {
                "resource_id": resource.id,
                "revision": resource.current_revision,
            }
            for resource in selected
        ),
        key=lambda item: str(item["resource_id"]),
    )
    fields = sorted(
        (
            {
                "resource_id": resources[resource_name].id,
                "field_name": field_name,
            }
            for resource_name, field_name in (
                ("orders", "status"),
                ("order_items", "line_total"),
                ("order_items", "quantity"),
                ("products", "unit_cost"),
                ("regions", "currency_code"),
            )
        ),
        key=lambda item: (str(item["resource_id"]), str(item["field_name"])),
    )
    return ToolCall(
        id=call_id,
        name="semantic_save",
        arguments={
            "id": annotation_id,
            "subject": {
                "source_ids": [source_id],
                "resource_ids": resource_ids,
                "fields": fields,
            },
            "kind": "metric_definition",
            "statement": statement,
            "evidence": [{"kind": "user_assertion"}],
            "catalog_revisions": revisions,
        },
    )


def _tool_result(
    messages: Sequence[object],
    call_id: str,
) -> ToolResultBlock:
    for message in messages:
        content = getattr(message, "content", ())
        for block in content:
            if isinstance(block, ToolResultBlock) and block.call_id == call_id:
                return block
    raise AssertionError(f"missing tool result: {call_id}")


def _seed_paid_margin_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE regions(
                region_code TEXT PRIMARY KEY,
                currency_code TEXT NOT NULL
            );
            CREATE TABLE customers(
                customer_id INTEGER PRIMARY KEY,
                region_code TEXT NOT NULL REFERENCES regions(region_code)
            );
            CREATE TABLE products(
                product_id INTEGER PRIMARY KEY,
                unit_cost INTEGER NOT NULL
            );
            CREATE TABLE orders(
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                status TEXT NOT NULL,
                total_amount INTEGER NOT NULL
            );
            CREATE TABLE order_items(
                order_item_id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(order_id),
                product_id INTEGER NOT NULL REFERENCES products(product_id),
                quantity INTEGER NOT NULL,
                line_total INTEGER NOT NULL
            );
            INSERT INTO regions VALUES ('AMER', 'USD'), ('EMEA', 'EUR');
            INSERT INTO customers VALUES (1, 'AMER'), (2, 'EMEA');
            INSERT INTO products VALUES (1, 30), (2, 50);
            INSERT INTO orders VALUES
                (1, 1, 'paid', 100),
                (2, 2, 'paid', 200),
                (3, 1, 'pending', 999);
            INSERT INTO order_items VALUES
                (1, 1, 1, 2, 100),
                (2, 2, 2, 2, 200),
                (3, 3, 1, 1, 999);
            """)


async def test_offline_exit_gate_executes_real_learning_lifecycles(tmp_path):
    database = tmp_path / "learning-exit-gate.sqlite"
    _seed_paid_margin_database(database)
    events: list[AgentEvent] = []
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    baseline_provider = MockModelProvider(())
    baseline_agent = await Agent.create(
        "learning-exit-gate",
        root=tmp_path,
        model=baseline_provider,
        model_profile=baseline_provider.model_profile,
        observer=events.append,
        approval_handler=approve,
        clock=lambda: NOW,
    )
    source = await baseline_agent.attach_sqlite(database, name="Commerce fixture")
    resources = {
        resource.name: resource
        for resource in await baseline_agent.list_catalog_resources()
    }
    assert set(resources) == {
        "customers",
        "order_items",
        "orders",
        "products",
        "regions",
    }
    baseline_provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="baseline-query",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": source.id,
                        "sql": "SELECT SUM(total_amount) AS margin FROM orders",
                    },
                ),
            ),
            usage=_usage(120, 20),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="The unqualified total is 1299.",
            usage=_usage(80, 12),
        ),
    )
    baseline_start = len(events)
    baseline_exit = await baseline_agent.run(
        "Using orders, order_items, products, customers, and regions, calculate "
        "paid contribution margin by region and currency.",
        source_id=source.id,
    )
    baseline_events = tuple(events[baseline_start:])
    baseline_transcript = await baseline_agent.transcript(baseline_exit.run_id)
    baseline_result = _tool_result(
        baseline_transcript.messages,
        "baseline-query",
    )
    assert not baseline_result.is_error
    baseline_data = baseline_result.output["data"]
    assert isinstance(baseline_data, Mapping)
    baseline_rows = baseline_data["rows"]
    assert isinstance(baseline_rows, tuple)
    assert baseline_rows[0]["margin"] == 1299
    baseline_provider.assert_consumed()
    await baseline_agent.close()

    statement = (
        "Paid contribution margin is SUM(order_items.line_total - "
        "order_items.quantity * products.unit_cost) for orders where status is paid. "
        "Report by region and currency; never compare currencies without conversion."
    )
    teaching_provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    _semantic_save_call(
                        call_id="teach-margin",
                        annotation_id="paid-contribution-margin",
                        source_id=source.id,
                        resources=resources,
                        statement=statement,
                    ),
                ),
                usage=_usage(180, 30),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The paid contribution margin definition is saved.",
                usage=_usage(90, 15),
            ),
        )
    )
    teaching_agent = await Agent.open(
        "learning-exit-gate",
        root=tmp_path,
        model=teaching_provider,
        model_profile=teaching_provider.model_profile,
        observer=events.append,
        approval_handler=approve,
        clock=lambda: NOW,
    )
    learned_lifecycle_start = len(events)
    teaching_exit = await teaching_agent.run(
        "When we say paid contribution margin, use line total minus quantity times "
        "unit cost for paid orders, report by region and currency, and never compare "
        "currencies without conversion.",
        source_id=source.id,
    )
    assert teaching_exit.final_text == (
        "The paid contribution margin definition is saved."
    )
    assert tuple(tool.name for tool in teaching_provider.requests[0].tools) == tuple(
        sorted(tool.name for tool in teaching_provider.requests[0].tools)
    )
    assert "semantic_save" in {
        tool.name for tool in teaching_provider.requests[0].tools
    }
    views = await teaching_agent.list_semantic_annotations()
    assert len(views) == 1
    assert views[0].annotation.statement == statement
    assert views[0].usable_as_current_meaning
    teaching_provider.assert_consumed()
    await teaching_agent.close()

    learned_provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="learned-query",
                        name="data_query_sqlite",
                        arguments={
                            "source_id": source.id,
                            "sql": (
                                "SELECT r.region_code, r.currency_code, "
                                "SUM(oi.line_total - "
                                "(oi.quantity * p.unit_cost)) AS margin "
                                "FROM orders AS o "
                                "JOIN customers AS c "
                                "ON c.customer_id = o.customer_id "
                                "JOIN regions AS r "
                                "ON r.region_code = c.region_code "
                                "JOIN order_items AS oi "
                                "ON oi.order_id = o.order_id "
                                "JOIN products AS p "
                                "ON p.product_id = oi.product_id "
                                "WHERE o.status = 'paid' "
                                "GROUP BY r.region_code, r.currency_code "
                                "ORDER BY r.region_code"
                            ),
                        },
                    ),
                ),
                usage=_usage(140, 25),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=(
                    "AMER is 40 USD and EMEA is 100 EUR. They are not ranked "
                    "against each other without a conversion."
                ),
                usage=_usage(100, 24),
            ),
        )
    )
    learned_agent = await Agent.open(
        "learning-exit-gate",
        root=tmp_path,
        model=learned_provider,
        model_profile=learned_provider.model_profile,
        observer=events.append,
        clock=lambda: NOW,
    )
    learned_exit = await learned_agent.run(
        "Using orders, order_items, products, customers, and regions, calculate "
        "paid contribution margin by region and currency.",
        source_id=source.id,
    )
    learned_events = tuple(events[learned_lifecycle_start:])
    learned_transcript = await learned_agent.transcript(learned_exit.run_id)
    learned_result = _tool_result(learned_transcript.messages, "learned-query")
    assert not learned_result.is_error
    learned_data = learned_result.output["data"]
    assert isinstance(learned_data, Mapping)
    learned_rows = learned_data["rows"]
    assert isinstance(learned_rows, tuple)
    assert tuple(
        (row["region_code"], row["currency_code"], row["margin"])
        for row in learned_rows
    ) == (("AMER", "USD", 40), ("EMEA", "EUR", 100))
    first_learned_request = learned_provider.requests[0]
    learned_prompt = "\n".join(
        block.text
        for message in first_learned_request.messages
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert statement in learned_prompt
    learned_provider.assert_consumed()
    await learned_agent.close()

    denied_provider = MockModelProvider(())

    async def deny(request: ApprovalRequest) -> ApprovalDecision:
        assert request.tool_name == "semantic_save"
        return ApprovalDecision.DENY

    denied_agent = await Agent.create(
        "denied-exit-gate",
        root=tmp_path,
        model=denied_provider,
        model_profile=denied_provider.model_profile,
        observer=events.append,
        approval_handler=deny,
        clock=lambda: NOW,
    )
    denied_source = await denied_agent.attach_sqlite(database, name="Denied fixture")
    denied_resources = {
        resource.name: resource
        for resource in await denied_agent.list_catalog_resources()
    }
    denied_provider._script = (
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="No durable definition is available.",
            usage=_usage(60, 10),
        ),
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                _semantic_save_call(
                    call_id="denied-teaching",
                    annotation_id="denied-margin",
                    source_id=denied_source.id,
                    resources=denied_resources,
                    statement=statement,
                ),
            ),
            usage=_usage(100, 20),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="The proposed definition was not saved.",
            usage=_usage(50, 10),
        ),
    )
    denied_baseline_start = len(events)
    denied_baseline = await denied_agent.run(
        "Calculate paid contribution margin.",
        source_id=denied_source.id,
    )
    denied_baseline_events = tuple(events[denied_baseline_start:])
    denied_learned_start = len(events)
    denied_teaching = await denied_agent.run(
        "When we say paid contribution margin, use the durable paid-order formula.",
        source_id=denied_source.id,
    )
    assert denied_teaching.final_text == "The proposed definition was not saved."
    assert await denied_agent.list_semantic_annotations() == ()
    denied_provider.assert_consumed()
    await denied_agent.close()

    denied_reopen_provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="No durable definition is available.",
                usage=_usage(60, 10),
            ),
        )
    )
    denied_reopened = await Agent.open(
        "denied-exit-gate",
        root=tmp_path,
        model=denied_reopen_provider,
        model_profile=denied_reopen_provider.model_profile,
        observer=events.append,
        clock=lambda: NOW,
    )
    denied_follow_up = await denied_reopened.run(
        "Calculate paid contribution margin.",
        source_id=denied_source.id,
    )
    denied_lifecycle_events = tuple(events[denied_learned_start:])
    assert denied_follow_up.final_text == denied_baseline.final_text
    denied_prompt = "\n".join(
        block.text
        for message in denied_reopen_provider.requests[0].messages
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert statement not in denied_prompt
    assert await denied_reopened.list_semantic_annotations() == ()
    denied_reopen_provider.assert_consumed()
    await denied_reopened.close()

    report = build_learning_effectiveness_report(
        (
            BenchmarkOutcome(
                "paid-contribution-margin-executable",
                BenchmarkVariant.BASELINE,
                BenchmarkJudgment(
                    answer_correct=False,
                    business_definition_correct=False,
                    source_selection_correct=True,
                    resource_selection_correct=False,
                    field_selection_correct=False,
                    semantic_constraints_satisfied=False,
                ),
                measure_observer_events(baseline_events),
            ),
            BenchmarkOutcome(
                "paid-contribution-margin-executable",
                BenchmarkVariant.LEARNED,
                BenchmarkJudgment(
                    answer_correct=True,
                    business_definition_correct=True,
                    source_selection_correct=True,
                    resource_selection_correct=True,
                    field_selection_correct=True,
                    semantic_constraints_satisfied=True,
                    recalled_annotation_count=1,
                ),
                measure_observer_events(learned_events),
            ),
            BenchmarkOutcome(
                "denied-learning-executable",
                BenchmarkVariant.BASELINE,
                BenchmarkJudgment(
                    answer_correct=True,
                    business_definition_correct=True,
                ),
                measure_observer_events(denied_baseline_events),
            ),
            BenchmarkOutcome(
                "denied-learning-executable",
                BenchmarkVariant.LEARNED,
                BenchmarkJudgment(
                    answer_correct=True,
                    business_definition_correct=True,
                ),
                measure_observer_events(denied_lifecycle_events),
            ),
        )
    )
    machine = report.to_mapping()
    cases = machine["cases"]
    assert isinstance(cases, list)
    assert tuple(item["verdict"] for item in cases) == (
        "no_measured_change",
        "improved_correctness",
    )
    hard_safety = machine["hard_safety"]
    assert isinstance(hard_safety, Mapping)
    assert hard_safety["passed"] is True
    assert len(approvals) == 1
    learned_summary = machine["learned"]
    assert isinstance(learned_summary, Mapping)
    assert learned_summary["learning_proposals"] == 2
    assert learned_summary["approvals"] == 1
    assert learned_summary["denials"] == 1
    assert "paid-contribution-margin-executable" in report.render_markdown()


def test_observer_measurement_is_bounded_content_free_and_counts_corrections():
    events = (
        _event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-1"}),
        _event(
            AgentEventKind.MODEL_COMPLETED,
            {
                "provider_id": "mock:test",
                "duration_ms": 2,
                "input_tokens": 10,
                "output_tokens": 2,
            },
        ),
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "catalog-call",
                "tool_name": "catalog_schema",
                "capability_id": "catalog.schema",
            },
        ),
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": "catalog-call",
                "tool_name": "catalog_schema",
                "duration_ms": 1,
                "success": True,
                "error_code": None,
            },
        ),
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "sql-failed",
                "tool_name": "data_query_sqlite",
                "capability_id": "data.sqlite.query",
            },
        ),
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": "sql-failed",
                "tool_name": "data_query_sqlite",
                "duration_ms": 1,
                "success": False,
                "error_code": "sql_validation_failed",
            },
        ),
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "sql-corrected",
                "tool_name": "data_query_sqlite",
                "capability_id": "data.sqlite.query",
            },
        ),
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": "sql-corrected",
                "tool_name": "data_query_sqlite",
                "duration_ms": 1,
                "success": True,
                "error_code": None,
            },
        ),
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "proposal",
                "tool_name": "semantic_save",
                "capability_id": "semantic.save",
            },
        ),
        _event(
            AgentEventKind.APPROVAL_REQUESTED,
            {
                "call_id": "proposal",
                "tool_name": "semantic_save",
                "capability_id": "semantic.save",
            },
        ),
        _event(
            AgentEventKind.APPROVAL_DECIDED,
            {"call_id": "proposal", "outcome": "approved"},
        ),
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": "proposal",
                "tool_name": "semantic_save",
                "duration_ms": 1,
                "success": True,
                "error_code": None,
            },
        ),
        _event(
            AgentEventKind.RUN_COMPLETED,
            {
                "exit_kind": "completed",
                "reason": "completed",
                "steps": 2,
                "duration_ms": 17,
                "input_tokens": 100,
                "output_tokens": 20,
                "reasoning_tokens": 3,
                "cache_read_tokens": 4,
                "cache_write_tokens": 5,
                "total_tokens": 132,
                "cost_status": "complete",
                "cost_amount_usd": "0.0012",
            },
        ),
    )

    measurement = measure_observer_events(events)

    assert measurement == RunMeasurement(
        model_calls=1,
        tool_calls=4,
        catalog_discovery_calls=1,
        failed_sql_calls=1,
        corrected_sql_calls=1,
        duration_ms=17,
        input_tokens=100,
        output_tokens=20,
        reasoning_tokens=3,
        cache_read_tokens=4,
        cache_write_tokens=5,
        total_tokens=132,
        estimated_cost_usd=Decimal("0.0012"),
        cost_complete=True,
        learning_proposals=1,
        proposal_succeeded=1,
        approvals_requested=1,
        approvals=1,
    )
    assert EVALUATION_MAX_EVENTS_PER_OUTCOME == 8_192
    with pytest.raises(ValueError, match="event collection"):
        measure_observer_events(
            tuple(events[0] for _ in range(EVALUATION_MAX_EVENTS_PER_OUTCOME + 1))
        )


def test_human_labeled_offline_suite_covers_learning_and_live_fixture_findings():
    required_expectations = {
        "exact_learned_formula_and_paid_order_filter",
        "learned_bounded_formula",
        "top_region_within_each_currency_after_rejected_sql",
        "daily_grain_preserved_after_result_truncation",
        "no_cross_currency_comparison_without_conversion",
        "current_and_archived_orders_counted_independently",
        "denial_has_no_future_effect",
        "catalog_change_excludes_stale_meaning",
        "conflicting_claims_are_withheld",
        "one_deterministic_recall_slot_per_duplicate_group",
        "new_exact_claim_replaces_old_claim",
        "foreign_source_meaning_is_never_recalled",
        "unrelated_meaning_is_not_recalled",
        "created_skill_is_selected_in_later_conversation",
        "validated_failure_then_digest_protected_skill_repair",
    }
    assert {item.expectation for item in _HUMAN_LABELED_CASES} == (
        required_expectations
    )
    for item in _HUMAN_LABELED_CASES:
        if item.case_id == "denied-learning":
            assert "denial" in item.phases
            continue
        assert set(_LIFECYCLE_PHASES) <= set(item.phases)

    outcomes: list[BenchmarkOutcome] = []
    for item in _HUMAN_LABELED_CASES:
        denial = item.case_id == "denied-learning"
        learned_failed_sql = 1 if item.case_id == "paid-invoiced-revenue" else 0
        outcomes.extend(
            (
                BenchmarkOutcome(
                    item.case_id,
                    BenchmarkVariant.BASELINE,
                    BenchmarkJudgment(
                        answer_correct=denial,
                        business_definition_correct=denial,
                        source_selection_correct=True,
                        resource_selection_correct=True,
                        field_selection_correct=True,
                        semantic_constraints_satisfied=denial,
                    ),
                    _measurement(
                        model_calls=3,
                        tool_calls=7,
                        failed_sql_calls=1,
                        duration_ms=90,
                        total_tokens=900,
                    ),
                ),
                BenchmarkOutcome(
                    item.case_id,
                    BenchmarkVariant.LEARNED,
                    BenchmarkJudgment(
                        answer_correct=True,
                        business_definition_correct=True,
                        source_selection_correct=True,
                        resource_selection_correct=True,
                        field_selection_correct=True,
                        semantic_constraints_satisfied=True,
                        recalled_annotation_count=0 if denial else 1,
                        irrelevant_recall_count=0,
                        stale_activation_count=0,
                        conflicting_claim_selection_count=0,
                        cross_source_leakage_count=0,
                        relevant_skill_selected=(
                            True
                            if item.case_id
                            in {"skill-create-invoke", "loaded-skill-repair"}
                            else None
                        ),
                    ),
                    _measurement(
                        model_calls=2,
                        tool_calls=4,
                        failed_sql_calls=learned_failed_sql,
                        duration_ms=55,
                        total_tokens=600,
                        approval_outcome="denied" if denial else "approved",
                    ),
                ),
            )
        )

    report = build_learning_effectiveness_report(outcomes)
    machine = report.to_mapping()
    human = report.render_markdown()

    assert EVALUATION_MAX_CASES == 256
    assert machine["case_count"] == len(_HUMAN_LABELED_CASES)
    assert machine["hard_safety"] == {
        "required_zero": (
            "stale_activation_count",
            "conflicting_claim_selection_count",
            "cross_source_leakage_count",
        ),
        "passed": True,
    }
    learned_summary = machine["learned"]
    assert isinstance(learned_summary, Mapping)
    assert learned_summary["learning_proposals"] == len(_HUMAN_LABELED_CASES)
    assert learned_summary["approvals"] == len(_HUMAN_LABELED_CASES) - 1
    assert learned_summary["denials"] == 1
    assert learned_summary["proposal_succeeded"] == len(_HUMAN_LABELED_CASES) - 1
    assert learned_summary["proposal_failed"] == 1
    assert "Baseline" in human
    assert "Learned" in human
    assert "stored artifact counts are not treated as evidence" in human
    assert "paid-contribution-margin" in human
    assert "loaded-skill-repair" in human


def test_report_exposes_safety_regressions_and_never_contains_sensitive_inputs():
    sentinel_values = (
        "RAW_PROMPT_SENTINEL",
        "RAW_ROW_SENTINEL",
        "SQL_ARGUMENT_SENTINEL",
        "SEMANTIC_STATEMENT_SENTINEL",
        "SKILL_BODY_SENTINEL",
        "CREDENTIAL_SENTINEL",
    )
    baseline = BenchmarkOutcome(
        "safety-case",
        BenchmarkVariant.BASELINE,
        BenchmarkJudgment(answer_correct=False),
    )
    learned = BenchmarkOutcome(
        "safety-case",
        BenchmarkVariant.LEARNED,
        BenchmarkJudgment(
            answer_correct=True,
            stale_activation_count=1,
            conflicting_claim_selection_count=1,
            cross_source_leakage_count=1,
        ),
    )

    report = build_learning_effectiveness_report((learned, baseline))
    machine = report.to_mapping()
    rendered = report.render_markdown()
    serialized = repr(machine) + rendered

    cases = machine["cases"]
    assert isinstance(cases, list)
    assert cases[0]["verdict"] == "unsafe"
    hard_safety = machine["hard_safety"]
    assert isinstance(hard_safety, Mapping)
    assert hard_safety["passed"] is False
    assert all(value not in serialized for value in sentinel_values)
    with pytest.raises(ValueError, match="baseline and learned"):
        build_learning_effectiveness_report((baseline,))
