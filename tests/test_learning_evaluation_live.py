from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from html import unescape
import json
import os
from pathlib import Path
import re

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    LoopLimits,
    SemanticAnnotationState,
    create_llm_provider,
)
from daita.evaluation import (
    BenchmarkJudgment,
    BenchmarkOutcome,
    BenchmarkVariant,
    build_learning_effectiveness_report,
    measure_observer_events,
)
from daita.llm.models import (
    CanonicalMessage,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ModelProvider, provider_has_complete_pricing
from daita.loop.models import Transcript
from daita.observation import AgentEvent
from daita.security import EnvironmentSecretProvider, SecretReference

_AUTHORIZATION = "DAITA_RUN_LIVE_LEARNING_EVAL"
_MODEL_ID = "DAITA_EVAL_MODEL_ID"
_MODEL_KEY = "DAITA_EVAL_LLM_API_KEY"
_OUTPUT_DIR = "DAITA_EVAL_OUTPUT_DIR"
_MAX_COST = "DAITA_EVAL_MAX_COST_USD"
_DATABASE_PASSWORD = "DAITA_FIXTURE_POSTGRES_PASSWORD"
_DATABASE_PORT = "DAITA_FIXTURE_POSTGRES_PORT"
_FIXTURE_SOURCE_NAME = "Fixture PostgreSQL"
_STATEMENT_REQUIREMENTS = (
    "line_total",
    "quantity",
    "unit_cost",
    "paid",
    "currency",
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after starting "
            "tests/fixtures/postgresql/compose.yaml and explicitly authorizing "
            "live model cost"
        ),
    ),
]


class _RecordingProvider:
    """Caller-owned request capture around one real provider."""

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return await self._delegate.generate(request)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} must be set for the authorized live evaluation")
    return value


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.50")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _text(messages: Sequence[CanonicalMessage]) -> str:
    return "\n".join(
        block.text
        for message in messages
        for block in message.content
        if isinstance(block, TextBlock)
    )


def _tool_calls(transcript: Transcript, name: str) -> tuple[ToolCall, ...]:
    return tuple(
        call
        for message in transcript.messages
        for call in message.tool_calls
        if call.name == name
    )


def _tool_results(transcript: Transcript) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _successful_sql_call(transcript: Transcript) -> ToolCall | None:
    results = _tool_results(transcript)
    for call in reversed(_tool_calls(transcript, "data_query_postgresql")):
        result = results.get(call.id)
        if result is not None and not result.is_error:
            return call
    return None


def _paid_margin_judgment(
    transcript: Transcript,
    *,
    final_text: str | None,
    source_id: str,
    definition_recalled: bool,
) -> BenchmarkJudgment:
    call = _successful_sql_call(transcript)
    arguments = {} if call is None else call.arguments
    sql_value = arguments.get("sql")
    sql = sql_value.casefold() if isinstance(sql_value, str) else ""
    parameters = repr(arguments.get("parameters", ())).casefold()
    source_selection = arguments.get("source_id") == source_id
    resource_selection = all(
        resource in sql
        for resource in (
            "analytics.orders",
            "analytics.order_items",
            "analytics.products",
            "analytics.customers",
            "analytics.regions",
        )
    )
    field_selection = all(
        field in sql
        for field in (
            "line_total",
            "quantity",
            "unit_cost",
            "currency_code",
        )
    )
    paid_filter = "paid" in sql or "paid" in parameters
    reserve = bool(
        re.search(r"\b0?\.0*3\b", sql)
        or re.search(r"\b0?\.9*7\b", sql)
        or ("3" in sql and "100" in sql)
    )
    currency_separation = "currency_code" in sql and "group by" in sql
    constraints = paid_filter and reserve and currency_separation
    answer_correct = bool(
        final_text
        and call is not None
        and source_selection
        and resource_selection
        and field_selection
        and constraints
    )
    return BenchmarkJudgment(
        answer_correct=answer_correct,
        business_definition_correct=constraints,
        source_selection_correct=source_selection,
        resource_selection_correct=resource_selection,
        field_selection_correct=field_selection,
        semantic_constraints_satisfied=constraints,
        recalled_annotation_count=1 if definition_recalled else 0,
        irrelevant_recall_count=0,
        stale_activation_count=0,
        conflicting_claim_selection_count=0,
        cross_source_leakage_count=0,
    )


def _write_report(report, output_directory: Path, model_id: str) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^a-z0-9._-]+", "-", model_id.casefold()).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"learning-effectiveness-{safe_model}-{stamp}"
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(report.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(report.render_markdown(), encoding="utf-8")
    return json_path, markdown_path


async def test_live_fixture_baseline_teaching_and_learned_report(tmp_path: Path):
    model_id = _required_environment(_MODEL_ID)
    api_key = _required_environment(_MODEL_KEY)
    _required_environment(_DATABASE_PASSWORD)
    output_directory = Path(
        os.environ.get(_OUTPUT_DIR, str(tmp_path / "learning-evaluation-reports"))
    ).expanduser()
    if not output_directory.is_absolute():
        pytest.fail(f"{_OUTPUT_DIR} must be absolute when supplied")
    profile = reviewed_model_profile(model_id)
    if profile is None:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")
    delegate = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 4_096),
    )
    provider = _RecordingProvider(delegate)
    events: list[AgentEvent] = []
    approval_requests: list[ApprovalRequest] = []

    async def approve_exact_definition(
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        approval_requests.append(request)
        statement = request.arguments.get("statement")
        normalized = statement.casefold() if isinstance(statement, str) else ""
        reserve = "3%" in normalized or "0.03" in normalized
        if (
            request.tool_name == "semantic_save"
            and request.arguments.get("kind") == "metric_definition"
            and all(value in normalized for value in _STATEMENT_REQUIREMENTS)
            and reserve
        ):
            return ApprovalDecision.APPROVE
        return ApprovalDecision.DENY

    limits = LoopLimits(
        max_steps=16,
        max_total_tokens=60_000,
        max_wall_time_seconds=240,
        max_estimated_cost_usd=_cost_limit(),
    )
    agent = await Agent.create(
        "live-learning-evaluation",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        observer=events.append,
        approval_handler=approve_exact_definition,
        secret_provider=EnvironmentSecretProvider(),
        limits=limits,
    )
    credential = SecretReference.environment(_DATABASE_PASSWORD)
    source = await agent.attach_postgresql(
        host="127.0.0.1",
        port=int(os.environ.get(_DATABASE_PORT, "55432")),
        database="daita_fixture",
        username="daita_reader",
        credential=credential,
        schemas=("analytics",),
        ssl_mode="disable",
        name=_FIXTURE_SOURCE_NAME,
    )
    baseline_start = len(events)
    baseline_exit = await agent.run(
        "Using analytics.orders, analytics.order_items, analytics.products, "
        "analytics.customers, and analytics.regions, calculate paid contribution "
        "margin by region and currency. Keep currencies separate.",
        source_id=source.id,
    )
    baseline_events = tuple(events[baseline_start:])
    baseline_transcript = await agent.transcript(baseline_exit.run_id)

    teaching_start = len(events)
    teaching_exit = await agent.run(
        "When we say paid contribution margin, we mean "
        "SUM(analytics.order_items.line_total - "
        "analytics.order_items.quantity * analytics.products.unit_cost - "
        "analytics.order_items.line_total * 0.03), restricted to "
        "analytics.orders.status = paid. The 3% term is our risk reserve. Report "
        "results by region and currency, and never compare currencies without an "
        "explicit conversion.",
        source_id=source.id,
    )
    teaching_views = await agent.list_semantic_annotations()
    active = tuple(
        view
        for view in teaching_views
        if view.state is SemanticAnnotationState.ACTIVE
        and view.annotation.kind.value == "metric_definition"
    )
    await agent.close()

    learned_request_start = len(provider.requests)
    reopened = await Agent.open(
        "live-learning-evaluation",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        observer=events.append,
        approval_handler=approve_exact_definition,
        secret_provider=EnvironmentSecretProvider(),
        limits=limits,
    )
    learned_exit = await reopened.run(
        "Using analytics.orders, analytics.order_items, analytics.products, "
        "analytics.customers, and analytics.regions, calculate paid contribution "
        "margin by region and currency. Keep currencies separate.",
        source_id=source.id,
    )
    learned_transcript = await reopened.transcript(learned_exit.run_id)
    learned_events = tuple(events[teaching_start:])
    learned_requests = provider.requests[learned_request_start:]
    learned_request_text = "\n".join(
        _text(request.messages) for request in learned_requests
    )
    recalled = bool(
        active and active[0].annotation.statement in unescape(learned_request_text)
    )
    await reopened.close()

    baseline_judgment = _paid_margin_judgment(
        baseline_transcript,
        final_text=baseline_exit.final_text,
        source_id=source.id,
        definition_recalled=False,
    )
    learned_judgment = _paid_margin_judgment(
        learned_transcript,
        final_text=learned_exit.final_text,
        source_id=source.id,
        definition_recalled=recalled,
    )
    report = build_learning_effectiveness_report(
        (
            BenchmarkOutcome(
                "paid-contribution-margin-live",
                BenchmarkVariant.BASELINE,
                baseline_judgment,
                measure_observer_events(baseline_events),
            ),
            BenchmarkOutcome(
                "paid-contribution-margin-live",
                BenchmarkVariant.LEARNED,
                learned_judgment,
                measure_observer_events(learned_events),
            ),
        )
    )
    json_path, markdown_path = _write_report(report, output_directory, model_id)
    print(report.render_markdown())
    print(f"Machine report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    assert teaching_exit.final_text is not None
    assert len(active) == 1
    assert (
        sum(
            request.tool_name == "semantic_save"
            and all(
                requirement in str(request.arguments.get("statement", "")).casefold()
                for requirement in _STATEMENT_REQUIREMENTS
            )
            for request in approval_requests
        )
        == 1
    )
    assert recalled
    assert learned_judgment.hard_safety_passed
    assert learned_judgment.answer_correct
    assert sum(
        value is True
        for value in (
            learned_judgment.answer_correct,
            learned_judgment.business_definition_correct,
            learned_judgment.source_selection_correct,
            learned_judgment.resource_selection_correct,
            learned_judgment.field_selection_correct,
            learned_judgment.semantic_constraints_satisfied,
        )
    ) > sum(
        value is True
        for value in (
            baseline_judgment.answer_correct,
            baseline_judgment.business_definition_correct,
            baseline_judgment.source_selection_correct,
            baseline_judgment.resource_selection_correct,
            baseline_judgment.field_selection_correct,
            baseline_judgment.semantic_constraints_satisfied,
        )
    )
