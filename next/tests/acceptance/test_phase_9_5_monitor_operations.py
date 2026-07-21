from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from daita import AgentHost, SQLiteSource
from daita.adapters.sqlite_query import SQLiteQueryBackend
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from daita.loop.models import LoopBudgets
from daita.monitors import (
    IntervalSchedule,
    MonitorBudgetOverrides,
    MonitorCondition,
    MonitorConditionKind,
    MonitorDefinition,
    MonitorRunStatus,
    MonitorScope,
)
from daita.operations.governance import DefaultPolicyProfile
from daita.operations.models import OperationStatus

NOW = datetime(2026, 7, 19, 18, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:phase-9-5-monitors",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


@dataclass
class MutableClock:
    current: datetime = NOW

    def __call__(self) -> datetime:
        return self.current


class ScriptedProvider:
    provider_id = PROFILE.id

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected monitor model call")
        return self.responses.pop(0)


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _database(path: Path, *, rows: int = 2) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, status TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO customers (status) VALUES (?)",
            [("active",) for _ in range(rows)],
        )


def _query(call_id: str, source_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id=call_id,
                name="data_query_sqlite",
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT COUNT(*) AS total FROM customers",
                },
            ),
        ),
    )


def _answer(evidence_id: str) -> ModelResponse:
    return ModelResponse(
        text=f"Count recorded. [evidence:{evidence_id}]",
        finish_reason=FinishReason.STOP,
    )


def _threshold_definition(
    source_id: str,
    *,
    name: str,
    value: int,
    max_turns: int = 3,
) -> MonitorDefinition:
    return MonitorDefinition(
        name=name,
        objective="Count the current customers.",
        scope=MonitorScope(source_ids=(source_id,)),
        schedule=IntervalSchedule(interval_seconds=300, anchor_at=NOW),
        condition=MonitorCondition(
            kind=MonitorConditionKind.THRESHOLD,
            expression="rows.0.total",
            configuration={"operator": "gt", "value": value},
        ),
        budget_overrides=MonitorBudgetOverrides(
            max_turns=max_turns,
            max_capability_calls=2,
            max_wall_time_seconds=30,
        ),
        policy_overrides={"mode": "read_only"},
        operation_template={"domain": "data"},
    )


async def _activate(
    host: AgentHost,
    monitor_id: str,
    definition: MonitorDefinition,
) -> None:
    proposal = await host.propose_monitor(
        monitor_id,
        definition,
        idempotency_key=f"propose-{monitor_id}",
    )
    await host.confirm_monitor(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="operator",
        reason="The typed condition, scope, and limits are correct.",
    )


async def test_default_host_matches_and_unmatches_typed_thresholds_atomically(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    provider = ScriptedProvider()
    defaults = LoopBudgets(max_turns=6, max_actions=6, max_wall_time_seconds=60)
    host = await AgentHost.create(
        "thresholds",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        budgets=defaults,
        clock=lambda: NOW,
        id_factory=_ids(),
        cadence_seconds=3_600,
    )
    try:
        await host.start()
        source = await host.attach(
            SQLiteSource(database),
            idempotency_key="attach-customers",
        )
        matched_definition = _threshold_definition(
            source.id,
            name="Customer count high",
            value=1,
        )
        unmatched_definition = _threshold_definition(
            source.id,
            name="Customer count extreme",
            value=10,
        )
        await _activate(host, "customer-high", matched_definition)
        await _activate(host, "customer-extreme", unmatched_definition)
        provider.responses.extend(
            (
                _query("query-matched", source.id),
                _answer("evidence-1"),
                _query("query-unmatched", source.id),
                _answer("evidence-2"),
            )
        )

        matched = await host.run_monitor_now(
            "customer-high",
            idempotency_key="run-customer-high",
        )
        unmatched = await host.run_monitor_now(
            "customer-extreme",
            idempotency_key="run-customer-extreme",
        )
        matched_inspection = await host.inspect_monitor("customer-high")
        unmatched_inspection = await host.inspect_monitor("customer-extreme")
        matched_operation = await host.inspect_operation(matched.operation_id or "")
        unmatched_operation = await host.inspect_operation(unmatched.operation_id or "")

        assert matched.finding_id is not None
        assert len(matched_inspection.findings) == 1
        assert matched_inspection.findings[0].evidence_id == "evidence-1"
        assert matched_inspection.findings[0].operation_id == matched.operation_id
        assert unmatched.finding_id is None
        assert unmatched_inspection.findings == ()
        assert matched_operation.budgets.max_turns == 3
        assert matched_operation.budgets.max_actions == 2
        assert matched_operation.budgets.max_wall_time_seconds == 30
        assert matched_operation.trigger.payload["monitor_definition_hash"] == (
            matched_definition.content_hash
        )
        retained_condition = matched_operation.trigger.payload["monitor_condition"]
        assert isinstance(retained_condition, Mapping)
        assert retained_condition["kind"] == "threshold"
        assert retained_condition["expression"] == "rows.0.total"
        configuration = retained_condition["configuration"]
        assert isinstance(configuration, Mapping)
        assert dict(configuration) == {"operator": "gt", "value": 1}
        policy = matched_operation.trigger.payload["monitor_effective_policy"]
        assert isinstance(policy, Mapping)
        assert policy["allow_destructive"] is False
        assert policy["fingerprint"] == DefaultPolicyProfile().fingerprint
        assert matched_operation.approvals == ()
        assert unmatched_operation.approvals == ()
        tool_names = {
            tool.name
            for call in matched_operation.model_calls
            for tool in call.request.tools
        }
        assert "catalog_search" not in tool_names
        assert "catalog_inspect" not in tool_names
        assert "catalog_traverse" not in tool_names
        assert "data_update_sqlite" not in tool_names
    finally:
        await host.stop(drain=False)


async def test_default_host_rejects_out_of_scope_monitor_before_query_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoped_database = tmp_path / "scoped.db"
    other_database = tmp_path / "other.db"
    _database(scoped_database)
    _database(other_database, rows=7)
    provider = ScriptedProvider()
    host = await AgentHost.create(
        "scope",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        budgets=LoopBudgets(max_turns=2),
        clock=lambda: NOW,
        id_factory=_ids(),
        cadence_seconds=3_600,
    )
    try:
        await host.start()
        scoped = await host.attach(
            SQLiteSource(scoped_database),
            idempotency_key="attach-scoped",
        )
        other = await host.attach(
            SQLiteSource(other_database),
            idempotency_key="attach-other",
        )
        definition = _threshold_definition(
            scoped.id,
            name="Scoped customer count",
            value=1,
            max_turns=2,
        )
        await _activate(host, "scoped-count", definition)
        provider.responses.extend(
            (
                _query("query-other", other.id),
                ModelResponse(
                    text="The requested source is outside scope.",
                    finish_reason=FinishReason.STOP,
                ),
            )
        )
        query_calls = 0

        async def forbidden_query_io(*args: object, **kwargs: object) -> object:
            nonlocal query_calls
            del args, kwargs
            query_calls += 1
            raise AssertionError("out-of-scope monitor reached SQLite query I/O")

        monkeypatch.setattr(SQLiteQueryBackend, "execute_read", forbidden_query_io)

        result = await host.run_monitor_now(
            "scoped-count",
            idempotency_key="run-out-of-scope",
        )
        snapshot = await host.inspect_operation(result.operation_id or "")

        assert result.run_status is MonitorRunStatus.FAILED
        assert result.finding_id is None
        assert snapshot.operation.status is OperationStatus.FAILED
        assert snapshot.tasks == ()
        assert snapshot.evidence == ()
        assert snapshot.approvals == ()
        assert query_calls == 0
    finally:
        await host.stop(drain=False)


async def test_default_host_persists_and_enforces_tight_monitor_turn_budget(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    provider = ScriptedProvider(
        ModelResponse(
            text="No grounded evidence was collected.",
            finish_reason=FinishReason.STOP,
        )
    )
    host = await AgentHost.create(
        "tight-budget",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        budgets=LoopBudgets(max_turns=5),
        clock=lambda: NOW,
        id_factory=_ids(),
        cadence_seconds=3_600,
    )
    try:
        await host.start()
        source = await host.attach(
            SQLiteSource(database),
            idempotency_key="attach-customers",
        )
        definition = _threshold_definition(
            source.id,
            name="One turn count",
            value=1,
            max_turns=1,
        )
        await _activate(host, "one-turn-count", definition)

        result = await host.run_monitor_now(
            "one-turn-count",
            idempotency_key="run-one-turn",
        )
        snapshot = await host.inspect_operation(result.operation_id or "")

        assert result.run_status is MonitorRunStatus.FAILED
        assert result.finding_id is None
        assert snapshot.operation.status is OperationStatus.FAILED
        assert snapshot.budgets.max_turns == 1
        assert snapshot.loop_state.turn_count == 1
        assert len(provider.requests) == 1
    finally:
        await host.stop(drain=False)


async def test_default_host_catches_up_once_and_deduplicates_after_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    clock = MutableClock()
    provider = ScriptedProvider()
    ids = _ids()
    root = tmp_path / "state"
    host = await AgentHost.create(
        "restart-monitor",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=clock,
        id_factory=ids,
        cadence_seconds=3_600,
    )
    await host.start()
    source = await host.attach(
        SQLiteSource(database),
        idempotency_key="attach-customers",
    )
    definition = _threshold_definition(
        source.id,
        name="Restart customer count",
        value=1,
    )
    await _activate(host, "restart-count", definition)
    await host.stop(drain=False)

    clock.current = NOW + timedelta(minutes=15, seconds=1)
    provider.responses.extend(
        (
            _query("query-after-restart", source.id),
            _answer("evidence-1"),
        )
    )
    reopened = await AgentHost.open(
        "restart-monitor",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=clock,
        id_factory=ids,
        cadence_seconds=3_600,
    )
    await reopened.start()
    first = await reopened.inspect_monitor("restart-count")
    await reopened.stop(drain=False)

    replay = await AgentHost.open(
        "restart-monitor",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=clock,
        id_factory=ids,
        cadence_seconds=3_600,
    )
    try:
        await replay.start()
        durable = await replay.inspect_monitor("restart-count")

        assert len(first.occurrences) == 1
        assert len(first.runs) == 1
        assert len(first.findings) == 1
        assert first.schedule_state.next_scheduled_at == NOW + timedelta(minutes=20)
        assert durable.occurrences == first.occurrences
        assert durable.runs == first.runs
        assert durable.findings == first.findings
        assert len(provider.requests) == 2
    finally:
        await replay.stop(drain=False)
