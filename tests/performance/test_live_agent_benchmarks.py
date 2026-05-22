"""
Live LLM performance benchmarks for the agent runtime.

These tests intentionally require real provider credentials. They measure the
agent framework in the only environment that captures real latency, token
usage, tool behavior, and from_db model/tool coordination.

Run:
    DAITA_BENCHMARK_LIVE_LLM=1 OPENAI_API_KEY=... pytest tests/performance \
        -m "performance and requires_llm" -v -s

Optional:
    DAITA_BENCHMARK_PROVIDER=openai
    DAITA_BENCHMARK_MODEL=gpt-4o-mini
    DAITA_BENCHMARK_OUTPUT=.daita/benchmarks/live-agent-benchmarks.jsonl
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daita.agents.agent import Agent
from daita.agents.db.config import ANSWER_EVIDENCE_DB_TOOLS
from daita.core.tools import AgentTool

pytestmark = [
    pytest.mark.performance,
    pytest.mark.requires_llm,
    pytest.mark.slow,
]


@dataclass(frozen=True)
class BenchmarkProvider:
    provider: str
    model_env: str
    default_model: str
    api_key_envs: tuple[str, ...]


@dataclass(frozen=True)
class FromDbBenchmarkCase:
    case_id: str
    prompt: str
    expected_terms: tuple[str, ...]
    max_iterations: int = 8
    max_tools: int = 6
    require_answer_tool: bool = True
    require_recovery: bool = False
    require_empty_result: bool = False
    human_labels: tuple[str, ...] = ()


PROVIDERS = {
    "openai": BenchmarkProvider(
        provider="openai",
        model_env="OPENAI_TEST_MODEL",
        default_model="gpt-4o-mini",
        api_key_envs=("OPENAI_API_KEY",),
    ),
    "anthropic": BenchmarkProvider(
        provider="anthropic",
        model_env="ANTHROPIC_TEST_MODEL",
        default_model="claude-haiku-4-5",
        api_key_envs=("ANTHROPIC_API_KEY",),
    ),
    "grok": BenchmarkProvider(
        provider="grok",
        model_env="GROK_TEST_MODEL",
        default_model="grok-4.20",
        api_key_envs=("XAI_API_KEY", "GROK_API_KEY"),
    ),
    "gemini": BenchmarkProvider(
        provider="gemini",
        model_env="GEMINI_TEST_MODEL",
        default_model="gemini-2.5-flash-lite",
        api_key_envs=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    ),
}


@pytest.fixture(scope="session")
def benchmark_provider() -> BenchmarkProvider:
    if os.environ.get("DAITA_BENCHMARK_LIVE_LLM") != "1":
        pytest.fail(
            "Live LLM benchmarks require DAITA_BENCHMARK_LIVE_LLM=1. "
            "These benchmarks do not run against mock LLMs."
        )
    provider_name = os.environ.get("DAITA_BENCHMARK_PROVIDER", "openai").lower()
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        pytest.fail(
            "Unsupported DAITA_BENCHMARK_PROVIDER "
            f"{provider_name!r}. Supported: {', '.join(sorted(PROVIDERS))}."
        )
    if not _first_env(provider.api_key_envs):
        pytest.fail(
            f"Missing API key for {provider.provider}. Set one of: "
            f"{', '.join(provider.api_key_envs)}."
        )
    return provider


@pytest.fixture
def benchmark_agent_kwargs(benchmark_provider: BenchmarkProvider) -> dict[str, Any]:
    return {
        "llm_provider": benchmark_provider.provider,
        "model": os.environ.get("DAITA_BENCHMARK_MODEL")
        or os.environ.get(benchmark_provider.model_env)
        or benchmark_provider.default_model,
        "api_key": _first_env(benchmark_provider.api_key_envs),
        "temperature": 0,
        "max_tokens": int(os.environ.get("DAITA_BENCHMARK_MAX_TOKENS", "160")),
    }


async def test_live_simple_answer_latency_and_tokens(benchmark_agent_kwargs):
    agent = Agent(name="BenchSimple", **benchmark_agent_kwargs)
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            "Reply with exactly this token and nothing else: benchmark-simple-ok",
        )
        _assert_result_contains(result, "benchmark-simple-ok")
        _assert_basic_metrics(result, elapsed_ms, max_iterations=1, max_tools=0)
        _record_benchmark("simple_answer", result, elapsed_ms)
    finally:
        await agent.stop()


async def test_live_single_tool_latency_tokens_and_tool_sequence(
    benchmark_agent_kwargs,
):
    async def revenue_lookup(_args):
        return {"customer": "Bob", "revenue": 150}

    tool = AgentTool(
        name="lookup_revenue",
        description="Return the top customer revenue from the benchmark fixture.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=revenue_lookup,
        replay_safe=True,
        retry_safe=True,
        side_effecting=False,
    )
    agent = Agent(name="BenchSingleTool", tools=[tool], **benchmark_agent_kwargs)
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            (
                "Use the lookup_revenue tool exactly once, then answer with the "
                "customer name and revenue."
            ),
            max_iterations=4,
        )
        _assert_result_contains(result, "Bob")
        _assert_tool_sequence(result, ["lookup_revenue"])
        _assert_basic_metrics(result, elapsed_ms, max_iterations=4, max_tools=1)
        _record_benchmark("single_tool", result, elapsed_ms)
    finally:
        await agent.stop()


async def test_live_multi_tool_chain_latency_tokens_and_tool_sequence(
    benchmark_agent_kwargs,
):
    async def fetch_customer(_args):
        return {"customer_id": 2, "name": "Bob"}

    async def fetch_revenue(args):
        return {"customer_id": args["customer_id"], "revenue": 150}

    tools = [
        AgentTool(
            name="fetch_top_customer",
            description="Return the top customer id and name.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=fetch_customer,
            replay_safe=True,
            retry_safe=True,
            side_effecting=False,
        ),
        AgentTool(
            name="fetch_customer_revenue",
            description="Return revenue for a customer id.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "integer"}},
                "required": ["customer_id"],
            },
            handler=fetch_revenue,
            replay_safe=True,
            retry_safe=True,
            side_effecting=False,
        ),
    ]
    agent = Agent(name="BenchMultiTool", tools=tools, **benchmark_agent_kwargs)
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            (
                "Use fetch_top_customer first. Then use fetch_customer_revenue "
                "with that customer_id. Answer with the customer name and revenue."
            ),
            max_iterations=6,
        )
        _assert_result_contains(result, "Bob")
        _assert_tool_sequence(result, ["fetch_top_customer", "fetch_customer_revenue"])
        _assert_basic_metrics(result, elapsed_ms, max_iterations=6, max_tools=2)
        _record_benchmark("multi_tool_chain", result, elapsed_ms)
    finally:
        await agent.stop()


async def test_live_focus_context_reduces_payload_and_records_tokens(
    benchmark_agent_kwargs,
):
    raw_rows = [
        {
            "order_id": index,
            "customer": "Bob" if index == 7 else f"Customer {index}",
            "region": "EMEA" if index % 3 == 0 else "NA",
            "status": "complete" if index % 2 == 0 else "pending",
            "revenue": 150 if index == 7 else index * 11,
            "debug_payload": "x" * 120,
        }
        for index in range(1, 30)
    ]

    async def get_orders(_args):
        return {"rows": raw_rows}

    tool = AgentTool(
        name="get_orders",
        description="Return benchmark order rows.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=get_orders,
        replay_safe=True,
        retry_safe=True,
        side_effecting=False,
    )
    focus = "rows[] | WHERE customer == 'Bob' | SELECT order_id, customer, revenue"
    raw_payload_tokens = _token_estimate({"rows": raw_rows})
    focused_payload_tokens = _token_estimate(
        {
            "rows": [
                {
                    "order_id": row["order_id"],
                    "customer": row["customer"],
                    "revenue": row["revenue"],
                }
                for row in raw_rows
                if row["customer"] == "Bob"
            ]
        }
    )
    assert focused_payload_tokens < raw_payload_tokens

    agent = Agent(
        name="BenchFocusContext",
        tools=[tool],
        focus={"get_orders": focus},
        **benchmark_agent_kwargs,
    )
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            "Use get_orders exactly once. Report Bob's order id and revenue.",
            max_iterations=4,
        )
        _assert_result_contains(result, "Bob")
        _assert_tool_sequence(result, ["get_orders"])
        _assert_basic_metrics(result, elapsed_ms, max_iterations=4, max_tools=1)
        _record_benchmark(
            "focused_context",
            result,
            elapsed_ms,
            extra={
                "raw_payload_token_estimate": raw_payload_tokens,
                "focused_payload_token_estimate": focused_payload_tokens,
            },
        )
    finally:
        await agent.stop()


FROM_DB_BENCHMARK_CASES = (
    FromDbBenchmarkCase(
        case_id="from_db_sqlite_single_table",
        prompt=(
            "Use db_query exactly once with this SQL to answer from the database: "
            "SELECT COUNT(*) AS customer_count FROM customers"
        ),
        expected_terms=("3",),
    ),
    FromDbBenchmarkCase(
        case_id="from_db_sqlite_join_entity_label",
        prompt=(
            "Use db_query exactly once with this SQL to answer from the database: "
            "SELECT c.name, SUM(o.total_amount) AS total_revenue "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.name ORDER BY total_revenue DESC LIMIT 1"
        ),
        expected_terms=("Bob", "150"),
        human_labels=("Bob",),
    ),
    FromDbBenchmarkCase(
        case_id="from_db_sqlite_aggregation",
        prompt=(
            "Use db_query exactly once with this SQL to answer from the database: "
            "SELECT SUM(total_amount) AS shipped_revenue "
            "FROM orders WHERE status = 'shipped'"
        ),
        expected_terms=("125",),
    ),
    FromDbBenchmarkCase(
        case_id="from_db_sqlite_repair_recovery",
        prompt=(
            "First use db_query with this SQL exactly: "
            "SELECT c.full_name FROM customers c. If that returns repair_required "
            "or preflight_failed, recover by querying the correct customer name "
            "column and answer with all customer names."
        ),
        expected_terms=("Alice", "Bob", "Cara"),
        max_iterations=10,
        max_tools=10,
        require_recovery=True,
        human_labels=("Alice", "Bob", "Cara"),
    ),
    FromDbBenchmarkCase(
        case_id="from_db_sqlite_empty_result",
        prompt=(
            "Use db_query exactly once with this SQL to answer from the database: "
            "SELECT name FROM customers WHERE signup_date > '2030-01-01'. "
            "If there are no matching rows, say there are no matching customers."
        ),
        expected_terms=("no",),
        require_empty_result=True,
    ),
)


OPEN_ENDED_FROM_DB_BENCHMARK_CASES = (
    FromDbBenchmarkCase(
        case_id="from_db_open_ended_single_table",
        prompt="How many customers are in the database?",
        expected_terms=("3",),
        max_iterations=10,
        max_tools=8,
    ),
    FromDbBenchmarkCase(
        case_id="from_db_open_ended_join_entity_label",
        prompt=(
            "Which customer has the highest total order revenue, and what is "
            "the amount? Include the customer name."
        ),
        expected_terms=("Bob", "150"),
        max_iterations=10,
        max_tools=8,
        human_labels=("Bob",),
    ),
    FromDbBenchmarkCase(
        case_id="from_db_open_ended_aggregation",
        prompt="What is the total shipped order revenue?",
        expected_terms=("125",),
        max_iterations=10,
        max_tools=8,
    ),
    FromDbBenchmarkCase(
        case_id="from_db_open_ended_empty_result",
        prompt=(
            "Which customers have signup_date after 2030-01-01? If there are "
            "no matching rows, say there are no matching customers."
        ),
        expected_terms=("no",),
        max_iterations=10,
        max_tools=8,
        require_empty_result=True,
    ),
)


@pytest.mark.parametrize(
    "benchmark_case",
    FROM_DB_BENCHMARK_CASES,
    ids=[case.case_id for case in FROM_DB_BENCHMARK_CASES],
)
async def test_live_from_db_sqlite_quality_latency_tokens_and_tool_usage(
    benchmark_agent_kwargs,
    tmp_path,
    monkeypatch,
    benchmark_case: FromDbBenchmarkCase,
):
    db_path = tmp_path / "benchmark.sqlite"
    _seed_sqlite(db_path)
    monkeypatch.chdir(tmp_path)

    build_start = time.perf_counter()
    agent = await Agent.from_db(
        str(db_path),
        name="BenchFromDbSQLite",
        cache_ttl=3600,
        query_default_limit=10,
        **benchmark_agent_kwargs,
    )
    build_elapsed_ms = _elapsed_ms(build_start)
    try:
        result, elapsed_ms = await _timed_run(
            agent,
            benchmark_case.prompt,
            max_iterations=benchmark_case.max_iterations,
        )
        for expected in benchmark_case.expected_terms:
            _assert_result_contains(result, expected)
        if benchmark_case.require_answer_tool:
            _assert_answer_db_tool_was_used(result)
        _assert_basic_metrics(
            result,
            elapsed_ms,
            max_iterations=benchmark_case.max_iterations,
            max_tools=benchmark_case.max_tools,
        )
        from_db_metrics = _from_db_benchmark_metrics(result, benchmark_case)
        if benchmark_case.require_recovery:
            assert from_db_metrics["sql_rejection_count"] > 0
            assert from_db_metrics["success_path"] == "recovered"
        if benchmark_case.require_empty_result:
            assert from_db_metrics["empty_result_observed"] is True
        for label in benchmark_case.human_labels:
            assert label in from_db_metrics["human_readable_labels_present"]
        _record_benchmark(
            benchmark_case.case_id,
            result,
            elapsed_ms,
            extra={
                "agent_build_ms": build_elapsed_ms,
                "from_db": from_db_metrics,
            },
        )
    finally:
        await agent.stop()


@pytest.mark.parametrize(
    "benchmark_case",
    OPEN_ENDED_FROM_DB_BENCHMARK_CASES,
    ids=[case.case_id for case in OPEN_ENDED_FROM_DB_BENCHMARK_CASES],
)
async def test_live_from_db_sqlite_open_ended_quality(
    benchmark_agent_kwargs,
    tmp_path,
    monkeypatch,
    benchmark_case: FromDbBenchmarkCase,
):
    db_path = tmp_path / "benchmark.sqlite"
    _seed_sqlite(db_path)
    monkeypatch.chdir(tmp_path)

    build_start = time.perf_counter()
    agent = await Agent.from_db(
        str(db_path),
        name="BenchFromDbSQLiteOpenEnded",
        cache_ttl=3600,
        query_default_limit=10,
        **benchmark_agent_kwargs,
    )
    build_elapsed_ms = _elapsed_ms(build_start)
    result: dict[str, Any] | None = None
    elapsed_ms = 0.0
    try:
        try:
            result, elapsed_ms = await _timed_run(
                agent,
                benchmark_case.prompt,
                max_iterations=benchmark_case.max_iterations,
            )
        except Exception as exc:
            _record_benchmark_failure(
                benchmark_case.case_id,
                _elapsed_ms(build_start),
                exc,
                result=result,
                agent=agent,
                benchmark_case=benchmark_case,
                extra={
                    "agent_build_ms": build_elapsed_ms,
                    "open_ended": True,
                    "failure_phase": "runtime",
                },
            )
            raise

        try:
            fast_path = _is_from_db_fast_path(result)
            for expected in benchmark_case.expected_terms:
                _assert_result_contains(result, expected)
            if benchmark_case.require_answer_tool:
                _assert_answer_db_tool_was_used(result)
            if fast_path:
                _assert_from_db_fast_path_result(result)
            _assert_basic_metrics(
                result,
                elapsed_ms,
                max_iterations=benchmark_case.max_iterations,
                max_tools=benchmark_case.max_tools,
                require_tokens=not fast_path,
            )
            from_db_metrics = _from_db_benchmark_metrics(result, benchmark_case)
            if benchmark_case.require_empty_result:
                assert from_db_metrics["empty_result_observed"] is True
            for label in benchmark_case.human_labels:
                assert label in from_db_metrics["human_readable_labels_present"]
        except AssertionError as exc:
            _record_benchmark_failure(
                benchmark_case.case_id,
                elapsed_ms,
                exc,
                result=result,
                agent=agent,
                benchmark_case=benchmark_case,
                extra={
                    "agent_build_ms": build_elapsed_ms,
                    "open_ended": True,
                    "failure_phase": "assertion",
                },
            )
            raise
        _record_benchmark(
            benchmark_case.case_id,
            result,
            elapsed_ms,
            extra={
                "agent_build_ms": build_elapsed_ms,
                "from_db": from_db_metrics,
                "open_ended": True,
            },
        )
    finally:
        await agent.stop()


async def _timed_run(
    agent: Agent,
    prompt: str,
    *,
    max_iterations: int = 3,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    result = await agent.run(prompt, detailed=True, max_iterations=max_iterations)
    elapsed_ms = _elapsed_ms(start)
    return result, elapsed_ms


def _assert_basic_metrics(
    result: dict[str, Any],
    elapsed_ms: float,
    *,
    max_iterations: int,
    max_tools: int,
    require_tokens: bool = True,
) -> None:
    assert elapsed_ms > 0
    assert elapsed_ms <= float(
        os.environ.get("DAITA_BENCHMARK_MAX_LATENCY_MS", "90000")
    )
    assert result["iterations"] <= max_iterations
    assert len(result.get("tool_calls") or []) <= max_tools
    tokens = result.get("tokens") or {}
    if require_tokens:
        assert tokens.get("total_tokens", 0) > 0


def _is_from_db_fast_path(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    fast_path = result.get("from_db_fast_path") or {}
    return isinstance(fast_path, dict) and bool(fast_path.get("used"))


def _assert_from_db_fast_path_result(result: dict[str, Any]) -> None:
    fast_path = result.get("from_db_fast_path") or {}
    assert fast_path.get("used") is True
    assert result.get("iterations") == 0
    tool_calls = result.get("tool_calls") or []
    assert len(tool_calls) <= 1
    if tool_calls:
        assert tool_calls[0].get("tool") == "db_compile_and_query"


def _assert_result_contains(result: dict[str, Any], expected: str) -> None:
    assert expected.lower() in str(result.get("result", "")).lower()


def _assert_tool_sequence(result: dict[str, Any], expected: list[str]) -> None:
    sequence = [call.get("tool") for call in result.get("tool_calls") or []]
    assert sequence == expected


def _assert_tool_was_used(result: dict[str, Any], expected_tool: str) -> None:
    sequence = [call.get("tool") for call in result.get("tool_calls") or []]
    assert expected_tool in sequence


def _assert_answer_db_tool_was_used(result: dict[str, Any]) -> None:
    sequence = [call.get("tool") for call in result.get("tool_calls") or []]
    assert any(tool in ANSWER_EVIDENCE_DB_TOOLS for tool in sequence), sequence


def _from_db_benchmark_metrics(
    result: dict[str, Any], benchmark_case: FromDbBenchmarkCase
) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") or {}
    completeness = diagnostics.get("db_completeness") or {}
    tool_calls = result.get("tool_calls") or []
    sql_rejections = [
        call
        for call in tool_calls
        if isinstance(call.get("result"), dict)
        and (
            call["result"].get("repair_required")
            or call["result"].get("preflight_failed")
            or call["result"].get("error_type") in {"schema_reference_error"}
        )
    ]
    executed_db_calls = [
        call
        for call in tool_calls
        if call.get("tool") in ANSWER_EVIDENCE_DB_TOOLS
        and not (
            isinstance(call.get("result"), dict)
            and (
                call["result"].get("repair_required")
                or call["result"].get("preflight_failed")
                or call["result"].get("error")
            )
        )
    ]
    empty_result_observed = any(
        _is_empty_db_result(call.get("result")) for call in tool_calls
    )
    labels_present = [
        label
        for label in benchmark_case.human_labels
        if label.lower() in str(result.get("result", "")).lower()
    ]
    sql_rejection_count = int(
        completeness.get("sql_rejected")
        if isinstance(completeness.get("sql_rejected"), int)
        else len(sql_rejections)
    )
    executed_query_count = int(
        completeness.get("queries_executed")
        if isinstance(completeness.get("queries_executed"), int)
        else len(executed_db_calls)
    )
    fast_path = _is_from_db_fast_path(result)
    return {
        "success_path": (
            "from_db_fast_path"
            if fast_path
            else "recovered" if sql_rejection_count else "llm_tool_path"
        ),
        "sql_rejection_count": sql_rejection_count,
        "executed_query_count": executed_query_count,
        "empty_result_observed": empty_result_observed,
        "human_readable_labels_present": labels_present,
        "db_completeness_status": completeness.get("status"),
        "db_can_answer": completeness.get("can_answer"),
        "returned_columns": completeness.get("returned_columns"),
        "missing_required_fields": completeness.get("missing_required_fields"),
        "unresolved_repair": completeness.get("unresolved_repair"),
    }


def _is_empty_db_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    rows = value.get("rows")
    if isinstance(rows, list):
        return len(rows) == 0
    if isinstance(value.get("row_count"), int):
        return value["row_count"] == 0
    return False


def _record_benchmark(
    case: str,
    result: dict[str, Any],
    elapsed_ms: float,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    output_path = Path(
        os.environ.get(
            "DAITA_BENCHMARK_OUTPUT",
            ".daita/benchmarks/live-agent-benchmarks.jsonl",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case,
        "elapsed_ms": round(elapsed_ms, 2),
        "processing_time_ms": round(float(result.get("processing_time_ms", 0)), 2),
        "iterations": result.get("iterations"),
        "tool_count": len(result.get("tool_calls") or []),
        "tool_sequence": [call.get("tool") for call in result.get("tool_calls") or []],
        "tool_durations_ms": [
            call.get("duration_ms") for call in result.get("tool_calls") or []
        ],
        "tokens": result.get("tokens"),
        "cost": result.get("cost"),
        "diagnostics": result.get("diagnostics"),
        **(extra or {}),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    print("[benchmark]", json.dumps(payload, sort_keys=True, default=str))


def _record_benchmark_failure(
    case: str,
    elapsed_ms: float,
    error: Exception,
    *,
    result: dict[str, Any] | None = None,
    agent: Agent | None = None,
    benchmark_case: FromDbBenchmarkCase | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    output_path = Path(
        os.environ.get(
            "DAITA_BENCHMARK_OUTPUT",
            ".daita/benchmarks/live-agent-benchmarks.jsonl",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case,
        "elapsed_ms": round(elapsed_ms, 2),
        "failed": True,
        "error_type": type(error).__name__,
        "error": str(error),
        "failure_reason": _classify_from_db_failure(
            error,
            result=result,
            agent=agent,
            benchmark_case=benchmark_case,
        ),
        "failure_diagnostics": _from_db_failure_diagnostics(
            result=result,
            agent=agent,
            benchmark_case=benchmark_case,
        ),
        **(extra or {}),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    print("[benchmark]", json.dumps(payload, sort_keys=True, default=str))


def _from_db_failure_diagnostics(
    *,
    result: dict[str, Any] | None,
    agent: Agent | None,
    benchmark_case: FromDbBenchmarkCase | None,
) -> dict[str, Any]:
    result = result if isinstance(result, dict) else {}
    diagnostics = result.get("diagnostics") or {}
    completeness = diagnostics.get("db_completeness") or {}
    tool_calls = _diagnostic_tool_calls(result, agent)
    context = (
        getattr(agent, "_db_last_context_metadata", {}) if agent is not None else {}
    ) or {}
    catalog_store_id = getattr(agent, "_db_catalog_store_id", None) if agent else None
    return {
        "success_path": (
            _from_db_benchmark_metrics(result, benchmark_case)["success_path"]
            if benchmark_case is not None and result
            else None
        ),
        "selected_tools": context.get("selected_tools"),
        "selected_tool_count": context.get("selected_tool_count"),
        "fast_path": context.get("fast_path") or _is_from_db_fast_path(result),
        "tool_calls_attempted": _summarize_tool_calls(tool_calls),
        "tool_error_fingerprints": _tool_error_fingerprints(tool_calls),
        "repeated_tool_error_fingerprints": _repeated_values(
            _tool_error_fingerprints(tool_calls)
        ),
        "db_completeness_status": completeness.get("status"),
        "db_can_answer": completeness.get("can_answer"),
        "final_answer_readiness_warnings": diagnostics.get("warnings"),
        "malformed_tool_argument_errors": _malformed_tool_argument_errors(
            tool_calls, diagnostics
        ),
        "max_iteration_termination": diagnostics.get("exit_reason")
        in {"max_iterations", "max_iterations_partial"},
        "catalog_unavailable_or_unprofiled": not bool(catalog_store_id),
        "catalog_tools_attempted": _catalog_tools_attempted(tool_calls),
    }


def _classify_from_db_failure(
    error: Exception,
    *,
    result: dict[str, Any] | None,
    agent: Agent | None,
    benchmark_case: FromDbBenchmarkCase | None,
) -> str:
    result = result if isinstance(result, dict) else {}
    diagnostics = result.get("diagnostics") or {}
    completeness = diagnostics.get("db_completeness") or {}
    tool_calls = _diagnostic_tool_calls(result, agent)
    error_text = str(error).lower()

    if _is_from_db_fast_path(result) and "total_tokens" in error_text:
        return "fast_path_misclassified"
    if "max iterations" in error_text:
        return "max_iterations"
    if "loop detected" in error_text:
        return "tool_routing_loop"
    if _malformed_tool_argument_errors(tool_calls, diagnostics):
        return "tool_argument_parse_failure"
    if diagnostics.get("exit_reason") in {"max_iterations", "max_iterations_partial"}:
        return "max_iterations"
    if _has_catalog_dead_end(tool_calls):
        return "catalog_dead_end"
    if _repeated_values(_tool_error_fingerprints(tool_calls)):
        return "tool_routing_loop"
    if _has_sql_validation_failure(tool_calls):
        return "sql_validation_failure"
    if completeness.get("can_answer") is True or completeness.get("status") in {
        "answerable",
        "answerable_empty",
        "answerable_with_caveats",
    }:
        return "finalization_failure"
    if isinstance(error, AssertionError):
        return "benchmark_assertion_failure"
    return "runtime_failure"


def _summarize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for call in tool_calls[-12:]:
        result = call.get("result", call.get("output"))
        result_summary = {}
        if isinstance(result, dict):
            result_summary = {
                "ok": result.get("ok"),
                "error_type": result.get("error_type"),
                "status": result.get("status"),
                "repair_required": bool(result.get("repair_required")),
                "preflight_failed": bool(result.get("preflight_failed")),
                "blocked_repeat": bool(result.get("blocked_repeat")),
                "guardrail": result.get("guardrail"),
                "suggested_next_tool": result.get("suggested_next_tool"),
                "row_count": result.get("row_count") or result.get("total_rows"),
            }
        summaries.append(
            {
                "tool": _tool_call_name(call),
                "argument_keys": sorted(
                    (call.get("arguments") or call.get("input") or {}).keys()
                ),
                "duration_ms": call.get("duration_ms"),
                "result": {
                    key: value
                    for key, value in result_summary.items()
                    if value not in (None, False)
                },
            }
        )
    return summaries


def _tool_error_fingerprints(tool_calls: list[dict[str, Any]]) -> list[str]:
    fingerprints = []
    for call in tool_calls:
        result = call.get("result", call.get("output"))
        if not isinstance(result, dict):
            continue
        markers = [
            result.get("error_type"),
            result.get("status"),
            result.get("guardrail"),
            "repair_required" if result.get("repair_required") else None,
            "preflight_failed" if result.get("preflight_failed") else None,
            "blocked_repeat" if result.get("blocked_repeat") else None,
        ]
        marker = next((str(item) for item in markers if item), None)
        if marker:
            fingerprints.append(f"{_tool_call_name(call)}:{marker}")
    return fingerprints


def _repeated_values(values: list[str]) -> list[str]:
    repeated = []
    seen = set()
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _malformed_tool_argument_errors(
    tool_calls: list[dict[str, Any]], diagnostics: dict[str, Any]
) -> list[str]:
    candidates = [json.dumps(diagnostics, default=str).lower()]
    for call in tool_calls:
        result = call.get("result", call.get("output"))
        if isinstance(result, dict):
            candidates.append(json.dumps(result, default=str).lower())
    parse_markers = ("malformed", "invalid json", "parse", "tool argument")
    return [
        _truncate_for_diagnostics(candidate, 240)
        for candidate in candidates
        if any(marker in candidate for marker in parse_markers)
        and any(marker in candidate for marker in ("error", "exception", "failed"))
    ][:5]


def _has_catalog_dead_end(tool_calls: list[dict[str, Any]]) -> bool:
    for call in tool_calls:
        tool_name = _tool_call_name(call)
        result = call.get("result", call.get("output"))
        if not tool_name.startswith("catalog_") or not isinstance(result, dict):
            continue
        text = json.dumps(result, default=str).lower()
        if result.get("error") or any(
            marker in text
            for marker in ("store", "profile", "not found", "unavailable", "unknown")
        ):
            return True
    return False


def _has_sql_validation_failure(tool_calls: list[dict[str, Any]]) -> bool:
    for call in tool_calls:
        result = call.get("result", call.get("output"))
        if not isinstance(result, dict):
            continue
        if result.get("repair_required") or result.get("preflight_failed"):
            return True
        if result.get("error_type") in {
            "schema_reference_error",
            "required_field_error",
            "sql_validation_error",
        }:
            return True
    return False


def _truncate_for_diagnostics(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


def _diagnostic_tool_calls(
    result: dict[str, Any], agent: Agent | None
) -> list[dict[str, Any]]:
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        return tool_calls
    history = getattr(agent, "_tool_call_history", None) if agent is not None else None
    return history if isinstance(history, list) else []


def _tool_call_name(call: dict[str, Any]) -> str:
    return str(call.get("tool") or call.get("name") or "")


def _catalog_tools_attempted(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [
        _tool_call_name(call)
        for call in tool_calls
        if _tool_call_name(call).startswith("catalog_")
    ]


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _token_estimate(data: Any) -> int:
    return len(json.dumps(data, default=str)) // 4


def _seed_sqlite(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                signup_date TEXT NOT NULL
            );

            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                total_amount REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE daily_metrics (
                metric_date TEXT PRIMARY KEY,
                revenue REAL NOT NULL,
                orders_count INTEGER NOT NULL,
                sessions INTEGER NOT NULL
            );
            """)
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            [
                (1, "Alice", "alice@example.com", "2026-01-01"),
                (2, "Bob", "bob@example.com", "2026-01-02"),
                (3, "Cara", "cara@example.com", "2026-01-03"),
            ],
        )
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            [
                (1, 1, 50.0, "shipped", "2026-02-01T10:00:00"),
                (2, 1, 75.0, "shipped", "2026-02-02T10:00:00"),
                (3, 2, 150.0, "pending", "2026-02-03T10:00:00"),
            ],
        )
        conn.executemany(
            "INSERT INTO daily_metrics VALUES (?, ?, ?, ?)",
            [
                ("2026-02-01", 100.0, 10, 100),
                ("2026-02-02", 120.0, 12, 120),
                ("2026-02-03", 140.0, 14, 140),
                ("2026-02-04", 160.0, 16, 160),
                ("2026-02-05", 180.0, 18, 180),
                ("2026-02-06", 1000.0, 20, 200),
            ],
        )
        conn.commit()
    finally:
        conn.close()
