"""
Live LLM performance benchmarks for the agent runtime.

These tests intentionally require real provider credentials. They measure the
agent framework in the only environment that captures real latency, token
usage, and tool behavior.

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daita.agents.agent import Agent
from daita.core.tools import LocalTool

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

    tool = LocalTool(
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
        LocalTool(
            name="fetch_top_customer",
            description="Return the top customer id and name.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=fetch_customer,
            replay_safe=True,
            retry_safe=True,
            side_effecting=False,
        ),
        LocalTool(
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

    tool = LocalTool(
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
) -> None:
    assert elapsed_ms > 0
    assert elapsed_ms <= float(
        os.environ.get("DAITA_BENCHMARK_MAX_LATENCY_MS", "90000")
    )
    assert result["iterations"] <= max_iterations
    assert len(result.get("tool_calls") or []) <= max_tools
    tokens = result.get("tokens") or {}
    assert tokens.get("total_tokens", 0) > 0


def _assert_result_contains(result: dict[str, Any], expected: str) -> None:
    assert expected.lower() in str(result.get("result", "")).lower()


def _assert_tool_sequence(result: dict[str, Any], expected: list[str]) -> None:
    sequence = [call.get("tool") for call in result.get("tool_calls") or []]
    assert sequence == expected


def _assert_tool_was_used(result: dict[str, Any], expected_tool: str) -> None:
    sequence = [call.get("tool") for call in result.get("tool_calls") or []]
    assert expected_tool in sequence


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
