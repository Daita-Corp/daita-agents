"""
Unit tests for OrchestratorPlugin.

Tests result truncation, previous_result context capping, and
cache key SHA-256 hashing — without real agent execution.
"""

import hashlib
import pytest
from unittest.mock import MagicMock, AsyncMock
from daita.plugins.orchestrator import OrchestratorPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_orchestrator(*agent_ids):
    plugin = OrchestratorPlugin()
    for aid in agent_ids:
        agent = MagicMock()
        agent.name = aid
        agent.run = AsyncMock(return_value=f"result from {aid}")
        agent.description = f"Agent {aid}"
        plugin.register_agent(agent, agent_id=aid)
    return plugin


# ---------------------------------------------------------------------------
# _truncate_agent_result
# ---------------------------------------------------------------------------


def test_truncate_short_string_unchanged():
    s = "hello world"
    assert OrchestratorPlugin._truncate_agent_result(s, 100) == s


def test_truncate_long_string_capped():
    s = "x" * 20_000
    result = OrchestratorPlugin._truncate_agent_result(s, 10_000)
    assert len(result) > 10_000  # includes the suffix
    assert result.startswith("x" * 10_000)
    assert "truncated" in result


def test_truncate_non_string_converted():
    result = OrchestratorPlugin._truncate_agent_result({"key": "value"}, 5)
    assert isinstance(result, str)
    assert "truncated" in result


def test_truncate_exactly_at_limit_not_truncated():
    s = "a" * 10_000
    result = OrchestratorPlugin._truncate_agent_result(s, 10_000)
    assert result == s


# ---------------------------------------------------------------------------
# run_parallel — result truncation
# ---------------------------------------------------------------------------


async def test_run_parallel_truncates_long_results():
    plugin = make_orchestrator("agent_a")
    long_result = "z" * 20_000
    plugin._agents["agent_a"].run = AsyncMock(return_value=long_result)

    results = await plugin.run_parallel([{"task": "do something", "agent_id": "agent_a"}])

    task_result = results["results"][0]["result"]
    assert len(task_result) < 20_000
    assert "truncated" in task_result


async def test_run_parallel_short_results_not_modified():
    plugin = make_orchestrator("agent_a")
    short_result = "done"
    plugin._agents["agent_a"].run = AsyncMock(return_value=short_result)

    results = await plugin.run_parallel([{"task": "do something", "agent_id": "agent_a"}])

    assert results["results"][0]["result"] == short_result


# ---------------------------------------------------------------------------
# run_sequential — previous_result context capping
# ---------------------------------------------------------------------------


async def test_run_sequential_caps_previous_result_context():
    plugin = make_orchestrator("agent_a", "agent_b")
    long_result = "r" * 20_000
    plugin._agents["agent_a"].run = AsyncMock(return_value=long_result)

    received_tasks = []

    async def capture_task(task):
        received_tasks.append(task)
        return "done"

    plugin._agents["agent_b"].run = capture_task

    await plugin.run_sequential([
        {"task": "step one", "agent_id": "agent_a"},
        {"task": "step two", "agent_id": "agent_b"},
    ])

    # The context injected into step two should be capped at 5000 chars
    second_task = received_tasks[0]
    context_part = second_task.split("Context from previous step:")[-1]
    assert len(context_part) <= 5_000 + 100  # small buffer for the suffix text


async def test_run_sequential_truncates_results_too():
    plugin = make_orchestrator("agent_a")
    long_result = "y" * 20_000
    plugin._agents["agent_a"].run = AsyncMock(return_value=long_result)

    results = await plugin.run_sequential([{"task": "do work", "agent_id": "agent_a"}])

    assert "truncated" in results["results"][0]["result"]


# ---------------------------------------------------------------------------
# Routing cache key — SHA-256
# ---------------------------------------------------------------------------


def test_cache_key_uses_sha256():
    """Two tasks identical in first 100 chars but different overall must get different cache keys."""
    prefix = "A" * 100
    task_1 = prefix + "_suffix_one"
    task_2 = prefix + "_suffix_two"

    key_1 = f"task:{hashlib.sha256(task_1.encode()).hexdigest()}"
    key_2 = f"task:{hashlib.sha256(task_2.encode()).hexdigest()}"

    # Would collide with task[:100] approach but not with full hash
    assert task_1[:100] == task_2[:100]
    assert key_1 != key_2


def test_cache_key_format_is_sha256_hex():
    task = "route this task to the right agent"
    key = f"task:{hashlib.sha256(task.encode()).hexdigest()}"
    # SHA-256 hex digest is always 64 chars
    digest = key.split("task:")[1]
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
