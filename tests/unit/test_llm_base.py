"""
Unit tests for daita/llm/base.py (BaseLLMProvider shared logic).

Uses MockLLMProvider as the concrete implementation since BaseLLMProvider
is abstract. Tests cover shared behaviour that every provider inherits.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from daita.core.tools import AgentTool
from daita.llm.anthropic import AnthropicProvider
from daita.llm.gemini import GeminiProvider
from daita.llm.grok import GrokProvider
from daita.llm.mock import MockLLMProvider
from daita.llm.openai import OpenAIProvider, _build_token_param

# ===========================================================================
# Helpers
# ===========================================================================


def _make_provider(**kwargs) -> MockLLMProvider:
    return MockLLMProvider(delay=0, **kwargs)


def _make_tool(name: str, return_value: Any = "result"):
    async def h(args):
        return return_value

    return AgentTool(
        name=name,
        description=f"Tool {name}",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=h,
    )


def _make_slow_tool(name: str, sleep: float = 10.0):
    async def h(args):
        await asyncio.sleep(sleep)
        return "never"

    return AgentTool(
        name=name,
        description="Slow",
        parameters={},
        handler=h,
        timeout_seconds=0.01,
    )


# ===========================================================================
# Provider defaults
# ===========================================================================


class TestProviderDefaults:
    def test_openai_default_model_is_current_cost_sensitive_model(self):
        assert OpenAIProvider().model == "gpt-5.4-mini"

    def test_anthropic_default_model_is_current_cost_sensitive_model(self):
        assert AnthropicProvider().model == "claude-haiku-4-5"

    def test_gemini_default_model_is_current_cost_sensitive_model(self):
        assert GeminiProvider().model == "gemini-2.5-flash-lite"

    def test_grok_default_model_is_current_cost_sensitive_model(self):
        assert GrokProvider().model == "grok-4.20"


# ===========================================================================
# OpenAI-compatible helper behavior
# ===========================================================================


class TestOpenAICompatibleHelpers:
    def test_openai_max_tokens_alias_uses_modern_parameter(self):
        assert _build_token_param(max_tokens=100) == {"max_completion_tokens": 100}

    def test_openai_explicit_max_completion_tokens_wins(self):
        assert _build_token_param(max_tokens=100, max_completion_tokens=200) == {
            "max_completion_tokens": 200
        }

    def test_openai_legacy_max_tokens_is_opt_in(self):
        assert _build_token_param(max_tokens=100, use_legacy_max_tokens=True) == {
            "max_tokens": 100
        }

    def test_converts_flat_tool_calls_to_openai_message_shape(self):
        p = OpenAIProvider(api_key="test")
        converted = p._convert_messages_to_openai(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "search",
                            "arguments": {"query": "daita"},
                        }
                    ],
                }
            ]
        )

        assert converted == [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "daita"}',
                        },
                    }
                ],
            }
        ]

    def test_safe_parse_tool_arguments_handles_bad_json(self):
        p = GrokProvider(api_key="test")
        assert p._safe_parse_tool_arguments("{bad json") == {}

    def test_gemini_usage_metadata_extracts_token_counts(self):
        p = GeminiProvider(api_key="test")
        usage = SimpleNamespace(
            total_token_count=12,
            prompt_token_count=5,
            candidates_token_count=7,
        )
        assert p._extract_tokens(usage) == {
            "total_tokens": 12,
            "prompt_tokens": 5,
            "completion_tokens": 7,
        }


# ===========================================================================
# _merge_params
# ===========================================================================


class TestMergeParams:
    def test_no_overrides_returns_defaults(self):
        p = _make_provider()
        merged = p._merge_params({})
        assert "temperature" in merged
        assert "max_tokens" in merged

    def test_override_replaces_default(self):
        p = _make_provider()
        merged = p._merge_params({"temperature": 0.0})
        assert merged["temperature"] == 0.0

    def test_new_key_added(self):
        p = _make_provider()
        merged = p._merge_params({"custom_key": "value"})
        assert merged["custom_key"] == "value"

    def test_original_defaults_unchanged(self):
        p = _make_provider()
        original_temp = p.default_params["temperature"]
        p._merge_params({"temperature": 0.0})
        assert p.default_params["temperature"] == original_temp


# ===========================================================================
# _estimate_cost
# ===========================================================================


class TestEstimateCost:
    def test_zero_tokens_returns_none(self):
        p = _make_provider()
        assert p._estimate_cost({"total_tokens": 0}) is None

    def test_positive_tokens_returns_positive_float(self):
        p = _make_provider()
        cost = p._estimate_cost({"total_tokens": 1000})
        assert isinstance(cost, float)
        assert cost > 0.0

    def test_cost_scales_with_tokens(self):
        p = _make_provider()
        cost_1k = p._estimate_cost({"total_tokens": 1000})
        cost_2k = p._estimate_cost({"total_tokens": 2000})
        assert cost_2k == pytest.approx(cost_1k * 2, rel=1e-6)


# ===========================================================================
# _convert_tools_to_format
# ===========================================================================


class TestConvertToolsToFormat:
    def test_returns_openai_function_format(self):
        p = _make_provider()
        t = _make_tool("search")
        result = p._convert_tools_to_format([t])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"

    def test_multiple_tools_converted(self):
        p = _make_provider()
        result = p._convert_tools_to_format([_make_tool("a"), _make_tool("b")])
        assert len(result) == 2


# ===========================================================================
# _update_accumulated_metrics
# ===========================================================================


class TestAccumulatedMetrics:
    def test_tokens_accumulate_across_calls(self):
        p = _make_provider()
        p._update_accumulated_metrics(
            {"total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40}
        )
        p._update_accumulated_metrics(
            {"total_tokens": 200, "prompt_tokens": 120, "completion_tokens": 80}
        )
        acc = p.get_accumulated_tokens()
        assert acc["total_tokens"] == 300
        assert acc["prompt_tokens"] == 180
        assert acc["completion_tokens"] == 120

    def test_cost_accumulates_across_calls(self):
        p = _make_provider()
        p._update_accumulated_metrics({"total_tokens": 1000})
        p._update_accumulated_metrics({"total_tokens": 1000})
        assert p.get_accumulated_cost() > 0.0

    def test_explicit_cost_overrides_estimate(self):
        p = _make_provider()
        p._update_accumulated_metrics({"total_tokens": 1000}, cost=9.99)
        assert p.get_accumulated_cost() == pytest.approx(9.99)


# ===========================================================================
# get_token_stats
# ===========================================================================


class TestGetTokenStats:
    def test_stats_structure_with_agent_id(self):
        p = _make_provider()
        p.set_agent_id("test-agent-123")
        stats = p.get_token_stats()
        for key in (
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "estimated_cost",
        ):
            assert key in stats

    def test_stats_without_agent_id_returns_zeros(self):
        p = _make_provider()
        # No agent_id set → fallback path
        stats = p.get_token_stats()
        assert stats["total_calls"] == 0
        assert stats["total_tokens"] == 0


# ===========================================================================
# set_agent_id
# ===========================================================================


class TestSetAgentId:
    def test_stores_agent_id(self):
        p = _make_provider()
        p.set_agent_id("abc-123")
        assert p.agent_id == "abc-123"


# ===========================================================================
# provider.info property
# ===========================================================================


class TestProviderInfo:
    def test_info_has_required_keys(self):
        p = _make_provider()
        info = p.info
        for key in (
            "provider",
            "model",
            "agent_id",
            "config",
            "default_params",
            "tracing_enabled",
        ):
            assert key in info

    def test_info_excludes_api_key(self):
        p = MockLLMProvider(delay=0)
        info = p.info
        # No key containing 'key' should appear in config
        for k in info.get("config", {}):
            assert "key" not in k.lower()

    def test_tracing_enabled_is_true(self):
        p = _make_provider()
        assert p.info["tracing_enabled"] is True

    def test_model_name_alias(self):
        p = MockLLMProvider(model="test-model-1", delay=0)
        assert p.model_name == "test-model-1"
