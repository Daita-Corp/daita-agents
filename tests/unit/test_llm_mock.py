"""
Unit tests for daita/llm/mock.py (MockLLMProvider).

Verifies that the mock provider behaves predictably so it can safely be
used as the test double throughout the rest of the suite.
"""

import pytest

from daita.core.streaming import LLMChunk
from daita.llm.mock import MockLLMProvider


# ===========================================================================
# Helpers
# ===========================================================================

def _user_message(content: str):
    return [{"role": "user", "content": content}]


# ===========================================================================
# Basic generation
# ===========================================================================

class TestMockGenerate:
    async def test_generate_returns_string_without_tools(self):
        mock = MockLLMProvider(delay=0)
        result = await mock.generate(_user_message("hello"))
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_generate_custom_response_via_set_response(self):
        mock = MockLLMProvider(delay=0)
        mock.set_response("hello", "world response")
        result = await mock.generate(_user_message("hello"))
        assert result == "world response"

    async def test_generate_keyword_analyze_triggers_default(self):
        mock = MockLLMProvider(delay=0)
        result = await mock.generate(_user_message("please analyze this data"))
        # The mock detects "analyze" in the prompt and uses the analyze template,
        # which contains "insights" and "trends" — verify the right template fired.
        assert "analyze" in result.lower() or "insights" in result.lower()

    async def test_generate_with_tools_returns_dict(self):
        from daita.core.tools import AgentTool

        async def h(args):
            return "ok"

        t = AgentTool(name="t", description="T", parameters={}, handler=h)
        mock = MockLLMProvider(delay=0)
        # _generate_impl is called via generate() after tool conversion
        result = await mock._generate_impl(
            messages=_user_message("use tool"),
            tools=[{"type": "function", "function": {"name": "t"}}],
        )
        assert isinstance(result, dict)
        assert "content" in result

    async def test_generate_string_prompt_accepted(self):
        mock = MockLLMProvider(delay=0)
        result = await mock.generate("plain string prompt")
        assert isinstance(result, str)


# ===========================================================================
# Call history tracking
# ===========================================================================

class TestCallHistory:
    async def test_call_history_recorded_after_generate(self):
        mock = MockLLMProvider(delay=0)
        await mock.generate(_user_message("first call"))
        assert len(mock.call_history) == 1

    async def test_call_history_accumulates(self):
        mock = MockLLMProvider(delay=0)
        await mock.generate(_user_message("call 1"))
        await mock.generate(_user_message("call 2"))
        assert len(mock.call_history) == 2

    async def test_clear_history_empties_list(self):
        mock = MockLLMProvider(delay=0)
        await mock.generate(_user_message("call"))
        mock.clear_history()
        assert len(mock.call_history) == 0

    async def test_get_last_call_returns_most_recent(self):
        mock = MockLLMProvider(delay=0)
        await mock.generate(_user_message("first"))
        await mock.generate(_user_message("second"))
        last = mock.get_last_call()
        assert last is not None

    def test_get_last_call_none_when_empty(self):
        mock = MockLLMProvider(delay=0)
        assert mock.get_last_call() is None


# ===========================================================================
# Streaming
# ===========================================================================

class TestMockStream:
    async def test_stream_yields_llm_chunks(self):
        mock = MockLLMProvider(delay=0)
        chunks = []
        async for chunk in mock._stream_impl(_user_message("hello"), tools=None):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, LLMChunk) for c in chunks)

    async def test_stream_chunks_are_text_type(self):
        mock = MockLLMProvider(delay=0)
        async for chunk in mock._stream_impl(_user_message("hello"), tools=None):
            assert chunk.type == "text"

    async def test_stream_content_nonempty_when_concatenated(self):
        mock = MockLLMProvider(delay=0)
        text = ""
        async for chunk in mock._stream_impl(_user_message("hello"), tools=None):
            text += chunk.content
        assert len(text) > 0


# ===========================================================================
# Token usage
# ===========================================================================

class TestTokenUsage:
    async def test_token_usage_nonzero_after_call(self):
        mock = MockLLMProvider(delay=0)
        await mock._generate_impl(messages=_user_message("hello world"))
        usage = mock._get_last_token_usage()
        assert usage["total_tokens"] > 0
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0


# ===========================================================================
# Info property
# ===========================================================================

class TestMockInfo:
    async def test_info_contains_call_count(self):
        mock = MockLLMProvider(delay=0)
        await mock._generate_impl(messages=_user_message("hi"))
        info = mock.info
        assert "call_count" in info
        assert info["call_count"] == 1

    def test_info_contains_configured_responses(self):
        mock = MockLLMProvider(delay=0)
        mock.set_response("x", "y")
        assert mock.info["configured_responses"] == 1
