"""
Unit tests for ConversationHistory (daita/agents/conversation.py).

Covers:
  - add_turn() appends user+assistant messages in order
  - max_turns windowing
  - max_tokens windowing
  - messages property returns a copy
  - clear() resets in-memory state only
  - save() / load() round-trip (async, atomic write)
  - save() before workspace raises ValueError
  - load() on missing file returns empty history
  - load() on corrupt file returns empty history (no exception)
  - auto_save triggers save() inside add_turn()
  - _set_workspace() derives workspace from agent_id
  - turn_count and session_path properties
  - Agent integration: history= parameter flows through run() correctly
"""

import json
from pathlib import Path
from typing import Any, List

import pytest

from daita.agents.agent import Agent
from daita.agents.conversation import (
    ConversationHistory,
    _derive_workspace,
    _estimate_tokens,
)

from tests.conftest import SequentialMockLLM

# ===========================================================================
# Helpers
# ===========================================================================


def _make_agent(responses: List[Any]) -> Agent:
    llm = SequentialMockLLM(response_sequence=responses)
    return Agent(name="support_bot", llm_provider=llm)


def _history(tmp_path: Path, **kwargs) -> ConversationHistory:
    """Return a ConversationHistory rooted in tmp_path to avoid polluting the repo."""
    return ConversationHistory(
        session_id=kwargs.pop("session_id", "test-session"),
        workspace=kwargs.pop("workspace", "test_ws"),
        base_dir=tmp_path,
        **kwargs,
    )


# ===========================================================================
# _derive_workspace helper
# ===========================================================================


class TestDeriveWorkspace:
    def test_strips_uuid_suffix(self):
        assert _derive_workspace("support_bot_a1b2c3ef") == "support_bot"

    def test_strips_short_suffix(self):
        assert _derive_workspace("my_agent_abc12345") == "my_agent"

    def test_no_underscore_returns_as_is(self):
        assert _derive_workspace("myagent") == "myagent"

    def test_short_id_returns_as_is(self):
        # len <= 9, must not strip
        assert _derive_workspace("a_b") == "a_b"


# ===========================================================================
# _estimate_tokens helper
# ===========================================================================


class TestEstimateTokens:
    def test_empty_returns_zero(self):
        assert _estimate_tokens([]) == 0

    def test_approximation(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        assert _estimate_tokens(msgs) == 100

    def test_missing_content_key(self):
        # Messages without content (e.g. tool_calls) don't crash
        assert _estimate_tokens([{"role": "assistant"}]) == 0


# ===========================================================================
# add_turn
# ===========================================================================


class TestAddTurn:
    async def test_appends_two_messages(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("Hello", "Hi there!")
        assert len(h._messages) == 2

    async def test_message_order(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("user msg", "assistant msg")
        assert h._messages[0] == {"role": "user", "content": "user msg"}
        assert h._messages[1] == {"role": "assistant", "content": "assistant msg"}

    async def test_multiple_turns_accumulate(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("u1", "a1")
        await h.add_turn("u2", "a2")
        assert len(h._messages) == 4
        assert h._messages[2]["content"] == "u2"
        assert h._messages[3]["content"] == "a2"


# ===========================================================================
# max_turns windowing
# ===========================================================================


class TestMaxTurns:
    async def test_windowing_drops_oldest(self, tmp_path):
        h = _history(tmp_path, max_turns=2)
        await h.add_turn("u1", "a1")
        await h.add_turn("u2", "a2")
        await h.add_turn("u3", "a3")
        assert h.turn_count == 2
        assert h._messages[0]["content"] == "u2"
        assert h._messages[-1]["content"] == "a3"

    async def test_none_keeps_all(self, tmp_path):
        h = _history(tmp_path, max_turns=None)
        for i in range(10):
            await h.add_turn(f"u{i}", f"a{i}")
        assert h.turn_count == 10

    async def test_max_turns_one(self, tmp_path):
        h = _history(tmp_path, max_turns=1)
        await h.add_turn("u1", "a1")
        await h.add_turn("u2", "a2")
        assert h.turn_count == 1
        assert h._messages[0]["content"] == "u2"


# ===========================================================================
# max_tokens windowing
# ===========================================================================


class TestMaxTokens:
    async def test_drops_oldest_when_over_budget(self, tmp_path):
        # Each turn ≈ 100 tokens (400 chars per message × 2 messages = 800 chars ÷ 4)
        long_content = "x" * 400
        h = _history(tmp_path, max_tokens=150)
        await h.add_turn(
            long_content, long_content
        )  # ~200 tokens — over budget on its own
        # After windowing: only the current turn, but it's still over 150.
        # Loop drops pairs until 0 messages remain (can't drop partial turns).
        # At that point len < 2 so the while exits — empty history is acceptable.
        assert h.turn_count == 0

    async def test_retains_turns_within_budget(self, tmp_path):
        h = _history(tmp_path, max_tokens=500)
        # Each turn is ~10 tokens (40 chars ÷ 4)
        for i in range(5):
            await h.add_turn("short user msg.", "short reply.")
        assert h.turn_count == 5

    async def test_applied_after_max_turns(self, tmp_path):
        # max_turns=3 runs first, then max_tokens trims further if still over budget
        long_content = "x" * 800  # ~200 tokens per message, ~400 per turn
        h = _history(tmp_path, max_turns=3, max_tokens=100)
        for _ in range(4):
            await h.add_turn(long_content, long_content)
        # max_turns=3 leaves 3 turns (~1200 tokens); max_tokens=100 drops all
        assert h.turn_count == 0

    async def test_none_applies_no_token_limit(self, tmp_path):
        h = _history(tmp_path, max_tokens=None)
        for _ in range(20):
            await h.add_turn("x" * 400, "x" * 400)
        assert h.turn_count == 20


# ===========================================================================
# messages property
# ===========================================================================


class TestMessagesProperty:
    async def test_returns_copy(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("hello", "world")
        snapshot = h.messages
        snapshot.append({"role": "user", "content": "injected"})
        assert len(h._messages) == 2

    def test_empty_initially(self, tmp_path):
        h = _history(tmp_path)
        assert h.messages == []


# ===========================================================================
# clear()
# ===========================================================================


class TestClear:
    async def test_clear_empties_messages(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("a", "b")
        h.clear()
        assert h.messages == []

    async def test_save_after_clear_writes_empty_list(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("a", "b")
        h.clear()
        path = await h.save()
        assert json.loads(path.read_text()) == []


# ===========================================================================
# save() / load()
# ===========================================================================


class TestSaveLoad:
    async def test_save_creates_file(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("hi", "hello")
        path = await h.save()
        assert path.exists()

    async def test_save_returns_path(self, tmp_path):
        h = _history(tmp_path)
        result = await h.save()
        assert isinstance(result, Path)

    async def test_save_creates_parent_directories(self, tmp_path):
        h = ConversationHistory(
            session_id="deep-session",
            workspace="deep_ws",
            base_dir=tmp_path / "nonexistent" / "nested",
        )
        path = await h.save()
        assert path.exists()

    async def test_atomic_write_no_tmp_file_left(self, tmp_path):
        h = _history(tmp_path)
        await h.save()
        tmp = h.session_path.with_suffix(".tmp")
        assert not tmp.exists()

    async def test_save_raises_without_workspace(self):
        h = ConversationHistory(session_id="s")
        with pytest.raises(ValueError, match="workspace is not set"):
            await h.save()

    async def test_load_restores_messages(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("msg1", "resp1")
        await h.add_turn("msg2", "resp2")
        await h.save()

        loaded = await ConversationHistory.load(
            session_id="test-session",
            workspace="test_ws",
            base_dir=tmp_path,
        )
        assert loaded.messages == h.messages

    async def test_load_missing_file_returns_empty(self, tmp_path):
        loaded = await ConversationHistory.load(
            session_id="no-such-session",
            workspace="test_ws",
            base_dir=tmp_path,
        )
        assert loaded.messages == []
        assert loaded.turn_count == 0

    async def test_load_corrupt_file_returns_empty(self, tmp_path):
        # Write malformed JSON directly
        path = tmp_path / "test_ws" / "corrupt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not valid json [[[")

        loaded = await ConversationHistory.load(
            session_id="corrupt",
            workspace="test_ws",
            base_dir=tmp_path,
        )
        assert loaded.messages == []

    async def test_load_preserves_params(self, tmp_path):
        loaded = await ConversationHistory.load(
            session_id="s",
            workspace="ws",
            max_turns=5,
            max_tokens=3000,
            auto_save=True,
            base_dir=tmp_path,
        )
        assert loaded.max_turns == 5
        assert loaded.max_tokens == 3000
        assert loaded.auto_save is True


# ===========================================================================
# auto_save
# ===========================================================================


class TestAutoSave:
    async def test_auto_save_true_writes_file(self, tmp_path):
        h = _history(tmp_path, auto_save=True)
        await h.add_turn("hi", "hello")
        assert (tmp_path / "test_ws" / "test-session.json").exists()

    async def test_auto_save_false_does_not_write_file(self, tmp_path):
        h = _history(tmp_path, auto_save=False)
        await h.add_turn("hi", "hello")
        assert not (tmp_path / "test_ws" / "test-session.json").exists()


# ===========================================================================
# _set_workspace()
# ===========================================================================


class TestSetWorkspace:
    def test_derives_workspace_when_not_set(self):
        h = ConversationHistory(session_id="s")
        h._set_workspace("my_agent_a1b2c3d4")
        assert h.workspace == "my_agent"

    def test_does_not_override_existing_workspace(self):
        h = ConversationHistory(session_id="s", workspace="explicit")
        h._set_workspace("other_agent_xyz12345")
        assert h.workspace == "explicit"


# ===========================================================================
# turn_count and session_path properties
# ===========================================================================


class TestProperties:
    def test_turn_count_zero_initially(self, tmp_path):
        assert _history(tmp_path).turn_count == 0

    async def test_turn_count_increments(self, tmp_path):
        h = _history(tmp_path)
        await h.add_turn("a", "b")
        assert h.turn_count == 1
        await h.add_turn("c", "d")
        assert h.turn_count == 2

    def test_session_path_none_without_workspace(self):
        assert ConversationHistory(session_id="s").session_path is None

    def test_session_path_set_with_workspace(self, tmp_path):
        h = _history(tmp_path)
        assert h.session_path is not None
        assert h.session_path.name == "test-session.json"


# ===========================================================================
# Agent integration tests
# ===========================================================================


class TestAgentIntegration:
    async def test_run_without_history_unchanged(self):
        agent = _make_agent(["Hello!"])
        result = await agent.run("Say hi")
        assert isinstance(result, str)

    async def test_run_with_history_adds_turn(self, tmp_path):
        agent = _make_agent(["Hello!"])
        h = _history(tmp_path)
        await agent.run("Say hi", history=h)
        assert h.turn_count == 1
        assert h._messages[0]["content"] == "Say hi"
        assert h._messages[1]["content"] == "Hello!"

    async def test_sequential_runs_accumulate(self, tmp_path):
        agent = _make_agent(["Turn 1 response.", "Turn 2 response."])
        h = _history(tmp_path)
        await agent.run("First message", history=h)
        await agent.run("Second message", history=h)
        assert h.turn_count == 2

    async def test_prior_messages_injected_into_llm(self, tmp_path):
        llm = SequentialMockLLM(response_sequence=["I remember!"])
        agent = Agent(name="support_bot", llm_provider=llm)
        h = _history(tmp_path)
        await h.add_turn("Hi, I'm Alice.", "Hello Alice!")

        await agent.run("What's my name?", history=h)

        messages = llm.call_history[0]["messages"]
        contents = [m["content"] for m in messages if "content" in m]
        assert "Hi, I'm Alice." in contents
        assert "Hello Alice!" in contents
        assert contents[-1] == "What's my name?"

    async def test_history_with_tool_call_stores_only_final_text(self, tmp_path):
        from daita.core.tools import AgentTool

        async def calc(args):
            return args["a"] + args["b"]

        tool = AgentTool(
            name="add",
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "a"},
                    "b": {"type": "integer", "description": "b"},
                },
                "required": ["a", "b"],
            },
            handler=calc,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Let me calculate.",
                    "tool_calls": [
                        {"id": "tc1", "name": "add", "arguments": {"a": 3, "b": 4}}
                    ],
                },
                "The answer is 7.",
            ]
        )
        agent = Agent(name="support_bot", llm_provider=llm, tools=[tool])
        h = _history(tmp_path)
        await agent.run("What is 3 + 4?", history=h)

        assert h.turn_count == 1
        assert h._messages[0] == {"role": "user", "content": "What is 3 + 4?"}
        assert h._messages[1] == {"role": "assistant", "content": "The answer is 7."}

    async def test_history_derives_workspace_from_agent_id(self):
        agent = _make_agent(["Hi!"])
        h = ConversationHistory(session_id="s")
        assert h.workspace is None
        await agent.run("Hello", history=h)
        assert h.workspace == "support_bot"

    async def test_run_with_retry_enabled_passes_initial_messages(self, tmp_path):
        from daita.config.base import AgentConfig, RetryPolicy, RetryStrategy

        config = AgentConfig(
            name="retry_bot",
            enable_retry=True,
            retry_policy=RetryPolicy(
                max_retries=2, strategy=RetryStrategy.FIXED, base_delay=0.1
            ),
        )
        llm = SequentialMockLLM(response_sequence=["Remembered!"])
        agent = Agent(name="retry_bot", llm_provider=llm, config=config)

        h = _history(tmp_path)
        await h.add_turn("My name is Bob.", "Nice to meet you, Bob!")

        await agent.run("Do you know my name?", history=h)

        messages = llm.call_history[0]["messages"]
        contents = [m["content"] for m in messages if "content" in m]
        assert "My name is Bob." in contents
