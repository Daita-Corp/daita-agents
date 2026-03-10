"""
Integration tests for ConversationHistory against the real OpenAI API.

Tests:
  - Auto UUID generation on new conversations
  - UUID persisted as the session file name on disk
  - Multi-turn memory: agent recalls context from earlier turns
  - Resume: load() restores a saved session, agent remembers prior context
  - Multiple independent conversations: separate histories stay isolated
  - clear() breaks context — agent no longer recalls prior turns
  - auto_save writes the file after every turn without a manual save()

Run with:
    OPENAI_API_KEY=<key> pytest tests/integration/test_conversation_history_live.py -v

Marked requires_llm so the standard safe-default pytest invocation skips them:
    pytest tests/ -m "not requires_llm and not requires_db"
"""

import os
import uuid
from pathlib import Path

import pytest

from daita import Agent, ConversationHistory

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"  # cheapest capable model; fast for tests


def _agent() -> Agent:
    return Agent(
        name="conv_test_bot",
        llm_provider="openai",
        model=MODEL,
        api_key=OPENAI_API_KEY,
    )


# ---------------------------------------------------------------------------
# 1. New conversation auto-generates a valid UUID session_id
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_new_conversation_generates_uuid_session_id(tmp_path):
    """ConversationHistory() with no session_id should create a valid UUID."""
    history = ConversationHistory(workspace="conv_test_bot", base_dir=tmp_path)

    # UUID must be set before any run — auto-generated at construction
    assert history.session_id is not None
    parsed = uuid.UUID(history.session_id)  # raises if not a valid UUID
    assert str(parsed) == history.session_id


# ---------------------------------------------------------------------------
# 2. UUID is used as the session file name on disk
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_uuid_persisted_as_session_file(tmp_path):
    """After save(), the file on disk should be named <uuid>.json."""
    agent = _agent()
    history = ConversationHistory(
        workspace="conv_test_bot",
        auto_save=True,
        base_dir=tmp_path,
    )

    await agent.run("Hello! My name is Dana.", history=history)

    expected_file = tmp_path / "conv_test_bot" / f"{history.session_id}.json"
    assert expected_file.exists(), f"Expected session file not found: {expected_file}"


# ---------------------------------------------------------------------------
# 3. Multi-turn memory: agent recalls earlier context in the same session
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_agent_recalls_name_within_session(tmp_path):
    """Agent should remember a name provided in an earlier turn of the same session."""
    agent = _agent()
    history = ConversationHistory(workspace="conv_test_bot", base_dir=tmp_path)

    await agent.run("My name is Jordan. Please just say OK.", history=history)
    response = await agent.run(
        "What is my name? Reply with just the name, nothing else.",
        history=history,
    )

    assert (
        "jordan" in response.lower()
    ), f"Expected agent to recall 'Jordan', got: {response!r}"


# ---------------------------------------------------------------------------
# 4. Resume: load() restores session, agent recalls prior context
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_resume_conversation_recalls_prior_context(tmp_path):
    """
    Saving a session and loading it in a new ConversationHistory object should
    let the agent recall information from the original session.
    """
    agent = _agent()

    # --- First session ---
    history = ConversationHistory(
        workspace="conv_test_bot",
        auto_save=True,
        base_dir=tmp_path,
    )
    session_id = history.session_id

    await agent.run("My favourite colour is indigo. Just say OK.", history=history)

    # --- Resume in a fresh history object ---
    resumed = await ConversationHistory.load(
        session_id=session_id,
        workspace="conv_test_bot",
        base_dir=tmp_path,
    )
    assert resumed.turn_count == 1, "Loaded history should have one prior turn"

    response = await agent.run(
        "What is my favourite colour? Reply with just the colour name.",
        history=resumed,
    )

    assert (
        "indigo" in response.lower()
    ), f"Expected agent to recall 'indigo' after resume, got: {response!r}"


# ---------------------------------------------------------------------------
# 5. Multiple independent conversations stay isolated
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_two_conversations_stay_independent(tmp_path):
    """
    Two ConversationHistory instances should each carry separate context;
    information told in one session must not bleed into the other.
    """
    agent = _agent()

    history_a = ConversationHistory(workspace="conv_test_bot", base_dir=tmp_path)
    history_b = ConversationHistory(workspace="conv_test_bot", base_dir=tmp_path)

    # Tell each session a different name
    await agent.run("My name is Alice. Just say OK.", history=history_a)
    await agent.run("My name is Ben. Just say OK.", history=history_b)

    # Each session should recall its own name
    resp_a = await agent.run(
        "What is my name? Reply with just the name.", history=history_a
    )
    resp_b = await agent.run(
        "What is my name? Reply with just the name.", history=history_b
    )

    assert "alice" in resp_a.lower(), f"Session A expected 'Alice', got: {resp_a!r}"
    assert "ben" in resp_b.lower(), f"Session B expected 'Ben', got: {resp_b!r}"

    # The two sessions must have different IDs and different files
    assert history_a.session_id != history_b.session_id


# ---------------------------------------------------------------------------
# 6. clear() breaks in-memory context — agent no longer recalls prior turns
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_clear_breaks_context(tmp_path):
    """After clear(), the agent should have no memory of what was said before."""
    agent = _agent()
    history = ConversationHistory(workspace="conv_test_bot", base_dir=tmp_path)

    await agent.run(
        "My secret code word is ZEPHYR. Please just say OK.", history=history
    )
    assert history.turn_count == 1

    history.clear()
    assert history.turn_count == 0

    response = await agent.run(
        "What was the secret code word I told you? If you don't know, say 'I don't know'.",
        history=history,
    )

    assert (
        "zephyr" not in response.lower()
    ), f"Agent should not recall ZEPHYR after clear(), got: {response!r}"


# ---------------------------------------------------------------------------
# 7. auto_save writes file after every turn without manual save()
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_auto_save_writes_after_each_turn(tmp_path):
    """With auto_save=True, the session file should exist after the first turn."""
    agent = _agent()
    history = ConversationHistory(
        workspace="conv_test_bot",
        auto_save=True,
        base_dir=tmp_path,
    )

    assert history.session_path is not None
    assert not history.session_path.exists(), "File should not exist before any turn"

    await agent.run("Hello. Just say hi back.", history=history)

    assert (
        history.session_path.exists()
    ), "File should exist after first turn with auto_save=True"
    assert history.turn_count == 1

    await agent.run("And again. Just say hi.", history=history)

    assert history.turn_count == 2
    # File should reflect the latest state
    import json

    saved = json.loads(history.session_path.read_text())
    assert len(saved) == 4  # 2 turns × 2 messages each


# ---------------------------------------------------------------------------
# 8. session_id is stable across the lifetime of a ConversationHistory object
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
async def test_session_id_stable_across_turns(tmp_path):
    """session_id must not change between turns."""
    agent = _agent()
    history = ConversationHistory(workspace="conv_test_bot", base_dir=tmp_path)

    sid = history.session_id
    await agent.run("Turn one. Just say OK.", history=history)
    assert history.session_id == sid

    await agent.run("Turn two. Just say OK.", history=history)
    assert history.session_id == sid
