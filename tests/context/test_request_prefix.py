"""A provider can cache a request prefix only if it renders the same bytes."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from daita import Agent
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from daita.llm.providers.anthropic.adapter import AnthropicProvider
from daita.llm.providers.mock import MockModelProvider
from tests.support.workspace import workspace_for

_RENDERER = AnthropicProvider("claude-sonnet-5", api_key="offline")


def _strip_cache_control(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_cache_control(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_strip_cache_control(item) for item in value]
    return value


def _rendered(request: ModelRequest) -> tuple[str, list[str]]:
    """Canonical bytes of tools+system, and of each rendered message."""

    arguments = _strip_cache_control(_RENDERER._request_arguments(request))
    assert isinstance(arguments, dict)

    def encode(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    return (
        encode([arguments.get("tools"), arguments.get("system")]),
        [encode(message) for message in arguments["messages"]],
    )


def _read(call_id: str, sql: str) -> ModelResponse:
    call = ToolCall(
        id=call_id, name="data_query", arguments={"source": "shop", "sql": sql}
    )
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,))


def _answer(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


async def _three_runs(tmp_path):
    """Run three questions of one conversation; return each run's requests."""

    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, status TEXT)")
        connection.executemany(
            "INSERT INTO orders(status) VALUES (?)", [("paid",), ("refunded",)]
        )
    provider = MockModelProvider(
        (
            _read("a1", "SELECT COUNT(*) AS n FROM orders"),
            _read("a2", "SELECT status FROM orders"),
            _answer("There are 2 orders."),
            _read("b1", "SELECT * FROM orders WHERE status = 'paid'"),
            _answer("One was paid."),
            _read("c1", "SELECT COUNT(*) AS n FROM orders"),
            _answer("Still 2."),
        ),
        provider_id="mock:prefix",
    )
    now = [datetime(2026, 3, 2, 9, 0, tzinfo=UTC)]
    agent = await Agent.create(
        "prefix",
        root=tmp_path / "home",
        workspace=workspace_for(tmp_path / "home"),
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=200_000,
            max_output_tokens=8_000,
            supports_tools=True,
        ),
        clock=lambda: now[0],
    )
    try:
        await agent.attach_sqlite(database, name="shop")
        runs: list[tuple[ModelRequest, ...]] = []
        conversation_id = None
        for question in (
            "How many orders are there?",
            "Export the paid orders from 2026-01-03 to a csv file.",
            "And how many in total now?",
        ):
            before = len(provider.requests)
            result = await agent.run(question, conversation_id=conversation_id)
            conversation_id = result.conversation_id
            runs.append(tuple(provider.requests[before:]))
            now[0] += timedelta(hours=7, minutes=13)
        return tuple(runs)
    finally:
        await agent.close()


async def test_request_prefix_is_stable_across_steps_and_runs(tmp_path):
    first, second, third = await _three_runs(tmp_path)
    assert len(first) == 3

    # Within a run, tools and system never change and every earlier request is
    # an exact prefix of the next, so a provider's cache extends step to step.
    for earlier, later in zip(first, first[1:]):
        (earlier_head, earlier_messages), (later_head, later_messages) = (
            _rendered(earlier),
            _rendered(later),
        )
        assert later_head == earlier_head
        assert later_messages[: len(earlier_messages)] == earlier_messages

    # Across runs asking different questions at different times, only the new
    # run's first user turn and what follows may differ.
    heads = {_rendered(request)[0] for request in (*first, *second, *third)}
    assert len(heads) == 1

    # An earlier run's projected turn renders the same bytes in every later run.
    _, closed_after_first = _rendered(second[0])
    _, closed_after_second = _rendered(third[0])
    one_turn = closed_after_first[:-1]
    two_turns = closed_after_second[:-1]
    assert one_turn and len(two_turns) > len(one_turn)
    assert two_turns[: len(one_turn)] == one_turn
    assert _rendered(second[-1])[1][: len(one_turn)] == one_turn
