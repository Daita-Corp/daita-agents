"""Gemini contents keep the shape the API accepts, with all HTTP kept offline."""

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from daita._json import canonical_json
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.gemini.messages import _gemini_contents
from tests.llm._token_counting_support import input_request, is_count, provider_at


def _text(role: MessageRole, text: str) -> CanonicalMessage:
    return CanonicalMessage(role=role, content=(TextBlock(text),))


_PROMPT = _text(MessageRole.SYSTEM, "Agent prompt.")
_RUN_START = _text(MessageRole.SYSTEM, "Machine run instruction.")
_CALL = ToolCall("call-1", "lookup", {})
_TOOL_STEP = (
    CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(_CALL,)),
    CanonicalMessage(
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id=_CALL.id, output={"rows": []}),),
    ),
)
_PRIOR_TURN = (
    _text(MessageRole.USER, "Earlier question."),
    _text(MessageRole.ASSISTANT, "Earlier answer."),
)


async def _sent_bodies(
    messages: tuple[CanonicalMessage, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return what the actual SDK sends to countTokens and generateContent."""
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if is_count(request.url.path):
            return httpx.Response(200, json={"totalTokens": 100})
        return httpx.Response(503, json={"error": {"message": "offline"}})

    async with provider_at("gemini", respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await provider.generate(replace(input_request(), messages=messages))
    assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert len(seen) == 2
    return seen[0]["generateContentRequest"], seen[1]


@pytest.mark.parametrize(
    ("messages", "roles"),
    [
        pytest.param((_PROMPT, _RUN_START), ["user"], id="first-step"),
        pytest.param(
            (_PROMPT, _RUN_START, *_TOOL_STEP),
            ["user", "model", "user"],
            id="after-tool-call",
        ),
        pytest.param(
            (_PROMPT, *_PRIOR_TURN, _RUN_START),
            ["user", "model", "user"],
            id="first-step-after-history",
        ),
        pytest.param(
            (_PROMPT, *_PRIOR_TURN, _RUN_START, *_TOOL_STEP),
            ["user", "model", "user", "model", "user"],
            id="after-tool-call-after-history",
        ),
    ],
)
async def test_machine_run_is_sendable_with_its_instruction_in_the_system_field(
    messages: tuple[CanonicalMessage, ...], roles: list[str]
) -> None:
    counted, generated = await _sent_bodies(messages)

    assert [content["role"] for content in generated["contents"]] == roles
    assert counted["contents"] == generated["contents"]
    assert "Machine run instruction." in canonical_json(generated["systemInstruction"])
    assert "Machine run instruction." not in canonical_json(generated["contents"])


def test_machine_run_opens_with_the_same_user_turn_at_every_step() -> None:
    first, _ = _gemini_contents((_PROMPT, _RUN_START), "gemini")
    later, _ = _gemini_contents((_PROMPT, _RUN_START, *_TOOL_STEP), "gemini")

    assert later[0] == first[0]


def test_machine_run_after_history_keeps_earlier_turns_as_a_stable_prefix() -> None:
    first, _ = _gemini_contents((_PROMPT, *_PRIOR_TURN, _RUN_START), "gemini")
    later, _ = _gemini_contents(
        (_PROMPT, *_PRIOR_TURN, _RUN_START, *_TOOL_STEP), "gemini"
    )

    assert later[:3] == first


def test_foreground_conversation_gains_no_turn() -> None:
    question = _text(MessageRole.USER, "Question.")

    contents, system = _gemini_contents(
        (_PROMPT, *_PRIOR_TURN, question, *_TOOL_STEP), "gemini"
    )

    assert system == "Agent prompt."
    assert [content["role"] for content in contents] == [
        "user",
        "model",
        "user",
        "model",
        "user",
    ]
    assert contents[:3] == [
        {"role": "user", "parts": [{"text": "Earlier question."}]},
        {"role": "model", "parts": [{"text": "Earlier answer."}]},
        {"role": "user", "parts": [{"text": "Question."}]},
    ]


def test_request_with_nothing_to_send_still_fails_locally() -> None:
    with pytest.raises(ModelProviderError, match="no Gemini contents"):
        _gemini_contents((), "gemini")
