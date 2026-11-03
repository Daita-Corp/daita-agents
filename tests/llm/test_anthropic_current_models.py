import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.llm.errors import ModelProviderError
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.anthropic.messages import _message_input


def test_anthropic_preserves_signed_omitted_thinking_blocks():
    provider = AnthropicMessagesProvider("claude-opus-5-5")

    response = provider._decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "message-1",
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "content": (
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "signed-opaque-thinking",
                },
                {"type": "text", "text": "Done."},
            ),
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
            },
        }
    )

    continuation = response.provider_metadata["anthropic_continuation"]

    assert isinstance(continuation, FrozenJsonObject)
    assert continuation.to_dict()["content_blocks"] == [
        {
            "type": "thinking",
            "thinking": "",
            "signature": "signed-opaque-thinking",
        }
    ]


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


def test_machine_run_instruction_stays_in_the_system_field() -> None:
    system, messages = _message_input((_PROMPT, _RUN_START), "anthropic")

    assert system == "Agent prompt.\n\nMachine run instruction."
    assert [message["role"] for message in messages] == ["user"]
    assert "Machine run instruction." not in canonical_json(messages)


def test_machine_run_opens_with_the_same_user_turn_at_every_step() -> None:
    _, first = _message_input((_PROMPT, _RUN_START), "anthropic")
    _, later = _message_input((_PROMPT, _RUN_START, *_TOOL_STEP), "anthropic")

    assert [message["role"] for message in later] == ["user", "assistant", "user"]
    assert later[0] == first[0]


def test_machine_run_after_history_never_ends_on_or_joins_an_old_answer() -> None:
    _, first = _message_input((_PROMPT, *_PRIOR_TURN, _RUN_START), "anthropic")
    _, later = _message_input(
        (_PROMPT, *_PRIOR_TURN, _RUN_START, *_TOOL_STEP), "anthropic"
    )

    assert [message["role"] for message in first] == ["user", "assistant", "user"]
    assert [message["role"] for message in later] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert later[:3] == first


def test_foreground_conversation_gains_no_turn() -> None:
    question = _text(MessageRole.USER, "Question.")

    _, messages = _message_input((_PROMPT, question, *_TOOL_STEP), "anthropic")

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == [{"type": "text", "text": "Question."}]


def test_request_with_nothing_to_send_still_fails_locally() -> None:
    with pytest.raises(ModelProviderError, match="no Anthropic messages"):
        _message_input((), "anthropic")
