"""Chat Completions serialization of non-leading system instructions."""

from unittest.mock import ANY

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.openai_compatible.messages import _chat_messages


def _text(role: MessageRole, text: str) -> CanonicalMessage:
    return CanonicalMessage(role=role, content=(TextBlock(text),))


def test_trailing_step_status_joins_the_leading_system_message():
    messages = (
        _text(MessageRole.SYSTEM, "Fixed prompt."),
        _text(MessageRole.USER, "Question."),
        _text(MessageRole.SYSTEM, "Token budget left for this run: 10 tokens."),
    )

    assert _chat_messages(messages, "compatible") == [
        {
            "role": "system",
            "content": "Fixed prompt.\n\nToken budget left for this run: 10 tokens.",
        },
        {"role": "user", "content": "Question."},
    ]


def test_leading_system_messages_stay_separate():
    messages = (
        _text(MessageRole.SYSTEM, "Fixed prompt."),
        _text(MessageRole.SYSTEM, "Setup."),
        _text(MessageRole.USER, "Question."),
    )

    assert [item["role"] for item in _chat_messages(messages, "compatible")] == [
        "system",
        "system",
        "user",
    ]


def test_a_run_note_follows_the_tool_results_as_user_text():
    call = ToolCall("call-1", "lookup", {})
    messages = (
        _text(MessageRole.USER, "Question."),
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(ToolResultBlock(call_id=call.id, output={"rows": []}),),
            run_note="Budget pressure.",
        ),
    )

    assert _chat_messages(messages, "compatible")[-2:] == [
        {"role": "tool", "tool_call_id": call.id, "content": ANY},
        {"role": "user", "content": "Budget pressure."},
    ]
