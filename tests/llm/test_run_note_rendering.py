"""Every adapter renders a run note after the tool result it was written with."""

from __future__ import annotations

from daita._json import canonical_json
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.anthropic.messages import _message_input
from daita.llm.providers.gemini.messages import _gemini_contents
from daita.llm.providers.openai.messages import _response_input
from daita.llm.providers.subscription_cli.envelope import _project_message

_NOTE = "Budget pressure: another round trip may not fit."
_CALL = ToolCall("call-1", "lookup", {})
_MESSAGES = (
    CanonicalMessage(role=MessageRole.USER, content=(TextBlock("Question."),)),
    CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(_CALL,)),
    CanonicalMessage(
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id=_CALL.id, output={"rows": []}),),
        run_note=_NOTE,
    ),
)


def test_anthropic_puts_the_note_after_the_tool_result_in_one_user_turn() -> None:
    _system, messages = _message_input(_MESSAGES, "anthropic:test")
    content = messages[-1]["content"]
    assert isinstance(content, list)
    assert [block["type"] for block in content] == ["tool_result", "text"]
    assert content[-1]["text"] == _NOTE


def test_responses_input_puts_the_note_after_the_function_output() -> None:
    items = _response_input(_MESSAGES, "openai:test")
    assert items[-2]["type"] == "function_call_output"
    assert items[-1] == {"role": "user", "content": _NOTE}


def test_gemini_puts_the_note_after_the_function_response() -> None:
    contents, _system = _gemini_contents(_MESSAGES, "gemini:test")
    parts = contents[-1]["parts"]
    assert isinstance(parts, list)
    assert "function_response" in parts[0]
    assert parts[-1] == {"text": _NOTE}


def test_subscription_envelope_puts_the_note_after_the_result() -> None:
    projected = _project_message(_MESSAGES[-1])
    assert canonical_json(projected["content"]).endswith(
        canonical_json({"type": "text", "text": _NOTE}) + "]"
    )
