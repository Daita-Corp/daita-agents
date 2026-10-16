"""Responses requests route a run's shared prompt prefix to one cache."""

from daita.llm.models import CanonicalMessage, MessageRole, ModelRequest, TextBlock
from daita.llm.providers.openai.adapter import OpenAIResponsesProvider


def _request(*messages: CanonicalMessage) -> ModelRequest:
    return ModelRequest(messages=messages)


def _text(role: MessageRole, text: str) -> CanonicalMessage:
    return CanonicalMessage(role=role, content=(TextBlock(text),))


def test_cache_key_follows_the_system_prompt_only():
    provider = OpenAIResponsesProvider("gpt-5.6-terra", api_key="offline")
    system = _text(MessageRole.SYSTEM, "Fixed prompt.")
    first = provider._request_arguments(
        _request(system, _text(MessageRole.USER, "Question."))
    )
    later = provider._request_arguments(
        _request(
            system,
            _text(MessageRole.USER, "Question."),
            _text(MessageRole.ASSISTANT, "Answer."),
            _text(MessageRole.SYSTEM, "Token budget left for this run: 10 tokens."),
        )
    )
    changed = provider._request_arguments(
        _request(_text(MessageRole.SYSTEM, "Other prompt."))
    )
    key = first["prompt_cache_key"]
    assert isinstance(key, str) and key.startswith("daita-") and len(key) <= 64
    assert later["prompt_cache_key"] == key
    assert changed["prompt_cache_key"] != key
    assert "Fixed prompt" not in key


def test_requests_without_a_system_prompt_send_no_cache_key():
    provider = OpenAIResponsesProvider("gpt-5.6-terra", api_key="offline")
    arguments = provider._request_arguments(
        _request(_text(MessageRole.USER, "Question."))
    )
    assert "prompt_cache_key" not in arguments
