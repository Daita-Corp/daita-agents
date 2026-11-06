"""A route's reasoning effort reaches each provider's wire request unchanged."""

import pytest

from daita.hosting.configuration import (
    _encode_model_route_contract,
    _model_execution_contracts,
    _model_route,
)
from daita.llm.factory import create_llm_provider
from daita.llm.models import CanonicalMessage, MessageRole, ModelRequest, TextBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.gemini.messages import _count_request
from daita.llm.routing import ModelRouteCandidate

_REQUEST = ModelRequest(
    messages=(CanonicalMessage(role=MessageRole.USER, content=(TextBlock("Hi."),)),)
)


def _field(mapping: object, *path: str) -> object:
    for key in path:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _anthropic(arguments: dict[str, object]) -> object:
    return _field(arguments, "output_config", "effort")


def _openai(arguments: dict[str, object]) -> object:
    return _field(arguments, "reasoning", "effort")


def _gemini(arguments: dict[str, object]) -> object:
    return _field(arguments, "config", "thinking_config", "thinking_level")


@pytest.mark.parametrize(
    ("model_id", "effort", "wire_effort"),
    (
        ("anthropic:claude-opus-5-5", "xhigh", _anthropic),
        ("openai:gpt-6-sol", "low", _openai),
        ("gemini:gemini-3.8-flash", "minimal", _gemini),
    ),
)
def test_effort_is_sent_only_when_the_route_sets_one(model_id, effort, wire_effort):
    pinned = create_llm_provider(model_id, api_key="offline", reasoning_effort=effort)
    default = create_llm_provider(model_id, api_key="offline")

    assert wire_effort(pinned._request_arguments(_REQUEST)) == effort  # type: ignore[attr-defined]
    assert wire_effort(default._request_arguments(_REQUEST)) is None  # type: ignore[attr-defined]


def test_anthropic_effort_and_structured_output_share_output_config():
    provider = create_llm_provider(
        "anthropic:claude-opus-5-5", api_key="offline", reasoning_effort="high"
    )
    arguments = provider._request_arguments(  # type: ignore[attr-defined]
        ModelRequest(messages=_REQUEST.messages, response_schema={"type": "object"})
    )
    assert arguments["output_config"]["effort"] == "high"
    assert arguments["output_config"]["format"]["type"] == "json_schema"


def test_gemini_counts_a_request_that_pins_a_thinking_level():
    provider = create_llm_provider(
        "gemini:gemini-3.8-flash", api_key="offline", reasoning_effort="high"
    )
    body = _count_request(provider._request_arguments(_REQUEST))  # type: ignore[attr-defined]
    assert body["contents"]


@pytest.mark.parametrize(
    ("model_id", "effort"),
    (
        ("gemini:gemini-3.8-flash", "max"),
        ("anthropic:claude-opus-5-5", "minimal"),
        ("claude-code:claude-opus-5-5", "high"),
    ),
)
def test_a_level_the_provider_does_not_accept_is_refused(model_id, effort):
    profile = reviewed_model_profile(model_id)
    assert profile is not None
    with pytest.raises(ValueError, match="reasoning effort"):
        ModelRouteCandidate(model_id, profile, reasoning_effort=effort)
    with pytest.raises(ValueError, match="reasoning effort"):
        create_llm_provider(model_id, reasoning_effort=effort)


def test_custom_providers_refuse_a_reasoning_effort():
    with pytest.raises(ValueError, match="reasoning effort"):
        create_llm_provider(
            "private:model",
            base_url="https://models.example.test/v1",
            reasoning_effort="high",
        )


def _route(effort: str | None):
    return _model_route(
        "anthropic",
        "claude-opus-5-5",
        base_url=None,
        secret_reference=None,
        context_window_tokens=None,
        max_output_tokens=None,
        reasoning_effort=effort,
    )


def test_machine_run_binding_changes_with_the_effort_and_not_without_one():
    # Routes saved before the setting existed keep their exact contract bytes.
    unset = _encode_model_route_contract(_route(None))
    assert "reasoning_effort" not in unset["candidates"][0]  # type: ignore[index]

    def contracts(effort: str | None) -> dict[str, str]:
        return _model_execution_contracts(None, None, _route(effort))

    assert contracts("high") != contracts(None)
    assert contracts("high") != contracts("low")
    assert contracts("high") == contracts("high")
