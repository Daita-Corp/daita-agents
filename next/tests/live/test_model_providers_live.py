"""Credential-backed smoke coverage for every retained model adapter.

Run explicitly with ``DAITA_RUN_LIVE_LLM=1``.  The tests never load a dotenv
file, print credentials, or persist requests/responses.  Each parameter is a
separate release-gate row; a skipped row is not live acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os

import pytest

from daita.llm import (
    AnthropicProvider,
    CanonicalMessage,
    FinishReason,
    GeminiProvider,
    GrokProvider,
    MessageRole,
    ModelRequest,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    StreamingModelProvider,
    TextBlock,
)


def _message(role: MessageRole, text: str) -> CanonicalMessage:
    return CanonicalMessage(
        agent_id="live-agent",
        operation_id="live-operation",
        turn_id="live-turn",
        role=role,
        content=(TextBlock(text),),
    )


def _request(*messages: CanonicalMessage) -> ModelRequest:
    return ModelRequest(
        operation_id="live-operation",
        turn_id="live-turn",
        messages=messages,
    )


@dataclass(frozen=True, slots=True)
class _LiveProviderCase:
    name: str
    model_env: str
    default_model: str
    credential_envs: tuple[str, ...] = ()


CASES = (
    _LiveProviderCase(
        "openai",
        "OPENAI_TEST_MODEL",
        "gpt-4.1-mini",
        ("OPENAI_API_KEY",),
    ),
    _LiveProviderCase(
        "anthropic",
        "ANTHROPIC_TEST_MODEL",
        "claude-haiku-4-5",
        ("ANTHROPIC_API_KEY",),
    ),
    _LiveProviderCase(
        "gemini",
        "GEMINI_TEST_MODEL",
        "gemini-2.5-flash-lite",
        ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    ),
    _LiveProviderCase(
        "grok",
        "GROK_TEST_MODEL",
        "grok-4.20",
        ("XAI_API_KEY", "GROK_API_KEY"),
    ),
    _LiveProviderCase(
        "ollama",
        "OLLAMA_TEST_MODEL",
        "llama3.1",
    ),
    _LiveProviderCase(
        "openai-compatible",
        "DAITA_COMPATIBLE_MODEL",
        "gpt-4.1-mini",
    ),
)


def _require_live_opt_in() -> None:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("requires DAITA_RUN_LIVE_LLM=1")


def _first_environment_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    pytest.skip(f"requires one of these environment variables: {', '.join(names)}")
    raise AssertionError("pytest.skip returned unexpectedly")


def _provider(case: _LiveProviderCase) -> StreamingModelProvider:
    _require_live_opt_in()
    model = os.environ.get(case.model_env, case.default_model).strip()
    if case.name == "ollama":
        enabled = os.environ.get("DAITA_RUN_OLLAMA") == "1"
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
        if not enabled and not base_url:
            pytest.skip("requires OLLAMA_BASE_URL or DAITA_RUN_OLLAMA=1")
        return OllamaProvider(
            model,
            base_url=base_url or "http://127.0.0.1:11434/v1",
            max_tokens=16,
        )

    if case.name == "openai-compatible":
        base_url = os.environ.get("DAITA_COMPATIBLE_BASE_URL", "").strip()
        if base_url:
            api_key = _first_environment_value("DAITA_COMPATIBLE_API_KEY")
            provider_name = os.environ.get(
                "DAITA_COMPATIBLE_PROVIDER",
                "compatible-live",
            ).strip()
        else:
            # The real OpenAI Chat Completions endpoint is a stable explicit
            # compatibility target distinct from the native Responses adapter.
            base_url = "https://api.openai.com/v1"
            api_key = _first_environment_value("OPENAI_API_KEY")
            provider_name = "openai-compatible"
            model = os.environ.get(
                case.model_env,
                os.environ.get("OPENAI_TEST_MODEL", case.default_model),
            ).strip()
        return OpenAICompatibleProvider(
            model,
            provider=provider_name,
            base_url=base_url,
            api_key=api_key,
            max_tokens=16,
        )

    api_key = _first_environment_value(*case.credential_envs)
    if case.name == "openai":
        return OpenAIProvider(model, api_key=api_key, max_output_tokens=16)
    if case.name == "anthropic":
        return AnthropicProvider(model, api_key=api_key, max_tokens=16)
    if case.name == "gemini":
        return GeminiProvider(model, api_key=api_key, max_output_tokens=16)
    if case.name == "grok":
        return GrokProvider(model, api_key=api_key, max_tokens=16)
    raise AssertionError(f"unknown retained live provider row: {case.name}")


async def _close_provider_client(provider: StreamingModelProvider) -> None:
    client = getattr(provider, "_client", None)
    if client is None:
        return
    asynchronous_client = getattr(client, "aio", None)
    close = getattr(asynchronous_client, "aclose", None)
    if not callable(close):
        close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_retained_provider_live_text_conformance(
    case: _LiveProviderCase,
) -> None:
    """Cross the real provider boundary and receive one canonical response."""

    provider = _provider(case)
    try:
        response = await provider.generate(
            _request(
                _message(
                    MessageRole.USER,
                    "Reply with exactly LIVE_OK and no other text.",
                )
            )
        )
    finally:
        await _close_provider_client(provider)

    assert response.provider_id == provider.provider_id
    assert response.finish_reason is FinishReason.STOP
    assert response.text is not None
    assert response.text.strip()
    assert len(response.text) <= 256
    assert response.tool_calls == ()
    assert response.provider_response_id is not None
    assert response.usage.total_tokens > 0

    rendered_metadata = repr(response.provider_metadata)
    for environment_name in (
        *case.credential_envs,
        "DAITA_COMPATIBLE_API_KEY",
        "OPENAI_API_KEY",
    ):
        credential = os.environ.get(environment_name)
        if credential and credential in rendered_metadata:
            pytest.fail("provider metadata retained credential material")
