from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace

import pytest

from daita.llm.anthropic import AnthropicProvider
from daita.llm.gemini import GeminiProvider
from daita.llm.grok import GrokProvider
from daita.llm.ollama import OllamaProvider
from daita.llm.openai import OpenAIProvider

ASYNC_PROVIDER_FACTORIES = (
    lambda: OpenAIProvider(model="test", api_key="test-key"),
    lambda: AnthropicProvider(model="test", api_key="test-key"),
    lambda: GrokProvider(model="test", api_key="test-key"),
    lambda: OllamaProvider(model="test"),
)


@pytest.mark.parametrize("provider_factory", ASYNC_PROVIDER_FACTORIES)
async def test_async_provider_closes_created_client_once(provider_factory):
    provider = provider_factory()
    client = AsyncMock()
    provider._client = client

    await provider.aclose()
    await provider.aclose()

    client.aclose.assert_awaited_once_with()
    assert provider._client is None


@pytest.mark.parametrize("provider_factory", ASYNC_PROVIDER_FACTORIES)
async def test_async_provider_close_before_first_use_stays_lazy(provider_factory):
    provider = provider_factory()

    await provider.aclose()

    assert provider._client is None


async def test_gemini_provider_uses_synchronous_client_close():
    provider = GeminiProvider(model="test", api_key="test-key")
    close = Mock()
    client = SimpleNamespace(close=close)
    provider._client = client

    await provider.aclose()
    await provider.aclose()

    close.assert_called_once_with()
    assert provider._client is None


async def test_provider_close_failure_is_surfaced_and_client_is_cleared():
    provider = OpenAIProvider(model="test", api_key="test-key")
    client = AsyncMock()
    client.aclose.side_effect = RuntimeError("client close failed")
    provider._client = client

    with pytest.raises(RuntimeError, match="client close failed"):
        await provider.aclose()

    assert provider._client is None
