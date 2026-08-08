"""Construct built-in providers and simple provider routes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
import re

from ..security import (
    KeychainStore,
    SecretProvider,
    SecretResolutionError,
    default_secret_provider,
)
from .errors import ModelProviderError, ProviderErrorCode
from .models import ModelRequest, ModelResponse, ModelStreamEvent
from .protocols import ModelProvider, StreamingModelProvider
from .providers import (
    AnthropicProvider,
    ClaudeCodeSubscriptionProvider,
    CodexSubscriptionProvider,
    GeminiProvider,
    GrokBuildSubscriptionProvider,
    GrokProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from .routing import (
    ModelProviderRegistration,
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
)

_PROVIDER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


def create_llm_provider(
    model_id: str,
    *,
    api_key: str | None = None,
    subscription_credential: str | None = None,
    credential_updater: Callable[[str], Awaitable[None]] | None = None,
    base_url: str | None = None,
    max_output_tokens: int = 1_024,
) -> ModelProvider:
    provider_name, separator, model = model_id.partition(":")
    if not separator or not _PROVIDER.fullmatch(provider_name) or not model:
        raise ValueError("model_id must use provider:model form")
    if not isinstance(max_output_tokens, int) or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if provider_name != "codex" and subscription_credential is not None:
        raise ValueError(
            "subscription_credential is only accepted by subscription providers"
        )
    if provider_name != "codex" and credential_updater is not None:
        raise ValueError("credential_updater is only accepted by Codex")
    if provider_name == "openai":
        _fixed_endpoint(provider_name, base_url)
        return OpenAIProvider(
            model, api_key=api_key, max_output_tokens=max_output_tokens
        )
    if provider_name == "anthropic":
        _fixed_endpoint(provider_name, base_url)
        return AnthropicProvider(model, api_key=api_key, max_tokens=max_output_tokens)
    if provider_name == "codex":
        _fixed_endpoint(provider_name, base_url)
        if api_key is not None:
            raise ValueError("codex does not accept an API key")
        if subscription_credential is None:
            raise ValueError("codex requires a Daita subscription login")
        return CodexSubscriptionProvider(
            model,
            credential=subscription_credential,
            credential_updater=credential_updater,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "claude-code":
        _subscription_auth_only(provider_name, api_key, base_url)
        return ClaudeCodeSubscriptionProvider(
            model,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "grok-build":
        _subscription_auth_only(provider_name, api_key, base_url)
        return GrokBuildSubscriptionProvider(
            model,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "gemini":
        _fixed_endpoint(provider_name, base_url)
        return GeminiProvider(
            model, api_key=api_key, max_output_tokens=max_output_tokens
        )
    if provider_name == "grok":
        _fixed_endpoint(provider_name, base_url)
        return GrokProvider(model, api_key=api_key, max_tokens=max_output_tokens)
    if provider_name == "ollama":
        return OllamaProvider(
            model,
            base_url=base_url or "http://127.0.0.1:11434/v1",
            api_key=api_key or "ollama",
            max_tokens=max_output_tokens,
        )
    if base_url is None:
        raise ValueError("custom providers require base_url")
    return OpenAICompatibleProvider(
        model,
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_output_tokens,
    )


def _fixed_endpoint(provider: str, base_url: str | None) -> None:
    if base_url is not None:
        raise ValueError(f"{provider} uses its fixed endpoint")


def _subscription_auth_only(
    provider: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    _fixed_endpoint(provider, base_url)
    if api_key is not None:
        raise ValueError(f"{provider} uses the official client's subscription login")


class _LazyProvider:
    def __init__(self, candidate: ModelRouteCandidate, secrets: SecretProvider) -> None:
        self._candidate = candidate
        self._secrets = secrets
        self._provider: ModelProvider | None = None

    @property
    def provider_id(self) -> str:
        return self._candidate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if self._provider is not None:
            return self._provider.supports_request_policy(request)
        provider_name = self.provider_id.partition(":")[0]
        return (
            request.allow_parallel_tool_calls is None
            or self._candidate.base_url is not None
            or provider_name
            in {
                "openai",
                "grok",
                "ollama",
                "codex",
                "claude-code",
                "grok-build",
            }
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        provider = await self._resolve(request)
        return await provider.generate(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        provider = await self._resolve(request)
        if not self._candidate.profile.supports_streaming or not isinstance(
            provider, StreamingModelProvider
        ):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "configured provider route does not support streaming",
                provider_id=self.provider_id,
            )
        async for event in provider.stream(request):
            yield event

    async def _resolve(self, request: ModelRequest) -> ModelProvider:
        if not self.supports_request_policy(request):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "provider cannot enforce the requested tool policy",
            )
        if self._provider is None:
            reference = self._candidate.secret_reference
            provider_name = self.provider_id.partition(":")[0]
            try:
                credential = (
                    None
                    if reference is None
                    else await self._secrets.resolve(reference)
                )
            except SecretResolutionError as error:
                raise ModelProviderError(
                    _secret_provider_error_code(error),
                    "The configured provider credential could not be resolved.",
                    provider_id=self.provider_id,
                ) from None
            credential_updater: Callable[[str], Awaitable[None]] | None = None
            if provider_name == "codex" and reference is not None:

                async def update_credential(value: str) -> None:
                    if not isinstance(self._secrets, KeychainStore):
                        raise SecretResolutionError(
                            "secret_provider_unavailable",
                            "The configured keychain cannot update the login.",
                        )
                    await self._secrets.set(reference, value)

                credential_updater = update_credential
            self._provider = create_llm_provider(
                self._candidate.provider_id,
                api_key=(credential if provider_name != "codex" else None),
                subscription_credential=(
                    credential if provider_name == "codex" else None
                ),
                credential_updater=credential_updater,
                base_url=self._candidate.base_url,
                max_output_tokens=self._candidate.profile.max_output_tokens,
            )
        return self._provider


def _secret_provider_error_code(
    error: SecretResolutionError,
) -> ProviderErrorCode:
    if error.code == "secret_not_found":
        return ProviderErrorCode.AUTHENTICATION_ERROR
    if error.code == "secret_provider_unavailable":
        return ProviderErrorCode.PROVIDER_UNAVAILABLE
    return ProviderErrorCode.CONFIGURATION_ERROR


def create_model_route_provider(
    route: ModelRoute,
    *,
    secret_provider: SecretProvider | None = None,
) -> ModelProvider:
    if not isinstance(route, ModelRoute):
        raise TypeError("route must be ModelRoute")
    secrets = default_secret_provider(secret_provider)
    registrations = tuple(
        ModelProviderRegistration(
            provider=_LazyProvider(candidate, secrets),
            profile=candidate.profile,
            allowed_sensitivities=candidate.allowed_sensitivities,
        )
        for candidate in route.candidates
    )
    if len(registrations) == 1 and route.retry_policy.attempts == 1:
        return registrations[0].provider
    return ModelRouter(registrations, retry_policy=route.retry_policy)


__all__ = ["create_llm_provider", "create_model_route_provider"]
