"""Explicit construction of retained built-in model-provider adapters."""

from __future__ import annotations

import re

from ..security.secrets import (
    SecretProvider,
    SecretReference,
    default_secret_provider,
)
from .models import ModelProfile, ModelRequest, ModelResponse, ModelSensitivity
from .protocols import ModelProvider
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    GrokProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenAIResponsesProvider,
)
from .routing import (
    ModelProviderRegistration,
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
    RetryPolicy,
)

_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


def create_llm_provider(
    model_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_output_tokens: int = 1_024,
) -> ModelProvider:
    """Create one explicit retained adapter from a canonical provider:model ID.

    Unknown provider names are accepted only with an explicit compatible API
    endpoint. The resulting adapter remains a normal registry/router input;
    this function does not mutate global process state.
    """

    if (
        not isinstance(model_id, str)
        or model_id != model_id.strip()
        or len(model_id) > 256
        or any(character.isspace() for character in model_id)
    ):
        raise ValueError("model_id must be canonical provider:model text")
    provider_name, separator, model = model_id.partition(":")
    if (
        not separator
        or not _PROVIDER_NAME.fullmatch(provider_name)
        or not model
        or model != model.strip()
    ):
        raise ValueError("model_id must use canonical provider:model form")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be a non-empty string when provided")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")

    if provider_name == "openai":
        _reject_base_url(provider_name, base_url)
        return OpenAIProvider(
            model,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "anthropic":
        _reject_base_url(provider_name, base_url)
        return AnthropicProvider(
            model,
            api_key=api_key,
            max_tokens=max_output_tokens,
        )
    if provider_name == "gemini":
        _reject_base_url(provider_name, base_url)
        return GeminiProvider(
            model,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "grok":
        _reject_base_url(provider_name, base_url)
        return GrokProvider(
            model,
            api_key=api_key,
            max_tokens=max_output_tokens,
        )
    if provider_name == "ollama":
        return OllamaProvider(
            model,
            base_url=("http://127.0.0.1:11434/v1" if base_url is None else base_url),
            api_key="ollama" if api_key is None else api_key,
            max_tokens=max_output_tokens,
        )
    if base_url is None:
        raise ValueError(
            "an explicit base_url is required for an OpenAI-compatible provider"
        )
    return OpenAICompatibleProvider(
        model,
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_output_tokens,
    )


def _reject_base_url(provider_name: str, base_url: str | None) -> None:
    if base_url is not None:
        raise ValueError(f"{provider_name} uses its fixed provider endpoint")


class _LazyRouteProvider:
    """Resolve one persisted secret and construct its retained adapter on use."""

    __slots__ = ("_candidate", "_provider", "_secrets")

    def __init__(
        self,
        candidate: ModelRouteCandidate,
        secrets: SecretProvider,
    ) -> None:
        self._candidate = candidate
        self._secrets = secrets
        self._provider: ModelProvider | None = None

    @property
    def provider_id(self) -> str:
        return self._candidate.provider_id

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if self._provider is None:
            reference = self._candidate.secret_reference
            api_key = (
                None if reference is None else await self._secrets.resolve(reference)
            )
            self._provider = create_llm_provider(
                self._candidate.provider_id,
                api_key=api_key,
                base_url=self._candidate.base_url,
                max_output_tokens=self._candidate.profile.max_output_tokens,
            )
        return await self._provider.generate(request)

    def __repr__(self) -> str:
        return f"_LazyRouteProvider(provider_id={self.provider_id!r})"


def create_model_route_provider(
    route: ModelRoute,
    *,
    secret_provider: SecretProvider | None = None,
) -> ModelProvider:
    """Reconstruct a persisted route without resolving any secret eagerly."""

    if not isinstance(route, ModelRoute):
        raise TypeError("route must be a ModelRoute")
    secrets = default_secret_provider(secret_provider)
    registrations = tuple(
        ModelProviderRegistration(
            provider=_LazyRouteProvider(candidate, secrets),
            profile=candidate.profile,
            allowed_sensitivities=candidate.allowed_sensitivities,
        )
        for candidate in route.candidates
    )
    if len(registrations) == 1 and route.retry_policy == RetryPolicy():
        return registrations[0].provider
    return ModelRouter(
        registrations[0],
        registrations[1:],
        retry_policy=route.retry_policy,
    )


def model_route_from_provider(
    provider: ModelProvider,
    profile: ModelProfile,
    *,
    revision: int = 1,
) -> ModelRoute | None:
    """Describe a reconstructable retained provider; return None for custom ones."""

    if not isinstance(profile, ModelProfile):
        raise TypeError("profile must be a ModelProfile")
    if isinstance(provider, ModelRouter):
        candidates: list[ModelRouteCandidate] = []
        for registration in provider.candidates:
            candidate = _route_candidate_from_provider(
                registration.provider,
                registration.profile,
                allowed_sensitivities=registration.allowed_sensitivities,
            )
            if candidate is None:
                return None
            candidates.append(candidate)
        route = ModelRoute(
            candidates=tuple(candidates),
            retry_policy=provider.retry_policy,
            revision=revision,
        )
        if route.model_profile != profile:
            raise ValueError("router profile does not match its route candidates")
        return route
    candidate = _route_candidate_from_provider(
        provider,
        profile,
        allowed_sensitivities=frozenset(
            {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
        ),
    )
    if candidate is None:
        return None
    return ModelRoute(candidates=(candidate,), revision=revision)


def _route_candidate_from_provider(
    provider: ModelProvider,
    profile: ModelProfile,
    *,
    allowed_sensitivities: frozenset[ModelSensitivity],
) -> ModelRouteCandidate | None:
    if provider.provider_id != profile.id:
        raise ValueError("provider identity does not match its model profile")
    base_url: str | None = None
    secret_reference: SecretReference | None
    if isinstance(provider, OllamaProvider):
        base_url = provider.base_url
        secret_reference = None
    elif isinstance(provider, GrokProvider):
        if getattr(provider, "_api_key", None) is not None:
            return None
        secret_reference = SecretReference.environment("XAI_API_KEY")
    elif isinstance(provider, OpenAICompatibleProvider):
        return None
    elif isinstance(provider, OpenAIResponsesProvider):
        if getattr(provider, "_api_key", None) is not None:
            return None
        secret_reference = SecretReference.environment("OPENAI_API_KEY")
    elif isinstance(provider, AnthropicProvider):
        if getattr(provider, "_api_key", None) is not None:
            return None
        secret_reference = SecretReference.environment("ANTHROPIC_API_KEY")
    elif isinstance(provider, GeminiProvider):
        if getattr(provider, "_api_key", None) is not None:
            return None
        secret_reference = SecretReference.environment("GEMINI_API_KEY")
    else:
        return None
    return ModelRouteCandidate(
        profile=profile,
        allowed_sensitivities=allowed_sensitivities,
        base_url=base_url,
        secret_reference=secret_reference,
    )


__all__ = [
    "create_llm_provider",
    "create_model_route_provider",
    "model_route_from_provider",
]
