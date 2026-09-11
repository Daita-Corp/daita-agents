"""Immutable definitions for Daita's statically supported model providers."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from .models import ModelRequest
from .protocols import ManagedModelProvider

_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class AuthenticationMode(str, Enum):
    API_KEY = "api_key"
    CODEX_SUBSCRIPTION = "codex_subscription"
    OFFICIAL_CLIENT = "official_client"
    LOCAL = "local"


class EndpointMode(str, Enum):
    FIXED = "fixed"
    OPTIONAL_OVERRIDE = "optional_override"


@dataclass(frozen=True, slots=True)
class ProviderConstruction:
    model: str
    api_key: str | None
    subscription_credential: str | None
    credential_updater: Callable[[str], Awaitable[None]] | None
    base_url: str | None
    max_output_tokens: int


ProviderConstructor = Callable[[ProviderConstruction], ManagedModelProvider]
RequestPolicySupport = Callable[[ModelRequest], bool]


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    id: str
    display_name: str
    authentication: AuthenticationMode
    endpoint: EndpointMode
    constructor: ProviderConstructor
    supports_request_policy: RequestPolicySupport
    profile_supports_parallel_tools: bool
    profile_supports_structured_output: bool
    profile_supports_streaming: bool
    profile_supports_reasoning: bool
    default_endpoint: str | None = None
    subscription_client: str | None = None
    subscription_login_command: str | None = None

    @property
    def requires_saved_credential(self) -> bool:
        return self.authentication in {
            AuthenticationMode.API_KEY,
            AuthenticationMode.CODEX_SUBSCRIPTION,
        }

    def construct(self, values: ProviderConstruction) -> ManagedModelProvider:
        if self.endpoint is EndpointMode.FIXED and values.base_url is not None:
            raise ValueError(f"{self.id} uses its fixed endpoint")
        if self.authentication is not AuthenticationMode.CODEX_SUBSCRIPTION:
            if values.subscription_credential is not None:
                raise ValueError(
                    "subscription_credential is only accepted by subscription providers"
                )
            if values.credential_updater is not None:
                raise ValueError("credential_updater is only accepted by Codex")
        if self.authentication is AuthenticationMode.CODEX_SUBSCRIPTION:
            if values.api_key is not None:
                raise ValueError("codex does not accept an API key")
            if values.subscription_credential is None:
                raise ValueError("codex requires a Daita subscription login")
        elif self.authentication is AuthenticationMode.OFFICIAL_CLIENT:
            if values.api_key is not None:
                raise ValueError(
                    f"{self.id} uses the official client's subscription login"
                )
        base_url = values.base_url
        if base_url is None:
            base_url = self.default_endpoint
        return self.constructor(
            ProviderConstruction(
                model=values.model,
                api_key=values.api_key,
                subscription_credential=values.subscription_credential,
                credential_updater=values.credential_updater,
                base_url=base_url,
                max_output_tokens=values.max_output_tokens,
            )
        )


def _all_requests(_request: ModelRequest) -> bool:
    return True


def _no_parallel_override(request: ModelRequest) -> bool:
    return request.allow_parallel_tool_calls is None


def _official_client_request(request: ModelRequest) -> bool:
    return request.response_schema is None or not request.tools


def _openai(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.openai import OpenAIProvider

    return OpenAIProvider(
        values.model,
        api_key=values.api_key,
        max_output_tokens=values.max_output_tokens,
    )


def _anthropic(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.anthropic import AnthropicProvider

    return AnthropicProvider(
        values.model,
        api_key=values.api_key,
        max_tokens=values.max_output_tokens,
    )


def _gemini(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.gemini import GeminiProvider

    return GeminiProvider(
        values.model,
        api_key=values.api_key,
        max_output_tokens=values.max_output_tokens,
    )


def _grok(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.grok import GrokProvider

    return GrokProvider(
        values.model,
        api_key=values.api_key,
        max_tokens=values.max_output_tokens,
    )


def _ollama(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.ollama import OllamaProvider

    assert values.base_url is not None
    return OllamaProvider(
        values.model,
        base_url=values.base_url,
        api_key=values.api_key or "ollama",
        max_tokens=values.max_output_tokens,
    )


def _codex(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.codex import CodexSubscriptionProvider

    assert values.subscription_credential is not None
    return CodexSubscriptionProvider(
        values.model,
        credential=values.subscription_credential,
        credential_updater=values.credential_updater,
        max_output_tokens=values.max_output_tokens,
    )


def _claude_code(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.subscription_cli import ClaudeCodeSubscriptionProvider

    return ClaudeCodeSubscriptionProvider(
        values.model,
        max_output_tokens=values.max_output_tokens,
    )


def _grok_build(values: ProviderConstruction) -> ManagedModelProvider:
    from .providers.subscription_cli import GrokBuildSubscriptionProvider

    return GrokBuildSubscriptionProvider(
        values.model,
        max_output_tokens=values.max_output_tokens,
    )


PROVIDER_DEFINITIONS = (
    ProviderDefinition(
        "openai",
        "OpenAI API",
        AuthenticationMode.API_KEY,
        EndpointMode.FIXED,
        _openai,
        _all_requests,
        False,
        False,
        True,
        False,
    ),
    ProviderDefinition(
        "anthropic",
        "Anthropic API",
        AuthenticationMode.API_KEY,
        EndpointMode.FIXED,
        _anthropic,
        _no_parallel_override,
        False,
        False,
        True,
        False,
    ),
    ProviderDefinition(
        "gemini",
        "Gemini API",
        AuthenticationMode.API_KEY,
        EndpointMode.FIXED,
        _gemini,
        _no_parallel_override,
        False,
        False,
        True,
        False,
    ),
    ProviderDefinition(
        "grok",
        "xAI (Grok) API",
        AuthenticationMode.API_KEY,
        EndpointMode.FIXED,
        _grok,
        _all_requests,
        False,
        False,
        True,
        False,
    ),
    ProviderDefinition(
        "ollama",
        "Ollama local",
        AuthenticationMode.LOCAL,
        EndpointMode.OPTIONAL_OVERRIDE,
        _ollama,
        _all_requests,
        False,
        False,
        True,
        False,
        default_endpoint="http://127.0.0.1:11434/v1",
    ),
    ProviderDefinition(
        "codex",
        "Codex subscription",
        AuthenticationMode.CODEX_SUBSCRIPTION,
        EndpointMode.FIXED,
        _codex,
        _all_requests,
        True,
        True,
        False,
        True,
        subscription_client="ChatGPT",
        subscription_login_command="sign in through Daita",
    ),
    ProviderDefinition(
        "claude-code",
        "Claude Code subscription",
        AuthenticationMode.OFFICIAL_CLIENT,
        EndpointMode.FIXED,
        _claude_code,
        _official_client_request,
        True,
        True,
        False,
        True,
        subscription_client="Claude Code",
        subscription_login_command="claude auth login",
    ),
    ProviderDefinition(
        "grok-build",
        "Grok Build subscription",
        AuthenticationMode.OFFICIAL_CLIENT,
        EndpointMode.FIXED,
        _grok_build,
        _official_client_request,
        True,
        True,
        False,
        True,
        subscription_client="Grok Build",
        subscription_login_command="grok login",
    ),
)

_BY_ID: Mapping[str, ProviderDefinition] = MappingProxyType(
    {definition.id: definition for definition in PROVIDER_DEFINITIONS}
)
if len(_BY_ID) != len(PROVIDER_DEFINITIONS):
    raise RuntimeError("provider definitions contain duplicate IDs")

BUILTIN_PROVIDER_IDS = frozenset(_BY_ID)
SUBSCRIPTION_PROVIDER_IDS = frozenset(
    definition.id
    for definition in PROVIDER_DEFINITIONS
    if definition.authentication
    in {
        AuthenticationMode.CODEX_SUBSCRIPTION,
        AuthenticationMode.OFFICIAL_CLIENT,
    }
)
SUBSCRIPTION_CREDENTIAL_PROVIDER_IDS = frozenset(
    definition.id
    for definition in PROVIDER_DEFINITIONS
    if definition.authentication is AuthenticationMode.CODEX_SUBSCRIPTION
)
PROVIDER_PRESENTATION = tuple(
    (definition.id, definition.display_name) for definition in PROVIDER_DEFINITIONS
)
SUBSCRIPTION_CLIENTS = MappingProxyType(
    {
        definition.id: (
            definition.subscription_client,
            definition.subscription_login_command,
        )
        for definition in PROVIDER_DEFINITIONS
        if definition.subscription_client is not None
        and definition.subscription_login_command is not None
    }
)


def provider_definition(provider_id: str) -> ProviderDefinition | None:
    return _BY_ID.get(provider_id)


def split_provider_model_id(model_id: str) -> tuple[str, str]:
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    provider, separator, model = model_id.partition(":")
    if not separator or not _PROVIDER_NAME.fullmatch(provider) or not model:
        raise ValueError("model_id must use provider:model form")
    return provider, model


def admit_model_selection(
    provider: str,
    model: str,
    base_url: str | None,
) -> tuple[str, str, str | None, bool]:
    """Normalize user-entered configuration before secret resolution."""
    if not isinstance(provider, str):
        raise TypeError("provider must be a string")
    provider_name = provider.strip().lower()
    if _PROVIDER_NAME.fullmatch(provider_name) is None:
        raise ValueError("provider must be a bounded lowercase identifier")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model identifier must be non-empty")
    model_name = model.strip()
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in model_name
    ):
        raise ValueError("model identifier cannot contain whitespace or controls")
    endpoint: str | None = None
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base URL must be non-empty when provided")
        endpoint = base_url.strip()
        if (
            len(endpoint) > 2_048
            or any(
                ord(character) < 32 or ord(character) == 127 for character in endpoint
            )
            or not endpoint.startswith(("http://", "https://"))
        ):
            raise ValueError("base URL must be a bounded HTTP or HTTPS URL")
    definition = provider_definition(provider_name)
    if (
        definition is not None
        and definition.endpoint is EndpointMode.FIXED
        and endpoint is not None
    ):
        raise ValueError(f"{provider_name} uses its fixed endpoint")
    if definition is None and endpoint is None:
        raise ValueError("custom providers require an explicit base URL")
    provider_id = f"{provider_name}:{model_name}"
    if len(provider_id) > 256:
        raise ValueError("model identity exceeds its 256 character bound")
    return (
        provider_name,
        model_name,
        endpoint,
        definition is None or definition.requires_saved_credential,
    )


def supports_builtin_request_policy(provider_id: str, request: ModelRequest) -> bool:
    definition = provider_definition(provider_id)
    if definition is None:
        raise ValueError("provider is not built in")
    return definition.supports_request_policy(request)


__all__ = [
    "AuthenticationMode",
    "admit_model_selection",
    "BUILTIN_PROVIDER_IDS",
    "EndpointMode",
    "PROVIDER_DEFINITIONS",
    "PROVIDER_PRESENTATION",
    "ProviderConstruction",
    "ProviderDefinition",
    "SUBSCRIPTION_CLIENTS",
    "SUBSCRIPTION_CREDENTIAL_PROVIDER_IDS",
    "SUBSCRIPTION_PROVIDER_IDS",
    "provider_definition",
    "split_provider_model_id",
    "supports_builtin_request_policy",
]
