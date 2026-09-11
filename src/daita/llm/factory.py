"""Build lazy model providers and ordered provider routes from configuration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from decimal import Decimal

from ..security import (
    KeychainStore,
    SecretProvider,
    SecretResolutionError,
    default_secret_provider,
)
from ._lifecycle import (
    AttemptLifecycle,
    NativeOwner,
    await_cleanup,
    closing_stream,
    materialize_request,
    shutdown_deadline,
)
from .errors import (
    ModelProviderError,
    ProviderErrorCode,
    before_generation,
    with_cancelled_model_usage,
)
from .models import ModelRequest, ModelResponse, ModelStreamEvent, ModelUsage
from .pricing import CostEstimate
from .protocols import (
    ManagedModelProvider,
    ModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
    provider_supports_request_policy,
)
from .provider_definitions import (
    AuthenticationMode,
    ProviderConstruction,
    provider_definition,
    split_provider_model_id,
)
from .routing import (
    ModelProviderRegistration,
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
)


def create_llm_provider(
    model_id: str,
    *,
    api_key: str | None = None,
    subscription_credential: str | None = None,
    credential_updater: Callable[[str], Awaitable[None]] | None = None,
    base_url: str | None = None,
    max_output_tokens: int = 1_024,
) -> ManagedModelProvider:
    provider_name, model = split_provider_model_id(model_id)
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be positive")
    values = ProviderConstruction(
        model=model,
        api_key=api_key,
        subscription_credential=subscription_credential,
        credential_updater=credential_updater,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
    )
    definition = provider_definition(provider_name)
    if definition is not None:
        return definition.construct(values)
    if subscription_credential is not None:
        raise ValueError(
            "subscription_credential is only accepted by subscription providers"
        )
    if credential_updater is not None:
        raise ValueError("credential_updater is only accepted by Codex")
    if base_url is None:
        raise ValueError("custom providers require base_url")
    from .providers.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        model,
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_output_tokens,
    )


class _LazyProvider:
    def __init__(self, candidate: ModelRouteCandidate, secrets: SecretProvider) -> None:
        self._candidate = candidate
        self._secrets = secrets
        self._provider: ModelProvider | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._native_owner = NativeOwner()

    @property
    def provider_id(self) -> str:
        return self._candidate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if request.sensitivity not in self._candidate.allowed_sensitivities:
            return False
        if self._provider is not None:
            return provider_supports_request_policy(self._provider, request)
        provider_name = self.provider_id.partition(":")[0]
        definition = provider_definition(provider_name)
        return (
            True if definition is None else definition.supports_request_policy(request)
        )

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if self._provider is not None:
            return provider_has_complete_pricing(self._provider, request)
        provider_name = self.provider_id.partition(":")[0]
        definition = provider_definition(provider_name)
        if (
            definition is not None
            and definition.authentication is AuthenticationMode.CODEX_SUBSCRIPTION
        ):
            return False
        provider = create_llm_provider(
            self._candidate.provider_id,
            base_url=self._candidate.base_url,
            max_output_tokens=self._candidate.profile.max_output_tokens,
        )
        return provider_has_complete_pricing(provider, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        request = materialize_request(request)
        provider = await self._resolve(request)
        return await provider.generate(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        request = materialize_request(request)
        provider = await self._resolve(request)
        if not self._candidate.profile.supports_streaming or not isinstance(
            provider, StreamingModelProvider
        ):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "configured provider route does not support streaming",
                provider_id=self.provider_id,
            )
        async with closing_stream(provider.stream(request)) as events:
            async for event in events:
                yield event

    async def _resolve(self, request: ModelRequest) -> ModelProvider:
        attempt = AttemptLifecycle(self._native_owner, request)
        try:
            async with attempt:
                return await self._resolve_before_generation(attempt.request, attempt)
        except TimeoutError:
            raise before_generation(
                ModelProviderError(
                    ProviderErrorCode.TIMEOUT,
                    "The model request deadline expired during provider resolution.",
                    provider_id=self.provider_id,
                ),
                code="provider_resolution_timeout",
            ) from None
        except asyncio.CancelledError as error:
            attempt.finish(error)
            raise with_cancelled_model_usage(
                error,
                ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
            ) from None
        except ModelProviderError as error:
            attempt.finish(error)
            raise before_generation(error, code="provider_resolution_failed") from None

    async def _resolve_before_generation(
        self, request: ModelRequest, attempt: AttemptLifecycle
    ) -> ModelProvider:
        request.remaining_after(
            ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
        )
        if self._close_task is not None:
            raise ModelProviderError(
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
                "configured provider route is closed",
                provider_id=self.provider_id,
            )
        if not self.supports_request_policy(request):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "provider cannot enforce the requested tool policy",
            )
        if self._provider is None:
            reference = self._candidate.secret_reference
            provider_name = self.provider_id.partition(":")[0]
            definition = provider_definition(provider_name)
            try:
                credential = (
                    None
                    if reference is None
                    else await attempt.run_native(self._secrets.resolve(reference))
                )
            except SecretResolutionError as error:
                raise ModelProviderError(
                    _secret_provider_error_code(error),
                    "The configured provider credential could not be resolved.",
                    provider_id=self.provider_id,
                ) from None
            attempt.check_execution()
            self._native_owner.require_available()
            credential_updater: Callable[[str], Awaitable[None]] | None = None
            if (
                definition is not None
                and definition.authentication is AuthenticationMode.CODEX_SUBSCRIPTION
                and reference is not None
            ):

                async def update_credential(value: str) -> None:
                    if not isinstance(self._secrets, KeychainStore):
                        raise SecretResolutionError(
                            "secret_provider_unavailable",
                            "The configured keychain cannot update the login.",
                        )
                    await self._secrets.set(reference, value)

                credential_updater = update_credential
            # Secret resolution yields: another caller may have initialized or
            # closed this owner while it was suspended. Never replace a live
            # delegate or activate a new one after shutdown starts.
            if self._close_task is not None:
                raise ModelProviderError(
                    ProviderErrorCode.PROVIDER_UNAVAILABLE,
                    "configured provider route is closed",
                    provider_id=self.provider_id,
                )
            if self._provider is None:
                self._provider = create_llm_provider(
                    self._candidate.provider_id,
                    api_key=(
                        None
                        if definition is not None
                        and definition.authentication
                        is AuthenticationMode.CODEX_SUBSCRIPTION
                        else credential
                    ),
                    subscription_credential=(
                        credential
                        if definition is not None
                        and definition.authentication
                        is AuthenticationMode.CODEX_SUBSCRIPTION
                        else None
                    ),
                    credential_updater=credential_updater,
                    base_url=self._candidate.base_url,
                    max_output_tokens=self._candidate.profile.max_output_tokens,
                )
        return self._provider

    async def close(self, *, deadline: float | None = None) -> None:
        """Join once-only cleanup without activating unused delegates."""

        deadline = shutdown_deadline(deadline)
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close(deadline))
        await await_cleanup(
            self._close_task, deadline=deadline, owner=self._native_owner
        )

    async def _finish_close(self, deadline: float) -> None:
        provider = self._provider
        if provider is not None:
            assert isinstance(provider, ManagedModelProvider)
            await provider.close(deadline=deadline)
        self._provider = None


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
) -> ManagedModelProvider:
    if not isinstance(route, ModelRoute):
        raise TypeError("route must be ModelRoute")
    secrets = default_secret_provider(secret_provider)
    registrations = tuple(
        ModelProviderRegistration(
            provider=_LazyProvider(candidate, secrets),
            profile=candidate.profile,
            allowed_sensitivities=candidate.allowed_sensitivities,
            close_with_router=True,
        )
        for candidate in route.candidates
    )
    return ModelRouter(registrations, retry_policy=route.retry_policy)


__all__ = ["create_llm_provider", "create_model_route_provider"]
