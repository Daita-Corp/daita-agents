"""Implement Daita-owned ChatGPT subscription transport for Codex models."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import datetime
from typing import cast

from ..._installation import repair_guidance
from .._lifecycle import await_cleanup, closing_stream
from ..errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from ..models import ModelRequest, ModelResponse, ModelStreamCompleted
from ..pricing import CostEstimate
from ..subscription_auth import (
    CodexOAuthCredential,
    refresh_codex_subscription,
)
from .openai import OpenAIResponsesProvider, _OpenAIClient

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_ATTEMPT_TIMEOUT_SECONDS = 120.0
_CODEX_CONNECT_TIMEOUT_SECONDS = 5.0
_CODEX_READ_TIMEOUT_SECONDS = 45.0
_CODEX_WRITE_TIMEOUT_SECONDS = 30.0
_CODEX_POOL_TIMEOUT_SECONDS = 5.0


class CodexSubscriptionProvider(OpenAIResponsesProvider):
    """Use a Daita-owned ChatGPT OAuth session through Responses.

    The provider performs only model-wire translation and token refresh. Daita's
    normal transcript loop remains the sole owner of tool execution.
    """

    def __init__(
        self,
        model: str,
        *,
        credential: str,
        max_output_tokens: int | None = None,
        credential_updater: Callable[[str], Awaitable[None]] | None = None,
        client: _OpenAIClient | None = None,
    ) -> None:
        try:
            oauth = CodexOAuthCredential.from_secret(credential)
        except ValueError as error:
            raise ValueError("Codex subscription credential is invalid") from error
        if credential_updater is not None and not callable(credential_updater):
            raise TypeError("credential_updater must be callable")
        super().__init__(
            model,
            api_key=oauth.access_token,
            max_output_tokens=max_output_tokens,
            client=client,
        )
        self._credential = oauth
        self._credential_updater = credential_updater
        self._refresh_lock = asyncio.Lock()

    @property
    def provider_id(self) -> str:
        return f"codex:{self.model}"

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return False

    @property
    def client(self) -> _OpenAIClient:
        if self._close_task is not None:
            raise RuntimeError("Codex subscription provider is closed")
        if self._client is None:
            try:
                from openai import AsyncOpenAI, Timeout
            except ImportError as error:
                raise ImportError(
                    "Daita's OpenAI runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from error
            self._client = cast(
                _OpenAIClient,
                AsyncOpenAI(
                    api_key=self._credential.access_token,
                    base_url=_CODEX_BASE_URL,
                    default_headers={
                        "ChatGPT-Account-ID": self._credential.account_id,
                        "OpenAI-Beta": "responses=experimental",
                        "User-Agent": "daita",
                        "originator": "daita",
                    },
                    timeout=Timeout(
                        connect=_CODEX_CONNECT_TIMEOUT_SECONDS,
                        read=_CODEX_READ_TIMEOUT_SECONDS,
                        write=_CODEX_WRITE_TIMEOUT_SECONDS,
                        pool=_CODEX_POOL_TIMEOUT_SECONDS,
                    ),
                    max_retries=0,
                ),
            )
        return self._client

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        try:
            async with asyncio.timeout(_CODEX_ATTEMPT_TIMEOUT_SECONDS):
                await self._ensure_current_credential()
                completed: ModelResponse | None = None
                async with closing_stream(super()._stream(request)) as events:
                    async for event in events:
                        if isinstance(event, ModelStreamCompleted):
                            completed = event.response
                if completed is None:
                    raise ModelProviderError(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        "Codex stream ended without a terminal response",
                        provider_id=self.provider_id,
                        diagnostic=ProviderFailureDiagnostic(
                            phase=ProviderFailurePhase.STREAM_TERMINAL,
                            code="terminal_completion_missing",
                        ),
                    )
                return completed
        except TimeoutError as error:
            raise ModelProviderError(
                ProviderErrorCode.TIMEOUT,
                "Codex subscription request exceeded its attempt deadline",
                provider_id=self.provider_id,
            ) from error

    async def _ensure_current_credential(self) -> None:
        if not self._credential.needs_refresh:
            return
        async with self._refresh_lock:
            if not self._credential.needs_refresh:
                return
            refreshed = await refresh_codex_subscription(self._credential)
            updater = self._credential_updater
            if updater is not None:
                try:
                    await updater(refreshed.to_secret())
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise ModelProviderError(
                        ProviderErrorCode.CONFIGURATION_ERROR,
                        "Daita could not persist the refreshed Codex login",
                    ) from error
            self._credential = refreshed
            self._api_key = refreshed.access_token
            if self._owns_client:
                previous_client = self._client
                self._client = None
                if previous_client is not None:
                    await await_cleanup(asyncio.create_task(previous_client.close()))

    def _request_arguments(self, request: ModelRequest) -> dict[str, object]:
        arguments = super()._request_arguments(request)
        # The ChatGPT Codex surface streams Responses and does not accept the
        # API-billing controls Daita applies to pay-as-you-go OpenAI requests.
        arguments.pop("service_tier", None)
        arguments.pop("max_output_tokens", None)

        native_input = arguments.get("input")
        if isinstance(native_input, list):
            instructions: list[str] = []
            retained: list[object] = []
            for item in native_input:
                if (
                    isinstance(item, Mapping)
                    and item.get("role") == "system"
                    and isinstance(item.get("content"), str)
                ):
                    instructions.append(cast(str, item["content"]))
                else:
                    retained.append(item)
            if instructions:
                arguments["instructions"] = "\n\n".join(instructions)
                arguments["input"] = retained
        if request.tools:
            arguments["tool_choice"] = "auto"
        return arguments

    def _decode_response(
        self,
        response: object,
        *,
        requested_at: datetime | None = None,
        canonical_ids_by_index: Mapping[int, str] | None = None,
        canonical_ids_by_provider_call_id: Mapping[str, str] | None = None,
    ) -> ModelResponse:
        decoded = super()._decode_response(
            response,
            requested_at=requested_at,
            canonical_ids_by_index=canonical_ids_by_index,
            canonical_ids_by_provider_call_id=canonical_ids_by_provider_call_id,
        )
        metadata = dict(decoded.provider_metadata)
        metadata.pop("pricing_dimensions", None)
        metadata.update(
            {
                "auth_mode": "subscription",
                "transport": "chatgpt_responses",
            }
        )
        return replace(
            decoded,
            usage=replace(
                decoded.usage,
                cost_estimate=CostEstimate.unavailable("subscription_billing"),
            ),
            provider_metadata=metadata,
        )


__all__ = ["CodexSubscriptionProvider"]
