"""Deterministic scripted provider for loop and recovery tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from ..models import (
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
)


class MockScriptExhausted(RuntimeError):
    """Raised when the loop makes an unexpected extra model call."""


ScriptItem = ModelResponse | Exception


class MockModelProvider:
    """Return or raise scripted items in order and capture canonical requests."""

    def __init__(
        self,
        script: Iterable[ScriptItem],
        *,
        provider_id: str = "mock:scripted",
        complete_pricing: bool = False,
    ) -> None:
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        items = tuple(script)
        if any(not isinstance(item, (ModelResponse, Exception)) for item in items):
            raise TypeError("mock script items must be ModelResponse or Exception")
        if not isinstance(complete_pricing, bool):
            raise TypeError("complete_pricing must be a boolean")
        self._provider_id = provider_id
        self._complete_pricing = complete_pricing
        self._script = items
        self._cursor = 0
        self._requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def model_profile(self) -> ModelProfile:
        """Return deterministic provider-owned facts for offline test requests."""

        return ModelProfile(
            id=self._provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_048,
            supports_tools=True,
            supports_parallel_tools=False,
        )

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        return tuple(self._requests)

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return self._complete_pricing

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._requests.append(request)
        if self._cursor >= len(self._script):
            raise MockScriptExhausted(
                f"Mock model script exhausted after {self._cursor} call(s)"
            )
        item = self._script[self._cursor]
        self._cursor += 1
        if isinstance(item, Exception):
            raise item
        return item

    def assert_consumed(self) -> None:
        remaining = len(self._script) - self._cursor
        if remaining:
            raise AssertionError(
                f"Mock model script has {remaining} unconsumed item(s)"
            )


StreamScriptEvent = ModelStreamEvent | Exception
StreamScriptItem = Iterable[StreamScriptEvent] | Exception


class MockStreamingModelProvider:
    """Yield deterministic canonical streams and capture their exact requests."""

    def __init__(
        self,
        script: Iterable[StreamScriptItem],
        *,
        provider_id: str = "mock:streaming",
        complete_pricing: bool = False,
    ) -> None:
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        items: list[tuple[StreamScriptEvent, ...] | Exception] = []
        for item in script:
            if isinstance(item, Exception):
                items.append(item)
                continue
            events = tuple(item)
            if any(
                not isinstance(
                    event,
                    (
                        ModelTextDelta,
                        ModelToolCallDelta,
                        ModelStreamCompleted,
                        Exception,
                    ),
                )
                for event in events
            ):
                raise TypeError(
                    "mock stream events must be canonical stream events or exceptions"
                )
            items.append(events)
        if not isinstance(complete_pricing, bool):
            raise TypeError("complete_pricing must be a boolean")
        self._provider_id = provider_id
        self._complete_pricing = complete_pricing
        self._script = tuple(items)
        self._cursor = 0
        self._requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def model_profile(self) -> ModelProfile:
        return ModelProfile(
            id=self._provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_048,
            supports_tools=True,
            supports_parallel_tools=False,
            supports_streaming=True,
        )

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        return tuple(self._requests)

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return self._complete_pricing

    async def generate(self, request: ModelRequest) -> ModelResponse:
        events = self._start(request)
        completion: ModelResponse | None = None
        for event in events:
            if isinstance(event, Exception):
                raise event
            if isinstance(event, ModelStreamCompleted):
                if completion is not None:
                    raise ValueError("mock stream contains duplicate completions")
                completion = event.response
        if completion is None:
            raise ValueError("mock stream has no canonical completion")
        return completion

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        events = self._start(request)
        for event in events:
            await asyncio.sleep(0)
            if isinstance(event, Exception):
                raise event
            yield event

    def _start(self, request: ModelRequest) -> tuple[StreamScriptEvent, ...]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._requests.append(request)
        if self._cursor >= len(self._script):
            raise MockScriptExhausted(
                f"Mock model stream exhausted after {self._cursor} call(s)"
            )
        item = self._script[self._cursor]
        self._cursor += 1
        if isinstance(item, Exception):
            raise item
        return item

    def assert_consumed(self) -> None:
        remaining = len(self._script) - self._cursor
        if remaining:
            raise AssertionError(
                f"Mock model stream has {remaining} unconsumed item(s)"
            )
