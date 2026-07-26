"""Deterministic scripted provider for loop and recovery tests."""

from __future__ import annotations

from collections.abc import Iterable

from ..models import ModelProfile, ModelRequest, ModelResponse


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
    ) -> None:
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        items = tuple(script)
        if any(not isinstance(item, (ModelResponse, Exception)) for item in items):
            raise TypeError("mock script items must be ModelResponse or Exception")
        self._provider_id = provider_id
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
