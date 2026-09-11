"""Shared helpers extracted from ``test_effectiveness.py``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from daita.llm._lifecycle import closing_stream
from daita.llm.models import (
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
)
from daita.llm.protocols import (
    ManagedModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.loop.models import LoopExit, LoopExitKind


class _RecordingProvider:
    """Caller-owned request capture around one real provider."""

    def __init__(self, delegate: ManagedModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return await self._delegate.generate(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(self._delegate, StreamingModelProvider):
            raise TypeError("the live learning provider must support streaming")
        self.requests.append(request)
        async with closing_stream(self._delegate.stream(request)) as events:
            async for event in events:
                yield event

    async def close(self, *, deadline: float | None = None) -> None:
        await self._delegate.close()


def _require_completed_phase(phase: str, result: LoopExit) -> None:
    if result.kind is not LoopExitKind.COMPLETED:
        pytest.fail(
            f"Learning comparison inconclusive: {phase} run {result.run_id} "
            f"ended {result.kind.value}/{result.reason} after {result.steps} steps. "
            "No comparative effectiveness score was produced."
        )
