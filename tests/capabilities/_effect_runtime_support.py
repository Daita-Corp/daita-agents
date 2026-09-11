"""Shared helpers extracted from ``test_effect_runtime.py``."""

from __future__ import annotations

from daita._json import FrozenJsonObject
from daita.capabilities import (
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ToolExecution,
    ToolOutput,
)


class _EffectExecutor:
    executor_id = "test.action.executor"

    def __init__(
        self, *, store, basis=EffectEvidenceBasis.SERVER_REPORTED, mode="success"
    ):
        self.store = store
        self.basis = basis
        self.mode = mode
        self.calls = 0
        self.preflights = 0

    async def preflight(self, request):
        self.preflights += 1
        return FrozenJsonObject.from_mapping({"current": True})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.calls += 1
        assert request.effect_receipt_id is not None
        started = await self.store.load_effect_receipt(
            "agent-effect", request.effect_receipt_id
        )
        assert started is not None and started.outcome is EffectOutcome.STARTED
        if self.mode == "disconnect":
            raise ConnectionError("acted then disconnected")
        payload = FrozenJsonObject.from_mapping({"target": request.arguments["target"]})
        observation = EffectObservation(EffectOutcome.SUCCEEDED, self.basis, payload)
        return ToolOutput(
            kind="bad-kind" if self.mode == "bad-output" else "test.action.result",
            data=payload,
            effect_observation=observation,
        )
