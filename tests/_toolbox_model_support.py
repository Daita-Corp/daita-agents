"""Test-only scripted provider support for ordinary on-demand tool loading."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal

from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider, ScriptItem


class ToolboxAwareMockModelProvider:
    """Load missing scripted tools through the real Phase 1 control path.

    Existing domain tests script the domain call they care about. This adapter
    contributes only the model turn that requests a missing working set; the
    loop and runtime still validate the receipt and execute the later call via
    the ordinary exact registry path. ``requests`` records every physical model
    turn. ``logical_requests`` is the explicit domain-test view containing only
    turns that consume one scripted item.
    """

    def __init__(
        self,
        script: Iterable[ScriptItem],
        *,
        provider_id: str = "mock:scripted",
        complete_pricing: bool = False,
        model_profile: ModelProfile | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._complete_pricing = complete_pricing
        if model_profile is not None and model_profile.id != provider_id:
            raise ValueError("model_profile.id must match provider_id")
        self._model_profile = model_profile
        self._requests: list[ModelRequest] = []
        self._logical_requests: list[ModelRequest] = []
        self._toolbox_call_count = 0
        self.replace_script(script)

    @property
    def provider_id(self) -> str:
        return self._scripted.provider_id

    @property
    def model_profile(self) -> ModelProfile:
        return self._model_profile or self._scripted.model_profile

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Return every physical request, including synthetic load turns."""

        return tuple(self._requests)

    @property
    def logical_requests(self) -> tuple[ModelRequest, ...]:
        """Return only requests that consumed one caller-scripted response."""

        return tuple(self._logical_requests)

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._scripted.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return self._scripted.has_complete_pricing(request)

    def replace_script(self, script: Iterable[ScriptItem]) -> None:
        """Replace the remaining logical script without hiding prior requests."""

        items = tuple(script)
        self._scripted = MockModelProvider(
            items,
            provider_id=self._provider_id,
            complete_pricing=self._complete_pricing,
        )
        self._logical_script = items
        self._logical_cursor = 0
        self._toolbox_load_attempt: tuple[int, tuple[str, ...]] | None = None

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._requests.append(request)
        if self._logical_cursor >= len(self._logical_script):
            return await self._scripted.generate(request)
        item = self._logical_script[self._logical_cursor]
        if isinstance(item, ModelResponse):
            desired_names = tuple(call.name for call in item.tool_calls)
            visible_names = frozenset(definition.name for definition in request.tools)
            missing_names = tuple(
                sorted(
                    name
                    for name in desired_names
                    if name not in visible_names
                    and name not in {"toolbox_search", "toolbox_load"}
                )
            )
            attempt = (self._logical_cursor, missing_names)
            if (
                missing_names
                and "toolbox_load" in visible_names
                and self._toolbox_load_attempt != attempt
            ):
                loaded_names = _loaded_names_from_transcript(request)
                target_names = tuple(
                    sorted(
                        set(missing_names).union(
                            name for name in desired_names if name in loaded_names
                        )
                    )
                )
                self._toolbox_load_attempt = attempt
                self._toolbox_call_count += 1
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id=f"test-toolbox-load-{self._toolbox_call_count}",
                            name="toolbox_load",
                            arguments={"tool_names": list(target_names)},
                        ),
                    ),
                    usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
                )
        self._logical_requests.append(request)
        try:
            return await self._scripted.generate(request)
        finally:
            self._logical_cursor += 1
            self._toolbox_load_attempt = None

    async def close(self) -> None:
        await self._scripted.close()

    def assert_consumed(self) -> None:
        self._scripted.assert_consumed()


def _loaded_names_from_transcript(request: ModelRequest) -> frozenset[str]:
    loaded: frozenset[str] = frozenset()
    for message in request.messages:
        for block in message.content:
            if (
                not isinstance(block, ToolResultBlock)
                or block.is_error
                or block.output.get("kind") != "toolbox_load_receipt"
            ):
                continue
            data = block.output.get("data")
            if not isinstance(data, Mapping):
                continue
            names = data.get("loaded_names")
            if isinstance(names, (tuple, list)) and all(
                isinstance(name, str) for name in names
            ):
                loaded = frozenset(names)
    return loaded


__all__ = ["ToolboxAwareMockModelProvider"]
