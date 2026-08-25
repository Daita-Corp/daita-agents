"""Define validated public configuration for an agent instance."""

from __future__ import annotations

from dataclasses import dataclass

from .llm.routing import ModelRoute
from .loop.models import LoopLimits


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model_route: ModelRoute | None = None
    limits: LoopLimits = LoopLimits()

    def __post_init__(self) -> None:
        if self.model_route is not None and not isinstance(
            self.model_route, ModelRoute
        ):
            raise TypeError("model_route must be ModelRoute or None")
        if not isinstance(self.limits, LoopLimits):
            raise TypeError("limits must be LoopLimits")


__all__ = ["AgentConfig"]
