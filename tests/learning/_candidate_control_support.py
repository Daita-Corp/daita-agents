from __future__ import annotations

import io
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from daita import Agent, AgentConfig, cli
from daita.cli_text import _write_learning_review_result
from daita.evaluation import CandidateReviewMeasurement
from daita.learning_candidates import (
    LEARNING_REVIEW_MAX_MODEL_CALLS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
    LearningReviewResult,
    LearningReviewStatus,
)
from daita.llm.models import ModelProfile
from daita.llm.routing import (
    ModelRoute,
    ModelRouteCandidate,
    RetryPolicy,
)
from daita.tui.controller import PresentationController
from tests.support.workspace import workspace_for


def _profile(
    provider_id: str = "openai:gpt-5.6-sol",
    *,
    maximum_output: int = 128_000,
) -> ModelProfile:
    return ModelProfile(
        id=provider_id,
        context_window_tokens=1_050_000,
        max_output_tokens=maximum_output,
        supports_tools=True,
        supports_structured_output=True,
        supports_reasoning=True,
    )


def _route() -> ModelRoute:
    return ModelRoute(
        (
            ModelRouteCandidate(
                provider_id="openai:gpt-5.6-sol",
                profile=_profile(),
            ),
            ModelRouteCandidate(
                provider_id="gemini:gemini-3.6-flash",
                profile=_profile(
                    "gemini:gemini-3.6-flash",
                    maximum_output=65_536,
                ),
            ),
        ),
        retry_policy=RetryPolicy(max_attempts_per_candidate=5, backoff_seconds=1),
    )
