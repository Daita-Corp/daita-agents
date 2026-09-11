from __future__ import annotations

__all__ = (
    "EAGER_LIMITS",
    "LEARNING_CANDIDATE_MAX_RECORDS",
    "LEARNING_CANDIDATE_MAX_SUPPORTING_RUNS",
    "LEARNING_REVIEW_MAX_MESSAGES",
    "LEARNING_REVIEW_MAX_MODEL_CALLS",
    "LEARNING_REVIEW_MAX_PROPOSALS",
    "LEARNING_REVIEW_MAX_RUNS",
    "LEARNING_REVIEW_MAX_TOTAL_TOKENS",
    "LEARNING_REVIEW_MAX_TRANSCRIPT_UTF8_BYTES",
    "LEARNING_REVIEW_MAX_WALL_TIME_SECONDS",
    "Agent",
    "ApprovalDecision",
    "ApprovalRequest",
    "CandidateReviewMeasurement",
    "CandidateReviewReport",
    "Decimal",
    "DocumentCandidateContent",
    "FinishReason",
    "LearningCandidateError",
    "LearningCandidateRejectionReason",
    "LearningCandidateStatus",
    "LearningCandidateTarget",
    "LearningReviewStatus",
    "Mapping",
    "MockModelProvider",
    "ModelResponse",
    "ModelUsage",
    "OneShotCandidateReviewer",
    "PresentationController",
    "SQLiteStateStore",
    "TextBlock",
    "ToolCall",
    "ToolResultBlock",
    "_BlockingReviewer",
    "_ids",
    "_response",
    "_review_response",
    "_sqlite_file",
    "_stored_candidate",
    "_write_memory_surface",
    "asyncio",
    "io",
    "json",
    "learning_candidate_content_from_mapping",
    "learning_candidate_content_to_mapping",
    "pytest",
    "sqlite3",
    "workspace_for",
)

import asyncio
import io
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    DocumentCandidateContent,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningReviewStatus,
)
from daita.cli_text import _write_memory_surface
from daita.evaluation import CandidateReviewMeasurement, CandidateReviewReport
from daita.learning_candidates import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    LEARNING_CANDIDATE_MAX_SUPPORTING_RUNS,
    LEARNING_REVIEW_MAX_MESSAGES,
    LEARNING_REVIEW_MAX_MODEL_CALLS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_RUNS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_TRANSCRIPT_UTF8_BYTES,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
    LearningCandidate,
    LearningCandidateError,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateTarget,
    OneShotCandidateReviewer,
    learning_candidate_content_from_mapping,
    learning_candidate_content_to_mapping,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopLimits
from daita.storage.sqlite import SQLiteStateStore
from tests.support.toolbox_model import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)
from tests.support.workspace import workspace_for

EAGER_LIMITS = LoopLimits()
from daita.tui.controller import PresentationController


class _BlockingReviewer:
    provider_id = "mock:blocking-reviewer"

    def __init__(self):
        self.started = asyncio.Event()
        self.requests: list[ModelRequest] = []

    @property
    def model_profile(self):
        return ModelProfile(
            id=self.provider_id,
            context_window_tokens=32_000,
            max_output_tokens=1_024,
            supports_structured_output=True,
        )

    def supports_request_policy(self, request):
        return isinstance(request, ModelRequest)

    def has_complete_pricing(self, request):
        return False

    async def generate(self, request):
        self.requests.append(request)
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


def _ids():
    counters: defaultdict[str, int] = defaultdict(int)

    def value(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]}"

    return value


def _response(text: str, *, usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text=text,
        usage=usage or ModelUsage(),
    )


def _review_response(run_id: str, *, text: str) -> ModelResponse:
    return _response(
        json.dumps(
            {
                "candidates": [
                    {
                        "target": "memory",
                        "source_ids": [],
                        "supporting_run_ids": [run_id],
                        "content": {"text": text},
                    }
                ]
            }
        )
    )


def _sqlite_file(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)")


def _stored_candidate(index: int, *, agent_id: str = "agent-one"):
    digest = f"{index + 1:064x}"
    run_id = f"run-{index}"
    reference = LearningCandidateRunReference(run_id, digest)
    candidate = LearningCandidate(
        id=f"candidate-{index}",
        agent_id=agent_id,
        target=LearningCandidateTarget.MEMORY,
        content=DocumentCandidateContent(f"Durable definition {index}."),
        source_ids=(),
        reviewed_runs=(reference,),
        supporting_run_ids=(run_id,),
        review_fingerprint=digest,
        artifact_state_sha256="a" * 64,
        catalog_revisions=(),
        candidate_fingerprint=digest,
        status=LearningCandidateStatus.AWAITING_REVIEW,
        created_at=datetime(2026, 7, 28, tzinfo=UTC),
        updated_at=datetime(2026, 7, 28, tzinfo=UTC),
    )
    stamp = LearningCandidateReviewStamp(
        run_id=run_id,
        transcript_sha256=digest,
        artifact_state_sha256="a" * 64,
        catalog_state_sha256="b" * 64,
    )
    return candidate, stamp
