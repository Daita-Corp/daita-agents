"""Shared helpers extracted from ``test_supervisor.py``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

import pytest

import daita.storage.sqlite as sqlite_module
from daita._json import FrozenJsonObject
from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import AccessMode, OperationalEffect, ToolOutput
from daita.capability_runtime import (
    CapabilityRuntime,
    InternalCapabilityOutcome,
    InternalCapabilityRequest,
)
from daita.distribution import DistributionOwner, OutcomeState
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import CostEstimate
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.routines.models import (
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    ResourceRevisionObservation,
    ResourceRevisionPrecheck,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineState,
    ScheduledRoutine,
    text_digest,
)
from daita.routines.owner import RoutineOwner
from daita.routines.supervisor import RoutineSupervisor
from daita.storage.sqlite import SQLiteStateStore
from tests.support.capability_runtime import frozen_execution_bindings
from tests.support.distribution import (
    inbox_distribution_plan,
    no_artifact_outcome_contract,
)


def _ids() -> Callable[[str], str]:
    counters: dict[str, int] = {}

    def create(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        if prefix == "run":
            return f"run-{counters[prefix]:032x}"
        return f"{prefix}-{counters[prefix]}"

    return create
