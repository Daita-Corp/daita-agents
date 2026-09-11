"""Shared helpers extracted from ``test_progression.py``."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import cast

import pytest

from daita._json import canonical_json
from daita.agent import Agent
from daita.capabilities import AccessMode, ExecutionScope, OperationalEffect
from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop import (
    AgentLoop,
    InMemoryTranscriptStore,
    InstructionAuthority,
    LoopExitKind,
    LoopLimits,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
    ToolBatchOutcome,
    ToolRuntime,
)
from daita.observation import AgentEvent, AgentEventKind
from tests.support.capability_runtime import (
    ContextToolProjectionAdapter,
    frozen_execution_bindings,
)
from tests.support.distribution import inbox_distribution_plan
from tests.support.workspace import workspace_for

NOW = datetime(2026, 7, 21, tzinfo=UTC)


class TranscriptContext:
    async def prepare(self, run, messages, tool_context, *, max_total_tokens=None):
        del run
        return messages[:-1], tool_context.initial_provider_definitions

    def project(
        self,
        snapshot,
        messages,
        *,
        step,
        tool_context,
        previous_request_input_tokens=None,
        remaining_tokens=None,
        request_input_growth_tokens=None,
        remaining_steps=None,
    ):
        del step, previous_request_input_tokens, tool_context
        sensitivity = ModelSensitivity.INTERNAL
        for message in messages:
            for block in message.content:
                if (
                    isinstance(block, ToolResultBlock)
                    and block.sensitivity is not None
                    and block.sensitivity.routing_rank > sensitivity.routing_rank
                ):
                    sensitivity = block.sensitivity
        static, tools = snapshot
        return ModelRequest(
            messages=(*static, *messages),
            tools=tools,
            sensitivity=sensitivity,
        )


class ScriptedTools:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
        self._projection = ContextToolProjectionAdapter(
            (
                ToolDefinition(
                    name="lookup",
                    description="look something up",
                    input_schema={"type": "object", "properties": {}},
                ),
            )
        )

    async def prepare_run(self, run):
        return await self._projection.prepare_run(run)

    def project(self, catalog, messages):
        return self._projection.project(catalog, messages)

    async def execute_all(self, run, calls, *, projection, messages, sensitivity):
        del run, projection, messages, sensitivity
        self.calls.extend(calls)
        return ToolBatchOutcome(tuple(self.outputs[call.id] for call in calls))
