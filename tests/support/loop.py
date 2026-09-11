"""Shared helpers extracted from ``test_progression.py``."""

from __future__ import annotations

from datetime import UTC, datetime

from daita.llm.models import (
    ModelRequest,
    ModelSensitivity,
    ToolDefinition,
    ToolResultBlock,
)
from daita.loop import (
    ToolBatchOutcome,
)
from tests.support.capability_runtime import (
    ContextToolProjectionAdapter,
)

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
