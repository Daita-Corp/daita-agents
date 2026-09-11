from __future__ import annotations

__all__ = (
    "CATALOG_SEARCH_CAPABILITY_ID",
    "NOW",
    "START_DATA_PROFILE_CAPABILITY_ID",
    "AgentContextBuilder",
    "AgentLoop",
    "CanonicalMessage",
    "ContextEvidencePressureExceeded",
    "ContextToolProjectionAdapter",
    "FinishReason",
    "InMemoryTranscriptStore",
    "LoopExit",
    "LoopExitKind",
    "LoopLimits",
    "Mapping",
    "MessageRole",
    "MockModelProvider",
    "ModelProfile",
    "ModelProviderRegistration",
    "ModelRequest",
    "ModelResponse",
    "ModelRouter",
    "ModelSensitivity",
    "RetryPolicy",
    "RunInput",
    "SQLiteStateStore",
    "TextBlock",
    "ToolBatchCertainty",
    "ToolBatchInterruption",
    "ToolBatchOutcome",
    "ToolCall",
    "ToolDefinition",
    "ToolResultBlock",
    "Transcript",
    "_CancellationResistantReadExecutor",
    "_ConcurrentReadExecutor",
    "_Context",
    "_FailingReadExecutor",
    "_InterruptibleSideEffect",
    "_NoTools",
    "_PayloadReadExecutor",
    "_SnapshotCatalog",
    "_error",
    "_estimate_input_tokens",
    "_run",
    "_runtime",
    "asyncio",
    "cast",
    "execute_projected",
    "pytest",
    "replace",
    "validate_completed_transcript",
)

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    Capability,
    OperationalEffect,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime
from daita.catalog.capabilities import CATALOG_SEARCH_CAPABILITY_ID
from daita.context import AgentContextBuilder, _estimate_input_tokens
from daita.domains.data.profile_jobs import START_DATA_PROFILE_CAPABILITY_ID
from daita.llm.errors import ContextEvidencePressureExceeded
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop import (
    AgentLoop,
    InMemoryTranscriptStore,
    LoopExit,
    LoopExitKind,
    LoopLimits,
    RunInput,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
    Transcript,
)
from daita.loop.models import validate_completed_transcript
from daita.memory.capabilities import MEMORY_SET_CAPABILITY_ID
from daita.storage.sqlite import SQLiteStateStore
from tests.support.capability_runtime import (
    ContextToolProjectionAdapter,
    StaticTestDomain,
    execute_projected,
    presentation_metadata,
    static_registry,
)

NOW = datetime(2026, 8, 18, tzinfo=UTC)


def _error(result: ToolResultBlock) -> Mapping[str, object]:
    return cast(Mapping[str, object], result.output["error"])


class _Context:
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
        static, tools = snapshot
        return ModelRequest(
            messages=(*static, *messages),
            tools=tools,
        )


class _NoTools:
    def __init__(self) -> None:
        self._projection = ContextToolProjectionAdapter(())

    async def prepare_run(self, run):
        return await self._projection.prepare_run(run)

    def project(self, catalog, messages):
        return self._projection.project(catalog, messages)

    async def execute_all(self, run, calls, *, projection, messages, sensitivity):
        del run, projection, messages, sensitivity
        assert calls == ()
        return ToolBatchOutcome(())


class _RuntimeCatalog:
    async def source_routing_facts(self, agent_id, source_ids=()):
        del agent_id, source_ids
        return ()


class _SnapshotCatalog:
    async def source_routing_facts(self, agent_id, source_ids=()):
        return ({"source_id": "source-snapshot", "adapter_id": "sqlite"},)

    async def readable_resource_ids(self, agent_id, source_ids=()):
        return frozenset(("resource-snapshot",))

    def __init__(self) -> None:
        self.context_reads = 0
        self.sensitivity_reads = 0
        self.revision = "one"

    async def admitted_model_sensitivity(self, agent_id, source_ids=()):
        del agent_id, source_ids
        self.sensitivity_reads += 1
        return ModelSensitivity.PUBLIC

    async def catalog_context(
        self,
        agent_id,
        query,
        *,
        prior_query=None,
        limit,
        source_ids=(),
        resource_ids=(),
        readable_resource_ids=None,
    ):
        del agent_id, query, prior_query, limit
        del source_ids, resource_ids, readable_resource_ids
        self.context_reads += 1
        return FrozenJsonObject.from_mapping(
            {
                "resources": (
                    {
                        "kind": "table",
                        "match_reasons": ("resource_name_exact_mention",),
                        "name": "snapshot",
                        "resource_id": "resource-snapshot",
                        "revision": self.revision,
                        "sensitivity": "public",
                        "source_id": "source-snapshot",
                    },
                ),
                "sources": (
                    {
                        "source_id": "source-snapshot",
                        "source_revision": "catalog:one",
                        "sync_id": "sync-one",
                    },
                ),
                "total_matches": 1,
                "returned_count": 1,
                "truncated": False,
                "trust_classification": "untrusted_external_data",
            }
        )


class _ReadExecutor:
    executor_id = "stage-a.read"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        return ToolOutput(kind="stage-a.read-result", data={"call": request.call_id})


class _ConcurrentReadExecutor(_ReadExecutor):
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return await super().execute(request)


class _PayloadReadExecutor(_ReadExecutor):
    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def execute(self, request: ToolExecution) -> ToolOutput:
        return ToolOutput(
            kind="stage-a.read-result",
            data={"call": request.call_id, "payload": self.payload},
        )


class _FailingReadExecutor(_ReadExecutor):
    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        raise RuntimeError("SECRET EXECUTOR DIAGNOSTIC")


class _CancellationResistantReadExecutor(_ReadExecutor):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue
        return await super().execute(request)


class _InterruptibleSideEffect:
    executor_id = "stage-a.write"

    def __init__(self, *, ignore_worker_cancellation: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.ignore_worker_cancellation = ignore_worker_cancellation

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        del request
        return FrozenJsonObject.from_mapping({"fingerprint": "current"})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.ignore_worker_cancellation:
                raise
            await self.release.wait()
        return ToolOutput(kind="stage-a.write-result", data={"call": request.call_id})


def _runtime(
    side_effect: _InterruptibleSideEffect,
    *,
    recovery_timeout: float = 1.0,
    read_executor: _ReadExecutor | None = None,
    limits: LoopLimits = LoopLimits(),
) -> CapabilityRuntime:
    resolved_read = read_executor or _ReadExecutor()
    read = Capability(
        id=CATALOG_SEARCH_CAPABILITY_ID,
        description="Stage A read.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        output_kind="stage-a.read-result",
        output_schema={
            "type": "object",
            "properties": {
                "call": {"type": "string"},
                "payload": {},
            },
            "required": ["call"],
        },
        executor_id=resolved_read.executor_id,
    )
    write = Capability(
        id=MEMORY_SET_CAPABILITY_ID,
        description="Stage A side effect.",
        input_schema={"type": "object", "properties": {}},
        output_kind="stage-a.write-result",
        output_schema={
            "type": "object",
            "properties": {"call": {"type": "string"}},
            "required": ["call"],
        },
        executor_id=side_effect.executor_id,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.CHANGE_ADVISORY_CONTEXT,
    )

    async def approve(request) -> ApprovalDecision:
        del request
        return ApprovalDecision.APPROVE

    views = (
        ToolView(
            name="stage_a_read",
            capability_id=read.id,
            description=read.description,
            presentation=presentation_metadata(),
        ),
        ToolView(
            name="stage_a_write",
            capability_id=write.id,
            description=write.description,
            presentation=presentation_metadata(load_mode=ToolLoadMode.ON_DEMAND),
        ),
    )
    domain = StaticTestDomain((read, write), views)
    return CapabilityRuntime(
        static_registry(domain, (resolved_read, side_effect)),
        (domain,),
        approval_handler=approve,
        limits=limits,
        side_effect_recovery_timeout_seconds=recovery_timeout,
    )


def _run(run_id: str) -> RunInput:
    return RunInput(
        id=run_id,
        agent_id="agent-stage-a",
        message="question",
        created_at=NOW,
        conversation_id="conversation-stage-a",
    )
