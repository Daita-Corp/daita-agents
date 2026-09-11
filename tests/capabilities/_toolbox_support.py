from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from daita import Agent
from daita._json import FrozenJsonObject, canonical_json
from daita.adapters.mcp import MCPCompletionSemantics, MCPToolBinding, MCPToolSelection
from daita.capabilities import (
    TOOLBOX_DEFINITIONS,
    AccessMode,
    ApprovalDecision,
    AutomationEligibility,
    Capability,
    CapabilityDeclarations,
    CapabilityRegistry,
    OperationalEffect,
    ToolboxDefinition,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from daita.capability_runtime import (
    CapabilityRuntime,
    RunToolCatalog,
    StepToolProjection,
)
from daita.llm.errors import (
    ToolCatalogLimitExceeded,
    ToolManifestLimitExceeded,
    ToolSurfaceLimitExceeded,
)
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
from daita.loop.models import LoopExitKind, LoopLimits, RunInput
from tests.support.capability_runtime import StaticTestDomain
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


class _Executor:
    def __init__(self, name: str, *, effectful: bool = False) -> None:
        self.executor_id = f"test.toolbox.{name}.executor"
        self.name = name
        self.effectful = effectful
        self.preflight_calls = 0
        self.execute_calls = 0

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        assert self.effectful
        self.preflight_calls += 1
        return FrozenJsonObject.from_mapping(
            {"call_id": request.call_id, "tool": self.name}
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.execute_calls += 1
        return ToolOutput(
            kind=f"test.toolbox.{self.name}.output",
            data={"value": request.arguments.get("value", self.name)},
        )


def _presentation(
    name: str,
    toolbox_id: ToolboxId,
    load_mode: ToolLoadMode,
    *,
    text_trust: ToolTextTrust = ToolTextTrust.CODE,
) -> ToolPresentation:
    return ToolPresentation(
        toolbox_id=toolbox_id,
        load_mode=load_mode,
        text_trust=text_trust,
        summary=f"Use {name.replace('_', ' ')} for an exact bounded operation.",
        when_to_use=f"Use when the request requires {name.replace('_', ' ')}.",
        keywords=tuple(dict.fromkeys((name.split("_")[0], "bounded", "exact"))),
    )


def _declaration(
    owner: str,
    specs: tuple[
        tuple[str, ToolboxId, ToolLoadMode, OperationalEffect, ToolTextTrust], ...
    ],
) -> tuple[StaticTestDomain, tuple[_Executor, ...]]:
    capabilities: list[Capability] = []
    views: list[ToolView] = []
    executors: list[_Executor] = []
    for name, toolbox_id, load_mode, effect, text_trust in specs:
        executor = _Executor(name, effectful=effect is not OperationalEffect.NONE)
        capability = Capability(
            id=f"test.toolbox.{owner}.{name}",
            description=f"Execute {name}.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            output_kind=f"test.toolbox.{name}.output",
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            executor_id=executor.executor_id,
            access_mode=(
                AccessMode.WRITE
                if effect is OperationalEffect.MUTATE_DATA
                else AccessMode.READ
            ),
            operational_effect=effect,
        )
        capabilities.append(capability)
        views.append(
            ToolView(
                name=name,
                capability_id=capability.id,
                description=capability.description,
                presentation=_presentation(
                    name,
                    toolbox_id,
                    load_mode,
                    text_trust=text_trust,
                ),
            )
        )
        executors.append(executor)
    return (
        StaticTestDomain(
            tuple(capabilities),
            tuple(views),
            domain_owner_id=owner,
        ),
        tuple(executors),
    )


def _runtime(
    *,
    limits: LoopLimits = LoopLimits(),
    approval_handler=None,
) -> tuple[
    CapabilityRuntime, CapabilityRegistry, StaticTestDomain, tuple[_Executor, ...]
]:
    domain, executors = _declaration(
        "toolbox_test",
        (
            (
                "pinned_read",
                ToolboxId.SOURCES,
                ToolLoadMode.PINNED,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
            (
                "on_demand_a",
                ToolboxId.ARTIFACTS,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
            (
                "on_demand_b",
                ToolboxId.KNOWLEDGE,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
            (
                "effect_write",
                ToolboxId.KNOWLEDGE,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.CHANGE_ADVISORY_CONTEXT,
                ToolTextTrust.CODE,
            ),
        ),
    )
    registry = CapabilityRegistry(
        declarations=(domain.declarations,),
        executors=executors,
    )
    return (
        CapabilityRuntime(
            registry,
            (domain,),
            limits=limits,
            approval_handler=approval_handler,
        ),
        registry,
        domain,
        executors,
    )


def _run(run_id: str = "run-toolbox") -> RunInput:
    return RunInput(
        id=run_id,
        agent_id="agent-toolbox",
        message="exercise toolbox loading",
        created_at=NOW,
        conversation_id="conversation-toolbox",
    )


def _data(result: ToolResultBlock) -> Mapping[str, object]:
    data = result.output.get("data")
    assert isinstance(data, Mapping)
    return data


def _error_code(result: ToolResultBlock) -> str:
    error = result.output.get("error")
    assert isinstance(error, Mapping)
    code = error.get("code")
    assert isinstance(code, str)
    return code


async def _execute(
    runtime: CapabilityRuntime,
    run: RunInput,
    projection: StepToolProjection,
    *calls: ToolCall,
    messages: tuple[CanonicalMessage, ...] = (),
):
    return await runtime.execute_all(
        run,
        calls,
        projection=projection,
        messages=messages,
        sensitivity=ModelSensitivity.INTERNAL,
    )


def _append_results(
    messages: tuple[CanonicalMessage, ...],
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResultBlock, ...],
) -> tuple[CanonicalMessage, ...]:
    return (
        *messages,
        CanonicalMessage(MessageRole.ASSISTANT, tool_calls=calls),
        *(CanonicalMessage(MessageRole.TOOL, content=(result,)) for result in results),
    )


async def _load(
    runtime: CapabilityRuntime,
    run: RunInput,
    catalog: RunToolCatalog,
    messages: tuple[CanonicalMessage, ...],
    names: tuple[str, ...],
    *,
    call_id: str,
) -> tuple[ToolResultBlock, tuple[CanonicalMessage, ...], StepToolProjection]:
    before = runtime.project(catalog, messages)
    call = ToolCall(
        id=call_id,
        name="toolbox_load",
        arguments={"tool_names": names},
    )
    outcome = await _execute(runtime, run, before, call, messages=messages)
    result = outcome.ordered_results[0]
    updated = _append_results(messages, (call,), (result,))
    return result, updated, runtime.project(catalog, updated)
