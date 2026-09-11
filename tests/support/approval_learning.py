import asyncio
import threading
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import cast

import pytest

from daita import Agent
from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalRequest,
    Capability,
    OperationalEffect,
    SideEffectExecutor,
    ToolExecution,
)
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import (
    LoopLimits,
    RunInput,
    ToolBatchInterruption,
)
from daita.memory import MEMORY_MAX_CHARACTERS, USER_MAX_CHARACTERS
from daita.memory.capabilities import (
    MEMORY_SET_CAPABILITY_ID,
    MEMORY_SET_EXECUTOR_ID,
    MEMORY_SET_OUTPUT_KIND,
    MEMORY_SET_TOOL_NAME,
)
from daita.observation import AgentEvent, AgentEventKind
from daita.skills.capabilities import (
    SKILL_DELETE_CAPABILITY_ID,
    SKILL_DELETE_EXECUTOR_ID,
    SKILL_DELETE_OUTPUT_KIND,
    SKILL_DELETE_TOOL_NAME,
    SKILL_SAVE_CAPABILITY_ID,
    SKILL_SAVE_EXECUTOR_ID,
    SKILL_SAVE_OUTPUT_KIND,
    SKILL_SAVE_TOOL_NAME,
)
from tests.support.capability_runtime import execute_projected
from tests.support.toolbox_model import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)
from tests.support.workspace import workspace_for

NOW = datetime(2026, 7, 22, tzinfo=UTC)
EAGER_LIMITS = LoopLimits()


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _call(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


def _memory_call(
    call_id: str = "write",
    *,
    target: str = "memory",
    content: str = "replacement",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=MEMORY_SET_TOOL_NAME,
        arguments={"target": target, "content": content},
    )


def _skill_save_call(
    call_id: str = "skill-save",
    *,
    name: str = "reusable-workflow",
    description: str = "Apply one reusable workflow.",
    instructions: str = "Follow the verified steps and report assumptions.",
    expected_sha256: str | None = None,
) -> ToolCall:
    arguments = {
        "name": name,
        "description": description,
        "instructions": instructions,
    }
    if expected_sha256 is not None:
        arguments["expected_sha256"] = expected_sha256
    return ToolCall(
        id=call_id,
        name=SKILL_SAVE_TOOL_NAME,
        arguments=arguments,
    )


def _skill_delete_call(
    call_id: str = "skill-delete",
    *,
    name: str = "reusable-workflow",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=SKILL_DELETE_TOOL_NAME,
        arguments={"name": name},
    )


def _run(agent: Agent, run_id: str = "approval-run") -> RunInput:
    return RunInput(
        id=run_id,
        agent_id=agent.id,
        message="test",
        created_at=NOW,
        conversation_id="approval-conversation",
    )


def _runtime(agent: Agent) -> CapabilityRuntime:
    loop = agent._embedded._loop
    assert loop is not None
    return cast(CapabilityRuntime, loop._tools)


async def _execute(agent: Agent, *calls: ToolCall):
    runtime = _runtime(agent)
    run = _run(agent)
    return await execute_projected(
        runtime,
        run,
        calls,
        sensitivity=ModelSensitivity.INTERNAL,
    )


async def _skill_digest(agent: Agent, name: str) -> str:
    skill, digest = await agent._embedded._skill_store.read_skill_with_digest(name)
    assert skill is not None
    return digest


def _error_code(result: ToolResultBlock) -> str:
    error = result.output["error"]
    assert isinstance(error, Mapping)
    code = error["code"]
    assert isinstance(code, str)
    return code


def _tool_results(provider: MockModelProvider) -> tuple[ToolResultBlock, ...]:
    return tuple(
        block
        for request in provider.requests
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.output.get("kind")
        not in {"toolbox_load_receipt", "toolbox_search_results"}
    )


def _system_text(request: ModelRequest) -> str:
    return "\n".join(
        block.text
        for message in request.messages
        if message.role is MessageRole.SYSTEM
        for block in message.content
        if isinstance(block, TextBlock)
    )


def _tool_event_kinds(events: list[AgentEvent]) -> tuple[AgentEventKind, ...]:
    return tuple(
        event.kind
        for event in events
        if event.data.get("tool_name") not in {"toolbox_search", "toolbox_load"}
        and event.kind
        in {
            AgentEventKind.TOOL_STARTED,
            AgentEventKind.APPROVAL_REQUESTED,
            AgentEventKind.APPROVAL_DECIDED,
            AgentEventKind.TOOL_COMPLETED,
        }
    )


async def _agent(
    tmp_path,
    name: str,
    *,
    approval_handler=None,
    observer=None,
    responses: tuple[ModelResponse, ...] = (_stop(),),
) -> Agent:
    provider = MockModelProvider(responses)
    return await Agent.create(
        name,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approval_handler,
        observer=observer,
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
