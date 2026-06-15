"""Runtime-native chat loop behind the public Agent facade."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from daita.core.exceptions import AgentError
from daita.core.tracing import TraceType
from daita.runtime import (
    ContextAudience,
    ContextBlock,
    Evidence,
    OperationStatus,
    RuntimeEventType,
    RuntimeKernelGovernanceBlocked,
    ToolView,
    register_local_tool,
)
from daita.plugins.manifest import PluginKind
from daita.skills import SkillResolution, SkillResolver

from .contextvars import active_run_state
from .evidence import add_evidence
from .exit import RunExitPolicy
from .guardrails import ToolCallGuardrails, has_terminal_tool_result
from .llm_turn import LLMResult
from .retry import mark_whole_run_retry_suppressed, run_model_turn_with_retry
from .state import FinalAnswerReadiness, RunPhase, RunState
from .tools import json_serializer

if TYPE_CHECKING:
    from daita.core.tools import LocalTool
    from daita.plugins.registry import ExtensionRegistry
    from daita.runtime import RuntimeKernel, RuntimeStore

logger = logging.getLogger(__name__)


class ChatRunPhase(str, Enum):
    """Chat-runtime phase labels."""

    INITIALIZING = "initializing"
    MODEL_TURN = "model_turn"
    TOOL_EXECUTION = "tool_execution"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True)
class ModelToolSpec:
    """Provider-neutral model-visible projection of one ToolView."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    capability_id: str
    owner: str
    tool_view: ToolView
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatToolCallResult:
    """Result of one model-selected tool view execution."""

    tool_name: str
    tool_call_id: str | None
    arguments: Mapping[str, Any]
    capability_id: str
    owner: str
    operation_id: str
    task_id: str | None
    evidence: tuple[Evidence, ...]
    rendered_result: Mapping[str, Any] | str | int | float | bool | None
    duration_ms: int
    blocked: bool = False
    error: str | None = None
    retry_safe: bool = False
    replay_safe: bool = False
    idempotent: bool = False
    side_effecting: bool = True

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the result shape older Agent diagnostics expect."""
        result: Any = self.rendered_result
        if self.error is not None and not (
            isinstance(result, Mapping) and result.get("error")
        ):
            result = {"error": self.error}
        return {
            "tool": self.tool_name,
            "arguments": dict(self.arguments),
            "result": result,
            "duration_ms": self.duration_ms,
            "capability_ids": (self.capability_id,),
            "operation_id": self.operation_id,
            "task_id": self.task_id,
            "retry_safe": self.retry_safe,
            "replay_safe": self.replay_safe,
            "idempotent": self.idempotent,
            "side_effecting": self.side_effecting,
            **({"blocked": True} if self.blocked else {}),
        }


@dataclass
class ChatRunState:
    """Conversation-scoped state for one ChatRuntime.run invocation."""

    runtime_id: str
    operation_id: str
    run_id: str
    phase: ChatRunPhase = ChatRunPhase.INITIALIZING
    iteration_count: int = 0
    model_turn_count: int = 0
    tool_call_count: int = 0
    tool_calls: list[ChatToolCallResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatRunResult:
    """Source-of-truth result produced by ChatRuntime."""

    text: str
    operation_id: str
    iterations: int
    tool_calls: tuple[ChatToolCallResult, ...]
    diagnostics: Mapping[str, Any]
    token_usage: Mapping[str, Any]
    cost: float
    partial: bool = False

    def to_agent_result(self) -> dict[str, Any]:
        """Adapt to the Agent.run detailed dictionary shape."""
        result = {
            "result": self.text,
            "operation_id": self.operation_id,
            "tool_calls": [call.to_legacy_dict() for call in self.tool_calls],
            "iterations": self.iterations,
            "tokens": dict(self.token_usage),
            "cost": self.cost,
            "diagnostics": dict(self.diagnostics),
        }
        if self.partial:
            result["partial"] = True
        return result


class ChatRuntime:
    """Runtime owner for generic Agent chat/model/tool execution."""

    runtime_kind = "chat"

    def __init__(
        self,
        *,
        runtime_id: str,
        agent_name: str,
        llm_getter: Callable[[], Any],
        llm_provider_name_getter: Callable[[], str | None],
        config: Any,
        prompt: str | Mapping[str, str] | None,
        default_focus: str | Mapping[str, str] | None,
        extension_registry: "ExtensionRegistry",
        runtime_kernel: "RuntimeKernel",
        runtime_store: "RuntimeStore",
        trace_manager: Any,
        setup_tools: Callable[[], Any],
        setup_extensions: Callable[[], Any],
        emit_event: Callable[..., None],
        retry_with_tracing: Callable[..., Any],
        final_answer_hooks_getter: Callable[[], list[Any]],
    ) -> None:
        self.runtime_id = runtime_id
        self.agent_name = agent_name
        self._llm_getter = llm_getter
        self._llm_provider_name_getter = llm_provider_name_getter
        self.config = config
        self.prompt = prompt
        self.default_focus = default_focus
        self.extension_registry = extension_registry
        self.kernel = runtime_kernel
        self.store = runtime_store
        self.trace_manager = trace_manager
        self._setup_tools = setup_tools
        self._setup_extensions = setup_extensions
        self._emit_event = emit_event
        self._retry_with_tracing = retry_with_tracing
        self._final_answer_hooks_getter = final_answer_hooks_getter
        self.guardrails = ToolCallGuardrails()

    @property
    def llm(self):
        return self._llm_getter()

    async def run_with_retry(self, **kwargs) -> ChatRunResult:
        """Run the chat loop with the facade's whole-run retry scaffold."""

        async def execute(attempt: int, max_attempts: int) -> ChatRunResult:
            result = await self.run(**kwargs)
            if attempt > 1:
                diagnostics = dict(result.diagnostics)
                diagnostics["retry_attempt"] = attempt
                return replace(result, diagnostics=diagnostics)
            return result

        return await self._retry_with_tracing(execute, "chat_runtime_retry_attempt")

    async def run(
        self,
        *,
        prompt: str,
        tools: list[Any] | None = None,
        max_iterations: int = 20,
        on_event: Callable | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ChatRunResult:
        """Execute one conversational run."""
        from daita.core.streaming import EventType

        if self.llm is None:
            provider_name = self._llm_provider_name_getter() or "openai"
            raise AgentError(
                f"Cannot execute: No API key found for '{provider_name}'. "
                f"Set {provider_name.upper()}_API_KEY environment variable "
                f"or pass api_key parameter to Agent."
            )

        await self._setup_tools()
        await self._setup_extensions()
        explicit_skills = kwargs.pop("skills", None)
        if explicit_skills is None:
            explicit_skills = kwargs.pop("selected_skills", None)
        selected_skill_names = _normalize_skill_selection(explicit_skills)

        operation = await self.kernel.create_operation(
            operation_type="chat.run",
            request={"prompt": prompt},
            metadata={"agent_id": self.runtime_id, "agent_name": self.agent_name},
        )
        skill_resolution = self._resolve_skills(
            prompt=prompt,
            explicit_skills=selected_skill_names,
        )
        operation = replace(
            operation,
            metadata={
                **operation.metadata,
                "skills": skill_resolution.to_metadata(),
                "skill_catalog": list(
                    SkillResolver(self.extension_registry).compact_catalog(
                        runtime_kind=self.runtime_kind
                    )
                ),
            },
        )
        await self.store.save_operation(operation)
        await self._record_skill_resolution(operation.id, skill_resolution)
        run_state = RunState(agent_id=self.runtime_id)
        run_state.diagnostics["operation_id"] = operation.id
        run_state.diagnostics["skills"] = skill_resolution.to_metadata()
        run_state.final_answer_readiness_hooks.extend(
            self._final_answer_hooks_getter() or []
        )
        chat_state = ChatRunState(
            runtime_id=self.runtime_id,
            operation_id=operation.id,
            run_id=run_state.run_id,
        )

        token = active_run_state.set(run_state)
        try:
            resolved_tools = await self._prepare_model_tools(tools)
            active_tools = list(resolved_tools)
            final_synthesis_without_tools = bool(
                kwargs.pop("final_synthesis_without_tools", False)
            )
            terminal_tools = set(kwargs.pop("terminal_tools", []) or [])
            exit_policy = RunExitPolicy(
                allow_partial=bool(kwargs.pop("partial_exit", False))
            )

            conversation = await self._build_initial_conversation(
                prompt,
                initial_messages,
                skill_resolution=skill_resolution,
            )
            legacy_tool_calls: list[dict[str, Any]] = []
            chat_tool_calls: list[ChatToolCallResult] = []
            terminal_evidence_seen = False
            latest_terminal_result: dict[str, Any] | None = None

            for iteration in range(max_iterations):
                run_state.iteration_count = iteration + 1
                chat_state.iteration_count = iteration + 1
                self._emit_event(
                    on_event,
                    EventType.ITERATION,
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                    runtime_id=self.runtime_id,
                    operation_id=operation.id,
                )

                run_state.set_phase(RunPhase.MODEL_TURN)
                chat_state.phase = ChatRunPhase.MODEL_TURN
                run_state.model_turn_count += 1
                chat_state.model_turn_count += 1
                if on_event:
                    llm_call = lambda: self._stream_llm_turn(
                        conversation,
                        active_tools,
                        on_event,
                        operation_id=operation.id,
                        run_state=run_state,
                        **kwargs,
                    )
                else:
                    llm_call = lambda: self._nonstream_llm_turn(
                        conversation,
                        active_tools,
                        operation_id=operation.id,
                        run_state=run_state,
                        **kwargs,
                    )
                if self.config.retry_enabled:
                    llm_result = await run_model_turn_with_retry(
                        llm_call,
                        policy=self.config.retry_policy,
                        run_state=run_state,
                    )
                else:
                    llm_result = await llm_call()
                run_state.warnings.extend(llm_result.warnings)

                if llm_result.tool_calls:
                    if terminal_evidence_seen and not active_tools:
                        if (
                            "tool_calls_after_terminal_evidence_ignored"
                            not in run_state.warnings
                        ):
                            run_state.warnings.append(
                                "tool_calls_after_terminal_evidence_ignored"
                            )
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "The necessary tool evidence is already present. "
                                    "Do not call tools. Provide the final answer now."
                                ),
                            }
                        )
                        continue
                    run_state.set_phase(RunPhase.TOOL_EXECUTION)
                    chat_state.phase = ChatRunPhase.TOOL_EXECUTION
                    results = []
                    for tool_call in llm_result.tool_calls:
                        result = await self._execute_tool_call(
                            tool_call,
                            active_tools,
                            operation_id=operation.id,
                            run_state=run_state,
                            on_event=on_event,
                        )
                        chat_tool_calls.append(result)
                        chat_state.tool_calls.append(result)
                        legacy_result = result.to_legacy_dict()
                        run_state.record_tool_call(legacy_result)
                        chat_state.tool_call_count = run_state.tool_call_count
                        legacy_tool_calls.append(legacy_result)
                        results.append(legacy_result)
                        self._record_tool_evidence(run_state, tool_call, legacy_result)

                        guardrail_decision = self.guardrails.observe_tool_result(
                            run_state, tool_call, legacy_result
                        )
                        if guardrail_decision.hard_stop_message:
                            run_state.set_phase(RunPhase.ERROR)
                            chat_state.phase = ChatRunPhase.ERROR
                            run_state.exit_reason = "repeated_tool_error"
                            raise AgentError(guardrail_decision.hard_stop_message)
                        if guardrail_decision.guidance_result is not None:
                            legacy_result["result"] = guardrail_decision.guidance_result
                            updated_result = replace(
                                result,
                                rendered_result=guardrail_decision.guidance_result,
                            )
                            chat_tool_calls[-1] = updated_result
                            chat_state.tool_calls[-1] = updated_result
                            run_state.warnings.append(
                                guardrail_decision.guidance_result["guardrail"]
                            )

                    self._append_tool_messages(
                        conversation, llm_result.tool_calls, results
                    )
                    terminal_result = self._terminal_result(results, terminal_tools)
                    if final_synthesis_without_tools and terminal_result:
                        terminal_evidence_seen = True
                        latest_terminal_result = terminal_result
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Use the latest tool result as answer evidence. "
                                    "Do not call more tools. Provide the final answer now."
                                ),
                            }
                        )
                        active_tools = []
                    elif suggested_decision := self._resolve_suggested_next_tool(
                        results, resolved_tools
                    ):
                        suggested_next_tool = suggested_decision.get("tool_name")
                        if suggested_decision.get("warning"):
                            warning = str(suggested_decision["warning"])
                            if warning not in run_state.warnings:
                                run_state.warnings.append(warning)
                        run_state.diagnostics.update(
                            suggested_decision.get("diagnostics") or {}
                        )
                        if not suggested_decision.get("allowed"):
                            guidance = suggested_decision.get("guidance")
                            if guidance:
                                conversation.append(
                                    {"role": "user", "content": guidance}
                                )
                            active_tools = list(resolved_tools)
                            continue
                        suggested_tools = [
                            spec
                            for spec in resolved_tools
                            if spec.name == suggested_next_tool
                        ]
                        if suggested_tools:
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"The previous tool result suggested calling "
                                        f"{suggested_next_tool}. Call that tool next "
                                        "using the provided suggested_next_arguments."
                                    ),
                                }
                            )
                            active_tools = suggested_tools
                    else:
                        active_tools = list(resolved_tools)
                    continue

                run_state.set_phase(RunPhase.SYNTHESIS)
                chat_state.phase = ChatRunPhase.SYNTHESIS
                readiness = self._evaluate_final_answer_readiness(
                    run_state, resolved_tools, llm_result.text
                )
                if not readiness.allow_final:
                    conversation.append(
                        {"role": "user", "content": readiness.guidance or ""}
                    )
                    active_tools = (
                        [] if terminal_evidence_seen else list(resolved_tools)
                    )
                    if iteration + 1 < max_iterations:
                        continue
                    break

                run_state.set_phase(RunPhase.COMPLETE)
                chat_state.phase = ChatRunPhase.COMPLETE
                run_state.exit_reason = "final_answer"
                await self.kernel.complete_operation(operation.id)
                return self._build_result(
                    llm_result.text,
                    operation.id,
                    chat_tool_calls,
                    iteration + 1,
                    run_state,
                    on_event,
                )

            run_state.set_phase(RunPhase.ERROR)
            chat_state.phase = ChatRunPhase.ERROR
            run_state.exit_reason = "max_iterations"
            if final_synthesis_without_tools and terminal_evidence_seen:
                run_state.set_phase(RunPhase.COMPLETE)
                chat_state.phase = ChatRunPhase.COMPLETE
                run_state.exit_reason = "terminal_evidence_synthesis"
                await self.kernel.complete_operation(operation.id)
                return self._build_result(
                    self._terminal_result_text(latest_terminal_result),
                    operation.id,
                    chat_tool_calls,
                    max_iterations,
                    run_state,
                    on_event,
                )
            exit_decision = exit_policy.decide_max_iterations(run_state, prompt)
            if exit_decision.should_return:
                run_state.set_phase(RunPhase.COMPLETE)
                chat_state.phase = ChatRunPhase.COMPLETE
                run_state.exit_reason = exit_decision.reason
                run_state.partial_result = exit_decision.result_text
                self._evaluate_final_answer_readiness(run_state, resolved_tools, "")
                await self.kernel.complete_operation(operation.id)
                return self._build_result(
                    exit_decision.result_text or "",
                    operation.id,
                    chat_tool_calls,
                    max_iterations,
                    run_state,
                    on_event,
                    partial=True,
                )
            self._evaluate_final_answer_readiness(run_state, resolved_tools, "")
            await self.kernel.complete_operation(
                operation.id,
                status=OperationStatus.FAILED,
            )
            raise AgentError(
                f"Max iterations ({max_iterations}) reached without final answer",
                agent_id=self.runtime_id,
                task=prompt,
                context={
                    "max_iterations": max_iterations,
                    "iterations": max_iterations,
                    "tool_calls": legacy_tool_calls,
                    "tokens": self._token_stats(),
                    "diagnostics": run_state.diagnostic_summary(),
                },
            )
        except Exception as error:
            mark_whole_run_retry_suppressed(error, run_state)
            await self.kernel.fail_operation_if_active(operation.id, error)
            raise
        finally:
            active_run_state.reset(token)

    async def _prepare_model_tools(
        self, selected_tools: list[Any] | None
    ) -> list[ModelToolSpec]:
        resolved_local_tools = await self._resolve_local_tools(selected_tools)
        selected_names = None
        if selected_tools is not None:
            selected_names = {
                item if isinstance(item, str) else getattr(item, "name", str(item))
                for item in selected_tools
            }

        specs: list[ModelToolSpec] = []
        seen: set[str] = set()
        for view in self.extension_registry.tool_views:
            if not view.model_visible:
                continue
            if selected_names is not None and view.name not in selected_names:
                continue
            owner = self.extension_registry.get_tool_view_owner(view.name)
            capability = self.extension_registry.get_capability(
                view.capability_id,
                owner=owner,
            )
            if capability.runtime_only:
                continue
            spec = ModelToolSpec(
                name=view.name,
                description=view.description,
                parameters=view.parameters,
                capability_id=view.capability_id,
                owner=owner,
                tool_view=view,
                metadata={
                    "owner": owner,
                    "capability_id": view.capability_id,
                    "capability": capability.to_dict(),
                },
            )
            if spec.name in seen:
                raise ValueError(f"duplicate model tool name: {spec.name}")
            seen.add(spec.name)
            specs.append(spec)

        registry_tool_names = {view.name for view in self.extension_registry.tool_views}
        local_tool_catalog = getattr(self, "local_tool_catalog", None)
        for tool in resolved_local_tools:
            if not bool(getattr(tool, "model_visible", True)) or bool(
                getattr(tool, "runtime_only", False)
            ):
                continue
            if (
                getattr(tool, "source", None) == "plugin"
                and tool.name in registry_tool_names
            ):
                continue
            if (
                tool.name in registry_tool_names
                and local_tool_catalog is not None
                and local_tool_catalog.get(tool.name) is tool
            ):
                continue
            spec = self._register_local_tool(tool)
            if selected_names is not None and spec.name not in selected_names:
                continue
            if spec.name in seen:
                raise ValueError(f"duplicate model tool name: {spec.name}")
            seen.add(spec.name)
            specs.append(spec)
        return specs

    async def _resolve_local_tools(self, selected_tools: list[Any] | None) -> list[Any]:
        await self._setup_tools()
        local_tool_catalog = getattr(self, "local_tool_catalog", None)
        if local_tool_catalog is None:
            return []
        if selected_tools is None:
            return [self._with_focus(tool) for tool in local_tool_catalog.tools]
        resolved = []
        seen: dict[str, Any] = {}
        for item in selected_tools:
            if isinstance(item, str):
                tool = local_tool_catalog.get(item)
                if tool is None:
                    raise ValueError(f"Tool '{item}' not found in registry")
            else:
                attached = local_tool_catalog.get(getattr(item, "name", ""))
                tool = attached if attached is not None else item
            name = str(getattr(tool, "name", ""))
            existing = seen.get(name)
            if existing is tool:
                continue
            seen[name] = tool
            resolved.append(tool)
        return [self._with_focus(tool) for tool in resolved]

    @property
    def local_tool_catalog(self):
        return getattr(self, "_local_tool_catalog", None)

    def bind_local_tool_catalog(self, catalog) -> None:
        self._local_tool_catalog = catalog

    def _with_focus(self, tool):
        effective = _resolve_tool_focus(self.default_focus, tool)
        if not effective:
            return tool
        return FocusedRuntimeTool(tool, effective)

    def _register_local_tool(self, tool) -> ModelToolSpec:
        registration = register_local_tool(self.extension_registry, tool)
        if registration.tool_view is None:
            raise ValueError(f"local tool {tool.name!r} is not model visible")
        return ModelToolSpec(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            capability_id=registration.capability_id,
            owner=registration.plugin_id,
            tool_view=registration.tool_view,
            metadata={"adapter": "local_tool", "owner": registration.plugin_id},
        )

    async def _build_initial_conversation(
        self,
        prompt: str,
        initial_messages: list[dict[str, Any]] | None = None,
        *,
        skill_resolution: SkillResolution | None = None,
    ) -> list[dict[str, Any]]:
        conversation: list[dict[str, Any]] = []
        system_parts: list[str] = []
        if self.prompt:
            system_parts.append(
                self.prompt if isinstance(self.prompt, str) else json.dumps(self.prompt)
            )
        skill_resolution = skill_resolution or self._resolve_skills(
            prompt=prompt,
            explicit_skills=(),
        )
        extension_blocks = [
            *await self._render_context_blocks(
                prompt,
                selected_skill_ids=skill_resolution.selected_ids(),
                selected_skill_names=skill_resolution.selected_names(),
            ),
            *[
                block
                for effect in skill_resolution.effects
                for block in effect.context_blocks
            ],
        ]
        extension_context = self._format_context(
            [
                block
                for block in extension_blocks
                if block.metadata.get("context_kind") != "skill_instructions"
            ]
        )
        if extension_context:
            system_parts.append(extension_context)
        skill_parts = [
            f"### {block.metadata.get('skill_name') or block.owner}\n{block.content}"
            for block in sorted(
                (
                    block
                    for block in extension_blocks
                    if block.metadata.get("context_kind") == "skill_instructions"
                ),
                key=lambda item: item.priority,
                reverse=True,
            )
        ]
        if skill_parts:
            system_parts.append("## Skills & Expertise\n" + "\n\n".join(skill_parts))
        catalog_context = self._format_skill_catalog(skill_resolution)
        if catalog_context:
            system_parts.append(catalog_context)
        if system_parts:
            conversation.append(
                {"role": "system", "content": "\n\n".join(system_parts)}
            )
        if initial_messages:
            conversation.extend(initial_messages)
        conversation.append({"role": "user", "content": prompt})
        return conversation

    async def render_context_blocks(
        self,
        prompt: str,
        *,
        audience: ContextAudience = ContextAudience.PRIMARY_MODEL,
        token_budget: int = 2000,
        selected_skill_ids: tuple[str, ...] = (),
        selected_skill_names: tuple[str, ...] = (),
    ) -> list[ContextBlock]:
        """Render context blocks from registry-owned context providers."""
        await self._setup_extensions()
        audience = ContextAudience(audience)

        blocks: list[ContextBlock] = []
        for provider in self.extension_registry.context_providers:
            if audience not in provider.audiences:
                continue
            try:
                provider_context = {
                    "prompt": prompt,
                    "runtime_id": self.runtime_id,
                    "agent_id": self.runtime_id,
                }
                if selected_skill_ids or selected_skill_names:
                    provider_context.update(
                        {
                            "selected_skill_ids": tuple(selected_skill_ids),
                            "selected_skill_names": tuple(selected_skill_names),
                        }
                    )
                block = await provider.render(
                    provider_context,
                    audience,
                    token_budget,
                )
            except Exception as error:
                provider_name = getattr(provider, "id", provider.__class__.__name__)
                logger.warning(
                    "context provider '%s' failed for agent '%s': %s",
                    provider_name,
                    self.agent_name,
                    error,
                )
                continue
            if block is not None and block.content:
                blocks.append(block)

        return blocks

    async def _render_context_blocks(
        self,
        prompt: str,
        *,
        selected_skill_ids: tuple[str, ...] = (),
        selected_skill_names: tuple[str, ...] = (),
    ) -> list[ContextBlock]:
        return await self.render_context_blocks(
            prompt,
            selected_skill_ids=selected_skill_ids,
            selected_skill_names=selected_skill_names,
        )

    def _format_context(self, blocks: list[ContextBlock]) -> str | None:
        if not blocks:
            return None

        rendered_blocks = [
            f"### {block.id}\n{block.content}"
            for block in sorted(blocks, key=lambda item: item.priority, reverse=True)
        ]
        return "## Runtime Context\n" + "\n\n".join(rendered_blocks)

    async def _nonstream_llm_turn(
        self, conversation, tools, *, operation_id, run_state, **kwargs
    ) -> LLMResult:
        await self.kernel.append_event(
            RuntimeEventType.LLM_REQUESTED,
            operation_id=operation_id,
            message="LLM turn requested.",
            payload={
                "chat_run_id": run_state.run_id,
                "model_turn": run_state.model_turn_count,
                "stream": False,
                "tool_count": len(tools),
            },
        )
        result = LLMResult.from_response(
            await self.llm.generate(
                messages=conversation,
                tools=tools,
                stream=False,
                **kwargs,
            )
        )
        await self.kernel.append_event(
            RuntimeEventType.LLM_COMPLETED,
            operation_id=operation_id,
            message="LLM turn completed.",
            payload={
                "chat_run_id": run_state.run_id,
                "model_turn": run_state.model_turn_count,
                "tool_call_count": len(result.tool_calls),
                "finish_reason": result.finish_reason,
                "text_length": len(result.text or ""),
            },
        )
        return result

    async def _stream_llm_turn(
        self, conversation, tools, on_event, *, operation_id, run_state, **kwargs
    ) -> LLMResult:
        from daita.core.streaming import EventType

        thinking_text = ""
        tool_calls = []
        finish_reason = None
        raw_metadata: dict[str, Any] = {}
        emitted_event = False
        await self.kernel.append_event(
            RuntimeEventType.LLM_REQUESTED,
            operation_id=operation_id,
            message="Streaming LLM turn requested.",
            payload={
                "chat_run_id": run_state.run_id,
                "model_turn": run_state.model_turn_count,
                "stream": True,
                "tool_count": len(tools),
            },
        )
        try:
            async for chunk in await self.llm.generate(
                messages=conversation,
                tools=tools,
                stream=True,
                **kwargs,
            ):
                if chunk.type == "text":
                    thinking_text += chunk.content
                    self._emit_event(
                        on_event,
                        EventType.THINKING,
                        content=chunk.content,
                        runtime_id=self.runtime_id,
                        runtime_kind=self.runtime_kind,
                        operation_id=operation_id,
                    )
                    emitted_event = True
                elif chunk.type == "tool_call_complete":
                    tool_calls.append(
                        {
                            "id": chunk.tool_call_id,
                            "name": chunk.tool_name,
                            "arguments": chunk.tool_args,
                        }
                    )
                    self._emit_event(
                        on_event,
                        EventType.TOOL_CALL,
                        tool_name=chunk.tool_name,
                        tool_args=chunk.tool_args,
                        runtime_id=self.runtime_id,
                        runtime_kind=self.runtime_kind,
                        operation_id=operation_id,
                    )
                    emitted_event = True
                metadata = getattr(chunk, "metadata", None)
                if isinstance(metadata, dict):
                    raw_metadata.update(metadata)
                    finish_reason = metadata.get("finish_reason") or finish_reason
        except Exception as error:
            if emitted_event:
                setattr(error, "_daita_stream_event_emitted", True)
            await self.kernel.append_event(
                RuntimeEventType.ERROR,
                operation_id=operation_id,
                message="Streaming LLM turn failed.",
                payload={
                    "chat_run_id": run_state.run_id,
                    "model_turn": run_state.model_turn_count,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
            raise
        await self.kernel.append_event(
            RuntimeEventType.LLM_COMPLETED,
            operation_id=operation_id,
            message="Streaming LLM turn completed.",
            payload={
                "chat_run_id": run_state.run_id,
                "model_turn": run_state.model_turn_count,
                "tool_call_count": len(tool_calls),
                "finish_reason": finish_reason,
                "text_length": len(thinking_text),
            },
        )
        return LLMResult(
            text=thinking_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_metadata=raw_metadata,
        )

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        specs: list[ModelToolSpec],
        *,
        operation_id: str,
        run_state: RunState,
        on_event: Callable | None,
    ) -> ChatToolCallResult:
        from daita.core.streaming import EventType

        tool_name = tool_call["name"]
        spec = next((item for item in specs if item.name == tool_name), None)
        start_time = time.time()
        if spec is None:
            rendered = {"error": f"Tool '{tool_name}' not found"}
            result = ChatToolCallResult(
                tool_name=tool_name,
                tool_call_id=tool_call.get("id"),
                arguments=tool_call.get("arguments", {}),
                capability_id="chat.missing_tool",
                owner="chat",
                operation_id=operation_id,
                task_id=None,
                evidence=(),
                rendered_result=rendered,
                duration_ms=0,
                error=rendered["error"],
            )
            self._emit_event(
                on_event, EventType.TOOL_RESULT, tool_name=tool_name, result=rendered
            )
            return result

        capability = self.extension_registry.get_capability(
            spec.capability_id,
            owner=spec.owner,
        )
        async with self.trace_manager.span(
            operation_name=f"tool_{tool_name}",
            trace_type=TraceType.TOOL_EXECUTION,
            agent_id=self.runtime_id,
            tool_name=tool_name,
            input_data=tool_call.get("arguments"),
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            operation_id=operation_id,
            capability_id=spec.capability_id,
            executor_id=capability.executor,
            plugin_id=spec.owner,
        ) as span_id:
            try:
                execution = await self.kernel.execute_capability(
                    spec.capability_id,
                    owner=spec.owner,
                    operation_id=operation_id,
                    operation_type="chat.tool_call",
                    input=tool_call.get("arguments", {}),
                    task_metadata={
                        "tool_view": spec.name,
                        "tool_call_id": tool_call.get("id"),
                        "model_turn": run_state.model_turn_count,
                    },
                    context={
                        "tool_view": spec.tool_view.to_dict(),
                        "chat_run_id": run_state.run_id,
                        "tool_owner": spec.owner,
                    },
                )
                evidence = tuple(execution.evidence)
                rendered = self._render_evidence_for_model(evidence)
                duration_ms = int((time.time() - start_time) * 1000)
                self.trace_manager.record_runtime_correlation(
                    span_id,
                    runtime_id=self.runtime_id,
                    runtime_kind=self.runtime_kind,
                    operation_id=operation_id,
                    task_id=execution.task.id,
                    capability_id=spec.capability_id,
                    executor_id=capability.executor,
                    plugin_id=spec.owner,
                    evidence_id=_first_evidence_id(evidence),
                )
                result = ChatToolCallResult(
                    tool_name=tool_name,
                    tool_call_id=tool_call.get("id"),
                    arguments=tool_call.get("arguments", {}),
                    capability_id=spec.capability_id,
                    owner=spec.owner,
                    operation_id=operation_id,
                    task_id=execution.task.id,
                    evidence=evidence,
                    rendered_result=rendered,
                    duration_ms=duration_ms,
                    retry_safe=capability.retry_safe,
                    replay_safe=capability.replay_safe,
                    idempotent=capability.idempotent,
                    side_effecting=capability.side_effecting,
                )
                self.trace_manager.record_output(span_id, rendered)
            except RuntimeKernelGovernanceBlocked as error:
                execution = error.result
                evidence = tuple(execution.evidence) if execution else ()
                rendered = {"error": "Tool call blocked by governance policy"}
                duration_ms = int((time.time() - start_time) * 1000)
                self.trace_manager.record_runtime_correlation(
                    span_id,
                    runtime_id=self.runtime_id,
                    runtime_kind=self.runtime_kind,
                    operation_id=operation_id,
                    task_id=execution.task.id if execution else None,
                    capability_id=spec.capability_id,
                    executor_id=capability.executor,
                    plugin_id=spec.owner,
                    evidence_id=_first_evidence_id(evidence),
                )
                result = ChatToolCallResult(
                    tool_name=tool_name,
                    tool_call_id=tool_call.get("id"),
                    arguments=tool_call.get("arguments", {}),
                    capability_id=spec.capability_id,
                    owner=spec.owner,
                    operation_id=operation_id,
                    task_id=execution.task.id if execution else None,
                    evidence=evidence,
                    rendered_result=rendered,
                    duration_ms=duration_ms,
                    blocked=True,
                    error=rendered["error"],
                    retry_safe=capability.retry_safe,
                    replay_safe=capability.replay_safe,
                    idempotent=capability.idempotent,
                    side_effecting=capability.side_effecting,
                )
                self.trace_manager.record_output(span_id, rendered)
            except Exception as error:
                rendered = {"error": f"Tool '{tool_name}' failed: {str(error)}"}
                duration_ms = int((time.time() - start_time) * 1000)
                result = ChatToolCallResult(
                    tool_name=tool_name,
                    tool_call_id=tool_call.get("id"),
                    arguments=tool_call.get("arguments", {}),
                    capability_id=spec.capability_id,
                    owner=spec.owner,
                    operation_id=operation_id,
                    task_id=None,
                    evidence=(),
                    rendered_result=rendered,
                    duration_ms=duration_ms,
                    error=rendered["error"],
                    retry_safe=capability.retry_safe,
                    replay_safe=capability.replay_safe,
                    idempotent=capability.idempotent,
                    side_effecting=capability.side_effecting,
                )
                self.trace_manager.record_output(span_id, rendered)

        if self._is_skill_owner(spec.owner) and result.task_id is not None:
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation_id,
                message=f"Skill tool {tool_name} executed.",
                task_id=result.task_id,
                capability_id=result.capability_id,
                executor_id=capability.executor,
                plugin_id=spec.owner,
                payload={
                    "diagnostic": "skill.tool_executed",
                    "skill_id": spec.owner,
                    "tool_view": tool_name,
                    "blocked": result.blocked,
                    "error": result.error,
                },
            )
        self._emit_event(
            on_event,
            EventType.TOOL_RESULT,
            tool_name=tool_name,
            result=result.rendered_result,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            operation_id=operation_id,
            task_id=result.task_id,
            capability_id=result.capability_id,
            executor_id=capability.executor,
            plugin_id=spec.owner,
            evidence_id=_first_evidence_id(result.evidence),
        )
        return result

    def _render_evidence_for_model(self, evidence: tuple[Evidence, ...]) -> Any:
        if len(evidence) == 1 and "result" in evidence[0].payload:
            return evidence[0].payload["result"]
        if len(evidence) == 1:
            return evidence[0].payload
        return {"evidence": [item.to_dict() for item in evidence]}

    def _resolve_skills(
        self,
        *,
        prompt: str,
        explicit_skills: tuple[str, ...],
    ) -> SkillResolution:
        return SkillResolver(self.extension_registry).resolve(
            runtime_kind=self.runtime_kind,
            prompt=prompt,
            explicit_skills=explicit_skills,
            runtime_context={
                "runtime_id": self.runtime_id,
                "available_capabilities": tuple(
                    capability.id for capability in self.extension_registry.capabilities
                ),
                "tool_view_names": tuple(
                    view.name for view in self.extension_registry.tool_views
                ),
            },
        )

    async def _record_skill_resolution(
        self,
        operation_id: str,
        resolution: SkillResolution,
    ) -> None:
        for activation in resolution.selected:
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation_id,
                message=f"Skill {activation.skill_name} selected.",
                plugin_id=activation.skill_id,
                payload={
                    "diagnostic": "skill.selected",
                    "activation": activation.to_dict(),
                },
            )
            if activation.context_loaded:
                await self.kernel.append_event(
                    RuntimeEventType.DIAGNOSTIC,
                    operation_id=operation_id,
                    message=f"Skill {activation.skill_name} context loaded.",
                    plugin_id=activation.skill_id,
                    payload={
                        "diagnostic": "skill.context_loaded",
                        "activation": activation.to_dict(),
                    },
                )
        for activation in resolution.skipped:
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation_id,
                message=f"Skill {activation.skill_name} activation skipped.",
                plugin_id=activation.skill_id,
                payload={
                    "diagnostic": "skill.activation_skipped",
                    "activation": activation.to_dict(),
                },
            )

    def _format_skill_catalog(self, resolution: SkillResolution) -> str | None:
        selected = set(resolution.selected_ids())
        cards = [
            activation
            for activation in resolution.activations
            if not activation.selected
            and activation.skill_id not in selected
            and activation.discovery.context_mode == "on_demand"
        ]
        if not cards:
            return None
        lines = []
        for activation in cards:
            hints = "; ".join(activation.discovery.when_to_use)
            suffix = f" Use when: {hints}" if hints else ""
            lines.append(
                f"- {activation.skill_name} ({activation.skill_id}): "
                f"{activation.discovery.description}{suffix}"
            )
        return "## Available Skills\n" + "\n".join(lines)

    def _is_skill_owner(self, plugin_id: str) -> bool:
        try:
            plugin = self.extension_registry.get_plugin(plugin_id)
        except KeyError:
            return False
        manifest = getattr(plugin, "manifest", None)
        return manifest is not None and manifest.kind is PluginKind.SKILL

    def _append_tool_messages(self, conversation, tool_calls, results) -> None:
        conversation.append({"role": "assistant", "tool_calls": tool_calls})
        for tool_call, result in zip(tool_calls, results):
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "content": json.dumps(result["result"], default=json_serializer),
                }
            )

    def _evaluate_final_answer_readiness(
        self, run_state: RunState, available_tools: list[ModelToolSpec], final_text: str
    ) -> FinalAnswerReadiness:
        combined = FinalAnswerReadiness()
        for hook in run_state.final_answer_readiness_hooks:
            try:
                decision = hook(run_state, final_text, available_tools)
            except Exception:
                continue
            if decision is None:
                continue
            run_state.diagnostics.update(decision.diagnostics)
            if decision.warning and decision.warning not in run_state.warnings:
                run_state.warnings.append(decision.warning)
            if not decision.allow_final:
                combined.allow_final = False
                combined.guidance = combined.guidance or decision.guidance
                combined.warning = combined.warning or decision.warning
        return combined

    def _record_tool_evidence(self, run_state, tool_call, result) -> None:
        raw = result.get("result")
        if isinstance(raw, dict) and raw.get("error"):
            return
        add_evidence(
            run_state,
            domain="generic",
            kind="tool_result",
            source_tool=str(tool_call.get("name") or ""),
            payload={
                "arguments": tool_call.get("arguments", {}),
                "result": raw,
                "duration_ms": result.get("duration_ms"),
            },
        )

    def _terminal_result(self, results, terminal_tools):
        if not has_terminal_tool_result(results, terminal_tools):
            return None
        return next(
            (
                result
                for result in reversed(results)
                if result.get("tool") in terminal_tools
            ),
            None,
        )

    def _terminal_result_text(self, result):
        raw = (result or {}).get("result")
        if isinstance(raw, dict):
            rows = raw.get("rows")
            if isinstance(rows, list):
                if not rows:
                    return "No matching rows were returned."
                return "Query result: " + json.dumps(rows[:5], default=str)
            row_count = raw.get("row_count")
            if row_count == 0:
                return "No matching rows were returned."
            return "Query result: " + json.dumps(raw, default=str)
        if raw is None:
            return "The terminal tool completed, but returned no result payload."
        return str(raw)

    def _suggested_next_tool(self, results):
        for result in reversed(results):
            raw = result.get("result")
            if not isinstance(raw, dict):
                continue
            tool_name = raw.get("suggested_next_tool")
            if tool_name:
                return str(tool_name)
        return None

    def _resolve_suggested_next_tool(self, results, resolved_tools):
        suggested_next_tool = self._suggested_next_tool(results)
        if not suggested_next_tool:
            return None
        available = {spec.name for spec in resolved_tools}
        if suggested_next_tool in available:
            return {"tool_name": suggested_next_tool, "allowed": True}
        guidance = (
            f"The previous tool result suggested {suggested_next_tool}, but that "
            "tool is not available in this run. Do not call unavailable tools. "
            "Synthesize from the available evidence"
        )
        if available:
            guidance += " or choose one of the available tools: " + ", ".join(
                sorted(available)
            )
        guidance += "."
        return {
            "tool_name": suggested_next_tool,
            "allowed": False,
            "guidance": guidance,
            "warning": f"suggested_next_tool_unavailable:{suggested_next_tool}",
            "diagnostics": {"unavailable_suggested_next_tool": suggested_next_tool},
        }

    def _build_result(
        self,
        text,
        operation_id,
        tool_calls,
        iterations,
        run_state,
        on_event,
        *,
        partial=False,
    ) -> ChatRunResult:
        from daita.core.streaming import EventType

        token_stats = self._token_stats()
        cost = token_stats.get("estimated_cost", 0.0)
        self._emit_event(
            on_event,
            EventType.COMPLETE,
            final_result=text,
            iterations=iterations,
            token_usage=token_stats,
            cost=cost,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            operation_id=operation_id,
        )
        return ChatRunResult(
            text=text,
            operation_id=operation_id,
            iterations=iterations,
            tool_calls=tuple(tool_calls),
            diagnostics=run_state.diagnostic_summary(),
            token_usage=token_stats,
            cost=cost,
            partial=partial,
        )

    def _token_stats(self) -> dict[str, Any]:
        if self.llm is None or not hasattr(self.llm, "get_token_stats"):
            return {}
        return self.llm.get_token_stats()


class FocusedRuntimeTool:
    """Runtime-local focused wrapper for LocalTool compatibility."""

    def __init__(self, tool, focus: str):
        self._tool = tool
        self._focus = focus

    async def handler(self, arguments: dict[str, Any]) -> Any:
        tool_props = self._tool.parameters.get("properties", {})
        if "focus" in tool_props:
            try:
                return await self._tool.handler({**arguments, "focus": self._focus})
            except Exception as error:
                logger.warning(
                    "Focus injection failed for %s (%s); falling back to Python evaluation",
                    self.name,
                    error,
                )
        result = await self._tool.handler(arguments)
        if result is None:
            return result
        try:
            from daita.core.focus import apply_focus

            if isinstance(result, dict) and isinstance(result.get("rows"), list):
                focused = apply_focus(result["rows"], self._focus)
                n = len(focused) if isinstance(focused, list) else 1
                return {**result, "rows": focused, "row_count": n}
            return apply_focus(result, self._focus)
        except Exception as error:
            logger.warning("Focus application failed for %s: %s", self.name, error)
            return result

    def __getattr__(self, name):
        return getattr(self._tool, name)


def _resolve_tool_focus(agent_focus, tool) -> str | None:
    tool_default = getattr(tool, "focus", None)
    if isinstance(agent_focus, dict):
        return agent_focus.get(tool.name) or tool_default
    return agent_focus or tool_default


def _first_evidence_id(evidence: tuple[Evidence, ...]) -> str | None:
    return next((item.id for item in evidence if item.id is not None), None)


def _normalize_skill_selection(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
