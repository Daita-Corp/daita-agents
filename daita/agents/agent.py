"""
Agent — Secondary generic chat facade.

The data-first production interface is ``Agent.from_db()`` / ``daita.db``.
Use this generic Agent for non-DB chat, local tools, skills, and lightweight
runtime-extension experiments.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Union, Callable

if TYPE_CHECKING:
    from .conversation import ConversationHistory
    from daita.db.agent import DbAgent
    from daita.db.llm_service import DbLLMConfig
    from daita.db.models import (
        DbMemoryConfig,
        DbRuntimeConfig,
        DbRuntimeOptions,
        DbSourceOptions,
    )
    from daita.plugins.base_db import BaseDatabasePlugin
    from daita.plugins.catalog import CatalogPlugin

import asyncio

from ..config.base import AgentConfig, AgentType, RetryPolicy
from ..core.interfaces import LLMProvider
from ..core.exceptions import AgentError
from ..runtime import InMemoryRuntimeStore, RuntimeKernel
from ..skills.base import BaseSkill
from ..core.tracing import TraceType
from .base import BaseAgent
from .chat.facade import ChatAgentFacadeMixin
from .chat.runtime import ChatRuntime, ChatRunResult, ChatToolCallResult, ModelToolSpec

logger = logging.getLogger(__name__)

__all__ = [
    "Agent",
    "ChatRunResult",
    "ChatToolCallResult",
    "ModelToolSpec",
]


# Import unified plugin access
from ..plugins import PluginAccess
from ..plugins.registry import ExtensionRegistry
from ..llm.factory import create_llm_provider
from ..config.settings import settings
from ..core.tools import LocalTool, LocalToolCatalog


class Agent(ChatAgentFacadeMixin, BaseAgent):
    """Generic chat facade with autonomous tool-calling and LLM execution."""

    # Class-level defaults for smart constructor
    _default_llm_provider = "openai"
    _default_model = "gpt-5.4-mini"
    _ALLOWED_DEFAULTS = frozenset({"llm_provider", "model"})

    @classmethod
    def configure_defaults(cls, **kwargs):
        """Set global defaults for all Agent instances. Allowed keys: llm_provider, model."""
        invalid = set(kwargs) - cls._ALLOWED_DEFAULTS
        if invalid:
            raise ValueError(
                f"configure_defaults() received unknown keys: {invalid}. "
                f"Allowed: {cls._ALLOWED_DEFAULTS}"
            )
        for key, value in kwargs.items():
            setattr(cls, f"_default_{key}", value)

    @classmethod
    async def from_db(
        cls,
        source: str | BaseDatabasePlugin,
        *,
        name: str | None = None,
        mode: str | None = None,
        config: DbRuntimeConfig | None = None,
        source_options: DbSourceOptions | None = None,
        llm: DbLLMConfig | None = None,
        runtime: DbRuntimeOptions | None = None,
        memory: DbMemoryConfig | None = None,
        catalog: CatalogPlugin | None | bool = None,
        lineage: Any | bool | None = None,
        quality: Any | bool | None = None,
        history: ConversationHistory | bool | None = None,
        stateful: bool = False,
        plugins: tuple[Any, ...] | list[Any] = (),
        skills: tuple[Any, ...] | list[Any] = (),
    ) -> DbAgent:
        """Create a DB-aware agent from typed binding records."""
        from daita.db import from_db

        return await from_db(
            source,
            name=name,
            mode=mode,
            config=config,
            source_options=source_options,
            llm=llm,
            runtime=runtime,
            memory=memory,
            catalog=catalog,
            lineage=lineage,
            quality=quality,
            history=history,
            stateful=stateful,
            plugins=plugins,
            skills=skills,
        )

    def __init__(
        self,
        name: Optional[str] = None,
        llm_provider: Optional[Union[str, LLMProvider]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        prompt: Optional[Union[str, Dict[str, str]]] = None,
        focus: Optional[Union[str, Dict[str, str]]] = None,
        display_reasoning: bool = False,
        # Extension/plugin sources
        plugins: Optional[List] = None,
        extension_registry: Optional[ExtensionRegistry] = None,
        skills: Optional[List[BaseSkill]] = None,
        # Tool sources (local tool projection: LocalTool or plugin-like objects)
        tools: Optional[List] = None,
        # Reliability
        enable_reliability: bool = False,
        max_concurrent_tasks: int = 10,
        max_queue_size: int = 100,
        # Retry (forwarded to AgentConfig when config is not provided)
        enable_retry: bool = False,
        retry_policy: Optional[RetryPolicy] = None,
        # LLM provider pass-through only (temperature, max_tokens, timeout, etc.)
        **kwargs,
    ):
        """Initialize Agent with smart constructor auto-configuration."""
        # Store LLM provider config for lazy initialization
        self._llm_kwargs = kwargs
        self._llm_provider_name: str | None
        self._llm_model: str | None
        self._llm_api_key: str | None
        self._llm: LLMProvider | None
        llm_provider_to_pass: LLMProvider | None
        if llm_provider is None or isinstance(llm_provider, str):
            # Defer LLM provider creation until first use
            self._llm_provider_name = llm_provider or self._default_llm_provider
            self._llm_model = model or self._default_model
            self._llm_api_key = api_key
            self._llm = None
            llm_provider_to_pass = None
        else:
            # User provided an actual LLM provider instance
            self._llm_provider_name = self._llm_model = self._llm_api_key = None
            self._llm = llm_provider
            llm_provider_to_pass = llm_provider

        # Create default config if none provided
        if config is None:
            config = AgentConfig(
                name=name or "Agent",
                type=AgentType.STANDARD,
                enable_retry=enable_retry,
                retry_policy=retry_policy,
            )

        # Initialize base agent (which handles automatic tracing)
        super().__init__(
            config,
            llm_provider_to_pass,
            agent_id,
            name,
            enable_reliability=enable_reliability,
            max_concurrent_tasks=max_concurrent_tasks,
            max_queue_size=max_queue_size,
        )

        # Store customization options
        self.prompt = prompt
        self.default_focus = focus

        # Decision display setup
        self.display_reasoning = display_reasoning
        self._decision_display: Any | None = None

        if display_reasoning:
            self._setup_decision_display()

        # Tool management (unified system)
        self.local_tool_catalog = LocalToolCatalog()
        self.extension_registry = extension_registry or ExtensionRegistry()
        self.runtime_store = InMemoryRuntimeStore()
        self.runtime_kernel = RuntimeKernel(
            runtime_id=self.agent_id,
            runtime_kind="chat",
            extension_registry=self.extension_registry,
            runtime_store=self.runtime_store,
        )
        self._runtime: ChatRuntime | None = None
        self._extension_setup_plugin_ids: set[str] = set()
        self._local_tool_sources: list[Any] = []
        self._tools_setup = False
        for plugin in plugins or []:
            self.add_plugin(plugin)
        for source in tools or []:
            self.add_plugin(source)
        for skill in skills or []:
            self.add_skill(skill)
        # Plugin access for direct plugin usage
        self.plugins = PluginAccess()

        logger.debug(f"Agent {self.name} initialized")

    @property
    def runtime(self) -> ChatRuntime:
        """Return the ChatRuntime implementation behind this Agent facade."""
        if self._runtime is None:
            runtime = ChatRuntime(
                runtime_id=self.agent_id,
                agent_name=self.name,
                llm_getter=lambda: self.llm,
                llm_provider_name_getter=lambda: self._llm_provider_name,
                config=self.config,
                prompt=self.prompt,
                default_focus=self.default_focus,
                extension_registry=self.extension_registry,
                runtime_kernel=self.runtime_kernel,
                runtime_store=self.runtime_store,
                trace_manager=self.trace_manager,
                setup_tools=self._setup_tools,
                setup_extensions=self.setup_extensions,
                emit_event=self._emit_event,
                retry_with_tracing=self._retry_with_tracing,
                final_answer_hooks_getter=lambda: getattr(
                    self, "_daita_final_answer_readiness_hooks", []
                ),
            )
            runtime.bind_local_tool_catalog(self.local_tool_catalog)
            self._runtime = runtime
        return self._runtime

    @runtime.setter
    def runtime(self, value: ChatRuntime) -> None:
        """Allow advanced users and tests to replace the ChatRuntime."""
        self._runtime = value

    @property
    def llm(self):
        """
        Lazily create LLM provider on first access.

        This defers API key validation until the LLM is actually needed,
        improving developer experience when loading .env files.
        """
        if (
            self._llm is None
            and self._llm_provider_name is not None
            and self._llm_model is not None
        ):
            # Try to get API key
            api_key = self._llm_api_key or settings.get_llm_api_key(
                self._llm_provider_name
            )
            if api_key:
                self._llm = create_llm_provider(
                    provider=self._llm_provider_name,
                    model=self._llm_model,
                    api_key=api_key,
                    agent_id=self.agent_id,
                    **self._llm_kwargs,  # Pass through kwargs (includes timeout, etc.)
                )
                # Set agent_id for automatic LLM tracing
                if self._llm:
                    self._llm.set_agent_id(self.agent_id)
        return self._llm

    @llm.setter
    def llm(self, value):
        """Allow setting LLM provider directly."""
        self._llm = value
        if value is not None and hasattr(self, "agent_id"):
            value.set_agent_id(self.agent_id)

    def _setup_decision_display(self):
        """Setup minimal decision display for local development."""
        try:
            from ..display.console import create_console_decision_display
            from ..core.decision_tracing import register_agent_decision_stream

            # Create display
            self._decision_display = create_console_decision_display(
                agent_name=self.name, agent_id=self.agent_id
            )

            # Register with decision streaming system
            register_agent_decision_stream(
                agent_id=self.agent_id, callback=self._decision_display.handle_event
            )

            logger.debug(f"Decision display enabled for agent {self.name}")

        except Exception as e:
            logger.warning(f"Failed to setup decision display: {e}")
            self.display_reasoning = False
            self._decision_display = None

    async def stop(self) -> None:
        """Stop agent and clean up runtime extension resources."""
        await self.teardown_extensions()

        # Call parent stop for standard cleanup
        await super().stop()

    @property
    def health(self) -> Dict[str, Any]:
        """Enhanced health information for Agent."""
        base_health = super().health
        llm = self.llm

        # Add Agent-specific health info
        base_health.update(
            {
                "tools": {
                    "count": len(self.available_tools),
                    "setup": self._tools_setup,
                    "names": self.tool_names,
                },
                "extensions": {
                    "plugin_ids": list(self.extension_registry.plugin_ids),
                    "manifest_ids": [manifest.id for manifest in self.plugin_manifests],
                    "capability_ids": [
                        capability.id for capability in self.capabilities
                    ],
                    "capability_count": len(self.capabilities),
                    "tool_view_names": [view.name for view in self.tool_views],
                    "tool_view_count": len(self.tool_views),
                    "context_provider_ids": [
                        provider.id for provider in self.context_providers
                    ],
                    "context_provider_count": len(self.context_providers),
                    "executor_ids": [executor.id for executor in self.executors],
                    "executor_count": len(self.executors),
                    "policy_ids": [policy.id for policy in self.policies],
                    "policy_count": len(self.policies),
                    "evidence_schema_kinds": [
                        schema.kind for schema in self.evidence_schemas
                    ],
                    "evidence_schema_count": len(self.evidence_schemas),
                    "worker_ids": [worker.id for worker in self.workers],
                    "worker_count": len(self.workers),
                    "diagnostic_ids": [
                        diagnostic.declaration_id
                        for diagnostic in self.extension_diagnostics
                    ],
                    "diagnostic_count": len(self.extension_diagnostics),
                    "setup_plugin_ids": self.extension_setup_plugin_ids,
                    "pending_setup_plugin_ids": self.pending_extension_setup_plugin_ids,
                    "setup_complete": self.extensions_setup_complete,
                },
                "llm": {
                    "available": llm is not None,
                    "provider": getattr(llm, "provider_name", None),
                },
            }
        )

        return base_health

    # ========================================================================
    # USER API - What developers call directly
    # ========================================================================

    async def run(
        self,
        prompt: str,
        tools: Optional[List[Union[str, LocalTool]]] = None,
        max_iterations: int = 20,
        timeout_seconds: Optional[int] = None,
        on_event: Optional[Callable] = None,
        history: Optional["ConversationHistory"] = None,
        detailed: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """Execute instruction with autonomous tool calling.

        Args:
            detailed: When ``True``, return the full execution dict (result,
                tool_calls, iterations, tokens, cost, processing_time_ms, …)
                instead of just the answer string.
        """
        result = await self._run_traced(
            prompt,
            tools,
            max_iterations,
            timeout_seconds,
            on_event,
            history=history,
            **kwargs,
        )
        return result if detailed else result["result"]

    async def stream(
        self,
        prompt: str,
        tools: Optional[List[Union[str, LocalTool]]] = None,
        max_iterations: int = 20,
        timeout_seconds: Optional[int] = None,
        history: Optional["ConversationHistory"] = None,
        **kwargs,
    ):
        """Execute instruction and yield :class:`AgentEvent` objects as they arrive.

        Async generator — use ``async for event in agent.stream(...)``.
        Yields events until a ``COMPLETE`` or ``ERROR`` event is emitted.

        Example::

            async for event in agent.stream("What are total sales?"):
                if event.type == EventType.THINKING:
                    print(event.content, end="", flush=True)
                elif event.type == EventType.COMPLETE:
                    print("\\nDone:", event.final_result)
        """
        from ..core.streaming import AgentEvent, EventType

        queue: asyncio.Queue = asyncio.Queue()

        def _on_event(event: AgentEvent) -> None:
            queue.put_nowait(event)

        async def _run_bg() -> None:
            try:
                await self._run_traced(
                    prompt,
                    tools,
                    max_iterations,
                    timeout_seconds,
                    _on_event,
                    history=history,
                    **kwargs,
                )
            except Exception as exc:
                queue.put_nowait(exc)
            finally:
                queue.put_nowait(None)  # sentinel

        task = asyncio.create_task(_run_bg())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
                if item.type in (EventType.COMPLETE, EventType.ERROR):
                    break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _run_traced(
        self,
        prompt: str,
        tools: Optional[List[Union[str, LocalTool]]],
        max_iterations: int,
        timeout_seconds: Optional[int],
        on_event: Optional[Callable],
        history: Optional["ConversationHistory"] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Internal: Execute with automatic tracing and optional event streaming."""
        start_time = time.time()

        # Create agent-level trace span (automatic, invisible to users)
        tools_requested = None
        if tools is not None:
            tools_requested = [
                t if isinstance(t, str) else getattr(t, "name", str(t)) for t in tools
            ]

        async with self.trace_manager.span(
            operation_name="agent_run",
            trace_type=TraceType.AGENT_EXECUTION,
            agent_id=self.agent_id,
            agent_name=self.name,
            prompt=prompt[:200],  # Truncate for storage
            tools_requested=tools_requested,
            max_iterations=max_iterations,
            entry_point="run",
            input_data={"prompt": prompt, "tools_requested": tools_requested},
        ) as span_id:
            # Resolve history workspace and extract prior messages before branching
            if history is not None:
                history._set_workspace(self.agent_id)
                kwargs["initial_messages"] = history.messages

            execute_fn = (
                self._execute_autonomous_with_retry
                if self.config.retry_enabled
                else self._execute_autonomous
            )
            coro = execute_fn(
                prompt=prompt,
                tools=tools,
                max_iterations=max_iterations,
                on_event=on_event,
                **kwargs,
            )

            if timeout_seconds is not None:
                try:
                    result = await asyncio.wait_for(coro, timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    raise AgentError(f"Run timed out after {timeout_seconds}s")
            else:
                result = await coro

            # Enrich result with metadata
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            result["agent_id"] = self.agent_id
            result["agent_name"] = self.name
            # Capture OTel trace_id while the span is still live
            result["_daita_trace_id"] = (
                self.trace_manager.trace_context.current_trace_id
            )

            # Append completed turn to conversation history
            if history is not None:
                await history.add_turn(prompt, result.get("result", ""))

            self.trace_manager.record_output(span_id, result)
            return result
