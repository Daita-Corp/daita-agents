"""
Agent — The primary agent class for building AI-driven applications.

Provides autonomous tool-calling, multi-provider LLM support, plugin integration,
and zero-configuration tracing. Use Agent for most use cases; subclass BaseAgent
for full control over the execution loop.
"""

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Union, Callable

if TYPE_CHECKING:
    from .conversation import ConversationHistory

import asyncio

from ..config.base import AgentConfig, AgentType, RetryPolicy
from ..core.interfaces import LLMProvider
from ..core.exceptions import AgentError, SkillError
from ..runtime import (
    AccessMode,
    Capability,
    ContextAudience,
    ContextBlock,
    ContextProvider,
    Evidence,
    EvidenceSchema,
    Executor,
    InMemoryRuntimeStore,
    MonitorRuntime,
    MonitorSpec,
    Operation,
    Policy,
    RiskLevel,
    RuntimeKernel,
    Task,
    ToolView,
    Worker,
)
from ..skills.base import BaseSkill
from ..core.tracing import TraceType
from .runtime.guardrails import (
    has_terminal_tool_result as _has_terminal_tool_result,
    make_error_fingerprint as _make_error_fingerprint,
    tool_loop_error_message as _tool_loop_error_message,
)
from .runtime.llm_turn import LLMResult
from .runtime.tools import (
    execute_tool_call as _execute_tool_call,
    json_serializer as _json_serializer,
)
from .base import BaseAgent
from .chat_runtime import ChatRuntime, ChatRunResult, ChatToolCallResult, ModelToolSpec

logger = logging.getLogger(__name__)

__all__ = [
    "Agent",
    "ChatRunResult",
    "ChatToolCallResult",
    "FocusedTool",
    "LLMResult",
    "ModelToolSpec",
    "_execute_tool_call",
    "_json_serializer",
    "_make_error_fingerprint",
    "_has_terminal_tool_result",
    "_tool_loop_error_message",
]


# Import unified plugin access
from ..plugins import PluginAccess
from ..plugins.base import PluginContext
from ..plugins.manifest import PluginKind, PluginManifest
from ..plugins.registry import ExtensionRegistry, RegistryDiagnostic
from ..llm.factory import create_llm_provider
from ..config.settings import settings
from ..core.tools import LocalTool, LocalToolCatalog


class FocusedTool:
    """Wrapper that applies a Focus DSL expression to tool results before they reach the LLM."""

    def __init__(self, tool: LocalTool, focus: str):
        self._tool = tool
        self._focus = focus

    async def handler(self, arguments: Dict[str, Any]) -> Any:
        # If the tool natively accepts a focus DSL (e.g. SQL plugin tools), inject
        # it into the args so pushdown happens at the source.  If that fails,
        # fall through to the Python-side evaluator below.
        tool_props = self._tool.parameters.get("properties", {})
        if "focus" in tool_props:
            try:
                return await self._tool.handler({**arguments, "focus": self._focus})
            except Exception as e:
                logger.warning(
                    f"Focus injection failed for {self._tool.name} ({e}); "
                    "falling back to Python evaluation"
                )
                # Fall through to Python-side application below

        result = await self._tool.handler(arguments)
        if result is None:
            return result
        try:
            from ..core.focus import apply_focus

            # Plugin tools wrap results in {"rows": [...], "row_count": N, ...}.
            # Apply focus to the rows list, not the wrapper dict, then put it back.
            if isinstance(result, dict) and isinstance(result.get("rows"), list):
                focused = apply_focus(result["rows"], self._focus)
                n = len(focused) if isinstance(focused, list) else 1
                logger.debug(
                    f"Focus applied to {self.name} rows: {result['row_count']} -> {n}"
                )
                return {**result, "rows": focused, "row_count": n}
            focused = apply_focus(result, self._focus)
            logger.debug(
                f"Focus applied to {self.name}: {type(result).__name__} -> {type(focused).__name__}"
            )
            return focused
        except Exception as e:
            logger.warning(f"Focus application failed for {self.name}: {e}")
            return result

    def __getattr__(self, name):
        return getattr(self._tool, name)

    def __repr__(self):
        return f"FocusedTool({self._tool.name}, focus={self._focus!r})"


class LocalWatchExecutor:
    """Runtime executor for one local @agent.watch handler."""

    def __init__(self, *, executor_id: str, capability_id: str, handler, config):
        self.id = executor_id
        self.capability_ids = frozenset({capability_id})
        self._handler = handler
        self._config = config

    async def execute(self, task: Task, operation: Operation, context):
        event = context.get("watch_event")
        if event is None:
            raise RuntimeError("watch handler context did not include watch_event")

        async def _invoke():
            return await self._handler(event)

        if self._config.handler_timeout is not None:
            result = await asyncio.wait_for(
                _invoke(),
                timeout=self._config.handler_timeout,
            )
        else:
            result = await _invoke()
        if isinstance(result, Evidence):
            return [result]
        if isinstance(result, (list, tuple)) and all(
            isinstance(item, Evidence) for item in result
        ):
            return list(result)
        payload: dict[str, Any] = {
            "watch_name": self._config.name,
            "handled": True,
        }
        if result is not None:
            payload["result"] = result
        return [
            Evidence(
                kind="agent.local.watch.result",
                owner=context.get("watch_owner"),
                payload=payload,
                metadata={"monitor_id": context.get("monitor_id")},
            )
        ]


class LocalWatchPlugin:
    """Hidden runtime declaration for one @agent.watch handler."""

    def __init__(self, *, plugin_id: str, capability_id: str, executor, config):
        self.manifest = PluginManifest(
            id=plugin_id,
            display_name=f"Watch Handler {config.name}",
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
            domains=frozenset({"monitor"}),
        )
        self._capability = Capability(
            id=capability_id,
            owner=plugin_id,
            description=f"Handle watch trigger {config.name}.",
            domains=frozenset({"monitor", "watch"}),
            operation_types=frozenset({"monitor.triggered"}),
            access=AccessMode.NONE,
            risk=RiskLevel.LOW,
            input_schema={"type": "object"},
            output_evidence=frozenset({"agent.local.watch.result"}),
            executor=executor.id,
            runtime_only=True,
            model_visible=False,
            side_effecting=True,
            metadata={"watch_name": config.name},
        )
        self._executor = executor

    def declare_capabilities(self):
        return (self._capability,)

    def get_executors(self):
        return (self._executor,)


def _resolve_tool_focus(
    agent_focus: Optional[Union[str, Dict[str, str]]],
    t: "LocalTool",
) -> Optional[str]:
    """Return the effective focus DSL for a single tool, applying precedence rules."""
    tool_default = getattr(t, "focus", None)
    if isinstance(agent_focus, dict):
        return agent_focus.get(t.name) or tool_default
    return agent_focus or tool_default


def _runtime_id_segment(value: str) -> str:
    segment = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    if not segment:
        segment = "handler"
    if segment[0].isdigit():
        segment = f"handler_{segment}"
    return segment


def _watch_event_payload(event: Any) -> dict[str, Any]:
    return {
        "value": _json_safe(getattr(event, "value", None)),
        "triggered_at": _json_safe(getattr(event, "triggered_at", None)),
        "source_type": getattr(event, "source_type", None),
        "resolved": bool(getattr(event, "resolved", False)),
        "previous_value": _json_safe(getattr(event, "previous_value", None)),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


class Agent(BaseAgent):
    """DAITA's primary agent with autonomous tool-calling and LLM-driven task execution."""

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
    async def from_db(cls, source, **kwargs):
        """Create a DB-aware agent from a database source.

        This public entry point is backed by the operation-centric DB runtime.
        It returns a ``DbAgent`` facade instead of a patched generic ``Agent``.

        Args:
            source: Connection string (e.g. "postgresql://user:pass@host/db") or
                    a BaseDatabasePlugin instance.
            **kwargs: Common options:
                name: Agent name override.
                model, api_key, llm_provider: LLM configuration.
                prompt: User context prepended to auto-generated schema prompt.
                db_schema: DB schema name override (e.g. "public" for PostgreSQL).
                lineage: True to auto-create LineagePlugin, or pass an instance.
                    Seeds FK relationships into a persistent graph.
                memory: True to auto-create MemoryPlugin, or pass an instance.
                    Enables conversational business context annotations.
                cache_ttl: Schema cache TTL in seconds. None disables caching.
                Additional kwargs forwarded to Agent.__init__.

        Example:
            agent = await Agent.from_db(
                "postgresql://localhost/mydb",
                lineage=True,
                memory=True,
                cache_ttl=3600,
            )
            result = await agent.run("What are our top customers?")
        """
        from daita.db import from_db

        return await from_db(source, **kwargs)

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
        relay: Optional[str] = None,
        mcp: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        mcp_servers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
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
        self.relay = relay

        # Decision display setup
        self.display_reasoning = display_reasoning
        self._decision_display = None

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
        self._local_tool_sources = []
        self._tools_setup = False
        self._watch_action_plugins: dict[str, tuple[str, str]] = {}
        for plugin in plugins or []:
            self.add_plugin(plugin)
        for source in tools or []:
            self.add_plugin(source)
        for skill in skills or []:
            self.add_skill(skill)
        self._focus_default_warned: set = (
            set()
        )  # tracks @tool focus defaults already logged

        # Tool call history tracking for operations metadata
        self._tool_call_history = []

        # MCP server integration — setup happens lazily on first use
        self.mcp_registry = None
        self.mcp_tools = []
        mcp_config = mcp_servers if mcp_servers is not None else mcp
        self._mcp_server_configs = (
            ([mcp_config] if isinstance(mcp_config, dict) else mcp_config)
            if mcp_config is not None
            else []
        )

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
                start_watches=self._start_watches,
                emit_event=self._emit_event,
                retry_with_tracing=self._retry_with_tracing,
                context_blocks_renderer=self._render_extension_context_blocks,
                context_formatter=self._format_extension_context,
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
        if self._llm is None and self._llm_provider_name is not None:
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

    async def _setup_mcp_tools(self):
        """Setup MCP servers and discover available tools. Called lazily on first process()."""
        if self.mcp_registry is not None:
            # Already setup
            return

        if not self._mcp_server_configs:
            # No MCP servers configured
            return

        try:
            from ..plugins.mcp import MCPServer, MCPToolRegistry

            logger.info(
                f"Setting up {len(self._mcp_server_configs)} MCP server(s) for {self.name}"
            )

            # Create registry
            self.mcp_registry = MCPToolRegistry()

            # Connect to each server and register tools
            for server_config in self._mcp_server_configs:
                server = MCPServer(
                    command=server_config.get("command"),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    server_name=server_config.get("name"),
                )

                # Add to registry (automatically connects and discovers tools)
                await self.mcp_registry.add_server(server)

            # Get all tools from registry
            self.mcp_tools = self.mcp_registry.get_all_tools()

            logger.info(
                f"MCP setup complete: {self.mcp_registry.tool_count} tools from {self.mcp_registry.server_count} server(s)"
            )

        except ImportError:
            logger.error(
                "MCP SDK not installed. Install with: pip install 'daita-agents[mcp]'"
            )
            raise

        except Exception as e:
            logger.error(f"Failed to setup MCP servers: {str(e)}")
            raise

    def _register_tool_source(self, source) -> None:
        """Register a local LocalTool source into the compatibility registry."""
        if isinstance(source, LocalTool):
            self.local_tool_catalog.register(source)
        else:
            logger.warning(
                "Ignoring non-manifest plugin %s; runtime plugins must declare a "
                "PluginManifest with capabilities and executors.",
                source.__class__.__name__,
            )

    async def _setup_tools(self):
        """Set up MCP tools. Non-MCP tools are registered eagerly in add_plugin()."""
        await self._setup_extension_plugins()

        if self._tools_setup:
            return  # Already setup

        # Setup MCP tools (requires async connection)
        if self._mcp_server_configs and self.mcp_registry is None:
            await self._setup_mcp_tools()
            for mcp_tool in self.mcp_tools:
                local_tool = LocalTool.from_mcp_tool(mcp_tool, self.mcp_registry)
                self.local_tool_catalog.register(local_tool)

        self._tools_setup = True
        logger.info(
            f"Agent {self.name} ready with {self.local_tool_catalog.tool_count} tools"
        )

    async def _setup_extension_plugins(self) -> None:
        """Set up runtime-aware plugins through the extension registry."""
        pending_plugin_ids = self._pending_extension_setup_plugin_ids()
        if not pending_plugin_ids:
            return

        context = PluginContext(
            runtime_id=self.agent_id,
            runtime_kind="agent",
            agent_id=self.agent_id,
        )
        for plugin_id in pending_plugin_ids:
            await self.extension_registry.setup_plugin(plugin_id, context)
            self._extension_setup_plugin_ids.add(plugin_id)

    def _pending_extension_setup_plugin_ids(self) -> List[str]:
        """Return registered plugin IDs that have not run PluginContext setup."""
        return [
            plugin_id
            for plugin_id in self.extension_registry.plugin_ids
            if plugin_id not in self._extension_setup_plugin_ids
        ]

    async def setup_extensions(self) -> None:
        """Set up attached registry plugins through ``PluginContext``."""
        await self._setup_extension_plugins()

    async def teardown_extensions(self) -> None:
        """Tear down attached registry plugins and clear setup state."""
        if not self.extension_registry.plugin_ids:
            return
        try:
            await self.extension_registry.teardown_all()
        finally:
            self._extension_setup_plugin_ids.clear()

    @property
    def extension_setup_plugin_ids(self) -> List[str]:
        """Return registry plugin IDs that have completed ``PluginContext`` setup."""
        return [
            plugin_id
            for plugin_id in self.extension_registry.plugin_ids
            if plugin_id in self._extension_setup_plugin_ids
        ]

    @property
    def pending_extension_setup_plugin_ids(self) -> List[str]:
        """Return registry plugin IDs pending ``PluginContext`` setup."""
        return self._pending_extension_setup_plugin_ids()

    @property
    def extensions_setup_complete(self) -> bool:
        """Return True when all registry plugins have completed setup."""
        return not self.pending_extension_setup_plugin_ids

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
            # Start watches lazily on first run (idempotent)
            await self._start_watches()

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

    def _resolve_tools(
        self, tools: Optional[List[Union[str, LocalTool]]]
    ) -> List[LocalTool]:
        """Resolve tool names to LocalTool instances. If None, returns all registered tools."""
        if tools is None:
            # Use all registered tools
            return list(self.available_tools)

        tool_list = []
        for t in tools:
            if isinstance(t, str):
                # Tool name - look up in registry
                tool = next(
                    (item for item in self.available_tools if item.name == t),
                    None,
                )
                if tool is None:
                    raise ValueError(f"Tool '{t}' not found in registry")
                tool_list.append(tool)
            else:
                # Already an LocalTool instance
                tool_list.append(t)

        return tool_list

    def _emit_event(self, on_event: Optional[Callable], event_type, **kwargs):
        """Emit event only if callback provided. Zero overhead when None."""
        if on_event:
            from ..core.streaming import AgentEvent

            on_event(AgentEvent(type=event_type, **kwargs))

    async def _prepare_tools_with_focus(
        self, tools: Optional[List[Union[str, "LocalTool"]]]
    ) -> List["LocalTool"]:
        """Resolve tools and wrap with FocusedTool using the focus precedence chain.

        Precedence (highest → lowest):
            1. Agent dict focus  — Agent(focus={"tool_name": "..."})
            2. Agent string focus — Agent(focus="...")
            3. @tool default      — @tool(focus="..."), warned on first use
            4. No focus           — result passes through unmodified
        """
        await self._setup_tools()
        resolved_tools = self._resolve_tools(tools)

        result = []
        for t in resolved_tools:
            effective = _resolve_tool_focus(self.default_focus, t)

            if effective:
                # Warn once when the only active focus is the @tool-level default
                if effective == getattr(t, "focus", None) and not self.default_focus:
                    if t.name not in self._focus_default_warned:
                        logger.warning(
                            f"Tool '{t.name}' has a built-in focus default: {effective!r}. "
                            f"Applied automatically. To be explicit, set "
                            f"Agent(focus={{'{t.name}': '...'}})"
                        )
                        self._focus_default_warned.add(t.name)
                else:
                    logger.debug(f"Focus applied to '{t.name}': {effective!r}")
                result.append(FocusedTool(t, effective))
            else:
                result.append(t)

        return result

    def _build_final_result(
        self,
        final_text: str,
        tools_called: List[Dict],
        iterations: int,
        on_event: Optional[Callable],
    ) -> Dict[str, Any]:
        """Build final result dictionary with metadata."""
        from ..core.streaming import EventType

        token_stats = self.llm.get_token_stats()
        cost = token_stats.get("estimated_cost", 0.0)

        result = {
            "result": final_text,
            "tool_calls": tools_called,
            "iterations": iterations,
            "tokens": token_stats,
            "cost": cost,
        }

        # Emit completion event with all metadata
        self._emit_event(
            on_event,
            EventType.COMPLETE,
            final_result=final_text,
            iterations=iterations,
            token_usage=token_stats,
            cost=cost,
        )

        return result

    async def _execute_autonomous_with_retry(
        self,
        prompt: str,
        tools: Optional[List[Union[str, "LocalTool"]]],
        max_iterations: int,
        on_event: Optional[Callable],
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute autonomous tool calling with retry logic via ChatRuntime."""
        result = await self.runtime.run_with_retry(
            prompt=prompt,
            tools=tools,
            max_iterations=max_iterations,
            on_event=on_event,
            **kwargs,
        )
        return result.to_agent_result()

    async def _build_initial_conversation(
        self,
        prompt: str,
        initial_messages: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Build the opening conversation list for a run.

        Assembles: system prompt + extension context + prior history + current
        user message.
        """
        conversation = []
        system_parts = []

        if self.prompt:
            system_parts.append(self.prompt)

        extension_blocks = await self._render_extension_context_blocks(prompt)
        extension_context = self._format_extension_context(
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

        if system_parts:
            conversation.append(
                {"role": "system", "content": "\n\n".join(system_parts)}
            )

        if initial_messages:
            conversation.extend(initial_messages)

        conversation.append({"role": "user", "content": prompt})
        return conversation

    async def _render_extension_context(self, prompt: str) -> Optional[str]:
        """Render primary-model context from extension registry providers."""
        return self._format_extension_context(
            await self._render_extension_context_blocks(prompt)
        )

    async def _render_extension_context_blocks(self, prompt: str) -> List[ContextBlock]:
        """Render primary-model context blocks from extension providers."""
        return await self.render_context_blocks(prompt)

    @property
    def context_providers(self) -> List[ContextProvider]:
        """Return context providers declared by attached registry plugins."""
        return list(self.extension_registry.context_providers)

    def get_context_provider(
        self, provider_id: str, *, owner: Optional[str] = None
    ) -> ContextProvider:
        """Return one context provider declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_context_provider(provider_id, owner=owner)

    async def render_context_blocks(
        self,
        prompt: str,
        *,
        audience: ContextAudience = ContextAudience.PRIMARY_MODEL,
        token_budget: int = 2000,
    ) -> List[ContextBlock]:
        """Render context blocks for a target runtime audience."""
        await self._setup_extension_plugins()
        audience = ContextAudience(audience)

        blocks: List[ContextBlock] = []
        for provider in self.extension_registry.context_providers:
            if audience not in provider.audiences:
                continue
            try:
                block = await provider.render(
                    {
                        "prompt": prompt,
                        "runtime_id": self.agent_id,
                        "agent_id": self.agent_id,
                    },
                    audience,
                    token_budget,
                )
            except Exception as e:
                provider_name = getattr(provider, "id", provider.__class__.__name__)
                logger.warning(
                    "context provider '%s' failed for agent '%s': %s",
                    provider_name,
                    self.name,
                    e,
                )
                continue
            if block is not None and block.content:
                blocks.append(block)

        return blocks

    def _format_extension_context(self, blocks: List[ContextBlock]) -> Optional[str]:
        """Format rendered extension context blocks for the generic Agent loop."""
        if not blocks:
            return None

        rendered_blocks = [
            f"### {block.id}\n{block.content}"
            for block in sorted(blocks, key=lambda item: item.priority, reverse=True)
        ]
        return "## Runtime Context\n" + "\n\n".join(rendered_blocks)

    async def _execute_autonomous(
        self,
        prompt: str,
        tools: Optional[List[Union[str, "LocalTool"]]],
        max_iterations: int,
        on_event: Optional[Callable],
        initial_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delegate generic chat execution to ChatRuntime."""
        result = await self.runtime.run(
            prompt=prompt,
            tools=tools,
            max_iterations=max_iterations,
            on_event=on_event,
            initial_messages=initial_messages,
            **kwargs,
        )
        return result.to_agent_result()

    def _frame_payload(self, data: Any, tag: str) -> str:
        """Wrap data in an XML-like tag for safe LLM framing (prevents prompt injection)."""
        if isinstance(data, (dict, list)):
            return f"\n\n<{tag}>{json.dumps(data, default=_json_serializer)}</{tag}>"
        elif data is not None:
            return f"\n\n<{tag}>{str(data)[:4000]}</{tag}>"
        return ""

    async def on_webhook(
        self, payload: Dict[str, Any], webhook_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle webhook trigger from external service. Called automatically by webhook system.

        Note: `instructions` in webhook_config must be developer-controlled configuration,
        not end-user-supplied content, as it is passed directly to the LLM.
        The payload is framed as structured data (not instructions) to prevent prompt injection.
        """
        instructions = str(webhook_config.get("instructions", "Process webhook data"))[
            :2000
        ]

        # Inject payload as structured content so the agent can act on it
        instructions += self._frame_payload(payload, "webhook_payload")

        result = await self.run(instructions, detailed=True)

        result["webhook_metadata"] = {
            "webhook_id": webhook_config.get("webhook_id"),
            "webhook_slug": webhook_config.get("webhook_slug"),
            "entry_point": "on_webhook",
        }

        return result

    async def on_schedule(self, schedule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scheduled task execution (cron jobs). Called automatically by scheduler."""
        task = schedule_config.get("task", "Execute scheduled task")

        result = await self.run(task, detailed=True)

        result["schedule_metadata"] = {
            "schedule_id": schedule_config.get("schedule_id"),
            "cron": schedule_config.get("cron"),
            "entry_point": "on_schedule",
        }

        return result

    # User customization methods

    def _attached_plugin_sources(self) -> List[Any]:
        """Return attached plugins with registry-owned plugins first."""
        return [
            *(
                self.extension_registry.get_plugin(plugin_id)
                for plugin_id in self.extension_registry.plugin_ids
            ),
            *self._local_tool_sources,
        ]

    @property
    def attached_plugins(self) -> List[Any]:
        """Return attached plugin objects using registry-first ownership order."""
        return self._attached_plugin_sources()

    @property
    def attached_plugin_ids(self) -> List[str]:
        """Return stable IDs for manifest plugins and class names for local tool sources."""
        plugin_ids: List[str] = []
        for source in self._attached_plugin_sources():
            manifest = getattr(source, "manifest", None)
            if isinstance(manifest, dict):
                plugin_id = manifest.get("id")
            else:
                plugin_id = getattr(manifest, "id", None)
            plugin_ids.append(plugin_id or source.__class__.__name__)
        return plugin_ids

    def get_attached_plugin(self, identifier: Union[str, type]) -> Optional[Any]:
        """Return one attached plugin by stable ID, class name, or type."""
        for source in self._attached_plugin_sources():
            if isinstance(identifier, type):
                if isinstance(source, identifier):
                    return source
                continue

            manifest = getattr(source, "manifest", None)
            plugin_id = (
                manifest.get("id")
                if isinstance(manifest, dict)
                else getattr(manifest, "id", None)
            )
            if identifier in {plugin_id, source.__class__.__name__}:
                return source
        return None

    @property
    def plugin_manifests(self) -> List[PluginManifest]:
        """Return manifests declared by attached registry plugins."""
        return list(self.extension_registry.manifests)

    def get_plugin_manifest(self, plugin_id: str) -> Optional[PluginManifest]:
        """Return one attached registry plugin manifest by stable plugin ID."""
        return next(
            (
                manifest
                for manifest in self.plugin_manifests
                if manifest.id == plugin_id
            ),
            None,
        )

    @property
    def capabilities(self) -> List[Capability]:
        """Return runtime capabilities declared by attached registry plugins."""
        return list(self.extension_registry.capabilities)

    def get_capability(
        self, capability_id: str, *, owner: Optional[str] = None
    ) -> Capability:
        """Return one runtime capability, optionally disambiguated by owner."""
        return self.extension_registry.get_capability(capability_id, owner=owner)

    def find_capabilities(
        self,
        *,
        domain: Optional[str] = None,
        operation_type: Optional[str] = None,
    ) -> List[Capability]:
        """Find declared runtime capabilities by domain or operation type."""
        return list(
            self.extension_registry.find_capabilities(
                domain=domain,
                operation_type=operation_type,
            )
        )

    @property
    def tool_views(self) -> List[ToolView]:
        """Return model-facing tool view declarations from registry plugins."""
        return list(self.extension_registry.tool_views)

    def get_tool_view(self, name: str) -> Optional[ToolView]:
        """Return one model-facing tool view declaration by tool name."""
        return next((view for view in self.tool_views if view.name == name), None)

    def get_tool_view_owner(self, name: str) -> str:
        """Return the stable plugin ID that owns a registry tool view."""
        return self.extension_registry.get_tool_view_owner(name)

    @property
    def executors(self) -> List[Executor]:
        """Return executors declared by attached registry plugins."""
        return list(self.extension_registry.executors)

    def get_executor(self, executor_id: str) -> Executor:
        """Return one executor declared by attached registry plugins."""
        return self.extension_registry.get_executor(executor_id)

    async def execute_capability(
        self,
        capability_id: str,
        arguments: Dict[str, Any],
        *,
        owner: Optional[str] = None,
        operation_type: Optional[str] = None,
        operation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        executor_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one registry capability through the shared runtime kernel."""
        await self.setup_extensions()

        capability = self.get_capability(capability_id, owner=owner)
        selected_operation_type = (
            operation_type
            or (
                sorted(capability.operation_types)[0]
                if capability.operation_types
                else None
            )
            or "capability.execute"
        )
        execution_metadata = {
            "capability_id": capability.id,
            **(metadata or {}),
        }
        context = {
            "agent_id": self.agent_id,
            "runtime_id": self.agent_id,
            "capability": capability.to_dict(),
        }
        context.update(executor_context or {})

        if operation_id is None:
            operation = await self.runtime_kernel.create_operation(
                operation_type=selected_operation_type,
                request=arguments,
                required_evidence=capability.output_evidence,
                metadata=execution_metadata,
            )
        else:
            operation = await self.runtime_store.load_operation(operation_id)
            if operation is None:
                operation = await self.runtime_kernel.create_operation(
                    operation_id=operation_id,
                    operation_type=selected_operation_type,
                    request=arguments,
                    required_evidence=capability.output_evidence,
                    metadata=execution_metadata,
                )
        task = await self.runtime_kernel.plan_task(
            task_id=task_id,
            operation_id=operation.id,
            capability_id=capability.id,
            owner=capability.owner,
            input=arguments,
            metadata=execution_metadata,
        )
        result = await self.runtime_kernel.execute_task(task.id, context=context)
        return {
            "capability_id": capability.id,
            "evidence": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in result.evidence
            ],
        }

    @property
    def policies(self) -> List[Policy]:
        """Return policies declared by attached registry plugins."""
        return list(self.extension_registry.policies)

    def get_policy(self, policy_id: str, *, owner: Optional[str] = None) -> Policy:
        """Return one policy declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_policy(policy_id, owner=owner)

    @property
    def evidence_schemas(self) -> List[EvidenceSchema]:
        """Return evidence schemas declared by attached registry plugins."""
        return list(self.extension_registry.evidence_schemas)

    def get_evidence_schema(
        self, kind: str, *, owner: Optional[str] = None
    ) -> EvidenceSchema:
        """Return one evidence schema declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_evidence_schema(kind, owner=owner)

    @property
    def workers(self) -> List[Worker]:
        """Return workers declared by attached registry plugins."""
        return list(self.extension_registry.workers)

    def get_worker(self, worker_id: str, *, owner: Optional[str] = None) -> Worker:
        """Return one worker declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_worker(worker_id, owner=owner)

    @property
    def extension_diagnostics(self) -> List[RegistryDiagnostic]:
        """Return diagnostics for registry-owned extension declarations."""
        return list(self.extension_registry.diagnostics)

    def add_plugin(self, plugin: Any):
        """
        Add a plugin or local LocalTool to the agent.

        Manifest-bearing plugins are registered with the extension registry so
        runtime semantics are keyed by stable plugin IDs. Non-manifest sources
        are accepted only when they are local LocalTool projection objects.
        """
        has_manifest = getattr(plugin, "manifest", None) is not None
        if has_manifest:
            self.extension_registry.register(plugin)
        else:
            self._local_tool_sources.append(plugin)
            self._register_tool_source(plugin)
        logger.debug(f"Added plugin: {plugin.__class__.__name__}")

    async def _invoke_watch_handler_through_runtime(
        self,
        handler: Callable,
        event: Any,
        config: Any,
    ) -> None:
        """Execute a watch handler as a runtime monitor action task."""
        plugin_id, capability_id = self._ensure_watch_handler_capability(
            handler,
            config,
        )
        payload = _watch_event_payload(event)
        monitor = MonitorRuntime(kernel=self.runtime_kernel)
        await monitor.tick(
            MonitorSpec(
                id=f"watch.{_runtime_id_segment(config.name)}",
                name=config.name,
                trigger={"truthy": True},
                action_capability_id=capability_id,
                action_input={"event": payload},
                metadata={
                    "watch_name": config.name,
                    "watch_owner": plugin_id,
                    "source_type": payload.get("source_type"),
                    "resolved": payload.get("resolved"),
                },
            ),
            value=payload,
            execute_actions=True,
            raise_action_errors=True,
            context={
                "agent_id": self.agent_id,
                "watch_event": event,
                "watch_owner": plugin_id,
            },
        )

    def _ensure_watch_handler_capability(
        self,
        handler: Callable,
        config: Any,
    ) -> tuple[str, str]:
        key = config.name
        existing = self._watch_action_plugins.get(key)
        if existing is not None:
            return existing
        suffix = _runtime_id_segment(config.name)
        plugin_id = f"watch_{suffix}_{len(self._watch_action_plugins) + 1}"
        capability_id = f"agent.local.watch.{suffix}"
        executor = LocalWatchExecutor(
            executor_id=f"{plugin_id}.executor",
            capability_id=capability_id,
            handler=handler,
            config=config,
        )
        self.extension_registry.register(
            LocalWatchPlugin(
                plugin_id=plugin_id,
                capability_id=capability_id,
                executor=executor,
                config=config,
            )
        )
        self._watch_action_plugins[key] = (plugin_id, capability_id)
        return plugin_id, capability_id

    def add_skill(self, skill: "BaseSkill"):
        """Add a skill to the agent.

        Capability requirements declared in ``skill.requires_capabilities()``
        are resolved against already-registered extension capabilities. Add
        capability-providing plugins before skills that need them.
        """
        requirements = skill.requires_capabilities()
        if requirements:
            capabilities_by_id: dict[str, list] = {}
            for capability in self.extension_registry.capabilities:
                capabilities_by_id.setdefault(capability.id, []).append(capability)
            unmet = [
                capability_id
                for capability_id in requirements
                if capability_id not in capabilities_by_id
            ]
            if unmet:
                raise SkillError(
                    f"Skill '{skill.name}' requires capabilities not yet available: "
                    f"{', '.join(unmet)}. Add capability-providing plugin(s) before "
                    f"adding this skill.",
                    plugin_name=skill.name,
                )
            skill._resolved_capabilities = {
                capability_id: tuple(capabilities_by_id[capability_id])
                for capability_id in requirements
            }

        self.add_plugin(skill)

    @property
    def skills(self) -> List["BaseSkill"]:
        """Return all attached skills."""
        return [
            source
            for source in self._attached_plugin_sources()
            if isinstance(source, BaseSkill)
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name with arguments."""
        await self._setup_tools()
        view = self.get_tool_view(name)
        if view is not None:
            owner = self.extension_registry.get_tool_view_owner(name)
            capability = self.extension_registry.get_capability(
                view.capability_id,
                owner=owner,
            )
            operation_type = (
                sorted(capability.operation_types)[0]
                if capability.operation_types
                else "tool.call"
            )
            return await self.execute_capability(
                view.capability_id,
                arguments,
                owner=owner,
                operation_type=operation_type,
                metadata={"tool_view": view.name},
                executor_context={"tool_view": view.to_dict(), "tool_owner": owner},
            )

        tool = self.local_tool_catalog.get(name)
        if tool is None:
            raise RuntimeError(f"Tool '{name}' not found in registry")
        spec = self.runtime._register_local_tool(tool)
        execution = await self.runtime_kernel.execute_capability(
            spec.capability_id,
            owner=spec.owner,
            operation_type="chat.tool_call",
            input=arguments,
            context={
                "tool_view": spec.tool_view.to_dict(),
                "tool_owner": spec.owner,
            },
        )
        return self.runtime._render_evidence_for_model(tuple(execution.evidence))

    @property
    def available_tools(self) -> List[LocalTool]:
        """Get model-visible local tools plus registry ToolView projections."""
        projected = []
        for view in self.extension_registry.tool_views:
            if not view.model_visible:
                continue
            owner = self.extension_registry.get_tool_view_owner(view.name)
            capability = self.extension_registry.get_capability(
                view.capability_id,
                owner=owner,
            )
            projected.append(
                LocalTool(
                    name=view.name,
                    description=view.description,
                    parameters=view.parameters,
                    handler=lambda arguments, tool_name=view.name: self.call_tool(
                        tool_name,
                        arguments,
                    ),
                    source="plugin",
                    plugin_name=owner,
                    capability_ids=(view.capability_id,),
                    output_evidence=tuple(capability.output_evidence),
                    timeout_seconds=capability.timeout_seconds,
                    retry_safe=capability.retry_safe,
                    replay_safe=capability.replay_safe,
                    idempotent=capability.idempotent,
                    side_effecting=capability.side_effecting,
                    metadata=dict(view.metadata),
                )
            )
        return [*projected, *self.local_tool_catalog.tools.copy()]

    @property
    def tools(self) -> List[LocalTool]:
        """Return model-visible tools using the registry-backed projection."""
        return self.available_tools

    @property
    def tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return [tool.name for tool in self.available_tools]

    async def __aenter__(self):
        """Support ``async with agent:`` for automatic lifecycle management."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Call stop() on exit to flush plugins and release resources."""
        try:
            await self.stop()
        except Exception as e:
            if exc_type is None:
                raise
            logger.error("Error during agent stop: %s", e, exc_info=True)
        return False

    async def stop(self) -> None:
        """Stop agent and clean up all resources including MCP connections."""
        # Clean up MCP connections first
        if self.mcp_registry:
            try:
                await self.mcp_registry.disconnect_all()
                logger.info(f"Cleaned up MCP connections for agent {self.name}")
            except Exception as e:
                logger.warning(f"Error cleaning up MCP connections: {e}")

        await self.teardown_extensions()

        # Call parent stop for standard cleanup
        await super().stop()

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics from automatic tracing."""
        if not self.llm or not hasattr(self.llm, "get_token_stats"):
            # Fallback for agents without LLM or tracing
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
            }

        return self.llm.get_token_stats()

    @property
    def health(self) -> Dict[str, Any]:
        """Enhanced health information for Agent."""
        base_health = super().health

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
                "relay": {"enabled": self.relay is not None, "channel": self.relay},
                "llm": {
                    "available": self.llm is not None,
                    "provider": (
                        self.llm.provider_name
                        if self.llm and hasattr(self.llm, "provider_name")
                        else None
                    ),
                },
            }
        )

        return base_health
