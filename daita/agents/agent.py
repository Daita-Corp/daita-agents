"""
Agent — The primary agent class for building AI-driven applications.

Provides autonomous tool-calling, multi-provider LLM support, plugin integration,
and zero-configuration tracing. Use Agent for most use cases; subclass BaseAgent
for full control over the execution loop.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Union, Callable

if TYPE_CHECKING:
    from .conversation import ConversationHistory

import asyncio

from ..config.base import AgentConfig, AgentType, RetryPolicy
from ..core.interfaces import LLMProvider
from ..core.exceptions import AgentError
from ..core.tracing import TraceType
from .base import BaseAgent

logger = logging.getLogger(__name__)


async def _execute_tool_call(
    tool_call: Dict[str, Any], tools: List["AgentTool"]
) -> Any:
    """
    Execute a single tool call with timeout and error handling.

    Intentionally lives in the agent layer — tool execution is an agent concern.
    The LLM layer is responsible for generating tool call requests; the agent
    layer is responsible for acting on them.
    """
    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]

    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        result = await asyncio.wait_for(
            tool.handler(arguments), timeout=tool.timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        return {"error": f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s"}
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {str(e)}"}


def _make_error_fingerprint(tool_call: Dict[str, Any]) -> str:
    """Stable hash of (tool_name, arguments) used for loop detection."""
    args_hash = hashlib.md5(
        json.dumps(tool_call.get("arguments", {}), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{tool_call['name']}:{args_hash}"


def _json_serializer(obj):
    """
    Custom JSON serializer for types commonly returned by database plugins.

    Handles datetime, Decimal, UUID, and other non-JSON-native types that
    plugins might return from queries.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        # Exclude private/internal attributes to avoid leaking credentials or state
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Import unified plugin access
from ..plugins import PluginAccess
from ..plugins.base import LifecyclePlugin
from ..llm.factory import create_llm_provider
from ..config.settings import settings
from ..core.tools import AgentTool, ToolRegistry


class FocusedTool:
    """Wrapper that applies a Focus DSL expression to tool results before they reach the LLM."""

    def __init__(self, tool: AgentTool, focus: str):
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


@dataclass
class LLMResult:
    """Unified LLM response format for both streaming and non-streaming."""

    text: str
    tool_calls: List[Dict[str, Any]]

    @classmethod
    def from_response(cls, response: Any) -> "LLMResult":
        """Create LLMResult from non-streaming response."""
        if isinstance(response, str):
            return cls(text=response, tool_calls=[])
        elif isinstance(response, dict):
            return cls(
                text=response.get("content", ""),
                tool_calls=response.get("tool_calls", []),
            )
        else:
            logger.warning(f"Unexpected response type: {type(response)}")
            return cls(text=str(response), tool_calls=[])


def _resolve_tool_focus(
    agent_focus: Optional[Union[str, Dict[str, str]]],
    t: "AgentTool",
) -> Optional[str]:
    """Return the effective focus DSL for a single tool, applying precedence rules."""
    tool_default = getattr(t, "focus", None)
    if isinstance(agent_focus, dict):
        return agent_focus.get(t.name) or tool_default
    return agent_focus or tool_default


class Agent(BaseAgent):
    """DAITA's primary agent with autonomous tool-calling and LLM-driven task execution."""

    # Class-level defaults for smart constructor
    _default_llm_provider = "openai"
    _default_model = "gpt-4"
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
        """Create a fully configured Agent from a database connection string or plugin.

        Connects to the database, discovers the schema, generates a system prompt,
        and returns an Agent with query tools ready to use.

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
        from .db import from_db

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
        display_reasoning: bool = False,
        # Tool sources
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
        self.tool_registry = ToolRegistry()
        self.tool_sources = []
        self._tools_setup = False
        for source in tools or []:
            self.add_plugin(source)
        self._focus_default_warned: set = (
            set()
        )  # tracks @tool focus defaults already logged

        # Tool call history tracking for operations metadata
        self._tool_call_history = []

        # MCP server integration — setup happens lazily on first use
        self.mcp_registry = None
        self.mcp_tools = []
        self._mcp_server_configs = (
            ([mcp] if isinstance(mcp, dict) else mcp) if mcp is not None else []
        )

        # Plugin access for direct plugin usage
        self.plugins = PluginAccess()

        logger.debug(f"Agent {self.name} initialized")

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
        """Register a single tool source (AgentTool or plugin) into the tool registry."""
        if isinstance(source, AgentTool):
            self.tool_registry.register(source)
        elif hasattr(source, "get_tools"):
            plugin_tools = source.get_tools()
            if plugin_tools:
                self.tool_registry.register_many(plugin_tools)
                logger.info(
                    f"Registered {len(plugin_tools)} tools from "
                    f"{source.__class__.__name__}"
                )
        else:
            logger.warning(
                f"Invalid tool source: {source}. "
                f"Expected AgentTool or plugin with get_tools() method."
            )

    async def _setup_tools(self):
        """Set up MCP tools. Non-MCP tools are registered eagerly in add_plugin()."""
        if self._tools_setup:
            return  # Already setup

        # Setup MCP tools (requires async connection)
        if self._mcp_server_configs and self.mcp_registry is None:
            await self._setup_mcp_tools()
            for mcp_tool in self.mcp_tools:
                agent_tool = AgentTool.from_mcp_tool(mcp_tool, self.mcp_registry)
                self.tool_registry.register(agent_tool)

        self._tools_setup = True
        logger.info(
            f"Agent {self.name} ready with {self.tool_registry.tool_count} tools"
        )

    # ========================================================================
    # USER API - What developers call directly
    # ========================================================================

    async def run(
        self,
        prompt: str,
        tools: Optional[List[Union[str, AgentTool]]] = None,
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
        tools: Optional[List[Union[str, AgentTool]]] = None,
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
        tools: Optional[List[Union[str, AgentTool]]],
        max_iterations: int,
        timeout_seconds: Optional[int],
        on_event: Optional[Callable],
        history: Optional["ConversationHistory"] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Internal: Execute with automatic tracing and optional event streaming."""
        start_time = time.time()

        # Create agent-level trace span (automatic, invisible to users)
        async with self.trace_manager.span(
            operation_name="agent_run",
            trace_type=TraceType.AGENT_EXECUTION,
            agent_id=self.agent_id,
            agent_name=self.name,
            prompt=prompt[:200],  # Truncate for storage
            tools_requested=tools,
            max_iterations=max_iterations,
            entry_point="run",  # Distinguishes from _process() calls
        ):
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

            return result

    def _resolve_tools(
        self, tools: Optional[List[Union[str, AgentTool]]]
    ) -> List[AgentTool]:
        """Resolve tool names to AgentTool instances. If None, returns all registered tools."""
        if tools is None:
            # Use all registered tools
            return list(self.tool_registry.tools)

        tool_list = []
        for t in tools:
            if isinstance(t, str):
                # Tool name - look up in registry
                tool = self.tool_registry.get(t)
                if not tool:
                    raise ValueError(f"Tool '{t}' not found in registry")
                tool_list.append(tool)
            else:
                # Already an AgentTool instance
                tool_list.append(t)

        return tool_list

    def _emit_event(self, on_event: Optional[Callable], event_type, **kwargs):
        """Emit event only if callback provided. Zero overhead when None."""
        if on_event:
            from ..core.streaming import AgentEvent

            on_event(AgentEvent(type=event_type, **kwargs))

    async def _prepare_tools_with_focus(
        self, tools: Optional[List[Union[str, "AgentTool"]]]
    ) -> List["AgentTool"]:
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

    async def _stream_llm_turn(
        self,
        conversation: List[Dict],
        tools: List["AgentTool"],
        on_event: Callable,
        **kwargs,
    ) -> LLMResult:
        """Execute streaming LLM turn with event emission."""
        from ..core.streaming import EventType

        thinking_text = ""
        tool_calls = []

        async for chunk in await self.llm.generate(
            messages=conversation, tools=tools, stream=True, **kwargs
        ):
            if chunk.type == "text":
                thinking_text += chunk.content
                self._emit_event(on_event, EventType.THINKING, content=chunk.content)

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
                )

        return LLMResult(text=thinking_text, tool_calls=tool_calls)

    async def _nonstream_llm_turn(
        self, conversation: List[Dict], tools: List["AgentTool"], **kwargs
    ) -> LLMResult:
        """Execute non-streaming LLM turn."""
        return LLMResult.from_response(
            await self.llm.generate(
                messages=conversation, tools=tools, stream=False, **kwargs
            )
        )

    async def _execute_and_track_tool(
        self,
        tool_call: Dict[str, Any],
        tools: List["AgentTool"],
        on_event: Optional[Callable],
    ) -> Dict[str, Any]:
        """Execute tool and emit result event."""
        from ..core.streaming import EventType

        # Track execution time
        start_time = time.time()

        result = await _execute_tool_call(tool_call, tools)

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Track this tool call in history
        tool_call_record = {
            "name": tool_call["name"],
            "duration_ms": duration_ms,
            "input": tool_call.get("arguments"),
            "output": result,
        }
        self._tool_call_history.append(tool_call_record)

        # Emit result event
        self._emit_event(
            on_event, EventType.TOOL_RESULT, tool_name=tool_call["name"], result=result
        )

        return {
            "tool": tool_call["name"],
            "arguments": tool_call["arguments"],
            "result": result,
        }

    def _append_tool_messages(
        self, conversation: List[Dict], tool_calls: List[Dict], results: List[Any]
    ):
        """Add tool calls and results to conversation history."""
        # Add assistant message with tool calls
        conversation.append({"role": "assistant", "tool_calls": tool_calls})

        # Add tool result messages
        for tool_call, result in zip(tool_calls, results):
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "content": json.dumps(result["result"], default=_json_serializer),
                }
            )

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
        tools: Optional[List[Union[str, "AgentTool"]]],
        max_iterations: int,
        on_event: Optional[Callable],
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute autonomous tool calling with retry logic via the shared scaffold."""

        async def execute(attempt: int, max_attempts: int) -> Dict[str, Any]:
            result = await self._execute_autonomous(
                prompt=prompt,
                tools=tools,
                max_iterations=max_iterations,
                on_event=on_event,
                **kwargs,
            )
            if attempt > 1:
                result["retry_attempt"] = attempt
            return result

        return await self._retry_with_tracing(execute, "autonomous_retry_attempt")

    async def _build_initial_conversation(
        self,
        prompt: str,
        initial_messages: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Build the opening conversation list for a run.

        Assembles: system prompt + memory context (from on_before_run hooks)
        + prior history + current user message.
        """
        conversation = []
        system_parts = []

        if self.prompt:
            system_parts.append(self.prompt)

        for source in self.tool_sources:
            if isinstance(source, LifecyclePlugin):
                try:
                    context = await source.on_before_run(prompt)
                    if context:
                        system_parts.append(context)
                except Exception:
                    pass

        if system_parts:
            conversation.append(
                {"role": "system", "content": "\n\n".join(system_parts)}
            )

        if initial_messages:
            conversation.extend(initial_messages)

        conversation.append({"role": "user", "content": prompt})
        return conversation

    async def _execute_autonomous(
        self,
        prompt: str,
        tools: Optional[List[Union[str, "AgentTool"]]],
        max_iterations: int,
        on_event: Optional[Callable],
        initial_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unified autonomous execution path for both streaming and non-streaming."""
        from ..core.streaming import EventType

        # Check if LLM provider is available
        if self.llm is None:
            provider_name = self._llm_provider_name or "openai"
            raise AgentError(
                f"Cannot execute: No API key found for '{provider_name}'. "
                f"Set {provider_name.upper()}_API_KEY environment variable "
                f"or pass api_key parameter to Agent."
            )

        # Prepare tools with focus wrapping
        resolved_tools = await self._prepare_tools_with_focus(tools)

        # Reset tool call history for this execution
        self._tool_call_history = []

        conversation = await self._build_initial_conversation(prompt, initial_messages)
        tools_called = []

        # Tracks consecutive error count per (tool_name, args) fingerprint.
        # Reset to 0 on any successful call with that fingerprint.
        # Raised to AgentError when the same call fails 3 times in a row —
        # the agent has demonstrated it cannot self-correct from this error.
        _error_fingerprints: Dict[str, int] = {}

        # Autonomous tool calling loop
        for iteration in range(max_iterations):
            # Emit iteration event
            self._emit_event(
                on_event,
                EventType.ITERATION,
                iteration=iteration + 1,
                max_iterations=max_iterations,
            )

            # Get LLM response (streaming or non-streaming based on on_event)
            if on_event:
                llm_result = await self._stream_llm_turn(
                    conversation, resolved_tools, on_event, **kwargs
                )
            else:
                llm_result = await self._nonstream_llm_turn(
                    conversation, resolved_tools, **kwargs
                )

            # Check if LLM wants to call tools
            if llm_result.tool_calls:
                # Execute each tool
                results = []
                for tool_call in llm_result.tool_calls:
                    result = await self._execute_and_track_tool(
                        tool_call, resolved_tools, on_event
                    )
                    tools_called.append(result)
                    results.append(result)

                    # Loop detection: identical (tool, args) producing consecutive errors
                    raw = result.get("result", {})
                    fp = _make_error_fingerprint(tool_call)
                    if isinstance(raw, dict) and "error" in raw:
                        _error_fingerprints[fp] = _error_fingerprints.get(fp, 0) + 1
                        if _error_fingerprints[fp] >= 3:
                            raise AgentError(
                                f"Loop detected: '{tool_call['name']}' returned an error "
                                f"{_error_fingerprints[fp]} consecutive times with identical "
                                f"arguments. Last error: {raw['error']}"
                            )
                    else:
                        _error_fingerprints.pop(fp, None)

                # Add to conversation and continue loop
                self._append_tool_messages(conversation, llm_result.tool_calls, results)
                continue

            # Final answer received
            return self._build_final_result(
                llm_result.text, tools_called, iteration + 1, on_event
            )

        # Max iterations reached without final answer
        raise AgentError(
            f"Max iterations ({max_iterations}) reached without final answer"
        )

    # ========================================================================
    # INTERNAL - Backward compatibility for system integration
    # ========================================================================

    async def _process(
        self,
        task: str,
        data: Any = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """INTERNAL: Legacy entry point — delegates to receive_message.

        Prefer receive_message() for new integrations; this exists for
        backward compatibility with older workflow and lambda callers.
        """
        result = await self.receive_message(
            data=data if data is not None else task,
            source_agent="internal",
            channel="process",
            workflow_name=task,
            **kwargs,
        )

        if context:
            result["context"] = {**result.get("context", {}), **context}

        # Legacy fields expected by older internal callers
        result["task"] = task
        result["status"] = "success" if "result" in result else "error"

        return result

    # ========================================================================
    # SYSTEM INTEGRATION API - What infrastructure calls
    # ========================================================================

    def _frame_payload(self, data: Any, tag: str) -> str:
        """Wrap data in an XML-like tag for safe LLM framing (prevents prompt injection)."""
        if isinstance(data, (dict, list)):
            return f"\n\n<{tag}>{json.dumps(data, default=_json_serializer)}</{tag}>"
        elif data is not None:
            return f"\n\n<{tag}>{str(data)[:4000]}</{tag}>"
        return ""

    async def receive_message(
        self,
        data: Any,
        source_agent: str,
        channel: str,
        workflow_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle workflow relay message from another agent. Called automatically by workflow system."""
        prompt = "A message has arrived from the workflow system. Process the input data below."
        prompt += self._frame_payload(data, "input_data")

        result = await self.run(prompt, detailed=True, **kwargs)

        # Add workflow metadata to result
        result["workflow_metadata"] = {
            "source_agent": source_agent,
            "channel": channel,
            "workflow": workflow_name,
            "entry_point": "receive_message",
        }

        return result

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

    def add_plugin(self, plugin: Any):
        """
        Add a plugin to agent's tool sources and immediately register its tools.

        Automatically initializes plugin with agent context if plugin has initialize() method.
        """
        if hasattr(plugin, "initialize") and callable(plugin.initialize):
            plugin.initialize(agent_id=self.agent_id)

        self.tool_sources.append(plugin)
        self._register_tool_source(plugin)
        logger.debug(f"Added plugin: {plugin.__class__.__name__}")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name with arguments."""
        await self._setup_tools()
        return await self.tool_registry.execute(name, arguments)

    @property
    def available_tools(self) -> List[AgentTool]:
        """Get list of all available tools."""
        return self.tool_registry.tools.copy()

    @property
    def tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return self.tool_registry.tool_names

    async def __aenter__(self):
        """Support ``async with agent:`` for automatic lifecycle management."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Call stop() on exit to flush plugins and release resources."""
        await self.stop()
        return False

    async def stop(self) -> None:
        """Stop agent and clean up all resources including MCP connections."""
        # Trigger auto-curation in memory plugins
        for source in self.tool_sources:
            if isinstance(source, LifecyclePlugin):
                try:
                    await source.on_agent_stop()
                except Exception as e:
                    logger.warning(f"Error during on_agent_stop: {e}")

        # Clean up MCP connections first
        if self.mcp_registry:
            try:
                await self.mcp_registry.disconnect_all()
                logger.info(f"Cleaned up MCP connections for agent {self.name}")
            except Exception as e:
                logger.warning(f"Error cleaning up MCP connections: {e}")

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
                    "count": self.tool_registry.tool_count,
                    "setup": self._tools_setup,
                    "names": self.tool_registry.tool_names if self._tools_setup else [],
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
