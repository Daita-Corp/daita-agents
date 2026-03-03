"""
Agent — The primary agent class for building AI-driven applications.

Provides autonomous tool-calling, multi-provider LLM support, plugin integration,
and zero-configuration tracing. Use Agent for most use cases; subclass BaseAgent
for full control over the execution loop.
"""
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID
from typing import Dict, Any, Optional, List, Union, Callable

from ..config.base import AgentConfig, AgentType
from ..core.interfaces import LLMProvider
from ..core.exceptions import AgentError
from ..core.tracing import TraceType
from .base import BaseAgent

logger = logging.getLogger(__name__)

# Import unified plugin access
from ..plugins import PluginAccess
from ..llm.factory import create_llm_provider
from ..config.settings import settings
from ..core.tools import AgentTool, ToolRegistry


class FocusedTool:
    """Wrapper that applies focus filtering to tool results before they reach the LLM."""

    def __init__(self, tool: AgentTool, focus_config):
        """Wrap tool with focus filtering."""
        self._tool = tool
        self._focus = focus_config

    async def handler(self, arguments: Dict[str, Any]) -> Any:
        """Execute tool handler and apply focus filtering to result."""
        # Execute original tool handler
        result = await self._tool.handler(arguments)

        # Apply focus to result (if applicable)
        if self._focus and result is not None:
            try:
                from ..core.focus import apply_focus
                from ..config.base import FocusConfig

                # Convert FocusConfig to format apply_focus expects
                focus_param = self._focus
                if isinstance(self._focus, FocusConfig):
                    # Convert FocusConfig to dict/str/list format
                    if self._focus.type == "column":
                        focus_param = self._focus.columns or []
                    elif self._focus.type == "jsonpath":
                        focus_param = self._focus.path
                    elif self._focus.type == "xpath":
                        focus_param = self._focus.path
                    elif self._focus.type == "css":
                        focus_param = self._focus.selector
                    elif self._focus.type == "regex":
                        focus_param = self._focus.pattern
                    else:
                        # For other types, convert to dict
                        focus_param = self._focus.dict()

                focused_result = apply_focus(result, focus_param)
                logger.debug(
                    f"Applied focus to {self.name} result: "
                    f"{type(result).__name__} -> {type(focused_result).__name__}"
                )
                return focused_result
            except Exception as e:
                logger.warning(f"Focus application failed for {self.name}: {e}")
                # Return original result if focus fails
                return result

        return result

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped tool."""
        return getattr(self._tool, name)

    def __repr__(self):
        return f"FocusedTool({self._tool.name}, focus={self._focus})"


@dataclass
class LLMResult:
    """Unified LLM response format for both streaming and non-streaming."""
    text: str
    tool_calls: List[Dict[str, Any]]

    @classmethod
    def from_stream(cls, thinking_text: str, tool_calls: List[Dict]) -> 'LLMResult':
        """Create LLMResult from streaming chunks."""
        return cls(text=thinking_text, tool_calls=tool_calls)

    @classmethod
    def from_response(cls, response: Any) -> 'LLMResult':
        """Create LLMResult from non-streaming response."""
        if isinstance(response, str):
            return cls(text=response, tool_calls=[])
        elif isinstance(response, dict):
            return cls(
                text=response.get('content', ''),
                tool_calls=response.get('tool_calls', [])
            )
        else:
            logger.warning(f"Unexpected response type: {type(response)}")
            return cls(text=str(response), tool_calls=[])


class Agent(BaseAgent):
    """DAITA's primary agent with autonomous tool-calling and LLM-driven task execution."""
    
    # Class-level defaults for smart constructor
    _default_llm_provider = "openai"
    _default_model = "gpt-4"
    _ALLOWED_DEFAULTS = frozenset({'llm_provider', 'model'})

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
            setattr(cls, f'_default_{key}', value)

    def __init__(
        self,
        name: Optional[str] = None,
        llm_provider: Optional[Union[str, LLMProvider]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        prompt: Optional[Union[str, Dict[str, str]]] = None,
        focus: Optional[Union[List[str], str, Dict[str, Any]]] = None,
        relay: Optional[str] = None,  # Relay channel name used by the workflow engine to route output
        mcp: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        display_reasoning: bool = False,
        **kwargs
    ):
        """Initialize Agent with smart constructor auto-configuration."""
        # Store LLM provider config for lazy initialization
        if isinstance(llm_provider, str) or llm_provider is None:
            # Defer LLM provider creation until first use
            self._llm_provider_name = llm_provider or self._default_llm_provider
            self._llm_model = model or self._default_model
            self._llm_api_key = api_key
            # Exclude known Agent-level kwargs that are not LLM provider parameters
            _non_llm_keys = {'tools', 'enable_reliability', 'max_concurrent_tasks', 'max_queue_size'}
            self._llm_kwargs = {k: v for k, v in kwargs.items() if k not in _non_llm_keys}
            self._llm = None
            self._llm_initialized = False
            llm_provider_to_pass = None
        else:
            # User provided an actual LLM provider instance
            self._llm_provider_name = None
            self._llm_model = None
            self._llm_api_key = None
            self._llm_kwargs = {}
            self._llm = llm_provider
            self._llm_initialized = True
            llm_provider_to_pass = llm_provider

        # Create default config if none provided
        if config is None:
            config = AgentConfig(
                name=name or "Agent",
                type=AgentType.STANDARD,
                **kwargs
            )

        # Initialize base agent (which handles automatic tracing)
        super().__init__(config, llm_provider_to_pass, agent_id, name)
        
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
        self.tool_sources = kwargs.get('tools', [])  # Plugins, AgentTool instances, or callables
        self._tools_setup = False

        # Tool call history tracking for operations metadata
        self._tool_call_history = []

        # MCP server integration
        self.mcp_registry = None
        self.mcp_tools = []
        if mcp is not None:
            # Normalize to list
            mcp_servers = [mcp] if isinstance(mcp, dict) else mcp
            self._mcp_server_configs = mcp_servers
            # MCP setup happens lazily on first use to avoid blocking init
        else:
            self._mcp_server_configs = []

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
        if self._llm is None and not self._llm_initialized:
            # Try to get API key
            api_key = self._llm_api_key or settings.get_llm_api_key(self._llm_provider_name)
            if api_key:
                self._llm = create_llm_provider(
                    provider=self._llm_provider_name,
                    model=self._llm_model,
                    api_key=api_key,
                    agent_id=self.agent_id,
                    **self._llm_kwargs  # Pass through kwargs (includes timeout, etc.)
                )
                # Set agent_id for automatic LLM tracing
                if self._llm:
                    self._llm.set_agent_id(self.agent_id)
            self._llm_initialized = True
        return self._llm

    @llm.setter
    def llm(self, value):
        """Allow setting LLM provider directly."""
        self._llm = value
        if value is not None:
            self._llm_initialized = True
            # Only set agent_id if it's already initialized (after super().__init__())
            if hasattr(self, 'agent_id'):
                value.set_agent_id(self.agent_id)

    def _setup_decision_display(self):
        """Setup minimal decision display for local development."""
        try:
            from ..display.console import create_console_decision_display
            from ..core.decision_tracing import register_agent_decision_stream

            # Create display
            self._decision_display = create_console_decision_display(
                agent_name=self.name,
                agent_id=self.agent_id
            )

            # Register with decision streaming system
            register_agent_decision_stream(
                agent_id=self.agent_id,
                callback=self._decision_display.handle_event
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

            logger.info(f"Setting up {len(self._mcp_server_configs)} MCP server(s) for {self.name}")

            # Create registry
            self.mcp_registry = MCPToolRegistry()

            # Connect to each server and register tools
            for server_config in self._mcp_server_configs:
                server = MCPServer(
                    command=server_config.get("command"),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    server_name=server_config.get("name")
                )

                # Add to registry (automatically connects and discovers tools)
                await self.mcp_registry.add_server(server)

            # Get all tools from registry
            self.mcp_tools = self.mcp_registry.get_all_tools()

            logger.info(f"MCP setup complete: {self.mcp_registry.tool_count} tools from {self.mcp_registry.server_count} server(s)")

        except ImportError:
            logger.error(
                "MCP SDK not installed. Install with: pip install mcp\n"
                "See: https://github.com/modelcontextprotocol/python-sdk"
            )
            raise

        except Exception as e:
            logger.error(f"Failed to setup MCP servers: {str(e)}")
            raise

    async def _setup_tools(self):
        """Discover and register tools from all sources. Called lazily on first process()."""
        if self._tools_setup:
            return  # Already setup

        # 1. Setup MCP tools first
        if self._mcp_server_configs and self.mcp_registry is None:
            await self._setup_mcp_tools()
            # Convert MCP tools to AgentTool format
            for mcp_tool in self.mcp_tools:
                agent_tool = AgentTool.from_mcp_tool(mcp_tool, self.mcp_registry)
                self.tool_registry.register(agent_tool)

        # 2. Register plugin tools
        for source in self.tool_sources:
            if isinstance(source, AgentTool):
                # Direct AgentTool registration
                self.tool_registry.register(source)
                logger.debug(f"Registered tool: {source.name}")

            elif hasattr(source, 'get_tools'):
                # Plugin with get_tools() method
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

        self._tools_setup = True
        logger.info(
            f"Agent {self.name} initialized with {self.tool_registry.tool_count} tools"
        )

    # ========================================================================
    # USER API - What developers call directly
    # ========================================================================

    async def run(
        self,
        prompt: str,
        tools: Optional[List[Union[str, AgentTool]]] = None,
        max_iterations: int = 5,
        on_event: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Execute instruction with autonomous tool calling, returns final answer string."""
        result = await self._run_traced(prompt, tools, max_iterations, on_event, **kwargs)
        return result['result']

    async def run_detailed(
        self,
        prompt: str,
        tools: Optional[List[Union[str, AgentTool]]] = None,
        max_iterations: int = 5,
        on_event: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Like run() but returns full execution details: result, tool_calls, iterations, tokens, cost, time."""
        return await self._run_traced(prompt, tools, max_iterations, on_event, **kwargs)

    async def _run_traced(
        self,
        prompt: str,
        tools: Optional[List[Union[str, AgentTool]]],
        max_iterations: int,
        on_event: Optional[Callable],
        **kwargs
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
            entry_point="run"  # Distinguishes from _process() calls
        ):
            # Execute with or without retry based on configuration
            if self.config.retry_enabled:
                result = await self._execute_autonomous_with_retry(
                    prompt=prompt,
                    tools=tools,
                    max_iterations=max_iterations,
                    on_event=on_event,
                    **kwargs
                )
            else:
                result = await self._execute_autonomous(
                    prompt=prompt,
                    tools=tools,
                    max_iterations=max_iterations,
                    on_event=on_event,
                    **kwargs
                )

            # Enrich result with metadata
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            result['agent_id'] = self.agent_id
            result['agent_name'] = self.name

            # Auto-log to memory plugins if present
            for source in self.tool_sources:
                if hasattr(source, '_auto_log'):
                    try:
                        await source._auto_log(
                            user_input=prompt,
                            agent_output=result.get('result', ''),
                            timestamp=datetime.now(),
                            source_metadata={
                                'workflow': result.get('workflow_metadata'),
                                'webhook': result.get('webhook_metadata'),
                                'schedule': result.get('schedule_metadata')
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Auto-logging failed: {e}")

            return result

    def _resolve_tools(self, tools: Optional[List[Union[str, AgentTool]]]) -> List[AgentTool]:
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
        self,
        tools: Optional[List[Union[str, 'AgentTool']]]
    ) -> List['AgentTool']:
        """Resolve tools and wrap with FocusedTool if focus configured."""
        # Ensure tools are set up from plugin sources
        await self._setup_tools()

        resolved_tools = self._resolve_tools(tools)

        if self.default_focus and resolved_tools:
            resolved_tools = [
                FocusedTool(tool, self.default_focus)
                for tool in resolved_tools
            ]
            logger.debug(
                f"Wrapped {len(resolved_tools)} tools with focus filter: "
                f"{self.default_focus}"
            )

        return resolved_tools

    async def _stream_llm_turn(
        self,
        conversation: List[Dict],
        tools: List['AgentTool'],
        on_event: Callable,
        **kwargs
    ) -> LLMResult:
        """Execute streaming LLM turn with event emission."""
        from ..core.streaming import EventType

        thinking_text = ""
        tool_calls = []

        async for chunk in await self.llm.generate(
            messages=conversation,
            tools=tools,
            stream=True,
            **kwargs
        ):
            if chunk.type == "text":
                thinking_text += chunk.content
                self._emit_event(on_event, EventType.THINKING, content=chunk.content)

            elif chunk.type == "tool_call_complete":
                tool_calls.append({
                    "id": chunk.tool_call_id,
                    "name": chunk.tool_name,
                    "arguments": chunk.tool_args
                })
                self._emit_event(
                    on_event,
                    EventType.TOOL_CALL,
                    tool_name=chunk.tool_name,
                    tool_args=chunk.tool_args
                )

        return LLMResult.from_stream(thinking_text, tool_calls)

    async def _nonstream_llm_turn(
        self,
        conversation: List[Dict],
        tools: List['AgentTool'],
        **kwargs
    ) -> LLMResult:
        """Execute non-streaming LLM turn."""
        response = await self.llm.generate(
            messages=conversation,
            tools=tools,
            stream=False,
            **kwargs
        )

        return LLMResult.from_response(response)

    async def _execute_and_track_tool(
        self,
        tool_call: Dict[str, Any],
        tools: List['AgentTool'],
        on_event: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute tool and emit result event."""
        from ..core.streaming import EventType

        # Track execution time
        start_time = time.time()

        # Execute via LLM provider (handles focus filtering via FocusedTool.handler)
        result = await self.llm._execute_tool_call(
            tool_call=tool_call,
            tools=tools
        )

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Track this tool call in history
        tool_call_record = {
            "name": tool_call["name"],
            "duration_ms": duration_ms,
            "input": tool_call.get("arguments"),
            "output": result
        }
        self._tool_call_history.append(tool_call_record)

        # Emit result event
        self._emit_event(
            on_event,
            EventType.TOOL_RESULT,
            tool_name=tool_call["name"],
            result=result
        )

        return {
            "tool": tool_call["name"],
            "arguments": tool_call["arguments"],
            "result": result
        }

    def _append_tool_messages(
        self,
        conversation: List[Dict],
        tool_calls: List[Dict],
        results: List[Any]
    ):
        """Add tool calls and results to conversation history."""
        def json_serializer(obj):
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
                return obj.decode('utf-8', errors='replace')
            elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                # Exclude private/internal attributes to avoid leaking credentials or state
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        # Add assistant message with tool calls
        conversation.append({
            "role": "assistant",
            "tool_calls": tool_calls
        })

        # Add tool result messages
        for tool_call, result in zip(tool_calls, results):
            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["name"],
                "content": json.dumps(result["result"], default=json_serializer)
            })

    def _build_final_result(
        self,
        final_text: str,
        tools_called: List[Dict],
        iterations: int,
        on_event: Optional[Callable]
    ) -> Dict[str, Any]:
        """Build final result dictionary with metadata."""
        from ..core.streaming import EventType

        token_stats = self.llm.get_token_stats()

        result = {
            "result": final_text,
            "tool_calls": tools_called,
            "iterations": iterations,
            "tokens": token_stats,
            "cost": token_stats.get("estimated_cost", 0.0)
        }

        # Emit completion event with all metadata
        self._emit_event(
            on_event,
            EventType.COMPLETE,
            final_result=final_text,
            iterations=iterations,
            token_usage=token_stats,
            cost=token_stats.get("estimated_cost", 0.0)
        )

        return result

    async def _execute_autonomous_with_retry(
        self,
        prompt: str,
        tools: Optional[List[Union[str, 'AgentTool']]],
        max_iterations: int,
        on_event: Optional[Callable],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute autonomous tool calling with retry logic via the shared scaffold."""
        async def execute(attempt: int, max_attempts: int) -> Dict[str, Any]:
            result = await self._execute_autonomous(
                prompt=prompt,
                tools=tools,
                max_iterations=max_iterations,
                on_event=on_event,
                **kwargs
            )
            if attempt > 1:
                result['retry_attempt'] = attempt
            return result

        return await self._retry_with_tracing(execute, "autonomous_retry_attempt")

    async def _execute_autonomous(
        self,
        prompt: str,
        tools: Optional[List[Union[str, 'AgentTool']]],
        max_iterations: int,
        on_event: Optional[Callable],
        **kwargs
    ) -> Dict[str, Any]:
        """Unified autonomous execution path for both streaming and non-streaming."""
        from ..core.streaming import EventType

        # Check if LLM provider is available
        if self.llm is None:
            provider_name = self._llm_provider_name or 'openai'
            raise AgentError(
                f"Cannot execute: No API key found for '{provider_name}'. "
                f"Set {provider_name.upper()}_API_KEY environment variable "
                f"or pass api_key parameter to Agent."
            )

        # Prepare tools with focus wrapping
        resolved_tools = await self._prepare_tools_with_focus(tools)

        # Reset tool call history for this execution
        self._tool_call_history = []

        # Build conversation, injecting system prompt and memory context
        conversation = []

        # Collect system content: agent's configured prompt + memory context
        system_parts = []
        if self.prompt:
            system_parts.append(self.prompt)

        # Auto-inject relevant memories from any memory plugins
        for source in self.tool_sources:
            if hasattr(source, 'on_before_run'):
                try:
                    context = await source.on_before_run(prompt)
                    if context:
                        system_parts.append(context)
                except Exception:
                    pass

        if system_parts:
            conversation.append({"role": "system", "content": "\n\n".join(system_parts)})

        conversation.append({"role": "user", "content": prompt})
        tools_called = []

        # Autonomous tool calling loop
        for iteration in range(max_iterations):
            # Emit iteration event
            self._emit_event(
                on_event,
                EventType.ITERATION,
                iteration=iteration + 1,
                max_iterations=max_iterations
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

                # Add to conversation and continue loop
                self._append_tool_messages(
                    conversation, llm_result.tool_calls, results
                )
                continue

            # Final answer received
            return self._build_final_result(
                llm_result.text,
                tools_called,
                iteration + 1,
                on_event
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
        **kwargs
    ) -> Dict[str, Any]:
        """INTERNAL: Process task with data and context. Used by framework for workflow/lambda/system integration."""
        # Convert task/data to prompt
        if data is not None:
            prompt = f"{task}: {data}"
        else:
            prompt = task

        # Use run_detailed as the core execution
        result = await self.run_detailed(
            prompt=prompt,
            **kwargs
        )

        # Merge context if provided (for internal tracking)
        if context:
            result['context'] = {**result.get('context', {}), **context}

        # Add legacy fields for backward compatibility with internal systems
        result['task'] = task
        result['status'] = 'success' if 'result' in result else 'error'

        return result

    # ========================================================================
    # SYSTEM INTEGRATION API - What infrastructure calls
    # ========================================================================

    async def receive_message(
        self,
        data: Any,
        source_agent: str,
        channel: str,
        workflow_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle workflow relay message from another agent. Called automatically by workflow system."""
        prompt = "A message has arrived from the workflow system. Process the input data below."

        # Frame data as content, not as instructions, to prevent prompt injection
        if isinstance(data, dict):
            prompt += f"\n\n<input_data>{json.dumps(data)}</input_data>"
        elif isinstance(data, list):
            prompt += f"\n\n<input_data>{json.dumps(data)}</input_data>"
        elif data is not None:
            prompt += f"\n\n<input_data>{str(data)[:4000]}</input_data>"

        result = await self.run_detailed(prompt)

        # Add workflow metadata to result
        result['workflow_metadata'] = {
            'source_agent': source_agent,
            'channel': channel,
            'workflow': workflow_name,
            'entry_point': 'receive_message'
        }

        return result

    async def on_webhook(
        self,
        payload: Dict[str, Any],
        webhook_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle webhook trigger from external service. Called automatically by webhook system.

        Note: `instructions` in webhook_config must be developer-controlled configuration,
        not end-user-supplied content, as it is passed directly to the LLM.
        """
        instructions = str(webhook_config.get('instructions', 'Process webhook data'))[:2000]

        result = await self.run_detailed(instructions)

        result['webhook_metadata'] = {
            'webhook_id': webhook_config.get('webhook_id'),
            'webhook_slug': webhook_config.get('webhook_slug'),
            'entry_point': 'on_webhook'
        }

        return result

    async def on_schedule(
        self,
        schedule_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle scheduled task execution (cron jobs). Called automatically by scheduler."""
        task = schedule_config.get('task', 'Execute scheduled task')

        result = await self.run_detailed(task)

        result['schedule_metadata'] = {
            'schedule_id': schedule_config.get('schedule_id'),
            'cron': schedule_config.get('cron'),
            'entry_point': 'on_schedule'
        }

        return result

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Manually call an MCP tool by name with arguments."""
        if not self.mcp_registry:
            raise RuntimeError("No MCP servers configured. Add mcp parameter to Agent.")

        return await self.mcp_registry.call_tool(tool_name, arguments)

    # User customization methods

    def add_plugin(self, plugin: Any):
        """
        Add a plugin to agent's tool sources.

        Automatically initializes plugin with agent context if plugin has initialize() method.
        Tools registered on next setup.
        """
        # Auto-inject agent context into plugin if it supports initialization
        if hasattr(plugin, 'initialize') and callable(plugin.initialize):
            plugin.initialize(agent_id=self.agent_id)

        self.tool_sources.append(plugin)
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

    async def stop(self) -> None:
        """Stop agent and clean up all resources including MCP connections."""
        # Trigger auto-curation in memory plugins
        for source in self.tool_sources:
            if hasattr(source, 'on_agent_stop'):
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
        if not self.llm or not hasattr(self.llm, 'get_token_stats'):
            # Fallback for agents without LLM or tracing
            return {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'requests': 0
            }

        return self.llm.get_token_stats()
    
    @property
    def health(self) -> Dict[str, Any]:
        """Enhanced health information for Agent."""
        base_health = super().health

        # Add Agent-specific health info
        base_health.update({
            'tools': {
                'count': self.tool_registry.tool_count,
                'setup': self._tools_setup,
                'names': self.tool_registry.tool_names if self._tools_setup else []
            },
            'relay': {
                'enabled': self.relay is not None,
                'channel': self.relay
            },
            'llm': {
                'available': self.llm is not None,
                'provider': self.llm.provider_name if self.llm and hasattr(self.llm, 'provider_name') else None
            }
        })

        return base_health