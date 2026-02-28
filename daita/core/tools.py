"""
Universal tool abstraction for Daita agents.

Tools are LLM-callable functions that can come from:
- Plugins (database queries, S3 operations, API calls)
- MCP servers (external tools via Model Context Protocol)
- Custom functions (user-defined Python functions)

This abstraction is provider-agnostic and supports both prompt-based
and native function calling (OpenAI, Anthropic, etc).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Awaitable, Union, Literal, get_origin, get_args, get_type_hints
import asyncio
import inspect
import logging
import re

logger = logging.getLogger(__name__)


def _parse_docstring_params(func: Callable) -> Dict[str, str]:
    """
    Extract parameter descriptions from Google/NumPy style docstrings.

    Supports formats:
        param: description
        param (type): description
        param : description

    Args:
        func: Function to extract docstring from

    Returns:
        Dict mapping parameter names to their descriptions
    """
    docstring = func.__doc__ or ""

    # Find Args: section
    args_match = re.search(
        r'Args?:(.*?)(?=Returns?:|Raises?:|Example:|Notes?:|$)',
        docstring,
        re.DOTALL | re.IGNORECASE
    )

    if not args_match:
        return {}

    descriptions = {}
    args_section = args_match.group(1)

    # Match parameter entries: "    param_name: description" or "    param_name (type): description"
    for match in re.finditer(
        r'^\s+(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=^\s+\w+\s*(?:\([^)]+\))?\s*:|$)',
        args_section,
        re.MULTILINE | re.DOTALL
    ):
        param_name = match.group(1)
        description = ' '.join(match.group(2).split())  # Clean up whitespace
        descriptions[param_name] = description

    return descriptions


def _type_hint_to_json_schema(hint: Any) -> Dict[str, Any]:
    """
    Convert Python type hints to JSON Schema.

    Supports:
        - Basic types: int, float, str, bool
        - Optional[T]: Makes field not required, uses T's schema
        - Literal["a", "b"]: Enum constraint
        - List[T]: Array with item type
        - Dict[K, V]: Object type
        - Union[A, B]: anyOf schema

    Args:
        hint: Type hint to convert

    Returns:
        JSON Schema dict for the type
    """
    origin = get_origin(hint)
    args = get_args(hint)

    # Handle Union types (includes Optional)
    if origin is Union:
        # Check if it's Optional (Union with None)
        non_none_args = [arg for arg in args if arg is not type(None)]

        if len(non_none_args) == 1:
            # Optional[T] - just return T's schema, optionality handled by 'required'
            return _type_hint_to_json_schema(non_none_args[0])
        else:
            # Union[A, B, ...] - use anyOf
            return {
                "anyOf": [_type_hint_to_json_schema(arg) for arg in non_none_args]
            }

    # Handle Literal types
    if origin is Literal:
        # Determine type from first value
        first_val = args[0] if args else ""
        val_type = type(first_val).__name__
        schema_type = {
            "int": "integer",
            "float": "number",
            "str": "string",
            "bool": "boolean"
        }.get(val_type, "string")

        return {
            "type": schema_type,
            "enum": list(args)
        }

    # Handle List types
    if origin is list or origin is List:
        schema = {"type": "array"}
        if args:
            schema["items"] = _type_hint_to_json_schema(args[0])
        return schema

    # Handle Dict types
    if origin is dict or origin is Dict:
        return {"type": "object"}

    # Handle basic Python types
    basic_type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    if hint in basic_type_map:
        return {"type": basic_type_map[hint]}

    # Default fallback
    return {"type": "string"}


def _extract_parameters_from_function(func: Callable) -> Dict[str, Any]:
    """
    Extract JSON Schema from function signature with type hints and docstring.

    Supports:
        - Type hints: int, str, float, bool, Optional, Literal, List, Dict, Union
        - Docstring parameter descriptions (Google/NumPy style)
        - Default values (automatically marks as optional)

    Args:
        func: Python function to analyze

    Returns:
        JSON Schema dict with properties and required fields
    """
    sig = inspect.signature(func)
    properties = {}
    required = []

    # Get type hints (resolves string annotations)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        # Fallback if get_type_hints fails
        type_hints = {}

    # Get parameter descriptions from docstring
    param_descriptions = _parse_docstring_params(func)

    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue

        # Get type hint for this parameter
        type_hint = type_hints.get(param_name, param.annotation)

        # Check if parameter is Optional (has None in Union)
        is_optional = False
        if type_hint != inspect.Parameter.empty:
            origin = get_origin(type_hint)
            args = get_args(type_hint)
            if origin is Union and type(None) in args:
                is_optional = True

        # Convert type hint to JSON Schema
        if type_hint != inspect.Parameter.empty:
            schema = _type_hint_to_json_schema(type_hint)
        else:
            schema = {"type": "string"}

        # Get description from docstring or use generic
        description = param_descriptions.get(param_name, f"Parameter: {param_name}")
        schema["description"] = description

        properties[param_name] = schema

        # Mark as required if no default value and not Optional
        has_default = param.default != inspect.Parameter.empty
        if not has_default and not is_optional:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def _make_async_handler(func: Callable) -> Callable[[Dict[str, Any]], Awaitable[Any]]:
    """
    Convert any function to async handler format.

    Args:
        func: Sync or async function to convert

    Returns:
        Async handler that accepts Dict[str, Any] and returns result
    """
    if asyncio.iscoroutinefunction(func):
        async def handler(args: Dict[str, Any]) -> Any:
            return await func(**args)
    else:
        async def handler(args: Dict[str, Any]) -> Any:
            return func(**args)
    return handler


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    category: Optional[str] = None,
    **kwargs
) -> Union['AgentTool', Callable]:
    """
    Convert a function into an AgentTool.
    Works as both decorator and function call.

    Automatically extracts parameter schema from type hints and docstring.

    Usage as decorator:
        @tool
        async def calculator(a: int, b: int) -> int:
            '''Add two numbers'''
            return a + b

    Usage with options:
        @tool(timeout_seconds=30, category="math")
        async def calculator(a: int, b: int) -> int:
            return a + b

    Usage as function:
        calc_tool = tool(calculator)
        calc_tool = tool(calculator, timeout_seconds=30)

    Args:
        func: Function to convert (when used as direct call)
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring first line)
        timeout_seconds: Execution timeout in seconds
        category: Tool category for organization
        **kwargs: Additional AgentTool fields

    Returns:
        AgentTool instance or decorator function
    """
    def create_tool(f: Callable) -> 'AgentTool':
        tool_name = name or f.__name__
        tool_description = description or (f.__doc__ or f"Execute {tool_name}").strip().split('\n')[0]
        tool_parameters = _extract_parameters_from_function(f)
        handler = _make_async_handler(f)

        return AgentTool(
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters,
            handler=handler,
            timeout_seconds=timeout_seconds,
            category=category,
            source="custom",
            **kwargs
        )

    # Used as @tool (no parentheses)
    if func is not None:
        return create_tool(func)

    # Used as @tool(...) (with arguments)
    return create_tool


@dataclass
class AgentTool:
    """
    Universal tool definition for agent-LLM integration.

    Represents any callable function that an agent can use, regardless of source.
    Designed to be compatible with industry standards (LangChain, AutoGen, etc).

    Example:
        ```python
        tool = AgentTool(
            name="search_database",
            description="Search for records in the database",
            parameters={
                "query": {
                    "type": "string",
                    "description": "SQL query to execute",
                    "required": True
                }
            },
            handler=async_search_function
        )
        ```
    """

    # Core fields (required)
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    handler: Callable[[Dict[str, Any]], Awaitable[Any]]  # async function

    # Optional metadata
    category: Optional[str] = None  # "database", "storage", "api", etc
    source: str = "custom"  # "plugin", "mcp", "custom"
    plugin_name: Optional[str] = None  # Which plugin provides this tool

    # Safety features
    timeout_seconds: Optional[int] = None  # Execution timeout

    def to_openai_function(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format.

        Returns:
            OpenAI function definition dict

        Reference:
            https://platform.openai.com/docs/guides/function-calling
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters  # Already in correct JSON Schema format
            }
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """
        Convert to Anthropic tool format.

        Returns:
            Anthropic tool definition dict

        Reference:
            https://docs.anthropic.com/claude/docs/tool-use
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters  # Already in correct JSON Schema format
        }

    def to_llm_function(self) -> Dict[str, Any]:
        """
        Generic LLM function format (works for most providers).

        For provider-specific formats, use to_openai_function() or to_anthropic_tool().
        This format is used for prompt-based tool calling.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters  # Already in correct JSON Schema format
        }

    def to_prompt_description(self) -> str:
        """
        Generate human-readable tool description for prompt injection.

        Used in prompt-based tool calling (non-native function calling).

        Returns:
            Formatted tool description string
        """
        params_desc = []
        properties = self.parameters.get("properties", {})
        required_params = self.parameters.get("required", [])

        for param_name, param_info in properties.items():
            required = " (required)" if param_name in required_params else " (optional)"
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            params_desc.append(
                f"  - {param_name} ({param_type}){required}: {param_desc}"
            )

        params_str = "\n".join(params_desc) if params_desc else "  (no parameters)"

        return f"{self.name}: {self.description}\nParameters:\n{params_str}"

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            arguments: Tool arguments matching the parameter schema

        Returns:
            Tool execution result

        Raises:
            RuntimeError: If tool execution fails or times out
        """
        if not callable(self.handler):
            raise RuntimeError(f"Tool '{self.name}' has non-callable handler")

        # Execute with timeout if specified
        if self.timeout_seconds:
            try:
                result = await asyncio.wait_for(
                    self.handler(arguments),
                    timeout=self.timeout_seconds
                )
                return result
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Tool '{self.name}' execution timed out after {self.timeout_seconds}s"
                )
        else:
            # No timeout
            return await self.handler(arguments)

    @classmethod
    def from_mcp_tool(cls, mcp_tool, mcp_registry) -> 'AgentTool':
        """
        Create AgentTool from an MCP tool.

        Args:
            mcp_tool: MCPTool instance from MCP plugin
            mcp_registry: MCPToolRegistry for routing calls

        Returns:
            AgentTool instance that wraps the MCP tool
        """
        # Create handler that routes to MCP registry
        async def mcp_handler(arguments: Dict[str, Any]) -> Any:
            return await mcp_registry.call_tool(mcp_tool.name, arguments)

        return cls(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=mcp_tool.input_schema.get("properties", {}),
            handler=mcp_handler,
            source="mcp",
            category="mcp"
        )


class ToolRegistry:
    """
    Registry for managing tools from multiple sources.

    Used internally by agents to aggregate tools from plugins, MCP servers,
    and custom functions.
    """

    def __init__(self):
        """Initialize empty tool registry"""
        self.tools: List[AgentTool] = []
        self._tool_map: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        """
        Register a tool.

        Args:
            tool: AgentTool to register
        """
        if tool.name in self._tool_map:
            logger.warning(
                f"Tool '{tool.name}' already registered. "
                f"Overwriting (old source: {self._tool_map[tool.name].source}, "
                f"new source: {tool.source})"
            )

        self.tools.append(tool)
        self._tool_map[tool.name] = tool

        logger.debug(f"Registered tool: {tool.name} (source: {tool.source})")

    def register_many(self, tools: List[AgentTool]) -> None:
        """
        Register multiple tools at once.

        Args:
            tools: List of AgentTool instances
        """
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[AgentTool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            AgentTool instance or None if not found
        """
        return self._tool_map.get(name)

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            RuntimeError: If tool not found or execution fails
        """
        tool = self.get(name)
        if not tool:
            available = [t.name for t in self.tools]
            raise RuntimeError(
                f"Tool '{name}' not found. Available tools: {', '.join(available)}"
            )

        return await tool.execute(arguments)

    @property
    def tool_count(self) -> int:
        """Total number of registered tools"""
        return len(self.tools)

    @property
    def tool_names(self) -> List[str]:
        """List of all tool names"""
        return list(self._tool_map.keys())
