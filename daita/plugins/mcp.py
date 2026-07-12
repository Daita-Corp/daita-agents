"""
MCP (Model Context Protocol) plugin for Daita Agents.

This plugin enables agents to connect to any MCP server and autonomously use
their tools via LLM function calling. Agents discover available tools and
decide which ones to use based on the task.

MCP integrations are owned by this plugin module. Agent-local MCP constructor
arguments were removed in the clean-break runtime architecture; MCP tool calls
should be exposed as registry declarations and executed through RuntimeKernel.

MCP Protocol:
    The Model Context Protocol (MCP) is Anthropic's open standard for connecting
    AI systems to external data sources and tools. This plugin provides native
    MCP client support for Daita agents.
"""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    ToolView,
)

from .base import PluginContext
from .manifest import PluginKind, PluginManifest

if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents a tool exposed by an MCP server"""

    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_llm_function(self) -> Dict[str, Any]:
        """
        Convert MCP tool schema to LLM function calling format.

        Returns OpenAI/Anthropic compatible function definition.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


class MCPServer:
    """
    MCP Server connection manager.

    Manages connection to a single MCP server via stdio transport,
    discovers available tools, and executes tool calls.
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        server_name: Optional[str] = None,
    ):
        """
        Initialize MCP server configuration.

        Args:
            command: Command to run MCP server (e.g., "uvx", "python", "npx")
            args: Arguments for the command (e.g., ["mcp-server-filesystem", "/data"])
            env: Environment variables for the server process
            server_name: Optional name for this server (for logging/debugging)
        """
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.server_name = server_name or f"mcp_{command}"

        # Connection state
        self._session: ClientSession | None = None
        self._read = None
        self._write = None
        self._tools: List[MCPTool] = []
        self._connected = False
        self._stdio_context_task: asyncio.Task[None] | None = (
            None  # Background task keeping context alive
        )
        self._session_lock = (
            asyncio.Lock()
        )  # Protects session access from concurrent calls
        self.manifest = PluginManifest(
            id=_plugin_id("mcp_server", self.server_name),
            display_name=f"MCP Server {self.server_name}",
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
            domains=frozenset({"chat"}),
        )

    async def setup(self, context: PluginContext) -> None:
        """Connect and register discovered MCP tools as runtime declarations."""
        await self.connect()
        registry = context.services.get("extension_registry")
        if registry is None:
            return
        declarations = MCPToolDeclarations(self)
        if declarations.manifest.id not in registry.plugin_ids:
            registry.register(declarations)

    async def teardown(self) -> None:
        """Disconnect the MCP transport."""
        await self.disconnect()

    async def _maintain_connection(self, server_params):
        """
        Background task that maintains the stdio connection context.

        This keeps the MCP SDK's anyio task group alive for the duration
        of the connection. We use an event to signal when to disconnect.
        """
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                # Create session as context manager (required by MCP SDK)
                async with ClientSession(read_stream, write_stream) as session:
                    self._session = session

                    # Initialize session
                    await session.initialize()

                    # Discover tools
                    await self._discover_tools()

                    # Mark as connected
                    self._connected = True
                    logger.info(
                        f"Connected to MCP server {self.server_name}: {len(self._tools)} tools available"
                    )

                    # Keep connection alive until disconnect is called
                    while self._connected:
                        await asyncio.sleep(0.1)

        except Exception as e:
            self._connected = False
            logger.error(f"MCP connection error for {self.server_name}: {str(e)}")
            raise
        finally:
            # Cleanup
            self._session = None
            self._read = None
            self._write = None

    async def connect(self) -> None:
        """
        Connect to the MCP server and discover available tools.

        Raises:
            ImportError: If MCP SDK is not installed
            ConnectionError: If connection to server fails
        """
        if self._connected:
            return

        try:
            # Import MCP SDK
            from mcp import StdioServerParameters

            # Create server parameters
            server_params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env if self.env else None
            )

            logger.info(f"Connecting to MCP server: {self.server_name}")

            # Start background task that maintains the connection
            self._stdio_context_task = asyncio.create_task(
                self._maintain_connection(server_params)
            )

            # Wait for connection to be established
            max_wait = 5.0  # seconds
            start_time = asyncio.get_event_loop().time()
            while not self._connected:
                if asyncio.get_event_loop().time() - start_time > max_wait:
                    raise ConnectionError(f"Connection timeout after {max_wait}s")
                if self._stdio_context_task.done():
                    # Task failed
                    try:
                        self._stdio_context_task.result()
                    except Exception as e:
                        raise ConnectionError(f"Connection task failed: {str(e)}")
                await asyncio.sleep(0.05)

        except ImportError as e:
            error_msg = (
                "MCP SDK not installed. Install with: pip install 'daita-agents[mcp]'\n"
                "Official SDK: https://github.com/modelcontextprotocol/python-sdk"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to connect to MCP server {self.server_name}: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server"""
        try:
            # List available tools from server
            tools_response = await self.session.list_tools()

            # Convert to MCPTool objects
            self._tools = []
            for tool in tools_response.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or f"Tool: {tool.name}",
                    input_schema=(
                        tool.inputSchema if hasattr(tool, "inputSchema") else {}
                    ),
                )
                self._tools.append(mcp_tool)
                pass

            logger.info(f"Discovered {len(self._tools)} tools from {self.server_name}")

        except Exception as e:
            logger.error(f"Failed to discover tools from {self.server_name}: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources"""
        # Thread-safe disconnect
        async with self._session_lock:
            if not self._connected:
                return

            try:
                # Signal the connection task to stop
                self._connected = False

                # Wait for background task to finish (with timeout)
                if self._stdio_context_task and not self._stdio_context_task.done():
                    try:
                        await asyncio.wait_for(self._stdio_context_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"MCP disconnect timeout for {self.server_name}, cancelling task"
                        )
                        self._stdio_context_task.cancel()
                        try:
                            await self._stdio_context_task
                        except asyncio.CancelledError:
                            pass

                self._stdio_context_task = None
                logger.info(f"Disconnected from MCP server: {self.server_name}")

            except Exception as e:
                logger.warning(f"Error during MCP server disconnect: {str(e)}")

    def list_tools(self) -> List[MCPTool]:
        """
        Get list of available tools from this MCP server.

        Returns:
            List of MCPTool objects

        Raises:
            RuntimeError: If not connected to server
        """
        if not self._connected:
            raise RuntimeError(
                f"MCP server {self.server_name} not connected. Call connect() first."
            )

        return self._tools.copy()

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server with thread-safe session access.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            RuntimeError: If not connected or tool not found
            Exception: If tool execution fails
        """
        # Thread-safe session access
        async with self._session_lock:
            if not self._connected:
                raise RuntimeError(f"MCP server {self.server_name} not connected")

            # Verify tool exists
            tool = next((t for t in self._tools if t.name == tool_name), None)
            if not tool:
                available_tools = [t.name for t in self._tools]
                raise RuntimeError(
                    f"Tool '{tool_name}' not found on server {self.server_name}. "
                    f"Available tools: {', '.join(available_tools)}"
                )

            try:
                # Call the tool via MCP session (protected by lock)
                result = await self.session.call_tool(tool_name, arguments=arguments)

                # Extract content from result
                if hasattr(result, "content"):
                    # MCP returns content as a list of content items
                    if isinstance(result.content, list) and len(result.content) > 0:
                        first_content = result.content[0]
                        # Text content
                        if first_content.type == "text":
                            return first_content.text
                        # Other content types
                        return str(first_content)
                    return result.content

                return result

            except Exception as e:
                error_msg = (
                    f"MCP tool call failed: {tool_name} on {self.server_name}: {str(e)}"
                )
                logger.error(error_msg)
                raise Exception(error_msg) from e

    @property
    def is_connected(self) -> bool:
        """Check if server is connected"""
        return self._connected

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(f"MCP server {self.server_name} not connected")
        return self._session

    @property
    def tool_names(self) -> List[str]:
        """Get list of tool names"""
        return [t.name for t in self._tools]

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        tool_count = len(self._tools) if self._tools else 0
        return f"MCPServer({self.server_name}, {status}, {tool_count} tools)"

    async def __aenter__(self):
        """Async context manager entry - automatically connect"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - automatically disconnect"""
        await self.disconnect()


@dataclass(frozen=True)
class MCPToolExecutor:
    """Runtime executor for one discovered MCP tool."""

    id: str
    capability_ids: frozenset[str]
    server: MCPServer
    tool_name: str

    async def execute(self, task, operation, context):
        result = await self.server._call_tool(self.tool_name, dict(task.input))
        return [
            Evidence(
                kind="mcp.tool.result",
                owner=context.get("tool_owner"),
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "server": self.server.server_name,
                    "tool": self.tool_name,
                    "arguments": dict(task.input),
                    "result": result,
                },
            )
        ]


class MCPToolDeclarations:
    """Declaration-only runtime plugin for one connected MCP server."""

    def __init__(self, server: MCPServer):
        self.server = server
        self.server_id = _identifier(server.server_name)
        self.manifest = PluginManifest(
            id=_plugin_id("mcp", server.server_name),
            display_name=f"MCP Tools {server.server_name}",
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
            domains=frozenset({"chat"}),
        )
        self._tools = tuple(server.list_tools())
        self._tool_ids = {tool.name: _identifier(tool.name) for tool in self._tools}

    def declare_capabilities(self):
        return tuple(
            Capability(
                id=f"mcp.{self.server_id}.{self._tool_ids[tool.name]}",
                owner=self.manifest.id,
                description=tool.description,
                domains=frozenset({"chat", "mcp"}),
                operation_types=frozenset({"chat.tool_call", "mcp.tool_call"}),
                access=AccessMode.READ,
                risk=RiskLevel.MEDIUM,
                input_schema=tool.input_schema or {"type": "object"},
                output_evidence=frozenset({"mcp.tool.result"}),
                executor=f"{self.manifest.id}.{self._tool_ids[tool.name]}",
                model_visible=True,
                side_effecting=True,
                metadata={"server": self.server.server_name, "tool": tool.name},
            )
            for tool in self._tools
        )

    def get_executors(self):
        return tuple(
            MCPToolExecutor(
                id=f"{self.manifest.id}.{self._tool_ids[tool.name]}",
                capability_ids=frozenset(
                    {f"mcp.{self.server_id}.{self._tool_ids[tool.name]}"}
                ),
                server=self.server,
                tool_name=tool.name,
            )
            for tool in self._tools
        )

    def declare_evidence_schemas(self):
        return (
            EvidenceSchema(
                kind="mcp.tool.result",
                owner=self.manifest.id,
                json_schema={"type": "object"},
                description="Result returned by an MCP tool.",
            ),
        )

    def get_tool_views(self):
        return tuple(
            ToolView(
                name=f"{self.server_id}_{self._tool_ids[tool.name]}",
                capability_id=f"mcp.{self.server_id}.{self._tool_ids[tool.name]}",
                description=tool.description,
                parameters=tool.input_schema or {"type": "object"},
                metadata={"server": self.server.server_name, "tool": tool.name},
            )
            for tool in self._tools
        )


# Factory function for clean server configuration
def server(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create MCP server configuration.

    This factory function provides a clean API for configuring MCP servers
    to be used with Agent.

    Args:
        command: Command to run MCP server (e.g., "uvx", "python", "npx")
        args: Arguments for the command
        env: Environment variables for the server process
        name: Optional name for the server

    Returns:
        Server configuration dict

    Example:
        ```python
        from daita.plugins import mcp

        # Filesystem server
        fs_server = mcp.server(
            command="uvx",
            args=["mcp-server-filesystem", "/data"]
        )

        # GitHub server with environment
        gh_server = mcp.server(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}
        )
        ```
    """
    return {"command": command, "args": args or [], "env": env or {}, "name": name}


def _plugin_id(prefix: str, value: str) -> str:
    return f"{_identifier(prefix)}_{_identifier(value)}"


def _identifier(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = "mcp"
    if not cleaned[0].isalpha():
        cleaned = f"mcp_{cleaned}"
    return cleaned


# Export public API
__all__ = ["MCPServer", "MCPTool", "MCPToolDeclarations", "server"]
