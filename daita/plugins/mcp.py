"""
MCP (Model Context Protocol) plugin for Daita Agents.

This plugin enables agents to connect to any MCP server and autonomously use
their tools via LLM function calling. Agents discover available tools and
decide which ones to use based on the task.

Usage:
    ```python
    from daita import Agent
    from daita.plugins import mcp

    # Agent with MCP tools
    agent = Agent(
        name="file_analyzer",
        mcp_servers=[
            mcp.server(command="uvx", args=["mcp-server-filesystem", "/data"])
        ]
    )

    # Agent autonomously discovers and uses MCP tools
    result = await agent.process("Read report.csv and calculate average revenue")
    ```

MCP Protocol:
    The Model Context Protocol (MCP) is Anthropic's open standard for connecting
    AI systems to external data sources and tools. This plugin provides native
    MCP client support for Daita agents.
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

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
            "parameters": self.input_schema
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
        server_name: Optional[str] = None
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
        self._session = None
        self._read = None
        self._write = None
        self._tools: List[MCPTool] = []
        self._connected = False
        self._stdio_context_task = None  # Background task keeping context alive
        self._session_lock = asyncio.Lock()  # Protects session access from concurrent calls

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
                    logger.info(f"Connected to MCP server {self.server_name}: {len(self._tools)} tools available")

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
                command=self.command,
                args=self.args,
                env=self.env if self.env else None
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
                "MCP SDK not installed. Install with: pip install mcp\n"
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
            tools_response = await self._session.list_tools()

            # Convert to MCPTool objects
            self._tools = []
            for tool in tools_response.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or f"Tool: {tool.name}",
                    input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
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
                        logger.warning(f"MCP disconnect timeout for {self.server_name}, cancelling task")
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
            raise RuntimeError(f"MCP server {self.server_name} not connected. Call connect() first.")

        return self._tools.copy()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
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
                result = await self._session.call_tool(tool_name, arguments=arguments)

                # Extract content from result
                if hasattr(result, 'content'):
                    # MCP returns content as a list of content items
                    if isinstance(result.content, list) and len(result.content) > 0:
                        first_content = result.content[0]
                        # Text content
                        if hasattr(first_content, 'text'):
                            return first_content.text
                        # Other content types
                        return str(first_content)
                    return result.content

                return result

            except Exception as e:
                error_msg = f"MCP tool call failed: {tool_name} on {self.server_name}: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg) from e

    @property
    def is_connected(self) -> bool:
        """Check if server is connected"""
        return self._connected

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


class MCPToolRegistry:
    """
    Registry for managing multiple MCP servers and their tools.

    Used internally by Agent to aggregate tools from multiple
    MCP servers and route tool calls to the appropriate server.
    """

    def __init__(self):
        """Initialize empty registry"""
        self.servers: List[MCPServer] = []
        self._tool_server_map: Dict[str, MCPServer] = {}

    async def add_server(self, server: MCPServer) -> None:
        """
        Add an MCP server to the registry.

        Args:
            server: MCPServer instance to add
        """
        # Connect if not already connected
        if not server.is_connected:
            await server.connect()

        # Add to registry
        self.servers.append(server)

        # Map tools to servers
        for tool in server.list_tools():
            if tool.name in self._tool_server_map:
                logger.warning(
                    f"Tool name collision: {tool.name} exists in multiple servers. "
                    f"Using tool from {server.server_name}"
                )
            self._tool_server_map[tool.name] = server

        logger.info(f"Added MCP server {server.server_name} to registry")

    def get_all_tools(self) -> List[MCPTool]:
        """Get aggregated list of all tools from all servers"""
        all_tools = []
        for server in self.servers:
            all_tools.extend(server.list_tools())
        return all_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool by routing to the appropriate server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool execution result
        """
        server = self._tool_server_map.get(tool_name)
        if not server:
            available_tools = list(self._tool_server_map.keys())
            raise RuntimeError(
                f"Tool '{tool_name}' not found in any MCP server. "
                f"Available tools: {', '.join(available_tools)}"
            )

        return await server.call_tool(tool_name, arguments)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers"""
        for server in self.servers:
            try:
                await server.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting from {server.server_name}: {str(e)}")

        self.servers.clear()
        self._tool_server_map.clear()

    @property
    def tool_count(self) -> int:
        """Total number of tools across all servers"""
        return len(self._tool_server_map)

    @property
    def server_count(self) -> int:
        """Number of connected servers"""
        return len(self.servers)


# Factory function for clean server configuration
def server(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    name: Optional[str] = None
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
    return {
        "command": command,
        "args": args or [],
        "env": env or {},
        "name": name
    }


# Export public API
__all__ = [
    "MCPServer",
    "MCPTool",
    "MCPToolRegistry",
    "server"
]
