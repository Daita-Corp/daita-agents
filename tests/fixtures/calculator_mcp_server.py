#!/usr/bin/env python3
"""
Simple Calculator MCP Server

A minimal MCP server for testing purposes.
Provides basic math operations as tools.
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create MCP server instance
app = Server("calculator-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available calculator tools"""
    return [
        Tool(
            name="add",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="subtract",
            description="Subtract second number from first number",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="multiply",
            description="Multiply two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="divide",
            description="Divide first number by second number",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Numerator"},
                    "b": {"type": "number", "description": "Denominator"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="power",
            description="Raise base to the power of exponent",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {"type": "number", "description": "Base number"},
                    "exponent": {"type": "number", "description": "Exponent"},
                },
                "required": ["base", "exponent"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute calculator tool"""
    try:
        if name == "add":
            result = arguments["a"] + arguments["b"]
        elif name == "subtract":
            result = arguments["a"] - arguments["b"]
        elif name == "multiply":
            result = arguments["a"] * arguments["b"]
        elif name == "divide":
            if arguments["b"] == 0:
                return [TextContent(type="text", text="Error: Division by zero")]
            result = arguments["a"] / arguments["b"]
        elif name == "power":
            result = arguments["base"] ** arguments["exponent"]
        else:
            return [TextContent(type="text", text=f"Error: Unknown tool {name}")]

        return [TextContent(type="text", text=str(result))]

    except KeyError as e:
        return [TextContent(type="text", text=f"Error: Missing argument {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the calculator MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
