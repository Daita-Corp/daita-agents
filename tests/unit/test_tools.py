"""
Unit tests for daita/core/tools.py

Covers:
  - @tool decorator: schema extraction, name/description, sync/async wrapping
  - AgentTool: format conversion methods, execute() with timeout
  - ToolRegistry: CRUD, duplicate handling, execute dispatch
"""

import asyncio
from typing import Dict, List, Literal, Optional, Union

import pytest

from daita.core.tools import (
    AgentTool,
    ToolRegistry,
    _extract_parameters_from_function,
    _type_hint_to_json_schema,
    tool,
)


# ===========================================================================
# _type_hint_to_json_schema — pure function, no I/O
# ===========================================================================

class TestTypeHintToJsonSchema:
    def test_int(self):
        assert _type_hint_to_json_schema(int) == {"type": "integer"}

    def test_str(self):
        assert _type_hint_to_json_schema(str) == {"type": "string"}

    def test_float(self):
        assert _type_hint_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _type_hint_to_json_schema(bool) == {"type": "boolean"}

    def test_list_of_int(self):
        assert _type_hint_to_json_schema(List[int]) == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_list_unparameterised(self):
        assert _type_hint_to_json_schema(list) == {"type": "array"}

    def test_dict(self):
        assert _type_hint_to_json_schema(Dict[str, int]) == {"type": "object"}

    def test_optional_str_unwraps_to_str(self):
        # Optional[str] == Union[str, None] → just the inner type schema
        schema = _type_hint_to_json_schema(Optional[str])
        assert schema == {"type": "string"}

    def test_literal_strings(self):
        schema = _type_hint_to_json_schema(Literal["a", "b"])
        assert schema["type"] == "string"
        assert schema["enum"] == ["a", "b"]

    def test_literal_ints(self):
        schema = _type_hint_to_json_schema(Literal[1, 2, 3])
        assert schema["type"] == "integer"
        assert schema["enum"] == [1, 2, 3]

    def test_union_two_types(self):
        schema = _type_hint_to_json_schema(Union[int, str])
        assert "anyOf" in schema
        types = {s["type"] for s in schema["anyOf"]}
        assert types == {"integer", "string"}

    def test_unknown_type_falls_back_to_string(self):
        class Custom:
            pass
        schema = _type_hint_to_json_schema(Custom)
        assert schema == {"type": "string"}


# ===========================================================================
# @tool decorator — schema extraction
# ===========================================================================

class TestToolDecoratorSchema:
    def test_basic_types_extracted(self):
        @tool
        def fn(x: int, y: str, z: float, flag: bool) -> str:
            """Do something.

            Args:
                x: An integer
                y: A string
                z: A float
                flag: A boolean
            """
            return ""

        props = fn.parameters["properties"]
        assert props["x"]["type"] == "integer"
        assert props["y"]["type"] == "string"
        assert props["z"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"

    def test_required_includes_non_defaulted_params(self):
        @tool
        def fn(a: int, b: str = "default") -> str:
            """Do something.

            Args:
                a: Required param
                b: Optional param with default
            """
            return ""

        required = fn.parameters["required"]
        assert "a" in required
        assert "b" not in required

    def test_optional_param_not_required(self):
        @tool
        def fn(name: Optional[str] = None) -> str:
            """Do something.

            Args:
                name: An optional name
            """
            return ""

        assert "name" not in fn.parameters.get("required", [])

    def test_literal_param_produces_enum(self):
        @tool
        def fn(mode: Literal["fast", "slow"]) -> str:
            """Do something.

            Args:
                mode: Speed mode
            """
            return ""

        schema = fn.parameters["properties"]["mode"]
        assert schema["enum"] == ["fast", "slow"]

    def test_list_param_produces_array(self):
        @tool
        def fn(items: List[str]) -> str:
            """Do something.

            Args:
                items: List of strings
            """
            return ""

        schema = fn.parameters["properties"]["items"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_docstring_args_become_descriptions(self):
        @tool
        def fn(query: str) -> str:
            """Search for things.

            Args:
                query: The search query to execute
            """
            return ""

        desc = fn.parameters["properties"]["query"]["description"]
        assert "search query" in desc.lower()

    def test_param_without_docstring_gets_fallback(self):
        @tool
        def fn(x: int) -> str:
            """No args section."""
            return ""

        desc = fn.parameters["properties"]["x"].get("description", "")
        assert desc  # fallback is non-empty

    def test_name_defaults_to_function_name(self):
        @tool
        def my_function() -> str:
            """Does something."""
            return ""

        assert my_function.name == "my_function"

    def test_description_defaults_to_docstring_first_line(self):
        @tool
        def fn() -> str:
            """First line description.

            More details here.
            """
            return ""

        assert fn.description == "First line description."

    def test_custom_name_overrides(self):
        @tool(name="custom_name")
        def fn() -> str:
            """Desc."""
            return ""

        assert fn.name == "custom_name"

    def test_custom_description_overrides(self):
        @tool(description="My custom description")
        def fn() -> str:
            """Original docstring."""
            return ""

        assert fn.description == "My custom description"

    def test_timeout_stored(self):
        @tool(timeout_seconds=30)
        def fn() -> str:
            """Desc."""
            return ""

        assert fn.timeout_seconds == 30

    def test_category_stored(self):
        @tool(category="math")
        def fn() -> str:
            """Desc."""
            return ""

        assert fn.category == "math"

    def test_no_parentheses_returns_agent_tool(self):
        @tool
        def fn() -> str:
            """Desc."""
            return ""

        assert isinstance(fn, AgentTool)

    def test_called_as_function_returns_agent_tool(self):
        def fn(x: int) -> int:
            """Desc.

            Args:
                x: A number
            """
            return x

        result = tool(fn)
        assert isinstance(result, AgentTool)

    def test_source_is_custom(self):
        @tool
        def fn() -> str:
            """Desc."""
            return ""

        assert fn.source == "custom"

    def test_wraps_sync_function_as_async(self):
        @tool
        def sync_fn(a: int) -> int:
            """Desc.

            Args:
                a: A number
            """
            return a * 2

        # handler must be awaitable
        assert asyncio.iscoroutinefunction(sync_fn.handler)

    def test_wraps_async_function(self):
        @tool
        async def async_fn(a: int) -> int:
            """Desc.

            Args:
                a: A number
            """
            return a * 2

        assert asyncio.iscoroutinefunction(async_fn.handler)


# ===========================================================================
# AgentTool — format conversion
# ===========================================================================

class TestAgentToolFormatConversion:
    @pytest.fixture
    def sample_tool(self):
        async def handler(args):
            return args

        return AgentTool(
            name="search",
            description="Search for records",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            handler=handler,
        )

    def test_to_openai_function_shape(self, sample_tool):
        result = sample_tool.to_openai_function()
        assert result["type"] == "function"
        assert "function" in result
        func = result["function"]
        assert func["name"] == "search"
        assert func["description"] == "Search for records"
        assert "parameters" in func

    def test_to_anthropic_tool_shape(self, sample_tool):
        result = sample_tool.to_anthropic_tool()
        assert result["name"] == "search"
        assert result["description"] == "Search for records"
        assert "input_schema" in result

    def test_to_prompt_description_contains_name(self, sample_tool):
        desc = sample_tool.to_prompt_description()
        assert "search" in desc

    def test_to_prompt_description_contains_description(self, sample_tool):
        desc = sample_tool.to_prompt_description()
        assert "Search for records" in desc

    def test_to_prompt_description_contains_param(self, sample_tool):
        desc = sample_tool.to_prompt_description()
        assert "query" in desc

    def test_to_prompt_description_required_label(self, sample_tool):
        desc = sample_tool.to_prompt_description()
        assert "(required)" in desc

    def test_to_prompt_description_optional_label(self):
        async def handler(args):
            return args

        t = AgentTool(
            name="fetch",
            description="Fetch data",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": [],
            },
            handler=handler,
        )
        desc = t.to_prompt_description()
        assert "(optional)" in desc

    def test_to_prompt_description_no_params(self):
        async def handler(args):
            return "ok"

        t = AgentTool(
            name="ping",
            description="Ping the server",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handler,
        )
        desc = t.to_prompt_description()
        assert "no parameters" in desc


# ===========================================================================
# AgentTool — execution
# ===========================================================================

class TestAgentToolExecution:
    async def test_execute_sync_handler(self):
        # @tool wraps sync functions in an async handler automatically.
        # Verify that execute() can await a sync-origin handler correctly.
        @tool
        def inc(x: int) -> int:
            """Increment.

            Args:
                x: Number to increment
            """
            return x + 1

        result = await inc.execute({"x": 5})
        assert result == 6

    async def test_execute_async_handler(self):
        @tool
        async def double(x: int) -> int:
            """Double.

            Args:
                x: Number
            """
            return x * 2

        result = await double.execute({"x": 4})
        assert result == 8

    async def test_execute_passes_arguments(self):
        received = {}

        @tool
        def capture(a: str, b: int) -> str:
            """Capture args.

            Args:
                a: First arg
                b: Second arg
            """
            received["a"] = a
            received["b"] = b
            return "ok"

        await capture.execute({"a": "hello", "b": 42})
        assert received == {"a": "hello", "b": 42}

    async def test_execute_timeout_raises(self):
        async def slow_handler(args):
            await asyncio.sleep(10)

        t = AgentTool(
            name="slow",
            description="Slow tool",
            parameters={},
            handler=slow_handler,
            timeout_seconds=0.01,
        )
        with pytest.raises(RuntimeError, match="timed out"):
            await t.execute({})

    async def test_execute_no_timeout(self):
        async def fast_handler(args):
            return "done"

        t = AgentTool(
            name="fast",
            description="Fast tool",
            parameters={},
            handler=fast_handler,
            timeout_seconds=None,
        )
        result = await t.execute({})
        assert result == "done"

    async def test_non_callable_handler_raises(self):
        t = AgentTool(
            name="broken",
            description="Broken tool",
            parameters={},
            handler="not_a_function",  # type: ignore
        )
        with pytest.raises(RuntimeError):
            await t.execute({})


# ===========================================================================
# ToolRegistry
# ===========================================================================

class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.fixture
    def tool_a(self):
        async def h(args):
            return "a"

        return AgentTool(name="tool_a", description="Tool A", parameters={}, handler=h)

    @pytest.fixture
    def tool_b(self):
        async def h(args):
            return "b"

        return AgentTool(name="tool_b", description="Tool B", parameters={}, handler=h)

    def test_starts_empty(self, registry):
        assert registry.tool_count == 0
        assert registry.tool_names == []

    def test_register_single(self, registry, tool_a):
        registry.register(tool_a)
        assert registry.tool_count == 1
        assert registry.get("tool_a") is tool_a

    def test_register_many(self, registry, tool_a, tool_b):
        registry.register_many([tool_a, tool_b])
        assert registry.tool_count == 2

    def test_get_missing_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_duplicate_name_overwrites(self, registry, tool_a):
        async def h(args):
            return "replacement"

        replacement = AgentTool(
            name="tool_a", description="Replacement", parameters={}, handler=h
        )
        registry.register(tool_a)
        registry.register(replacement)
        # Last write wins
        assert registry.get("tool_a") is replacement

    async def test_execute_calls_handler(self, registry, tool_a):
        registry.register(tool_a)
        result = await registry.execute("tool_a", {})
        assert result == "a"

    async def test_execute_missing_raises(self, registry):
        with pytest.raises(RuntimeError, match="not found"):
            await registry.execute("nonexistent", {})

    async def test_execute_missing_lists_available(self, registry, tool_a):
        registry.register(tool_a)
        with pytest.raises(RuntimeError, match="tool_a"):
            await registry.execute("missing", {})

    def test_tool_names_property(self, registry, tool_a, tool_b):
        registry.register_many([tool_a, tool_b])
        names = registry.tool_names
        assert "tool_a" in names
        assert "tool_b" in names

    def test_tool_count_property(self, registry, tool_a, tool_b):
        registry.register_many([tool_a, tool_b])
        assert registry.tool_count == 2
