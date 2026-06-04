"""Reference tests for generic Agent adapter boundaries."""

from dataclasses import dataclass

import pytest

from daita.agents.agent import Agent
from daita.core.exceptions import SkillError
from daita.core.tools import LocalTool
from daita.plugins.base import BasePlugin
from daita.plugins.manifest import PluginKind, PluginManifest
from daita.plugins.base import RuntimeExtensionPlugin
from daita.runtime import AccessMode, Capability, RiskLevel
from daita.skills.base import BaseSkill


def _tool(name: str, result: str, *, source: str = "plugin") -> LocalTool:
    async def handler(args):
        return result

    return LocalTool(
        name=name,
        description=f"{name} reference tool",
        parameters={"type": "object", "properties": {}},
        handler=handler,
        source=source,
    )


class RecordingPlugin(BasePlugin):
    def __init__(self, tool_name: str = "reference_tool", result: str = "ok"):
        self.received_agent_id = None
        self._tool_name = tool_name
        self._result = result

    def initialize(self, agent_id: str):
        self.received_agent_id = agent_id


def test_non_manifest_plugin_tools_and_initialize_are_not_runtime_contracts(mock_llm):
    agent = Agent(name="reference-agent", llm_provider=mock_llm)
    plugin = RecordingPlugin()

    agent.add_plugin(plugin)

    assert plugin.received_agent_id is None
    assert "reference_tool" not in agent.tool_names
    assert agent.available_tools == []


async def test_local_agent_tool_remains_explicit_adapter_boundary(mock_llm):
    agent = Agent(name="reference-agent", llm_provider=mock_llm)

    agent.add_plugin(_tool("shared", "first", source="custom"))
    agent.add_plugin(_tool("shared", "second", source="custom"))

    assert agent.tool_names == ["shared"]
    assert len(agent.available_tools) == 1
    assert await agent.call_tool("shared", {}) == "second"


@dataclass
class FakeMCPTool:
    name: str = "mcp_lookup"
    description: str = "Look up data through MCP."
    input_schema: dict = None

    def __post_init__(self):
        if self.input_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }


class FakeMCPRegistry:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"called": name, "arguments": arguments}


async def test_current_behavior_mcp_tools_adapt_to_agent_tools():
    registry = FakeMCPRegistry()
    agent_tool = LocalTool.from_mcp_tool(FakeMCPTool(), registry)

    result = await agent_tool.execute({"query": "orders"})

    assert agent_tool.name == "mcp_lookup"
    assert agent_tool.source == "mcp"
    assert agent_tool.category == "mcp"
    assert agent_tool.parameters == {"query": {"type": "string"}}
    assert registry.calls == [("mcp_lookup", {"query": "orders"})]
    assert result["called"] == "mcp_lookup"


class RequiredExecutor:
    id = "required.execute"
    capability_ids = frozenset({"required.lookup"})

    async def execute(self, task, operation, context):
        return []


class RequiredPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="required",
        display_name="Required",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def declare_capabilities(self):
        return (
            Capability(
                id="required.lookup",
                owner="required",
                description="Reference capability.",
                domains=frozenset({"agent"}),
                operation_types=frozenset({"reference.lookup"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset(),
                executor="required.execute",
                side_effecting=False,
            ),
        )

    def get_executors(self):
        return (RequiredExecutor(),)


class PluginDependentSkill(BaseSkill):
    name = "plugin_dependent"

    def requires_capabilities(self):
        return ("required.lookup",)


def test_current_behavior_skills_require_capabilities_before_registration(mock_llm):
    agent = Agent(name="reference-agent", llm_provider=mock_llm)

    with pytest.raises(SkillError, match="requires capabilities not yet available"):
        agent.add_skill(PluginDependentSkill())

    plugin = RequiredPlugin()
    skill = PluginDependentSkill()
    agent.add_plugin(plugin)
    agent.add_skill(skill)

    assert skill.resolved_capabilities["required.lookup"] == (
        plugin.declare_capabilities()[0],
    )
