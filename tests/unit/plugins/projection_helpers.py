from typing import Any

from daita.agents.agent import Agent
from daita.core.tools import LocalTool
from daita.llm.mock import MockLLMProvider


def projected_tools(plugin: Any) -> dict[str, LocalTool]:
    agent = Agent(
        name="ProjectionTestAgent",
        llm_provider=MockLLMProvider(delay=0),
        plugins=[plugin],
    )
    return {tool.name: tool for tool in agent.tools}


def projected_tool_names(plugin: Any) -> set[str]:
    return set(projected_tools(plugin))
