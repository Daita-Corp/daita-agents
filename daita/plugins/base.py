"""
Base classes for Daita plugins.

Plugins are infrastructure utilities (databases, APIs, storage) that can
optionally expose their capabilities as agent tools.
"""

from abc import ABC
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tools import AgentTool


class BasePlugin(ABC):
    """
    Base class for plugins that expose tools to the LLM.

    Plugins provide infrastructure utilities (S3, Slack, REST APIs, databases,
    etc.) and optionally expose their capabilities as agent-callable tools.

    For plugins that also need to react to agent lifecycle events (before each
    run, on shutdown), extend LifecyclePlugin instead.
    """

    def initialize(self, agent_id: str):
        """
        Inject agent context into the plugin.

        Called automatically by Agent.add_plugin(). Override when your plugin
        needs the agent ID (e.g., to scope memory to a specific agent).

        Args:
            agent_id: Unique identifier of the agent using this plugin
        """
        pass

    def get_tools(self) -> List["AgentTool"]:
        """
        Return agent-callable tools exposed by this plugin.

        Override to surface plugin capabilities as LLM tools.

        Returns:
            List of AgentTool instances (empty list by default)
        """
        return []

    @property
    def has_tools(self) -> bool:
        """True if this plugin exposes at least one tool."""
        return len(self.get_tools()) > 0


class LifecyclePlugin(BasePlugin):
    """
    Base class for plugins that participate in the agent run lifecycle.

    Extend this instead of BasePlugin when your plugin needs to:
    - Inject context before each run (e.g., retrieve relevant memories)
    - Clean up or persist state when the agent stops

    Agent calls these hooks automatically — no manual wiring needed.

    Example::

        class MemoryPlugin(LifecyclePlugin):
            async def on_before_run(self, prompt: str) -> str:
                return await self._retrieve_relevant_memories(prompt)

            async def on_agent_stop(self) -> None:
                await self._persist_memories()
    """

    async def on_before_run(self, prompt: str) -> Optional[str]:
        """
        Called before each agent run, after the user prompt is known.

        Return a string to append to the system prompt (e.g., retrieved
        memories or contextual state), or None to add nothing.

        Args:
            prompt: The user prompt for the upcoming run

        Returns:
            Context string to inject into the system prompt, or None
        """
        return None

    async def on_agent_stop(self) -> None:
        """
        Called when Agent.stop() is invoked.

        Override to persist state, flush buffers, or close connections.
        """
        pass
