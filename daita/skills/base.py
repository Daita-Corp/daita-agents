"""
Base classes for Daita skills.

Skills are reusable units of agent capability that bundle related tools
with domain-specific instructions. Unlike plugins (infrastructure connectors),
skills carry behavioral intelligence.
"""

import inspect
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from ..plugins.base import BasePlugin, LifecyclePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool


class BaseSkill(LifecyclePlugin):
    """Reusable, composable unit of agent capability.

    Unlike plugins (infrastructure connectors), skills carry behavioral
    intelligence: domain-specific instructions and related tools.

    Subclass this for skills that need dynamic instruction generation,
    plugin dependencies, or custom tool construction.

    For simple static skills, use the ``Skill`` convenience class instead.
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    instructions: str = ""
    instructions_file: Optional[str] = None

    def __init__(self, **config):
        self.config = config
        self._resolved_plugins: Dict[str, BasePlugin] = {}
        self._cached_instructions: Optional[str] = None

    async def on_before_run(self, prompt: str) -> Optional[str]:
        """Inject skill instructions into the system prompt."""
        return self.get_instructions(prompt)

    def get_instructions(self, user_prompt: str = "") -> Optional[str]:
        """Return instructions for this skill.

        Resolution order:
        1. instructions_file (read once, cached)
        2. instructions (inline string)
        3. None

        Override to generate dynamic, prompt-aware instructions.
        The default implementation ignores ``user_prompt``.
        """
        if self.instructions_file:
            if self._cached_instructions is None:
                path = Path(self.instructions_file)
                if not path.is_absolute():
                    module_dir = Path(inspect.getfile(self.__class__)).parent
                    path = module_dir / path
                self._cached_instructions = path.read_text()
            return self._cached_instructions
        return self.instructions or None

    def get_tools(self) -> List["AgentTool"]:
        """Override to return tools this skill provides."""
        return []

    def requires(self) -> Dict[str, type]:
        """Declare required plugin types.

        Returns a mapping of logical name to plugin type, e.g.
        ``{"db": BaseDatabasePlugin}``. Resolved plugins are available
        via ``self._resolved_plugins`` after ``add_skill()`` completes.
        """
        return {}


class Skill(BaseSkill):
    """Concrete skill built from instructions and a list of tools.

    Use this for simple skills that don't need subclassing::

        report_skill = Skill(
            name="report_gen",
            instructions="Format all reports with headers and tables.",
            tools=[format_report, generate_chart],
        )
    """

    def __init__(
        self,
        name: str,
        *,
        instructions: str = "",
        instructions_file: Optional[str] = None,
        tools: Optional[List["AgentTool"]] = None,
        description: str = "",
        version: str = "0.1.0",
        **config,
    ):
        super().__init__(**config)
        if instructions and instructions_file:
            raise ValueError(
                f"Skill('{name}'): provide either 'instructions' or "
                f"'instructions_file', not both."
            )
        if instructions_file and not Path(instructions_file).is_absolute():
            raise ValueError(
                f"Skill('{name}'): instructions_file must be an absolute path. "
                f"Use a BaseSkill subclass for relative path resolution."
            )
        self.name = name
        self.description = description
        self.version = version
        self.instructions = instructions
        self.instructions_file = instructions_file
        self._tools = tools or []

    def get_tools(self) -> List["AgentTool"]:
        return self._tools
