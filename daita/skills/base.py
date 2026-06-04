"""
Base classes for Daita skills.

Skills are reusable units of agent capability that bundle runtime declarations
with domain-specific instructions. Unlike plugins (infrastructure connectors),
skills carry behavioral intelligence.
"""

import inspect
from pathlib import Path
import re
from typing import Optional

from ..plugins.base import SkillPlugin
from ..plugins.manifest import PluginKind, PluginManifest
from ..runtime import Capability, ContextAudience, ContextBlock, Executor, ToolView

_SKILL_ID_RE = re.compile(r"[^a-z0-9_]+")


def _skill_manifest_id(name: str, class_name: str) -> str:
    raw = name or class_name or "skill"
    value = _SKILL_ID_RE.sub("_", raw.lower()).strip("_")
    if not value:
        value = "skill"
    if not value[0].isalpha():
        value = f"skill_{value}"
    if not value.startswith("skill_"):
        value = f"skill_{value}"
    return value


class _SkillInstructionContextProvider:
    """Context provider that renders one skill's prompt instructions."""

    audiences = frozenset({ContextAudience.PRIMARY_MODEL})

    def __init__(self, skill: "BaseSkill") -> None:
        self._skill = skill
        self.owner = skill.manifest.id
        self.id = f"{self.owner}.instructions"

    async def render(
        self,
        context: dict,
        audience: ContextAudience,
        token_budget: int,
    ) -> ContextBlock | None:
        if audience is not ContextAudience.PRIMARY_MODEL:
            return None
        instructions = self._skill.get_instructions(str(context.get("prompt") or ""))
        if not instructions:
            return None
        return ContextBlock(
            id=self.id,
            owner=self.owner,
            audience=audience,
            content=instructions,
            priority=20,
            metadata={
                "context_kind": "skill_instructions",
                "skill_name": self._skill.name,
            },
        )


class BaseSkill(SkillPlugin):
    """Reusable, composable unit of agent capability.

    Unlike plugins (infrastructure connectors), skills carry behavioral
    intelligence: domain-specific instructions and related runtime declarations.

    Subclass this for skills that need dynamic instruction generation,
    capability requirements, or custom runtime declarations.

    For simple static skills, use the ``Skill`` convenience class instead.
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    instructions: str = ""
    instructions_file: Optional[str] = None

    def __init__(self, **config):
        self.config = config
        self._resolved_capabilities: dict[str, tuple[Capability, ...]] = {}
        self._cached_instructions: Optional[str] = None

    @property
    def manifest(self) -> PluginManifest:
        skill_name = self.name or self.__class__.__name__
        return PluginManifest(
            id=_skill_manifest_id(skill_name, self.__class__.__name__),
            display_name=skill_name,
            version=self.version,
            kind=PluginKind.SKILL,
            domains=frozenset({"agent", "skill"}),
            provides=frozenset({"skill_instructions"}),
        )

    def get_context_providers(self) -> tuple[_SkillInstructionContextProvider, ...]:
        """Expose skill instructions through the runtime context surface."""
        return (_SkillInstructionContextProvider(self),)

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

    def requires_capabilities(self) -> tuple[str, ...]:
        """Declare capability IDs this skill needs from already-attached plugins."""
        return ()

    @property
    def resolved_capabilities(self) -> dict[str, tuple[Capability, ...]]:
        """Capabilities resolved for ``requires_capabilities()`` at attachment."""
        return dict(self._resolved_capabilities)


class Skill(BaseSkill):
    """Concrete skill built from instructions and runtime declarations.

    Use this for simple skills that don't need subclassing::

        report_skill = Skill(
            name="report_gen",
            instructions="Format all reports with headers and tables.",
            capabilities=[...],
            executors=[...],
            tool_views=[...],
        )
    """

    def __init__(
        self,
        name: str,
        *,
        instructions: str = "",
        instructions_file: Optional[str] = None,
        requires_capabilities: tuple[str, ...] = (),
        capabilities: tuple[Capability, ...] = (),
        executors: tuple[Executor, ...] = (),
        tool_views: tuple[ToolView, ...] = (),
        description: str = "",
        version: str = "0.1.0",
        **config,
    ):
        if "tools" in config:
            raise TypeError(
                "Skill no longer accepts 'tools'. Declare skill-owned "
                "capabilities, executors, and tool_views instead."
            )
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
        self._requires_capabilities = tuple(requires_capabilities)
        self._capabilities = tuple(capabilities)
        self._executors = tuple(executors)
        self._tool_views = tuple(tool_views)

    def requires_capabilities(self) -> tuple[str, ...]:
        return self._requires_capabilities

    def declare_capabilities(self) -> tuple[Capability, ...]:
        return self._capabilities

    def get_executors(self) -> tuple[Executor, ...]:
        return self._executors

    def get_tool_views(self) -> tuple[ToolView, ...]:
        return self._tool_views
