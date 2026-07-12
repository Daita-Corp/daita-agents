"""
Base classes for Daita skills.

Skills are reusable units of agent capability that bundle runtime declarations
with domain-specific instructions. Unlike plugins (infrastructure connectors),
skills carry behavioral intelligence.
"""

from dataclasses import replace
import inspect
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from ..plugins.base import SkillPlugin
from ..plugins.manifest import PluginKind, PluginManifest
from ..runtime import (
    Capability,
    ContextAudience,
    ContextBlock,
    ContextProvider,
    EvidenceSchema,
    Executor,
    LocalToolRuntimeAdapter,
    Policy,
    ToolView,
    Worker,
)
from .runtime import SkillActivationRules, SkillDiscovery, SkillRuntimeEffects

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
        context: Mapping[str, Any],
        audience: ContextAudience,
        token_budget: int,
    ) -> ContextBlock | None:
        if audience is not ContextAudience.PRIMARY_MODEL:
            return None
        instructions = self._skill.get_instructions(str(context.get("prompt") or ""))
        if not instructions:
            return None
        discovery = self._skill.discovery()
        selected_ids = set(context.get("selected_skill_ids") or ())
        selected_names = set(context.get("selected_skill_names") or ())
        selected = (
            self._skill.manifest.id in selected_ids
            or self._skill.name in selected_names
            or discovery.name in selected_names
        )
        if discovery.context_mode in {"none", "runtime_only"}:
            return None
        if discovery.context_mode == "on_demand" and not selected:
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

    def discovery(self) -> SkillDiscovery:
        """Return compact discovery metadata for this skill."""
        return SkillDiscovery(
            name=self.name or self.__class__.__name__,
            description=self.description,
            runtime_kinds=("chat", "db"),
            requires_capabilities=self.requires_capabilities(),
            provides_capabilities=tuple(
                capability.id for capability in self.declare_capabilities()
            ),
            context_mode=(
                "always" if self.instructions or self.instructions_file else "none"
            ),
        )

    def activation_rules(self) -> SkillActivationRules:
        """Return runtime activation rules for this skill."""
        return SkillActivationRules(
            requires_capabilities=self.requires_capabilities(),
            always_on=bool(self.instructions or self.instructions_file),
        )

    def runtime_effects(
        self,
        request: Any = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> SkillRuntimeEffects:
        """Return declarative effects for a selected runtime run."""
        return SkillRuntimeEffects(skill_id=self.manifest.id)

    def declare_capabilities(self) -> tuple[Capability, ...]:
        return ()

    def get_executors(self) -> tuple[Executor, ...]:
        return ()

    def get_tool_views(self) -> tuple[ToolView, ...]:
        return ()

    def declare_policies(self) -> tuple[Policy, ...]:
        return ()

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        return ()

    def get_workers(self) -> tuple[Worker, ...]:
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
        policies: tuple[Policy, ...] = (),
        evidence_schemas: tuple[EvidenceSchema, ...] = (),
        context_providers: tuple[ContextProvider, ...] = (),
        workers: tuple[Worker, ...] = (),
        activation_rules: SkillActivationRules | Mapping[str, Any] | None = None,
        runtime_effects: SkillRuntimeEffects | Mapping[str, Any] | None = None,
        discovery: SkillDiscovery | Mapping[str, Any] | None = None,
        context_mode: str = "always",
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
        self._policies = tuple(policies)
        self._evidence_schemas = tuple(evidence_schemas)
        self._extra_context_providers = tuple(context_providers)
        self._workers = tuple(workers)
        self._discovery = _coerce_discovery(
            discovery,
            name=name,
            description=description,
            context_mode=context_mode,
            requires_capabilities=self._requires_capabilities,
            capabilities=self._capabilities,
        )
        self._activation_rules = _coerce_activation_rules(
            activation_rules,
            default_always_on=bool(
                runtime_effects
                or (context_mode == "always" and (instructions or instructions_file))
            ),
            requires_capabilities=self._requires_capabilities,
        )
        self._runtime_effects = _coerce_runtime_effects(
            runtime_effects,
            skill_id=self.manifest.id,
            tool_view_names=tuple(view.name for view in self._tool_views),
        )

    @classmethod
    def with_tools(
        cls,
        name: str,
        *,
        tools: tuple[Any, ...] | list[Any],
        instructions: str = "",
        description: str = "",
        version: str = "0.1.0",
        context_mode: str = "always",
        requires_capabilities: tuple[str, ...] = (),
        runtime_effects: SkillRuntimeEffects | Mapping[str, Any] | None = None,
        **config,
    ) -> "Skill":
        """Create a skill from local tools using the runtime tool adapter."""
        manifest_id = _skill_manifest_id(name, cls.__name__)
        adapter = LocalToolRuntimeAdapter(owner_namespace=manifest_id)
        capabilities: list[Capability] = []
        executors: list[Executor] = []
        evidence_schemas: list[EvidenceSchema] = []
        tool_views: list[ToolView] = []
        for tool in tools:
            plugin = adapter.plugin_for(tool)
            capability = plugin.declare_capabilities()[0]
            rehomed_capability = replace(
                capability,
                owner=manifest_id,
                metadata={
                    **capability.metadata,
                    "skill_id": manifest_id,
                    "tool_owner": plugin.manifest.id,
                },
            )
            capabilities.append(rehomed_capability)
            executors.extend(plugin.get_executors())
            evidence_schemas.extend(
                replace(schema, owner=manifest_id)
                for schema in plugin.declare_evidence_schemas()
            )
            tool_views.extend(
                replace(
                    view,
                    metadata={
                        **view.metadata,
                        "skill_id": manifest_id,
                        "tool_owner": plugin.manifest.id,
                    },
                )
                for view in plugin.get_tool_views()
            )
        return cls(
            name=name,
            instructions=instructions,
            requires_capabilities=requires_capabilities,
            capabilities=tuple(capabilities),
            executors=tuple(executors),
            tool_views=tuple(tool_views),
            evidence_schemas=tuple(evidence_schemas),
            runtime_effects=runtime_effects,
            description=description,
            version=version,
            context_mode=context_mode,
            **config,
        )

    def requires_capabilities(self) -> tuple[str, ...]:
        return self._requires_capabilities

    def get_context_providers(self) -> tuple[ContextProvider, ...]:
        return (*super().get_context_providers(), *self._extra_context_providers)

    def discovery(self) -> SkillDiscovery:
        return self._discovery

    def activation_rules(self) -> SkillActivationRules:
        return self._activation_rules

    def runtime_effects(
        self,
        request: Any = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> SkillRuntimeEffects:
        return self._runtime_effects

    def declare_capabilities(self) -> tuple[Capability, ...]:
        return self._capabilities

    def get_executors(self) -> tuple[Executor, ...]:
        return self._executors

    def get_tool_views(self) -> tuple[ToolView, ...]:
        return self._tool_views

    def declare_policies(self) -> tuple[Policy, ...]:
        return self._policies

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        return self._evidence_schemas

    def get_workers(self) -> tuple[Worker, ...]:
        return self._workers


def _coerce_discovery(
    value: SkillDiscovery | Mapping[str, Any] | None,
    *,
    name: str,
    description: str,
    context_mode: str,
    requires_capabilities: tuple[str, ...],
    capabilities: tuple[Capability, ...],
) -> SkillDiscovery:
    if isinstance(value, SkillDiscovery):
        return value
    values = dict(value or {})
    return SkillDiscovery(
        name=str(values.pop("name", name)),
        description=str(values.pop("description", description)),
        requires_capabilities=tuple(
            values.pop("requires_capabilities", requires_capabilities)
        ),
        provides_capabilities=tuple(
            values.pop(
                "provides_capabilities",
                tuple(capability.id for capability in capabilities),
            )
        ),
        context_mode=str(values.pop("context_mode", context_mode)),
        **values,
    )


def _coerce_activation_rules(
    value: SkillActivationRules | Mapping[str, Any] | None,
    *,
    default_always_on: bool,
    requires_capabilities: tuple[str, ...],
) -> SkillActivationRules:
    if isinstance(value, SkillActivationRules):
        return value
    values = dict(value or {})
    return SkillActivationRules(
        requires_capabilities=tuple(
            values.pop("requires_capabilities", requires_capabilities)
        ),
        always_on=bool(values.pop("always_on", default_always_on)),
        **values,
    )


def _coerce_runtime_effects(
    value: SkillRuntimeEffects | Mapping[str, Any] | None,
    *,
    skill_id: str,
    tool_view_names: tuple[str, ...],
) -> SkillRuntimeEffects:
    if isinstance(value, SkillRuntimeEffects):
        if value.skill_id == skill_id:
            return value
        return replace(value, skill_id=skill_id)
    values = dict(value or {})
    return SkillRuntimeEffects(
        skill_id=skill_id,
        tool_view_names=tuple(values.pop("tool_view_names", tool_view_names)),
        **values,
    )
