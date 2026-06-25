"""Runtime selection and effect contracts for Daita skills."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import os
from typing import Any, Mapping

from daita.core.exceptions import SkillError
from daita.plugins.manifest import PluginKind
from daita.runtime import ContextBlock

_CONTEXT_MODES = frozenset({"none", "always", "on_demand", "runtime_only"})


def _tuple_strings(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    items = tuple(values)
    for value in items:
        if not isinstance(value, str):
            raise TypeError("skill runtime values must be strings")
    return items


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError("skill runtime mappings must be JSON serializable") from exc
    return copied


@dataclass(frozen=True)
class SkillDiscovery:
    """Compact skill card used for run-time selection."""

    name: str
    description: str
    domains: tuple[str, ...] = ()
    when_to_use: tuple[str, ...] = ()
    runtime_kinds: tuple[str, ...] = ("chat", "db")
    requires_capabilities: tuple[str, ...] = ()
    provides_capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    risk: str = "low"
    context_mode: str = "on_demand"

    def __post_init__(self) -> None:
        if self.context_mode not in _CONTEXT_MODES:
            raise ValueError(f"unknown skill context_mode: {self.context_mode!r}")
        object.__setattr__(self, "domains", _tuple_strings(self.domains))
        object.__setattr__(self, "when_to_use", _tuple_strings(self.when_to_use))
        object.__setattr__(self, "runtime_kinds", _tuple_strings(self.runtime_kinds))
        object.__setattr__(
            self,
            "requires_capabilities",
            _tuple_strings(self.requires_capabilities),
        )
        object.__setattr__(
            self,
            "provides_capabilities",
            _tuple_strings(self.provides_capabilities),
        )
        object.__setattr__(self, "tags", _tuple_strings(self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domains": list(self.domains),
            "when_to_use": list(self.when_to_use),
            "runtime_kinds": list(self.runtime_kinds),
            "requires_capabilities": list(self.requires_capabilities),
            "provides_capabilities": list(self.provides_capabilities),
            "tags": list(self.tags),
            "risk": self.risk,
            "context_mode": self.context_mode,
        }


@dataclass(frozen=True)
class SkillActivationRules:
    """Eligibility rules evaluated before a runtime run starts."""

    runtime_kinds: tuple[str, ...] = ("chat", "db")
    modes: tuple[str, ...] = ()
    allow_prompt_match: bool = True
    always_on: bool = False
    requires_capabilities: tuple[str, ...] = ()
    requires_config: tuple[str, ...] = ()
    requires_env: tuple[str, ...] = ()
    requires_packages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_kinds", _tuple_strings(self.runtime_kinds))
        object.__setattr__(self, "modes", _tuple_strings(self.modes))
        object.__setattr__(
            self,
            "requires_capabilities",
            _tuple_strings(self.requires_capabilities),
        )
        object.__setattr__(
            self, "requires_config", _tuple_strings(self.requires_config)
        )
        object.__setattr__(self, "requires_env", _tuple_strings(self.requires_env))
        object.__setattr__(
            self, "requires_packages", _tuple_strings(self.requires_packages)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_kinds": list(self.runtime_kinds),
            "modes": list(self.modes),
            "allow_prompt_match": self.allow_prompt_match,
            "always_on": self.always_on,
            "requires_capabilities": list(self.requires_capabilities),
            "requires_config": list(self.requires_config),
            "requires_env": list(self.requires_env),
            "requires_packages": list(self.requires_packages),
        }


@dataclass(frozen=True)
class SkillRuntimeEffects:
    """Declarative effects that selected skills contribute to runtimes."""

    skill_id: str
    context_blocks: tuple[ContextBlock, ...] = ()
    requested_capabilities: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    policy_ids: tuple[str, ...] = ()
    contract_metadata: dict[str, Any] = field(default_factory=dict)
    verifier_metadata: dict[str, Any] = field(default_factory=dict)
    synthesis_metadata: dict[str, Any] = field(default_factory=dict)
    tool_view_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_blocks", tuple(self.context_blocks))
        for block in self.context_blocks:
            if not isinstance(block, ContextBlock):
                raise TypeError("context_blocks must contain ContextBlock items")
        object.__setattr__(
            self,
            "requested_capabilities",
            _tuple_strings(self.requested_capabilities),
        )
        object.__setattr__(
            self, "required_evidence", _tuple_strings(self.required_evidence)
        )
        object.__setattr__(self, "policy_ids", _tuple_strings(self.policy_ids))
        object.__setattr__(
            self, "contract_metadata", _json_dict(self.contract_metadata)
        )
        object.__setattr__(
            self, "verifier_metadata", _json_dict(self.verifier_metadata)
        )
        object.__setattr__(
            self, "synthesis_metadata", _json_dict(self.synthesis_metadata)
        )
        object.__setattr__(
            self, "tool_view_names", _tuple_strings(self.tool_view_names)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "context_blocks": [block.to_dict() for block in self.context_blocks],
            "requested_capabilities": list(self.requested_capabilities),
            "required_evidence": list(self.required_evidence),
            "policy_ids": list(self.policy_ids),
            "contract_metadata": self.contract_metadata,
            "verifier_metadata": self.verifier_metadata,
            "synthesis_metadata": self.synthesis_metadata,
            "tool_view_names": list(self.tool_view_names),
        }


@dataclass(frozen=True)
class SkillActivation:
    """Resolved activation decision for one skill."""

    skill_id: str
    skill_name: str
    discovery: SkillDiscovery
    rules: SkillActivationRules
    effects: SkillRuntimeEffects
    selected: bool
    reason: str
    context_loaded: bool = False
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "discovery": self.discovery.to_dict(),
            "rules": self.rules.to_dict(),
            "effects": self.effects.to_dict(),
            "selected": self.selected,
            "reason": self.reason,
            "context_loaded": self.context_loaded,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True)
class SkillResolution:
    """Skill resolver output for one runtime run."""

    activations: tuple[SkillActivation, ...]
    skipped: tuple[SkillActivation, ...]

    @property
    def selected(self) -> tuple[SkillActivation, ...]:
        return tuple(item for item in self.activations if item.selected)

    @property
    def effects(self) -> tuple[SkillRuntimeEffects, ...]:
        return tuple(item.effects for item in self.selected)

    def selected_ids(self) -> tuple[str, ...]:
        return tuple(item.skill_id for item in self.selected)

    def selected_names(self) -> tuple[str, ...]:
        return tuple(item.skill_name for item in self.selected)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selected": [item.to_dict() for item in self.selected],
            "skipped": [item.to_dict() for item in self.skipped],
        }


class SkillResolver:
    """Resolve registered SkillPlugin declarations for one runtime run."""

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def skills(self) -> tuple[Any, ...]:
        skills: list[Any] = []
        for plugin_id in self.registry.plugin_ids:
            plugin = self.registry.get_plugin(plugin_id)
            manifest = getattr(plugin, "manifest", None)
            if manifest is not None and manifest.kind is PluginKind.SKILL:
                skills.append(plugin)
        return tuple(skills)

    def compact_catalog(self, *, runtime_kind: str) -> tuple[dict[str, Any], ...]:
        cards = []
        for skill in self.skills():
            discovery = skill.discovery()
            if runtime_kind not in discovery.runtime_kinds:
                continue
            cards.append({"skill_id": skill.manifest.id, **discovery.to_dict()})
        return tuple(cards)

    def resolve(
        self,
        *,
        runtime_kind: str,
        prompt: str,
        request: Any | None = None,
        explicit_skills: tuple[str, ...] | list[str] | set[str] | None = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> SkillResolution:
        context = dict(runtime_context or {})
        explicit = {str(item) for item in (explicit_skills or ())}
        matched_explicit: set[str] = set()
        has_capability_catalog = "available_capabilities" in context
        available_capabilities = set(context.get("available_capabilities") or ())
        mode = str(context.get("mode") or "")
        activations: list[SkillActivation] = []
        skipped: list[SkillActivation] = []

        for skill in self.skills():
            skill_id = skill.manifest.id
            discovery = skill.discovery()
            rules = skill.activation_rules()
            explicit_aliases = _matched_explicit_aliases(skill, discovery, explicit)
            matched_explicit.update(explicit_aliases)
            missing = [
                capability_id
                for capability_id in (
                    *discovery.requires_capabilities,
                    *rules.requires_capabilities,
                )
                if has_capability_catalog
                and capability_id not in available_capabilities
            ]
            missing_config = [
                key
                for key in rules.requires_config
                if not _has_config_key(context, key)
            ]
            missing_env = [
                name for name in rules.requires_env if name not in os.environ
            ]
            missing_packages = [
                package
                for package in rules.requires_packages
                if importlib.util.find_spec(package) is None
            ]
            selected = False
            reason = "not_selected"
            skipped_reason = None

            if runtime_kind not in rules.runtime_kinds or runtime_kind not in (
                discovery.runtime_kinds
            ):
                skipped_reason = "runtime_kind"
            elif rules.modes and mode not in rules.modes:
                skipped_reason = "mode"
            elif missing:
                skipped_reason = "missing_capabilities"
            elif missing_config:
                skipped_reason = "missing_config"
            elif missing_env:
                skipped_reason = "missing_env"
            elif missing_packages:
                skipped_reason = "missing_packages"
            elif explicit_aliases:
                selected = True
                reason = "explicit"
            elif rules.always_on:
                selected = True
                reason = "always_on"
            elif _prompt_matches(prompt, discovery, rules):
                selected = True
                reason = "prompt_match"

            effects = (
                skill.runtime_effects(
                    request,
                    {
                        **context,
                        "prompt": prompt,
                        "runtime_kind": runtime_kind,
                        "selected": selected,
                        "selection_reason": reason,
                    },
                )
                if selected
                else SkillRuntimeEffects(skill_id=skill_id)
            )
            context_loaded = selected and _loads_context(discovery, effects)
            activation = SkillActivation(
                skill_id=skill_id,
                skill_name=discovery.name or getattr(skill, "name", skill_id),
                discovery=discovery,
                rules=rules,
                effects=effects,
                selected=selected,
                reason=reason,
                context_loaded=context_loaded,
                skipped_reason=skipped_reason,
            )
            if skipped_reason is not None:
                skipped.append(activation)
            else:
                activations.append(activation)

        unmatched = explicit - matched_explicit
        if unmatched:
            raise SkillError(
                "Unknown skill selection(s): " + ", ".join(sorted(unmatched))
            )

        return SkillResolution(tuple(activations), tuple(skipped))


def _matched_explicit_aliases(
    skill: Any,
    discovery: SkillDiscovery,
    explicit: set[str],
) -> set[str]:
    if not explicit:
        return set()
    aliases = {
        str(getattr(skill.manifest, "id", "")),
        str(discovery.name),
        str(getattr(skill, "name", "")),
    }
    return aliases & explicit


def _has_config_key(context: Mapping[str, Any], key: str) -> bool:
    for value in (
        context.get("config"),
        context.get("runtime_config"),
        context.get("request_metadata"),
    ):
        if isinstance(value, Mapping) and key in value:
            return True
    return key in context


def _prompt_matches(
    prompt: str,
    discovery: SkillDiscovery,
    rules: SkillActivationRules,
) -> bool:
    if not rules.allow_prompt_match:
        return False
    if discovery.context_mode == "on_demand":
        return False
    haystack = prompt.lower()
    needles = (
        discovery.name,
        *discovery.domains,
        *discovery.when_to_use,
        *discovery.tags,
    )
    return any(str(item).lower() in haystack for item in needles if item)


def _loads_context(
    discovery: SkillDiscovery,
    effects: SkillRuntimeEffects,
) -> bool:
    if effects.context_blocks:
        return True
    return discovery.context_mode in {"always", "on_demand"}
