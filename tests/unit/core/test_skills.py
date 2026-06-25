"""
Unit tests for the skills system (daita/skills/, agent.add_skill()).
"""

import logging
import pytest

from daita.agents.agent import Agent
from daita.core.exceptions import SkillError
from daita.llm.mock import MockLLMProvider
from daita.plugins import (
    ExtensionRegistry,
    PluginKind,
    PluginManifest,
    RuntimeExtensionPlugin,
    SkillPlugin,
)
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    Operation,
    RiskLevel,
    Task,
    ToolView,
)
from daita.skills import (
    SkillActivationRules,
    SkillDiscovery,
    SkillResolver,
    SkillRuntimeEffects,
)
from daita.skills.base import BaseSkill, Skill
from tests.conftest import SequentialMockLLM

# ===========================================================================
# Helpers
# ===========================================================================


class RecordingExecutor:
    def __init__(self, executor_id: str, capability_ids: tuple[str, ...]):
        self.id = executor_id
        self.capability_ids = frozenset(capability_ids)
        self.calls = []

    async def execute(self, task: Task, operation: Operation, context: dict):
        self.calls.append((task, operation, context))
        return [
            Evidence(
                kind="skill.result",
                owner=task.executor_id.split(".")[0],
                payload={"ok": True, "input": task.input},
            )
        ]


def _capability(
    capability_id: str,
    owner: str,
    executor_id: str,
    *,
    description: str = "Run capability.",
) -> Capability:
    return Capability(
        id=capability_id,
        owner=owner,
        description=description,
        domains=frozenset({"agent", "skill"}),
        operation_types=frozenset({"skill.execute"}),
        access=AccessMode.READ,
        risk=RiskLevel.LOW,
        input_schema={"type": "object"},
        output_evidence=frozenset({"skill.result"}),
        executor=executor_id,
        model_visible=True,
        side_effecting=False,
    )


class StubPlugin(RuntimeExtensionPlugin):
    """Minimal plugin for capability dependency resolution tests."""

    manifest = PluginManifest(
        id="stub",
        display_name="Stub",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self):
        self.executor = RecordingExecutor("stub.execute", ("db.sql.execute_read",))

    def declare_capabilities(self):
        return (
            _capability(
                "db.sql.execute_read",
                "stub",
                "stub.execute",
                description="Execute a read query.",
            ),
        )

    def get_executors(self):
        return (self.executor,)


class ManifestDependencyPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="manifest_dependency",
        display_name="Manifest Dependency",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self):
        self.executor = RecordingExecutor(
            "manifest_dependency.execute",
            ("dependency.lookup",),
        )

    def declare_capabilities(self):
        return (
            _capability(
                "dependency.lookup",
                "manifest_dependency",
                "manifest_dependency.execute",
            ),
        )

    def get_executors(self):
        return (self.executor,)


class CountingSkill(BaseSkill):
    name = "finance"
    description = "Finance skill."

    def __init__(
        self,
        *,
        rules: SkillActivationRules | None = None,
        discovery: SkillDiscovery | None = None,
        effects: SkillRuntimeEffects | None = None,
    ):
        super().__init__()
        self.rules = rules or SkillActivationRules(always_on=True)
        self.discovery_card = discovery or SkillDiscovery(
            name=self.name,
            description=self.description,
            context_mode="none",
        )
        self.effects = effects
        self.effect_calls = 0

    def discovery(self):
        return self.discovery_card

    def activation_rules(self):
        return self.rules

    def runtime_effects(self, request=None, runtime_context=None):
        self.effect_calls += 1
        return self.effects or SkillRuntimeEffects(
            skill_id=self.manifest.id,
            contract_metadata={"effect_calls": self.effect_calls},
        )


# ===========================================================================
# BaseSkill basics
# ===========================================================================


class TestBaseSkill:
    def test_default_instructions_returns_none(self):
        skill = BaseSkill()
        assert skill.get_instructions() is None

    def test_inline_instructions(self):
        class MySkill(BaseSkill):
            name = "my"
            instructions = "Do the thing."

        skill = MySkill()
        assert skill.get_instructions() == "Do the thing."

    def test_instructions_file_relative(self, tmp_path):
        # Write a skill module + instructions file in tmp_path
        instructions_content = "Profile all columns carefully."
        prompt_file = tmp_path / "prompts" / "profile.md"
        prompt_file.parent.mkdir()
        prompt_file.write_text(instructions_content)

        # BaseSkill resolves relative paths against the subclass module,
        # so we test via an absolute path here (relative resolution is
        # tested indirectly through Skill factory rejection).
        class FileSkill(BaseSkill):
            name = "file_skill"
            instructions_file = str(prompt_file)

        skill = FileSkill()
        assert skill.get_instructions() == instructions_content

    def test_instructions_file_cached(self, tmp_path):
        prompt_file = tmp_path / "inst.md"
        prompt_file.write_text("v1")

        class CachedSkill(BaseSkill):
            name = "cached"
            instructions_file = str(prompt_file)

        skill = CachedSkill()
        assert skill.get_instructions() == "v1"

        # Overwrite file — cached value should persist
        prompt_file.write_text("v2")
        assert skill.get_instructions() == "v1"

    def test_skill_has_no_local_get_tools_surface(self):
        skill = BaseSkill()
        assert not hasattr(skill, "get_tools")

    def test_requires_capabilities_default_empty(self):
        skill = BaseSkill()
        assert skill.requires_capabilities() == ()
        assert skill.resolved_capabilities == {}

    def test_config_passthrough(self):
        skill = BaseSkill(sample_size=500, mode="fast")
        assert skill.config == {"sample_size": 500, "mode": "fast"}

    def test_skill_uses_extension_contract_not_lifecycle_base(self):
        skill = Skill(name="report_builder", instructions="Use tables.")

        assert isinstance(skill, SkillPlugin)
        assert skill.manifest.id == "skill_report_builder"
        assert skill.manifest.kind is PluginKind.SKILL

    async def test_skill_declares_instruction_context_provider(self):
        skill = Skill(name="report", instructions="Use markdown.")
        provider = skill.get_context_providers()[0]

        block = await provider.render(
            {"prompt": "draft a report"},
            next(iter(provider.audiences)),
            1000,
        )

        assert provider.id == "skill_report.instructions"
        assert block.content == "Use markdown."
        assert block.metadata["context_kind"] == "skill_instructions"
        assert block.metadata["skill_name"] == "report"


# ===========================================================================
# Skill convenience factory
# ===========================================================================


class TestSkillFactory:
    def test_basic_creation(self):
        executor = RecordingExecutor("skill_report.format", ("skill.report.format",))
        capability = _capability(
            "skill.report.format",
            "skill_report",
            "skill_report.format",
        )
        tool_view = ToolView(
            name="format_report",
            capability_id="skill.report.format",
            description="Format a report.",
            parameters={"type": "object"},
        )
        skill = Skill(
            name="report",
            instructions="Use markdown.",
            capabilities=(capability,),
            executors=(executor,),
            tool_views=(tool_view,),
        )
        assert skill.name == "report"
        assert skill.get_instructions() == "Use markdown."
        assert skill.declare_capabilities() == (capability,)
        assert skill.get_executors() == (executor,)
        assert skill.get_tool_views() == (tool_view,)

    def test_runtime_declarations_default_empty(self):
        skill = Skill(name="empty", instructions="Nothing.")
        assert skill.declare_capabilities() == ()
        assert skill.get_executors() == ()
        assert skill.get_tool_views() == ()

    def test_rejects_legacy_tools_keyword(self):
        with pytest.raises(TypeError, match="no longer accepts 'tools'"):
            Skill(name="legacy", tools=[object()])

    def test_rejects_relative_instructions_file(self):
        with pytest.raises(ValueError, match="absolute path"):
            Skill(name="bad", instructions_file="relative/path.md")

    def test_rejects_both_instructions_and_file(self, tmp_path):
        f = tmp_path / "inst.md"
        f.write_text("content")
        with pytest.raises(ValueError, match="not both"):
            Skill(name="bad", instructions="inline", instructions_file=str(f))

    def test_absolute_instructions_file(self, tmp_path):
        f = tmp_path / "inst.md"
        f.write_text("From file.")
        skill = Skill(name="file_skill", instructions_file=str(f))
        assert skill.get_instructions() == "From file."

    def test_version_and_description(self):
        skill = Skill(name="v", description="desc", version="2.0.0")
        assert skill.description == "desc"
        assert skill.version == "2.0.0"

    def test_runtime_effects_are_declarative(self):
        effects = SkillRuntimeEffects(
            skill_id="skill_finance",
            requested_capabilities=("catalog.schema.search",),
            required_evidence=("schema.asset_profile",),
            policy_ids=("skill_finance:aggregate_only",),
            contract_metadata={"planning_hints": {"prefer_aggregate_queries": True}},
            verifier_metadata={"checks": ["sql.validation"]},
            synthesis_metadata={"style": "finance_summary"},
        )
        skill = Skill(
            name="finance",
            runtime_effects=effects,
            activation_rules=SkillActivationRules(always_on=True),
        )

        resolved = skill.runtime_effects()

        assert resolved.skill_id == "skill_finance"
        assert resolved.requested_capabilities == ("catalog.schema.search",)
        assert (
            resolved.contract_metadata["planning_hints"]["prefer_aggregate_queries"]
            is True
        )

    def test_on_demand_discovery_is_compact_by_default(self):
        skill = Skill(
            name="finance",
            description="Finance analysis.",
            instructions="Full finance instructions.",
            context_mode="on_demand",
            discovery=SkillDiscovery(
                name="finance",
                description="Finance analysis.",
                context_mode="on_demand",
                when_to_use=("revenue questions",),
            ),
        )

        assert skill.discovery().context_mode == "on_demand"
        assert skill.activation_rules().always_on is False


# ===========================================================================
# agent.add_skill() — registration and dependency resolution
# ===========================================================================


class TestAddSkill:
    def test_constructor_skills_register_with_extension_registry(self, mock_llm):
        skill = Skill(name="s", instructions="Do stuff.")

        agent = Agent(name="A", llm_provider=mock_llm, skills=[skill])

        assert agent.skills == [skill]
        assert "skill_s" in agent.attached_plugin_ids
        assert "skills" not in agent._llm_kwargs

    def test_constructor_skills_resolve_capabilities_from_extension_registry(
        self, mock_llm
    ):
        class DepSkill(BaseSkill):
            name = "dep"

            def requires_capabilities(self):
                return ("dependency.lookup",)

        plugin = ManifestDependencyPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        skill = DepSkill()
        agent = Agent(
            name="A",
            llm_provider=mock_llm,
            extension_registry=registry,
            skills=[skill],
        )

        assert skill.resolved_capabilities["dependency.lookup"] == (
            plugin.declare_capabilities()[0],
        )

    def test_add_skill_registers_skill(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        skill = Skill(name="s", instructions="Do stuff.")
        agent.add_skill(skill)
        assert skill in agent.skills
        assert "skill_s" in agent.attached_plugin_ids

    def test_skills_property(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        s1 = Skill(name="s1", instructions="a")
        s2 = Skill(name="s2", instructions="b")
        agent.add_skill(s1)
        agent.add_skill(s2)
        assert agent.skills == [s1, s2]

    def test_skills_property_uses_constructor_extension_registry(self, mock_llm):
        registry = ExtensionRegistry()
        skill = Skill(name="s", instructions="Do stuff.")
        registry.register(skill)

        agent = Agent(name="A", llm_provider=mock_llm, extension_registry=registry)

        assert agent.skills == [skill]

    def test_skills_property_excludes_plugins(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_plugin(StubPlugin())
        agent.add_skill(Skill(name="s", instructions="x"))
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "s"

    async def test_skill_owned_tool_view_registers_as_agent_tool(self, mock_llm):
        executor = RecordingExecutor("skill_s.lookup", ("skill.lookup",))
        skill = Skill(
            name="s",
            capabilities=(_capability("skill.lookup", "skill_s", "skill_s.lookup"),),
            executors=(executor,),
            tool_views=(
                ToolView(
                    name="skill_lookup",
                    capability_id="skill.lookup",
                    description="Run skill lookup.",
                    parameters={"type": "object"},
                ),
            ),
        )
        agent = Agent(name="A", llm_provider=mock_llm)

        agent.add_skill(skill)

        tools = agent.available_tools
        assert [tool.name for tool in tools] == ["skill_lookup"]
        result = await agent.call_tool("skill_lookup", {"query": "abc"})
        assert result["capability_id"] == "skill.lookup"
        assert result["evidence"][0]["payload"]["input"] == {"query": "abc"}
        assert executor.calls[0][0].capability_id == "skill.lookup"

    async def test_tool_backed_skill_uses_local_tool_runtime_adapter_semantics(self):
        from daita.core.tools import tool

        calls = {"handler": 0}

        @tool(id="report.format", output_evidence="report.format.result")
        async def format_report(title: str) -> str:
            """Format a report."""
            calls["handler"] += 1
            return f"# {title}"

        skill = Skill.with_tools(
            name="report_builder",
            instructions="Use report tools.",
            tools=[format_report],
        )
        agent = Agent(name="A", llm_provider=SequentialMockLLM(["unused"]))
        agent.add_skill(skill)

        result = await agent.call_tool("format_report", {"title": "Revenue"})

        assert calls["handler"] == 1
        assert agent.get_tool_view_owner("format_report") == "skill_report_builder"
        assert result["capability_id"] == "report.format"
        assert result["evidence"][0]["owner"] == "skill_report_builder"

    def test_capability_requirement_resolution_success(self, mock_llm):
        class DepSkill(BaseSkill):
            name = "dep"

            def requires_capabilities(self):
                return ("db.sql.execute_read",)

        agent = Agent(name="A", llm_provider=mock_llm)
        plugin = StubPlugin()
        agent.add_plugin(plugin)

        skill = DepSkill()
        agent.add_skill(skill)
        assert skill.resolved_capabilities["db.sql.execute_read"] == (
            plugin.declare_capabilities()[0],
        )

    def test_capability_requirement_uses_extension_registry_for_manifest_plugins(
        self, mock_llm
    ):
        class DepSkill(BaseSkill):
            name = "dep"

            def requires_capabilities(self):
                return ("dependency.lookup",)

        agent = Agent(name="A", llm_provider=mock_llm)
        plugin = ManifestDependencyPlugin()
        agent.add_plugin(plugin)

        skill = DepSkill()
        agent.add_skill(skill)

        assert skill.resolved_capabilities["dependency.lookup"] == (
            plugin.declare_capabilities()[0],
        )

    def test_capability_requirement_resolution_failure(self, mock_llm):
        class DepSkill(BaseSkill):
            name = "dep"

            def requires_capabilities(self):
                return ("db.sql.execute_read",)

        agent = Agent(name="A", llm_provider=mock_llm)
        with pytest.raises(SkillError, match="requires capabilities not yet available"):
            agent.add_skill(DepSkill())

    def test_capability_requirement_resolution_partial_failure(self, mock_llm):
        class MultiDepSkill(BaseSkill):
            name = "multi"

            def requires_capabilities(self):
                return ("db.sql.execute_read", "db.sql.execute_write")

        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_plugin(StubPlugin())
        with pytest.raises(SkillError, match="db.sql.execute_write"):
            agent.add_skill(MultiDepSkill())


# ===========================================================================
# SkillResolver activation semantics
# ===========================================================================


class TestSkillResolver:
    def test_unknown_explicit_skill_selection_raises_clear_error(self):
        registry = ExtensionRegistry()
        registry.register(CountingSkill())

        with pytest.raises(SkillError, match="Unknown skill selection\\(s\\): finacne"):
            SkillResolver(registry).resolve(
                runtime_kind="chat",
                prompt="hello",
                explicit_skills=("finacne",),
            )

    def test_valid_explicit_skill_selection_matches_manifest_and_name_aliases(self):
        registry = ExtensionRegistry()
        skill = CountingSkill()
        registry.register(skill)

        by_manifest = SkillResolver(registry).resolve(
            runtime_kind="chat",
            prompt="hello",
            explicit_skills=("skill_finance",),
        )
        by_name = SkillResolver(registry).resolve(
            runtime_kind="chat",
            prompt="hello",
            explicit_skills=("finance",),
        )

        assert by_manifest.selected_ids() == ("skill_finance",)
        assert by_name.selected_ids() == ("skill_finance",)

    def test_empty_capability_catalog_enforces_missing_requirements(self):
        registry = ExtensionRegistry()
        skill = CountingSkill(
            rules=SkillActivationRules(
                always_on=True,
                requires_capabilities=("db.sql.execute_read",),
            ),
            discovery=SkillDiscovery(
                name="finance",
                description="Finance skill.",
                requires_capabilities=("catalog.schema.search",),
                context_mode="none",
            ),
        )
        registry.register(skill)

        resolution = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="revenue",
            runtime_context={"available_capabilities": ()},
        )

        assert resolution.selected == ()
        assert resolution.skipped[0].skipped_reason == "missing_capabilities"
        assert skill.effect_calls == 0

    def test_absent_capability_catalog_does_not_gate_activation(self):
        registry = ExtensionRegistry()
        skill = CountingSkill(
            rules=SkillActivationRules(
                always_on=True,
                requires_capabilities=("db.sql.execute_read",),
            )
        )
        registry.register(skill)

        resolution = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="revenue",
            runtime_context={},
        )

        assert resolution.selected_ids() == ("skill_finance",)
        assert skill.effect_calls == 1

    def test_runtime_effects_run_only_for_selected_skills(self):
        registry = ExtensionRegistry()
        unselected = CountingSkill(
            rules=SkillActivationRules(always_on=False, allow_prompt_match=False)
        )
        registry.register(unselected)

        implicit = SkillResolver(registry).resolve(
            runtime_kind="chat",
            prompt="hello",
        )
        assert implicit.selected == ()
        assert unselected.effect_calls == 0

        explicit = SkillResolver(registry).resolve(
            runtime_kind="chat",
            prompt="hello",
            explicit_skills=("finance",),
        )

        assert unselected.effect_calls == 1
        assert explicit.selected_ids() == ("skill_finance",)

    def test_modes_gate_skill_activation(self):
        registry = ExtensionRegistry()
        skill = CountingSkill(
            rules=SkillActivationRules(always_on=True, modes=("schema.query",))
        )
        registry.register(skill)

        selected = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="schema",
            runtime_context={"mode": "schema.query"},
        )
        skipped = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="schema",
            runtime_context={"mode": "data.query"},
        )

        assert selected.selected_ids() == ("skill_finance",)
        assert skipped.skipped[0].skipped_reason == "mode"

    def test_config_env_and_package_requirements_skip_without_effects(
        self, monkeypatch
    ):
        registry = ExtensionRegistry()
        skill = CountingSkill(
            rules=SkillActivationRules(
                always_on=True,
                requires_config=("tenant_id",),
                requires_env=("DAITA_SKILL_TEST_ENV",),
                requires_packages=("definitely_missing_daita_skill_pkg",),
            )
        )
        registry.register(skill)
        monkeypatch.delenv("DAITA_SKILL_TEST_ENV", raising=False)

        resolution = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="schema",
            runtime_context={"config": {}},
        )

        assert resolution.skipped[0].skipped_reason == "missing_config"
        assert skill.effect_calls == 0

    def test_env_requirement_skip_without_effects(self, monkeypatch):
        registry = ExtensionRegistry()
        skill = CountingSkill(
            rules=SkillActivationRules(
                always_on=True,
                requires_env=("DAITA_SKILL_TEST_ENV",),
            )
        )
        registry.register(skill)
        monkeypatch.delenv("DAITA_SKILL_TEST_ENV", raising=False)

        resolution = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="schema",
        )

        assert resolution.skipped[0].skipped_reason == "missing_env"
        assert skill.effect_calls == 0

    def test_package_requirement_skip_without_effects(self):
        registry = ExtensionRegistry()
        skill = CountingSkill(
            rules=SkillActivationRules(
                always_on=True,
                requires_packages=("definitely_missing_daita_skill_pkg",),
            )
        )
        registry.register(skill)

        resolution = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="schema",
        )

        assert resolution.skipped[0].skipped_reason == "missing_packages"
        assert skill.effect_calls == 0

    def test_package_requirement_uses_find_spec_without_importing(
        self, tmp_path, monkeypatch
    ):
        module = tmp_path / "probe_skill_package.py"
        module.write_text("raise AssertionError('module was imported')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        registry = ExtensionRegistry()
        skill = CountingSkill(
            rules=SkillActivationRules(
                always_on=True,
                requires_packages=("probe_skill_package",),
            )
        )
        registry.register(skill)

        resolution = SkillResolver(registry).resolve(
            runtime_kind="db",
            prompt="schema",
        )

        assert resolution.selected_ids() == ("skill_finance",)
        assert skill.effect_calls == 1


# ===========================================================================
# Structured prompt injection in ChatRuntime._build_initial_conversation
# ===========================================================================


class TestPromptInjection:
    async def test_skill_instructions_in_system_prompt(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm, prompt="Base prompt.")
        agent.add_skill(Skill(name="report", instructions="Use tables."))

        conversation = await agent.runtime._build_initial_conversation("Hello")
        system_msg = conversation[0]["content"]

        assert "## Skills & Expertise" in system_msg
        assert "### report" in system_msg
        assert "Use tables." in system_msg
        assert "Base prompt." in system_msg

    async def test_multiple_skills_grouped(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_skill(Skill(name="alpha", instructions="Alpha instructions."))
        agent.add_skill(Skill(name="beta", instructions="Beta instructions."))

        conversation = await agent.runtime._build_initial_conversation("Hi")
        system_msg = conversation[0]["content"]

        assert "### alpha" in system_msg
        assert "### beta" in system_msg
        # Both under the same section header
        assert system_msg.count("## Skills & Expertise") == 1

    async def test_skill_with_no_instructions_omitted(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_skill(Skill(name="empty"))

        conversation = await agent.runtime._build_initial_conversation("Hi")
        # No system message if no content
        if conversation[0]["role"] == "system":
            assert "## Skills & Expertise" not in conversation[0]["content"]

    async def test_on_demand_skill_omits_full_context_until_explicitly_selected(self):
        skill = Skill(
            name="finance",
            description="Finance analysis.",
            instructions="Full finance instructions.",
            context_mode="on_demand",
            discovery=SkillDiscovery(
                name="finance",
                description="Finance analysis.",
                context_mode="on_demand",
                when_to_use=("revenue questions",),
            ),
        )
        agent = Agent(
            name="A", llm_provider=SequentialMockLLM(["done"]), skills=[skill]
        )

        unselected = await agent.runtime._build_initial_conversation("Hello")
        unselected_system = unselected[0]["content"]
        selected_resolution = agent.runtime._resolve_skills(
            prompt="Hello",
            explicit_skills=("finance",),
        )
        selected = await agent.runtime._build_initial_conversation(
            "Hello",
            skill_resolution=selected_resolution,
        )
        selected_system = selected[0]["content"]

        assert "Full finance instructions." not in unselected_system
        assert "Finance analysis." in unselected_system
        assert "Full finance instructions." in selected_system


# ===========================================================================
# Exception logging fix (no longer silently swallowed)
# ===========================================================================


class TestSkillInstructionExceptionLogging:
    async def test_failing_skill_logs_warning(self, mock_llm, caplog):
        class FailingSkill(BaseSkill):
            name = "bad_skill"

            def get_instructions(self, user_prompt=""):
                raise ValueError("bad instructions")

        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_skill(FailingSkill())

        with caplog.at_level(logging.WARNING):
            conversation = await agent.runtime._build_initial_conversation("Hi")

        assert "bad_skill" in caplog.text
        assert "bad instructions" in caplog.text


# ===========================================================================
# SkillError inherits from PluginError
# ===========================================================================


class TestSkillError:
    def test_inherits_plugin_error(self):
        from daita.core.exceptions import PluginError

        err = SkillError("msg", plugin_name="my_skill")
        assert isinstance(err, PluginError)
        assert err.plugin_name == "my_skill"
