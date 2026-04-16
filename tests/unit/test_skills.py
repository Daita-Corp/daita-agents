"""
Unit tests for the skills system (daita/skills/, agent.add_skill()).
"""

import logging
import os
import tempfile

import pytest

from daita.agents.agent import Agent
from daita.core.exceptions import SkillError
from daita.core.tools import AgentTool
from daita.llm.mock import MockLLMProvider
from daita.plugins.base import BasePlugin, LifecyclePlugin
from daita.skills.base import BaseSkill, Skill


# ===========================================================================
# Helpers
# ===========================================================================


def _tool(name: str):
    async def h(args):
        return f"result_from_{name}"

    return AgentTool(name=name, description=f"Tool {name}", parameters={}, handler=h)


class StubPlugin(BasePlugin):
    """Minimal plugin for dependency resolution tests."""

    def get_tools(self):
        return [_tool("stub_query")]


class StubDatabasePlugin(BasePlugin):
    """A different plugin type for negative-match tests."""

    pass


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

    def test_get_tools_default_empty(self):
        skill = BaseSkill()
        assert skill.get_tools() == []

    def test_requires_default_empty(self):
        skill = BaseSkill()
        assert skill.requires() == {}

    def test_config_passthrough(self):
        skill = BaseSkill(sample_size=500, mode="fast")
        assert skill.config == {"sample_size": 500, "mode": "fast"}

    async def test_on_before_run_returns_instructions(self):
        class MySkill(BaseSkill):
            name = "my"
            instructions = "Be helpful."

        skill = MySkill()
        result = await skill.on_before_run("any prompt")
        assert result == "Be helpful."


# ===========================================================================
# Skill convenience factory
# ===========================================================================


class TestSkillFactory:
    def test_basic_creation(self):
        t = _tool("fmt")
        skill = Skill(name="report", instructions="Use markdown.", tools=[t])
        assert skill.name == "report"
        assert skill.get_instructions() == "Use markdown."
        assert skill.get_tools() == [t]

    def test_no_tools_defaults_empty(self):
        skill = Skill(name="empty", instructions="Nothing.")
        assert skill.get_tools() == []

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


# ===========================================================================
# agent.add_skill() — registration and dependency resolution
# ===========================================================================


class TestAddSkill:
    def test_skill_appears_in_tool_sources(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        skill = Skill(name="s", instructions="Do stuff.")
        agent.add_skill(skill)
        assert skill in agent.tool_sources

    def test_skills_property(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        s1 = Skill(name="s1", instructions="a")
        s2 = Skill(name="s2", instructions="b")
        agent.add_skill(s1)
        agent.add_skill(s2)
        assert agent.skills == [s1, s2]

    def test_skills_property_excludes_plugins(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_plugin(StubPlugin())
        agent.add_skill(Skill(name="s", instructions="x"))
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "s"

    def test_skill_tools_registered(self, mock_llm):
        t = _tool("skill_tool")
        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_skill(Skill(name="s", tools=[t]))
        assert "skill_tool" in agent.tool_names

    def test_dependency_resolution_success(self, mock_llm):
        class DepSkill(BaseSkill):
            name = "dep"

            def requires(self):
                return {"plug": StubPlugin}

        agent = Agent(name="A", llm_provider=mock_llm)
        plugin = StubPlugin()
        agent.add_plugin(plugin)

        skill = DepSkill()
        agent.add_skill(skill)
        assert skill._resolved_plugins["plug"] is plugin

    def test_dependency_resolution_failure(self, mock_llm):
        class DepSkill(BaseSkill):
            name = "dep"

            def requires(self):
                return {"db": StubDatabasePlugin}

        agent = Agent(name="A", llm_provider=mock_llm)
        with pytest.raises(SkillError, match="requires plugins not yet added"):
            agent.add_skill(DepSkill())

    def test_dependency_resolution_partial_failure(self, mock_llm):
        class MultiDepSkill(BaseSkill):
            name = "multi"

            def requires(self):
                return {"a": StubPlugin, "b": StubDatabasePlugin}

        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_plugin(StubPlugin())
        with pytest.raises(SkillError, match="StubDatabasePlugin"):
            agent.add_skill(MultiDepSkill())


# ===========================================================================
# Structured prompt injection in _build_initial_conversation
# ===========================================================================


class TestPromptInjection:
    async def test_skill_instructions_in_system_prompt(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm, prompt="Base prompt.")
        agent.add_skill(Skill(name="report", instructions="Use tables."))

        conversation = await agent._build_initial_conversation("Hello")
        system_msg = conversation[0]["content"]

        assert "## Skills & Expertise" in system_msg
        assert "### report" in system_msg
        assert "Use tables." in system_msg
        assert "Base prompt." in system_msg

    async def test_multiple_skills_grouped(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_skill(Skill(name="alpha", instructions="Alpha instructions."))
        agent.add_skill(Skill(name="beta", instructions="Beta instructions."))

        conversation = await agent._build_initial_conversation("Hi")
        system_msg = conversation[0]["content"]

        assert "### alpha" in system_msg
        assert "### beta" in system_msg
        # Both under the same section header
        assert system_msg.count("## Skills & Expertise") == 1

    async def test_plugin_context_not_grouped_with_skills(self, mock_llm):
        class ContextPlugin(LifecyclePlugin):
            async def on_before_run(self, prompt):
                return "Plugin context here."

        agent = Agent(name="A", llm_provider=mock_llm, prompt="Base.")
        agent.add_plugin(ContextPlugin())
        agent.add_skill(Skill(name="s", instructions="Skill inst."))

        conversation = await agent._build_initial_conversation("Hi")
        system_msg = conversation[0]["content"]

        # Plugin context should be separate from the skills section
        assert "Plugin context here." in system_msg
        assert "### s\nSkill inst." in system_msg

    async def test_skill_with_no_instructions_omitted(self, mock_llm):
        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_skill(Skill(name="empty"))

        conversation = await agent._build_initial_conversation("Hi")
        # No system message if no content
        if conversation[0]["role"] == "system":
            assert "## Skills & Expertise" not in conversation[0]["content"]


# ===========================================================================
# Exception logging fix (no longer silently swallowed)
# ===========================================================================


class TestLifecycleExceptionLogging:
    async def test_failing_plugin_logs_warning(self, mock_llm, caplog):
        class FailingPlugin(LifecyclePlugin):
            async def on_before_run(self, prompt):
                raise RuntimeError("boom")

        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_plugin(FailingPlugin())

        with caplog.at_level(logging.WARNING):
            conversation = await agent._build_initial_conversation("Hi")

        assert "on_before_run failed" in caplog.text
        assert "boom" in caplog.text

    async def test_failing_skill_logs_warning(self, mock_llm, caplog):
        class FailingSkill(BaseSkill):
            name = "bad_skill"

            def get_instructions(self, user_prompt=""):
                raise ValueError("bad instructions")

        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_skill(FailingSkill())

        with caplog.at_level(logging.WARNING):
            conversation = await agent._build_initial_conversation("Hi")

        assert "bad_skill" in caplog.text
        assert "bad instructions" in caplog.text

    async def test_failing_plugin_does_not_block_others(self, mock_llm):
        class FailPlugin(LifecyclePlugin):
            async def on_before_run(self, prompt):
                raise RuntimeError("fail")

        agent = Agent(name="A", llm_provider=mock_llm)
        agent.add_plugin(FailPlugin())
        agent.add_skill(Skill(name="ok", instructions="Still works."))

        conversation = await agent._build_initial_conversation("Hi")
        system_msg = conversation[0]["content"]
        assert "Still works." in system_msg


# ===========================================================================
# SkillError inherits from PluginError
# ===========================================================================


class TestSkillError:
    def test_inherits_plugin_error(self):
        from daita.core.exceptions import PluginError

        err = SkillError("msg", plugin_name="my_skill")
        assert isinstance(err, PluginError)
        assert err.plugin_name == "my_skill"
