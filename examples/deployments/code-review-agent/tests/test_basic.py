"""
Tests for the code review agent and its skills.

Unit tests run without API keys. Integration tests require OPENAI_API_KEY.
"""

import os
import sys

import pytest

# Add project root so imports match the run.py layout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample.py")
SAMPLE_CODE = open(SAMPLE_FILE).read()


# ===========================================================================
# Agent creation
# ===========================================================================


class TestAgentCreation:
    def test_creates_agent(self):
        from agents.reviewer import create_agent

        agent = create_agent()
        assert agent.name == "Code Reviewer"

    def test_has_skills(self):
        from agents.reviewer import create_agent

        agent = create_agent()
        skill_names = [s.name for s in agent.skills]
        assert "security_review" in skill_names
        assert "code_quality" in skill_names

    def test_has_read_file_tool(self):
        from agents.reviewer import create_agent

        agent = create_agent()
        assert "read_file" in agent.tool_names

    def test_has_skill_tools(self):
        from agents.reviewer import create_agent

        agent = create_agent()
        names = agent.tool_names
        assert "scan_security_patterns" in names
        assert "check_input_validation" in names
        assert "analyze_complexity" in names
        assert "check_naming_conventions" in names


# ===========================================================================
# SecurityReviewSkill
# ===========================================================================


class TestSecuritySkill:
    def test_skill_loads_instructions_from_file(self):
        from skills.security import SecurityReviewSkill

        skill = SecurityReviewSkill()
        instructions = skill.get_instructions()
        assert instructions is not None
        assert "severity classification" in instructions.lower()

    def test_skill_provides_tools(self):
        from skills.security import SecurityReviewSkill

        skill = SecurityReviewSkill()
        tool_names = [t.name for t in skill.get_tools()]
        assert "scan_security_patterns" in tool_names
        assert "check_input_validation" in tool_names

    @pytest.mark.asyncio
    async def test_scan_detects_sql_injection(self):
        from skills.security import scan_security_patterns

        result = await scan_security_patterns.execute({"code": SAMPLE_CODE})
        types = [f["type"] for f in result["findings"]]
        assert "sql_injection" in types

    @pytest.mark.asyncio
    async def test_scan_detects_hardcoded_secrets(self):
        from skills.security import scan_security_patterns

        result = await scan_security_patterns.execute({"code": SAMPLE_CODE})
        types = [f["type"] for f in result["findings"]]
        assert "hardcoded_secrets" in types

    @pytest.mark.asyncio
    async def test_scan_detects_command_injection(self):
        from skills.security import scan_security_patterns

        result = await scan_security_patterns.execute({"code": SAMPLE_CODE})
        types = [f["type"] for f in result["findings"]]
        assert "command_injection" in types

    @pytest.mark.asyncio
    async def test_scan_detects_insecure_deserialization(self):
        from skills.security import scan_security_patterns

        result = await scan_security_patterns.execute({"code": SAMPLE_CODE})
        types = [f["type"] for f in result["findings"]]
        assert "insecure_deserialization" in types

    @pytest.mark.asyncio
    async def test_scan_clean_code_returns_empty(self):
        from skills.security import scan_security_patterns

        clean = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = await scan_security_patterns.execute({"code": clean})
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_check_input_validation_finds_unvalidated_params(self):
        from skills.security import check_input_validation

        result = await check_input_validation.execute({"code": SAMPLE_CODE})
        func_names = [f["function"] for f in result["functions_without_validation"]]
        assert "getUserData" in func_names

    @pytest.mark.asyncio
    async def test_check_input_validation_handles_syntax_error(self):
        from skills.security import check_input_validation

        result = await check_input_validation.execute({"code": "def bad(:"})
        assert "error" in result


# ===========================================================================
# Code quality skill
# ===========================================================================


class TestCodeQualitySkill:
    def test_skill_has_inline_instructions(self):
        from skills.code_quality import create_code_quality_skill

        skill = create_code_quality_skill()
        instructions = skill.get_instructions()
        assert "complexity" in instructions.lower()
        assert "naming" in instructions.lower()

    def test_skill_provides_tools(self):
        from skills.code_quality import create_code_quality_skill

        skill = create_code_quality_skill()
        tool_names = [t.name for t in skill.get_tools()]
        assert "analyze_complexity" in tool_names
        assert "check_naming_conventions" in tool_names

    @pytest.mark.asyncio
    async def test_complexity_analysis(self):
        from skills.code_quality import analyze_complexity

        result = await analyze_complexity.execute({"code": SAMPLE_CODE})
        assert result["total_functions"] > 0

        # calculate_discount should be flagged as complex
        funcs = {f["name"]: f for f in result["functions"]}
        assert "calculate_discount" in funcs
        assert funcs["calculate_discount"]["flags"]["too_complex"]

    @pytest.mark.asyncio
    async def test_complexity_handles_syntax_error(self):
        from skills.code_quality import analyze_complexity

        result = await analyze_complexity.execute({"code": "def bad(:"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_naming_conventions_detects_camelcase_function(self):
        from skills.code_quality import check_naming_conventions

        result = await check_naming_conventions.execute({"code": SAMPLE_CODE})
        names = [v["name"] for v in result["violations"]]
        assert "getUserData" in names

    @pytest.mark.asyncio
    async def test_naming_conventions_detects_bad_class_name(self):
        from skills.code_quality import check_naming_conventions

        result = await check_naming_conventions.execute({"code": SAMPLE_CODE})
        names = [v["name"] for v in result["violations"]]
        assert "userData" in names

    @pytest.mark.asyncio
    async def test_naming_conventions_clean_code(self):
        from skills.code_quality import check_naming_conventions

        clean = "class MyClass:\n    def my_method(self):\n        pass\n"
        result = await check_naming_conventions.execute({"code": clean})
        assert result["total"] == 0


# ===========================================================================
# read_file tool
# ===========================================================================


class TestReadFile:
    @pytest.mark.asyncio
    async def test_reads_sample_file(self):
        from agents.reviewer import read_file

        result = await read_file.execute({"file_path": SAMPLE_FILE})
        assert "error" not in result
        assert result["line_count"] > 0
        assert "getUserData" in result["content"]

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self):
        from agents.reviewer import read_file

        result = await read_file.execute({"file_path": "/nonexistent/file.py"})
        assert "error" in result


# ===========================================================================
# Integration — requires OPENAI_API_KEY
# ===========================================================================


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_review(self):
        from agents.reviewer import create_agent

        agent = create_agent()
        await agent.start()
        try:
            result = await agent.run(
                f"Review the Python file at: {SAMPLE_FILE}",
                detailed=True,
            )
            answer = result.get("result", "")
            assert len(answer) > 100
            # Should mention at least one security issue
            assert any(
                term in answer.lower()
                for term in ("sql injection", "injection", "security", "vulnerability")
            )
        finally:
            await agent.stop()
