"""
Code review agent — demonstrates the daita-agents skills system.

This agent composes two skills:
- **SecurityReviewSkill** (BaseSkill subclass) — file-based instructions + tools
- **code_quality skill** (Skill factory) — inline instructions + tools

Each skill contributes its own tools and domain-specific instructions.
The agent's system prompt receives all skill instructions under a
structured "## Skills & Expertise" header automatically.
"""

import os
import sys
from typing import Any, Dict

from daita import Agent, tool

# Add project root to path so skill imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.security import SecurityReviewSkill
from skills.code_quality import create_code_quality_skill


@tool
async def read_file(file_path: str) -> Dict[str, Any]:
    """Read a file and return its contents with line numbers.

    Args:
        file_path: Path to the file to read

    Returns:
        Dict with the file contents, line count, and file name
    """
    try:
        with open(file_path) as f:
            content = f.read()
        lines = content.splitlines()
        numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, 1))
        return {
            "file": os.path.basename(file_path),
            "path": file_path,
            "content": content,
            "numbered_content": numbered,
            "line_count": len(lines),
        }
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": str(e)}


def create_agent() -> Agent:
    """Create the code review agent with security and quality skills."""
    agent = Agent(
        name="Code Reviewer",
        model="gpt-4o-mini",
        prompt=(
            "You are a thorough, constructive code reviewer. "
            "When given code or a file path, use your skills to produce a "
            "complete review covering both security and code quality.\n\n"
            "Structure your review as:\n"
            "1. **Summary** — one paragraph overview\n"
            "2. **Security findings** — from your security review skill\n"
            "3. **Quality findings** — from your code quality skill\n"
            "4. **Recommendations** — prioritised action items\n\n"
            "Be specific, cite line numbers, and suggest concrete fixes. "
            "Acknowledge what the code does well — reviews should be balanced."
        ),
        tools=[read_file],
    )

    agent.add_skill(SecurityReviewSkill())
    agent.add_skill(create_code_quality_skill())

    return agent
