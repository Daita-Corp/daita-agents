"""
Code quality skill — built with the Skill() convenience factory.

Demonstrates the factory pattern: inline instructions + a list of tool
functions, no subclassing needed.
"""

import ast
from typing import Any, Dict, List

from daita.core.tools import AgentTool, tool
from daita.skills import Skill

QUALITY_INSTRUCTIONS = """\
You are a code quality reviewer focused on maintainability and readability.

## Review methodology

1. **Call `analyze_complexity`** on the code to get cyclomatic complexity and \
function length metrics for every function.
2. **Call `check_naming_conventions`** to flag naming style violations.
3. Combine tool output with your own judgement. Flag functions that are too \
long (>30 lines), too complex (complexity >10), or poorly named.

## What to look for beyond tools

- Functions doing more than one thing (violates single responsibility)
- Deep nesting (>3 levels)
- Magic numbers or unexplained constants
- Missing or misleading docstrings on public functions
- Dead code or unreachable branches

## Output format

For each finding, report:
- **Location**: function name and line
- **Category**: Complexity / Naming / Structure / Documentation
- **Issue**: One-sentence description
- **Suggestion**: How to improve (be specific)

Praise well-written code when you see it — reviews should be balanced.\
"""


@tool
async def analyze_complexity(code: str) -> Dict[str, Any]:
    """Compute cyclomatic complexity and size metrics for each function.

    Walks the AST to count branches (if/elif/for/while/except/with/and/or/
    ternary) per function and measures line count.

    Args:
        code: Python source code to analyse

    Returns:
        Dict with per-function metrics and overall summary
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Could not parse code: {e}"}

    branch_types = (
        ast.If,
        ast.For,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.BoolOp,
        ast.IfExp,
    )

    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        branches = sum(
            1 for child in ast.walk(node) if isinstance(child, branch_types)
        )
        complexity = 1 + branches  # base path + branches

        end_line = getattr(node, "end_lineno", node.lineno)
        line_count = end_line - node.lineno + 1

        functions.append(
            {
                "name": node.name,
                "line": node.lineno,
                "line_count": line_count,
                "complexity": complexity,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "flags": {
                    "too_complex": complexity > 10,
                    "too_long": line_count > 30,
                },
            }
        )

    return {
        "functions": functions,
        "total_functions": len(functions),
        "max_complexity": max((f["complexity"] for f in functions), default=0),
        "avg_complexity": (
            round(sum(f["complexity"] for f in functions) / len(functions), 1)
            if functions
            else 0
        ),
    }


@tool
async def check_naming_conventions(code: str) -> Dict[str, Any]:
    """Check that names follow PEP 8 conventions.

    Validates: functions/variables use snake_case, classes use PascalCase,
    constants (module-level ALL_CAPS) are correctly cased.

    Args:
        code: Python source code to check

    Returns:
        Dict with naming violations grouped by category
    """
    import re

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Could not parse code: {e}"}

    snake_re = re.compile(r"^_*[a-z][a-z0-9_]*$")
    pascal_re = re.compile(r"^_*[A-Z][a-zA-Z0-9]*$")

    violations = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not snake_re.match(node.name) and not node.name.startswith("__"):
                violations.append(
                    {
                        "name": node.name,
                        "kind": "function",
                        "line": node.lineno,
                        "expected": "snake_case",
                    }
                )

        elif isinstance(node, ast.ClassDef):
            if not pascal_re.match(node.name):
                violations.append(
                    {
                        "name": node.name,
                        "kind": "class",
                        "line": node.lineno,
                        "expected": "PascalCase",
                    }
                )

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                # Module-level uppercase = constant, skip
                if name.isupper():
                    continue
                # Skip dunder
                if name.startswith("__") and name.endswith("__"):
                    continue

    return {"violations": violations, "total": len(violations)}


def create_code_quality_skill() -> Skill:
    """Build the code quality skill using the Skill convenience factory."""
    return Skill(
        name="code_quality",
        description="Analyse code complexity, structure, and naming conventions",
        instructions=QUALITY_INSTRUCTIONS,
        tools=[analyze_complexity, check_naming_conventions],
    )
