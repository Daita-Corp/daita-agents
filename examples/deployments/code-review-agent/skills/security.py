"""
Security review skill — BaseSkill subclass with file-based instructions.

Demonstrates the subclass pattern: instructions live in a separate .md file
(editable by prompt engineers), while callable behavior is exposed through
skill-owned capabilities, executors, and ToolViews.
"""

import re
from typing import Any, Dict

from daita.core.tools import tool
from daita.runtime import AccessMode, Capability, Evidence, RiskLevel, ToolView
from daita.skills import BaseSkill

# -- Vulnerability patterns (regex) ------------------------------------------

SECURITY_PATTERNS = {
    "sql_injection": {
        "patterns": [
            r"""f['\"].*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\b.*\{""",
            r"""(?:execute|cursor\.execute)\s*\(\s*f['\"]""",
            r"""\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)""",
            r"""%s.*%\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)""",
        ],
        "severity": "critical",
        "description": "Possible SQL injection via string interpolation",
    },
    "command_injection": {
        "patterns": [
            r"""os\.system\s*\(.*f['\"]""",
            r"""subprocess\.(?:call|run|Popen)\s*\(.*(?:shell\s*=\s*True)""",
            r"""os\.popen\s*\(""",
        ],
        "severity": "critical",
        "description": "Possible command injection",
    },
    "hardcoded_secrets": {
        "patterns": [
            r"""(?:password|secret|api_key|token)\s*=\s*['\"][^'\"]{8,}['\"]""",
            r"""(?:AWS_SECRET|PRIVATE_KEY)\s*=\s*['\"]""",
        ],
        "severity": "high",
        "description": "Hardcoded secret or credential",
    },
    "insecure_deserialization": {
        "patterns": [
            r"""pickle\.loads?\s*\(""",
            r"""yaml\.(?:load|unsafe_load)\s*\((?!.*Loader)""",
            r"""marshal\.loads?\s*\(""",
        ],
        "severity": "high",
        "description": "Insecure deserialization — untrusted data may execute code",
    },
    "path_traversal": {
        "patterns": [
            r"""open\s*\(.*(?:request|user|input|param)""",
            r"""os\.path\.join\s*\(.*(?:request|user|input|param)""",
        ],
        "severity": "high",
        "description": "Potential path traversal from user-controlled input",
    },
    "ssrf": {
        "patterns": [
            r"""requests\.(?:get|post|put|delete)\s*\(.*(?:request|user|input|param|url)""",
            r"""urllib\.request\.urlopen\s*\(.*(?:request|user|input|param)""",
        ],
        "severity": "high",
        "description": "Potential SSRF — URL derived from user input",
    },
}


class _SecurityToolExecutor:
    def __init__(self, executor_id: str, capability_id: str, tool_ref):
        self.id = executor_id
        self.capability_ids = frozenset({capability_id})
        self._tool = tool_ref

    async def execute(self, task, operation, context):
        result = await self._tool.execute(task.input)
        return [
            Evidence(
                kind="skill.tool_result",
                owner="skill_security_review",
                payload={"tool": self._tool.name, "result": result},
                operation_id=operation.id,
                task_id=task.id,
                metadata={"capability_id": task.capability_id},
            )
        ]


def _security_capability(capability_id: str, executor_id: str, tool_ref) -> Capability:
    return Capability(
        id=capability_id,
        owner="skill_security_review",
        description=tool_ref.description,
        domains=frozenset({"agent", "code_review", "security"}),
        operation_types=frozenset({"code.review.security"}),
        access=AccessMode.READ,
        risk=RiskLevel.LOW,
        input_schema=tool_ref.parameters,
        output_evidence=frozenset({"skill.tool_result"}),
        executor=executor_id,
        model_visible=True,
        side_effecting=False,
    )


def _security_tool_view(tool_ref, capability_id: str) -> ToolView:
    return ToolView(
        name=tool_ref.name,
        capability_id=capability_id,
        description=tool_ref.description,
        parameters=tool_ref.parameters,
    )


class SecurityReviewSkill(BaseSkill):
    """Scans Python code for common security vulnerabilities.

    Uses regex pattern matching to detect mechanical issues, then lets the
    LLM reason about semantic vulnerabilities that patterns can't catch.
    """

    name = "security_review"
    description = "Detect security vulnerabilities in Python code"
    version = "1.0.0"
    instructions_file = "prompts/security_review.md"

    def declare_capabilities(self):
        return (
            _security_capability(
                "skill.security.scan_patterns",
                "skill_security_review.scan_patterns",
                scan_security_patterns,
            ),
            _security_capability(
                "skill.security.check_input_validation",
                "skill_security_review.check_input_validation",
                check_input_validation,
            ),
        )

    def get_executors(self):
        return (
            _SecurityToolExecutor(
                "skill_security_review.scan_patterns",
                "skill.security.scan_patterns",
                scan_security_patterns,
            ),
            _SecurityToolExecutor(
                "skill_security_review.check_input_validation",
                "skill.security.check_input_validation",
                check_input_validation,
            ),
        )

    def get_tool_views(self):
        return (
            _security_tool_view(scan_security_patterns, "skill.security.scan_patterns"),
            _security_tool_view(
                check_input_validation,
                "skill.security.check_input_validation",
            ),
        )


@tool
async def scan_security_patterns(code: str) -> Dict[str, Any]:
    """Scan Python source code for known vulnerability patterns.

    Runs regex-based checks for SQL injection, command injection,
    hardcoded secrets, insecure deserialization, path traversal, and SSRF.

    Args:
        code: Python source code to scan

    Returns:
        Dict with findings list and summary counts by severity
    """
    findings = []
    lines = code.splitlines()

    for vuln_name, vuln_info in SECURITY_PATTERNS.items():
        for pattern in vuln_info["patterns"]:
            regex = re.compile(pattern, re.IGNORECASE)
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    findings.append(
                        {
                            "type": vuln_name,
                            "severity": vuln_info["severity"],
                            "description": vuln_info["description"],
                            "line": i,
                            "code": line.strip(),
                        }
                    )

    severity_counts = {}
    for f in findings:
        severity_counts[f["severity"]] = severity_counts.get(f["severity"], 0) + 1

    return {
        "findings": findings,
        "total": len(findings),
        "by_severity": severity_counts,
    }


@tool
async def check_input_validation(code: str) -> Dict[str, Any]:
    """Identify functions that accept string parameters without validation.

    Looks for function signatures with ``str`` parameters and checks whether
    the function body contains any validation (isinstance, assert, regex
    match, strip, len check, etc.).

    Args:
        code: Python source code to analyse

    Returns:
        Dict with a list of functions lacking input validation
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Could not parse code: {e}"}

    validation_keywords = {
        "isinstance",
        "assert",
        "re.match",
        "re.search",
        "re.fullmatch",
        "validate",
        "sanitize",
        "strip",
        "escape",
        "quote",
        "parameterize",
    }

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        str_params = []
        for arg in node.args.args:
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                if arg.annotation.id == "str":
                    str_params.append(arg.arg)

        if not str_params:
            continue

        body_source = ast.dump(node)
        has_validation = any(kw in body_source for kw in validation_keywords)

        if not has_validation:
            results.append(
                {
                    "function": node.name,
                    "line": node.lineno,
                    "unvalidated_str_params": str_params,
                }
            )

    return {"functions_without_validation": results, "total": len(results)}
