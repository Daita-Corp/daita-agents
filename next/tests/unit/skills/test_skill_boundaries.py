from __future__ import annotations

import ast
from pathlib import Path

SKILLS_ROOT = Path(__file__).parents[3] / "src" / "daita" / "skills"


def test_skills_depend_on_no_runtime_execution_or_policy_owners() -> None:
    forbidden_import_fragments = {
        "adapters",
        "capabilities",
        "governance",
        "loop",
        "operations",
        "policy",
        "runtime",
    }
    imports: list[tuple[str, str]] = []
    execute_calls: list[tuple[str, int]] = []
    for path in sorted(SKILLS_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((path.name, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path.name, node.module))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
            ):
                execute_calls.append((path.name, node.lineno))

    violations = [
        (filename, module)
        for filename, module in imports
        if any(
            fragment in module.casefold().split(".")
            for fragment in forbidden_import_fragments
        )
    ]
    assert violations == []
    assert execute_calls == []


def test_skill_records_have_no_runtime_effect_surfaces() -> None:
    tree = ast.parse(
        (SKILLS_ROOT / "models.py").read_text(encoding="utf-8"),
        filename="models.py",
    )
    forbidden_fields = {
        "executor",
        "executor_id",
        "executors",
        "policies",
        "policy_ids",
        "runtime_effects",
        "tool_views",
        "tools",
        "workers",
    }
    declared_fields: set[str] = set()
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                declared_fields.add(node.target.id)

    assert declared_fields.isdisjoint(forbidden_fields)
