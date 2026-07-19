from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "daita"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> tuple[str, ...]:
    modules: list[str] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return tuple(modules)


def test_generic_loop_does_not_own_context_memory_learning_or_skills() -> None:
    forbidden = {
        "context.contributors",
        "context.session",
        "learning",
        "memory",
        "skills",
        "storage.sqlite",
    }
    for path in sorted((SOURCE_ROOT / "loop").glob("*.py")):
        imported = _imports(path)
        assert not {
            fragment
            for fragment in forbidden
            if any(fragment in module for module in imported)
        }, path


def test_context_projection_does_not_import_concrete_state_owners() -> None:
    forbidden = {
        "adapters",
        "catalog.service",
        "memory.service",
        "operations.runtime",
        "skills.service",
        "storage",
    }
    for path in sorted((SOURCE_ROOT / "context").glob("*.py")):
        imported = _imports(path)
        assert not {
            fragment
            for fragment in forbidden
            if any(fragment in module for module in imported)
        }, path


def test_learning_and_skills_have_no_execution_path() -> None:
    paths = [SOURCE_ROOT / "learning.py"]
    paths.extend(sorted((SOURCE_ROOT / "memory").glob("*.py")))
    paths.extend(sorted((SOURCE_ROOT / "skills").glob("*.py")))
    execute_callers: list[str] = []
    for path in paths:
        for node in ast.walk(_tree(path)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
            ):
                execute_callers.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert execute_callers == []


def test_skills_do_not_import_runtime_policy_adapters_or_storage() -> None:
    forbidden = {
        "adapters",
        "operations.governance",
        "operations.runtime",
        "storage",
    }
    for path in sorted((SOURCE_ROOT / "skills").glob("*.py")):
        imported = _imports(path)
        assert not {
            fragment
            for fragment in forbidden
            if any(fragment in module for module in imported)
        }, path
