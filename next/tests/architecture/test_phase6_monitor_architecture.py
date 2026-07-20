from __future__ import annotations

import ast
from pathlib import Path

MONITORS = Path(__file__).parents[2] / "src" / "daita" / "monitors"


def _modules(tree: ast.AST) -> tuple[str, ...]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def _attribute_parts(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_attribute_parts(node.value), node.attr)
    return ()


def test_monitor_package_has_no_alternate_execution_or_provider_path() -> None:
    forbidden_imports = {
        "adapters",
        "hosting",
        "llm",
        "loop",
        "operations.runtime",
        "storage",
    }
    import_violations: list[tuple[str, str]] = []
    execute_violations: list[tuple[str, int]] = []

    for path in sorted(MONITORS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _modules(tree):
            normalized = module.lstrip(".")
            if path.name in {"scheduler.py", "service.py"} and normalized == (
                "loop.models"
            ):
                # Monitors restrict the canonical LoopBudgets value only; the
                # host remains the runner and sole background-work owner.
                continue
            if any(
                normalized == forbidden or normalized.startswith(forbidden + ".")
                for forbidden in forbidden_imports
            ):
                import_violations.append((path.name, module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _attribute_parts(node.func)[-1:] == (
                "execute",
            ):
                execute_violations.append((path.name, node.lineno))

    assert import_violations == []
    assert execute_violations == []


def test_monitor_package_does_not_start_background_work() -> None:
    for name in ("models.py", "store.py", "service.py", "scheduler.py"):
        tree = ast.parse(
            (MONITORS / name).read_text(encoding="utf-8"),
            filename=name,
        )
        assert not any(
            isinstance(node, ast.Call)
            and _attribute_parts(node.func)[-1:]
            in {("create_task",), ("ensure_future",)}
            for node in ast.walk(tree)
        )
