from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "daita"
PROVIDER_ROOT = SOURCE_ROOT / "llm" / "providers"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_provider_adapters_do_not_import_runtime_or_execution_owners() -> None:
    forbidden = {
        "agent",
        "capabilities",
        "domains",
        "loop",
        "operations",
        "runtime",
    }
    violations: list[tuple[str, int, str]] = []
    for path in sorted(PROVIDER_ROOT.glob("*.py")):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            parts = set(node.module.split("."))
            overlap = sorted(parts & forbidden)
            if overlap:
                violations.append((path.name, node.lineno, overlap[0]))

    assert violations == []


def test_optional_provider_sdks_are_not_imported_at_module_scope() -> None:
    optional_roots = {"anthropic", "google", "openai"}
    violations: list[tuple[str, int, str]] = []
    for path in sorted(PROVIDER_ROOT.glob("*.py")):
        for node in _tree(path).body:
            imported: tuple[str, ...]
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
            ):
                imported = (node.module,)
            else:
                continue
            for module in imported:
                if module.split(".", 1)[0] in optional_roots:
                    violations.append((path.name, node.lineno, module))

    assert violations == []


def test_router_depends_only_on_canonical_model_contracts() -> None:
    path = SOURCE_ROOT / "llm" / "routing.py"
    imported_modules = {
        node.module
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module == "providers"
        or module.startswith("providers.")
        or "operations" in module.split(".")
        or "loop" in module.split(".")
        or "domains" in module.split(".")
        for module in imported_modules
    )


def test_generic_loop_has_no_router_or_provider_implementation_branch() -> None:
    path = SOURCE_ROOT / "loop" / "driver.py"
    source = path.read_text(encoding="utf-8")
    imports = {
        node.module
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any("providers" in module or "routing" in module for module in imports)
    assert "ModelRouter" not in source
