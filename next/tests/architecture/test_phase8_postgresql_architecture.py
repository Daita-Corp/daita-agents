from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "daita"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return tuple(modules)


def test_generic_loop_has_no_postgresql_or_sql_parser_branch() -> None:
    for path in sorted((SOURCE_ROOT / "loop").glob("*.py")):
        imports = tuple(item.casefold() for item in _imports(path))
        source = path.read_text(encoding="utf-8").casefold()
        assert not any("postgres" in item or "sqlglot" in item for item in imports)
        assert "postgresql" not in source


def test_asyncpg_is_lazy_and_confined_to_the_adapter_boundary() -> None:
    importers = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if any("asyncpg" in item.casefold() for item in _imports(path)):
            importers.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert importers == []
    source = (SOURCE_ROOT / "adapters" / "postgresql.py").read_text(encoding="utf-8")
    assert 'import_module("asyncpg")' in source


def test_postgresql_adapter_exposes_only_control_plane_source_lifecycle() -> None:
    tree = ast.parse(
        (SOURCE_ROOT / "adapters" / "postgresql.py").read_text(encoding="utf-8")
    )
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PostgreSQLResourceAdapter"
    )
    methods = {
        node.name
        for node in adapter.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"discover", "inspect", "health", "declarations", "close"} <= methods
    assert not {"execute", "execute_read", "query"} & methods
