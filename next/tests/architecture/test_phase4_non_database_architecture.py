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


def test_generic_loop_has_no_local_file_or_source_specific_dependency() -> None:
    forbidden_fragments = {
        "adapters.local_files",
        "csv",
        "file_capabilities",
        "pathlib",
        "sqlite",
        "tabular.compare",
    }
    for path in sorted((SOURCE_ROOT / "loop").glob("*.py")):
        imported = _imports(path)
        assert not {
            fragment
            for fragment in forbidden_fragments
            if any(fragment in module.casefold() for module in imported)
        }, path


def test_operation_runtime_remains_the_only_executor_invocation_boundary() -> None:
    callers: list[tuple[str, str]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            if (
                node.func.attr == "execute"
                and isinstance(owner, ast.Name)
                and owner.id == "executor"
            ):
                callers.append((relative, "executor.execute"))
    assert callers == [("operations/runtime.py", "executor.execute")]


def test_resource_adapter_exposes_control_plane_not_file_execution() -> None:
    module = _tree(SOURCE_ROOT / "adapters" / "local_files.py")
    classes = {
        node.name: node for node in module.body if isinstance(node, ast.ClassDef)
    }
    adapter = classes["LocalDirectoryResourceAdapter"]
    methods = {
        node.name for node in adapter.body if isinstance(node, ast.FunctionDef)
    } | {node.name for node in adapter.body if isinstance(node, ast.AsyncFunctionDef)}
    assert {"discover", "inspect", "health", "declarations", "close"} <= methods
    assert not {"execute", "execute_read", "read"} & methods
    backend = classes["LocalDirectoryReadBackend"]
    assert any(
        isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_read"
        for node in backend.body
    )


def test_embedded_composition_binds_one_registry_and_one_blob_store() -> None:
    source = (SOURCE_ROOT / "hosting" / "embedded.py").read_text(encoding="utf-8")
    assert source.count("CapabilityRegistry(") == 2
    assert source.count('LocalBlobStore(home / "blobs")') == 1
    assert "local_file_read_declarations(" in source
    assert "tabular_comparison_declarations(" in source
    assert "PersistedAcceptedEvidenceDatasetReader(" in source


def test_public_surface_exports_the_sandboxed_source_not_its_backend() -> None:
    assignment = next(
        node
        for node in _tree(SOURCE_ROOT / "__init__.py").body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    exports = ast.literal_eval(assignment.value)
    assert isinstance(exports, list)
    assert "LocalDirectorySource" in exports
    assert "LocalDirectoryReadBackend" not in exports
