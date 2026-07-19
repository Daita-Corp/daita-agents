from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

NEXT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = NEXT_ROOT / "src"
PROVIDER = SOURCE_ROOT / "daita" / "llm" / "providers" / "openai.py"


def _imports_module(node: ast.Import | ast.ImportFrom, module: str) -> bool:
    if isinstance(node, ast.Import):
        return any(
            alias.name == module or alias.name.startswith(f"{module}.")
            for alias in node.names
        )
    return node.module == module or (
        node.module is not None and node.module.startswith(f"{module}.")
    )


def _nested_imports(
    node: ast.AST,
) -> tuple[ast.Import | ast.ImportFrom, ...]:
    return tuple(
        child
        for child in ast.walk(node)
        if isinstance(child, (ast.Import, ast.ImportFrom))
    )


def test_openai_sdk_is_optional_and_never_a_core_dependency() -> None:
    configuration = tomllib.loads(
        (NEXT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert configuration["project"]["dependencies"] == []
    assert configuration["project"]["optional-dependencies"]["openai"] == [
        "openai>=1.99.9"
    ]


def test_openai_sdk_import_is_confined_to_runtime_methods() -> None:
    tree = ast.parse(PROVIDER.read_text(encoding="utf-8"), filename=str(PROVIDER))
    module_imports = tuple(
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    runtime_methods = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    sdk_imports = tuple(
        imported
        for method in runtime_methods
        for imported in _nested_imports(method)
        if _imports_module(imported, "openai")
    )

    assert not any(_imports_module(node, "openai") for node in module_imports)
    assert sdk_imports, "the optional OpenAI SDK must be imported lazily at runtime"


def test_provider_module_import_succeeds_when_openai_sdk_is_unavailable() -> None:
    script = """
import importlib.abc
import json
from pathlib import Path
import sys

class RejectOpenAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openai" or fullname.startswith("openai."):
            raise ModuleNotFoundError("blocked optional OpenAI SDK")
        return None

sys.meta_path.insert(0, RejectOpenAI())
import daita.llm.providers.openai as provider
print(json.dumps({"module": provider.__name__, "path": str(Path(provider.__file__).resolve())}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=NEXT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "module": "daita.llm.providers.openai",
        "path": str(PROVIDER.resolve()),
    }


def test_openai_adapter_has_no_execution_or_loop_ownership() -> None:
    tree = ast.parse(PROVIDER.read_text(encoding="utf-8"), filename=str(PROVIDER))
    forbidden_import_roots = {
        "agent",
        "capabilities",
        "hosting",
        "loop",
        "operations",
        "runtime",
    }
    forbidden_imports: list[tuple[int, str]] = []
    execution_methods: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            root = node.module.lstrip(".").split(".", maxsplit=1)[0]
            if root in forbidden_import_roots:
                forbidden_imports.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root in forbidden_import_roots:
                    forbidden_imports.append((node.lineno, alias.name))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "execute",
            "submit",
            "resume",
            "run",
        }:
            execution_methods.append((node.lineno, node.name))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"execute", "submit", "resume", "run"}
        ):
            execution_methods.append((node.lineno, node.func.attr))

    assert forbidden_imports == []
    assert execution_methods == []
