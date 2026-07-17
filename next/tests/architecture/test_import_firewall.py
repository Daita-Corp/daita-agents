from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import daita

NEXT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = NEXT_ROOT.parent
V2_SOURCE_ROOT = (NEXT_ROOT / "src").resolve()
V2_PACKAGE_ROOT = (V2_SOURCE_ROOT / "daita").resolve()
V1_PACKAGE_ROOT = (REPOSITORY_ROOT / "daita").resolve()
ARCHITECTURE_PLAN = REPOSITORY_ROOT / "docs" / "DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md"
ARCHITECTURE_PLAN_SHA256 = (
    "403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935"
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _loaded_daita_module_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, module in tuple(sys.modules.items()):
        if name != "daita" and not name.startswith("daita."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file:
            paths[name] = Path(module_file).resolve()
    return paths


def test_pytest_import_resolves_only_to_v2_source() -> None:
    package_file = Path(daita.__file__).resolve()

    assert _is_relative_to(package_file, V2_PACKAGE_ROOT), (
        f"Imported daita from {package_file}; run tests from {NEXT_ROOT} in an "
        "isolated v2 environment"
    )
    assert daita.__version__ == "2.0.0a0"

    loaded_paths = _loaded_daita_module_paths()
    assert loaded_paths
    assert all(_is_relative_to(path, V2_PACKAGE_ROOT) for path in loaded_paths.values())
    assert not any(
        _is_relative_to(path, V1_PACKAGE_ROOT) for path in loaded_paths.values()
    )


def test_site_disabled_subprocess_cannot_resolve_v1() -> None:
    script = """
import json
from pathlib import Path
import sys

import daita

modules = {}
for name, module in tuple(sys.modules.items()):
    if name == "daita" or name.startswith("daita."):
        module_file = getattr(module, "__file__", None)
        modules[name] = str(Path(module_file).resolve()) if module_file else None

print(json.dumps({"package_file": str(Path(daita.__file__).resolve()), "modules": modules}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(V2_SOURCE_ROOT)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=NEXT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    package_file = Path(payload["package_file"]).resolve()
    module_paths = [
        Path(path).resolve() for path in payload["modules"].values() if path is not None
    ]

    assert _is_relative_to(package_file, V2_PACKAGE_ROOT)
    assert module_paths
    assert all(_is_relative_to(path, V2_PACKAGE_ROOT) for path in module_paths)
    assert not any(_is_relative_to(path, V1_PACKAGE_ROOT) for path in module_paths)


def test_v2_production_contains_no_symlinks() -> None:
    symlinks = [path for path in V2_PACKAGE_ROOT.rglob("*") if path.is_symlink()]
    assert symlinks == []


def test_v2_production_uses_no_absolute_self_or_known_v1_imports() -> None:
    violations: list[str] = []
    known_v1_fragments = (
        "daita.agents.chat.runtime",
        "daita.core.",
        "daita.db.",
        "daita.plugins.registry",
        "daita.runtime.kernel",
        "daita.runtime.primitives",
    )

    for source_file in sorted(V2_PACKAGE_ROOT.rglob("*.py")):
        source = source_file.read_text(encoding="utf-8")
        relative_path = source_file.relative_to(NEXT_ROOT)
        tree = ast.parse(source, filename=str(relative_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "daita" or alias.name.startswith("daita."):
                        violations.append(
                            f"{relative_path}:{node.lineno}: absolute import {alias.name}"
                        )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
                and (node.module == "daita" or node.module.startswith("daita."))
            ):
                violations.append(
                    f"{relative_path}:{node.lineno}: absolute import {node.module}"
                )

        for fragment in known_v1_fragments:
            if fragment in source:
                violations.append(f"{relative_path}: known v1 reference {fragment}")

    assert violations == []


def test_root_distribution_configuration_does_not_include_next() -> None:
    root_configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_find = root_configuration["tool"]["setuptools"]["packages"]["find"]
    search_roots = package_find.get("where", ["."])
    include_patterns = package_find.get("include", ["*"])

    assert all(
        not Path(root).parts or Path(root).parts[0] != "next" for root in search_roots
    )
    assert not any(
        fnmatch.fnmatchcase("next.src.daita", pattern) for pattern in include_patterns
    )


def test_local_architecture_plan_matches_recorded_fingerprint() -> None:
    if ARCHITECTURE_PLAN.exists():
        digest = hashlib.sha256(ARCHITECTURE_PLAN.read_bytes()).hexdigest()
        assert digest == ARCHITECTURE_PLAN_SHA256
        return

    status = (NEXT_ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert ARCHITECTURE_PLAN_SHA256 in status
