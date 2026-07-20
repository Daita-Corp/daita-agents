from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

NEXT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = NEXT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "daita"
OPTIONAL_IMPORT_ROOTS = {
    "anthropic",
    "asyncpg",
    "google",
    "keyring",
    "openai",
    "sqlglot",
}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.partition(".")[0])
    return imported


def test_optional_sdks_are_absent_from_every_production_module_top_level() -> None:
    violations = {
        path.relative_to(PACKAGE_ROOT).as_posix(): sorted(
            _top_level_imports(path) & OPTIONAL_IMPORT_ROOTS
        )
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if _top_level_imports(path) & OPTIONAL_IMPORT_ROOTS
    }

    assert violations == {}


def test_minimal_import_succeeds_when_every_optional_sdk_is_unavailable() -> None:
    script = r"""
import importlib.abc
import json
from pathlib import Path
import sys

blocked = {"anthropic", "asyncpg", "google", "keyring", "openai", "sqlglot"}

class RejectOptionalSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise ModuleNotFoundError("blocked optional dependency")
        return None

sys.meta_path.insert(0, RejectOptionalSdk())

import daita
import daita.adapters.postgresql
import daita.domains.data.sql
import daita.llm.providers.anthropic
import daita.llm.providers.gemini
import daita.llm.providers.openai
import daita.llm.providers.openai_compatible
import daita.security.secrets

loaded_optional = sorted(
    name for name in sys.modules if name.partition(".")[0] in blocked
)
print(json.dumps({
    "loaded_optional": loaded_optional,
    "origin": str(Path(daita.__file__).resolve()),
    "version": daita.__version__,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=NEXT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "loaded_optional": [],
        "origin": str((PACKAGE_ROOT / "__init__.py").resolve()),
        "version": "2.0.0a0",
    }
