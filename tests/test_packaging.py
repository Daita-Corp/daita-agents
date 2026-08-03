from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any
from unittest.mock import patch

import pytest

from daita.adapters import postgresql
from daita.artifacts import renderers
from daita.domains.data import sql
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.security import KeychainSecretProvider
from daita import terminal_selection, terminal_tui

ROOT = Path(__file__).parents[1]
PIPX_REPAIR = "pipx reinstall daita-agents"


def _project_metadata() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as source:
        return tomllib.load(source)["project"]


def test_default_distribution_contains_every_supported_production_dependency():
    project = _project_metadata()

    assert project["version"] == "1.0.0"
    assert set(project["dependencies"]) == {
        "anthropic>=0.116.0",
        "asyncpg>=0.30.0",
        "google-genai>=1.73.1",
        "keyring>=25.0.0",
        "openai>=1.99.9",
        "prompt-toolkit>=3.0.52,<4.0.0",
        "rich>=15.0.0,<16.0.0",
        "sqlglot>=25.0.0",
        "XlsxWriter>=3.2.5,<4.0.0",
    }
    assert set(project["optional-dependencies"]) == {"dev"}
    assert project["scripts"] == {"daita": "daita.cli:main"}


def test_customer_and_fixture_documentation_use_the_complete_pipx_install():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    fixture = (ROOT / "tests/fixtures/postgresql/README.md").read_text(encoding="utf-8")
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "pipx install daita-agents" in readme
    assert "pipx upgrade daita-agents" in readme
    assert PIPX_REPAIR in readme
    assert "pipx uninstall daita-agents" in readme
    assert "Version 1.0.0 establishes the first supported agent home format" in readme
    assert "immediately preceding release" in readme
    assert "agent home was preserved" in readme
    assert "supported downgrade path" in readme
    assert "cp -a ~/.daita ~/.daita-backup-before-upgrade" in readme
    assert "\ndaita\n" in readme
    assert "## Advanced/headless CLI" in readme
    assert "pipx install daita-agents" in fixture
    assert "\ndaita --root /private/tmp/daita-live" in fixture
    assert "↑/↓" in fixture
    assert "Space" in fixture
    assert "Enter" in fixture
    assert "Escape" in fixture
    assert "live provider call" in fixture
    assert "developer-operated" in fixture
    assert "tests/pipx_lifecycle_smoke.py" in readme
    assert "first launch" in readme
    assert "returning launch" in readme
    assert "default production dependencies" in instructions
    assert "`prompt-toolkit`" in instructions
    assert "`rich`" in instructions
    assert "imported lazily" in instructions

    for document in (readme, fixture, instructions):
        assert "daita-agents[" not in document


def test_release_smoke_is_isolated_and_covers_the_complete_pipx_lifecycle():
    smoke = (ROOT / "tests/pipx_lifecycle_smoke.py").read_text(encoding="utf-8")

    assert "python -m build" in smoke
    assert "PIPX_HOME" in smoke
    assert "PIPX_BIN_DIR" in smoke
    assert "daita --help" in smoke
    assert "daita.cli:main" in smoke
    assert "pipx install" in smoke
    assert "pipx reinstall" in smoke
    assert "pipx uninstall" in smoke
    assert "--baseline-wheel" in smoke
    assert "--candidate-wheel" in smoke
    assert "cross-version smoke requires distinct baseline" in smoke
    assert "candidate wheels" in smoke
    assert "force-installs the candidate" in smoke
    assert '"create",' in smoke
    assert '"preservation-agent"' in smoke
    assert '"agent.toml"' in smoke
    assert '"state.db"' in smoke
    assert '"config.json"' in smoke
    assert '"MEMORY.md"' in smoke
    assert '"USER.md"' in smoke
    assert '"SKILL.md"' in smoke
    assert "_sha256" in smoke
    assert "_home_hashes" in smoke
    assert "candidate_projection != baseline_projection" in smoke
    assert '"Append after upgrade."' in smoke
    assert "pipx upgrade daita-agents" not in smoke
    assert "pip install" not in smoke
    assert "twine" not in smoke
    assert "publish" not in smoke
    assert "$HOME" not in smoke
    assert "~/" not in smoke


def _missing_import(module: str, action: Callable[[], object]) -> ImportError:
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name.split(".")[0] == module:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=guarded_import):
        with pytest.raises(ImportError) as caught:
            action()
    return caught.value


@pytest.mark.parametrize(
    ("module", "action"),
    (
        ("openai", lambda: OpenAIResponsesProvider("test-model").client),
        (
            "openai",
            lambda: OpenAICompatibleProvider(
                "test-model",
                provider="custom",
                base_url="https://models.example.test/v1",
            ).client,
        ),
        ("anthropic", lambda: AnthropicMessagesProvider("test-model").client),
        ("google", lambda: GeminiProvider("test-model").client),
        ("keyring", lambda: KeychainSecretProvider().client),
        ("sqlglot", lambda: sql._load_sqlglot("sqlite")),
        ("sqlglot", lambda: sql._load_sqlglot("postgresql")),
        ("prompt_toolkit", terminal_selection._load_prompt_toolkit),
        ("prompt_toolkit", terminal_tui._load_terminal_runtime),
        ("rich", terminal_tui._load_terminal_runtime),
    ),
)
def test_missing_default_runtime_dependencies_use_pipx_repair_guidance(
    module: str,
    action: Callable[[], object],
):
    error = _missing_import(module, action)

    assert PIPX_REPAIR in str(error)
    assert "daita-agents[" not in str(error)


def test_missing_asyncpg_uses_pipx_repair_guidance():
    with patch.object(postgresql, "import_module", side_effect=ImportError):
        with pytest.raises(ImportError) as caught:
            postgresql._load_asyncpg()

    assert PIPX_REPAIR in str(caught.value)
    assert "daita-agents[" not in str(caught.value)


def test_missing_xlsxwriter_uses_exact_pipx_repair_guidance():
    with patch.object(renderers, "import_module", side_effect=ImportError):
        with pytest.raises(ImportError) as caught:
            renderers._load_xlsxwriter()

    assert PIPX_REPAIR in str(caught.value)
    assert "daita-agents[" not in str(caught.value)


def test_package_cli_imports_and_headless_command_keep_integrations_lazy(tmp_path):
    script = """
import builtins
import sys

blocked = {
    "anthropic",
    "asyncpg",
    "google",
    "keyring",
    "openai",
    "prompt_toolkit",
    "rich",
    "sqlglot",
    "xlsxwriter",
}
original = builtins.__import__

def guarded(name, *args, **kwargs):
    level = kwargs.get("level", args[3] if len(args) >= 4 else 0)
    if level == 0 and name.split(".")[0] in blocked:
        raise AssertionError(f"eager integration import: {name}")
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
import daita
import daita.cli

raise SystemExit(
    daita.cli.main(["--root", sys.argv[1], "create", "packaging-smoke"])
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"name": "packaging-smoke"' in completed.stdout
