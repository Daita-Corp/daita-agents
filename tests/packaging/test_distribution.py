from __future__ import annotations

import builtins
import os
import shutil
import site
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daita import __version__, terminal
from daita._installation import (
    MANAGED_REPAIR_GUIDANCE,
    PIPX_REPAIR_GUIDANCE,
    _is_trusted_managed_runtime,
)
from daita.adapters import postgresql
from daita.artifacts import renderers
from daita.domains.data import sql
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.security import KeychainSecretProvider
from scripts.release_identity import read_project_identity
from scripts.render_managed_installer import inspect_wheel
from tests.support.paths import REPO_ROOT

ROOT = REPO_ROOT
PIPX_REPAIR = "pipx reinstall daita-agents"
_CANDIDATE_WHEEL_ENV = "DAITA_TEST_CANDIDATE_WHEEL"


def _project_metadata() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as source:
        return tomllib.load(source)["project"]


def _candidate_wheel(tmp_path: Path) -> Path:
    supplied = os.environ.get(_CANDIDATE_WHEEL_ENV)
    if supplied is not None:
        path = Path(supplied)
        if not path.is_absolute():
            raise AssertionError(f"{_CANDIDATE_WHEEL_ENV} must be an absolute path")
        return path.resolve(strict=True)
    if os.environ.get("CI") == "true":
        raise AssertionError(
            f"CI must supply the producer wheel through {_CANDIDATE_WHEEL_ENV}"
        )

    project = tmp_path / "project"
    project.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, project / filename)
    shutil.copytree(
        ROOT / "src",
        project / "src",
        ignore=shutil.ignore_patterns("*.egg-info", "__pycache__", "*.pyc"),
    )
    distribution = tmp_path / "dist"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(distribution),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert built.returncode == 0, built.stderr
    wheels = tuple(distribution.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_default_distribution_contains_every_supported_production_dependency():
    project = _project_metadata()

    identity = read_project_identity()
    assert project["version"] == identity.version
    assert __version__ == distribution_version("daita-agents") == identity.version
    assert project["requires-python"] == ">=3.11,<3.13"
    assert set(project["dependencies"]) == {
        "anthropic>=0.116.0,<1.0.0",
        "asyncpg>=0.30.0,<1.0.0",
        "google-genai>=2.22.0,<3.0.0",
        "keyring>=25.0.0,<26.0.0",
        "openai>=2.45.0,<3.0.0",
        "rich>=15.0.0,<16.0.0",
        "textual>=8.2.8,<9.0.0",
        "sqlglot>=30.14.0,<30.15.0",
        "XlsxWriter>=3.2.5,<4.0.0",
        "httpx>=0.28.1,<1.0.0",
        "duckdb==1.5.5",
    }
    assert set(project["optional-dependencies"]) == {"dev"}
    assert project["scripts"] == {"daita": "daita.cli:main"}


def test_local_no_isolation_build_requirements_are_in_the_dev_extra() -> None:
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)

    assert set(project["build-system"]["requires"]) <= set(
        project["project"]["optional-dependencies"]["dev"]
    )


def test_import_fails_explicitly_without_installed_distribution_metadata(
    tmp_path: Path,
) -> None:
    isolated_source = tmp_path / "src"
    shutil.copytree(ROOT / "src" / "daita", isolated_source / "daita")
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            f"import sys; sys.path.insert(0, {str(isolated_source)!r}); import daita",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert completed.returncode != 0
    assert "PackageNotFoundError" in completed.stderr
    assert "daita-agents" in completed.stderr


def test_stale_editable_metadata_is_detected_instead_of_hidden(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "daita_agents-9.9.9.dist-info"
    stale.mkdir()
    (stale / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: daita-agents\nVersion: 9.9.9\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(ROOT), str(ROOT / "src"))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.release_identity import read_project_identity; "
            "import daita; project = read_project_identity(); "
            "raise SystemExit(0 if daita.__version__ == project.version else "
            "f'stale installed metadata: {daita.__version__} != {project.version}')",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
    )

    assert completed.returncode != 0
    assert "stale installed metadata: 9.9.9 !=" in completed.stderr


def test_real_wheel_metadata_drives_installed_runtime_cli_tui_and_mcp(
    tmp_path: Path,
) -> None:
    wheel = _candidate_wheel(tmp_path)
    metadata = inspect_wheel(wheel)
    identity = read_project_identity()
    assert metadata.version == identity.version
    assert sorted(metadata.requires_python.split(",")) == sorted(
        identity.requires_python.split(",")
    )
    assert wheel.name == f"daita_agents-{identity.version}-py3-none-any.whl"

    runtime = tmp_path / "runtime"
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(runtime)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0, created.stderr
    python = runtime / "bin" / "python"
    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    runtime_environment = dict(os.environ)
    runtime_environment["PYTHONNOUSERSITE"] = "1"
    runtime_environment["PYTHONPATH"] = site.getsitepackages()[0]
    surfaces = subprocess.run(
        [
            str(python),
            "-c",
            "import daita; "
            "from daita.adapters import mcp; "
            "from daita.tui.widgets import welcome; "
            "print(daita.__version__); print(mcp.__version__); "
            "print(welcome.__version__)",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=runtime_environment,
    )
    assert surfaces.returncode == 0, surfaces.stderr
    assert surfaces.stdout.splitlines() == [metadata.version] * 3
    cli = subprocess.run(
        [str(runtime / "bin" / "daita"), "--version"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert cli.returncode == 0, cli.stderr
    assert cli.stdout == f"daita {metadata.version}\n"


def test_documented_local_markdown_links_resolve() -> None:
    import re
    from urllib.parse import unquote

    documents = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "examples/README.md",
    ]
    documents.extend(sorted((ROOT / "docs").glob("*.md")))
    documents.extend(sorted((ROOT / "tests/fixtures").glob("*/README.md")))
    missing: list[str] = []
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in documents:
        for raw_target in pattern.findall(document.read_text(encoding="utf-8")):
            target = unquote(raw_target.split("#", 1)[0].strip())
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / target).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert missing == []


def test_lifecycle_smoke_modules_are_importable_entry_points() -> None:
    for module in (
        "tests.packaging.pipx_lifecycle_smoke",
        "tests.packaging.managed_installer_lifecycle_smoke",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--candidate-wheel" in completed.stdout


def test_ci_requires_clean_pipx_wheel_smoke_on_each_supported_python():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "release-artifact:" in workflow
    assert "pipx-lifecycle:" in workflow
    assert "managed-lifecycle:" in workflow
    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert "python -m tests.packaging.pipx_lifecycle_smoke" in workflow
    assert "python -m tests.packaging.managed_installer_lifecycle_smoke" in workflow
    assert "not requires_network and not slow" in workflow
    assert "--candidate-wheel" in workflow
    assert workflow.count("python -m build --wheel") == 1
    assert (
        workflow.count(
            "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
        )
        == 3
    )
    assert "DAITA_TEST_CANDIDATE_WHEEL:" in workflow
    assert "needs: release-artifact" in workflow
    lifecycle_jobs = workflow[workflow.index("  pipx-lifecycle:") :]
    assert "python -m build --wheel" not in lifecycle_jobs
    assert "[dev,sqlite]" not in workflow


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
        ("textual", terminal._load_textual_app),
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


def test_managed_repair_guidance_requires_a_verified_runtime_topology(tmp_path: Path):
    home = tmp_path / "home"
    generation = home / ".local" / "share" / "daita" / "generations" / "1.0.0-fixture-1"
    runtime = generation / "tool" / "daita-agents"
    runtime.mkdir(parents=True)
    root = home / ".local" / "share" / "daita"
    state = root / "install-state"
    state.mkdir()
    (state / "owner").write_text(
        "marker=daita-managed-install-v1\n" f"root={root}\n",
        encoding="utf-8",
    )
    (generation / "manifest").write_text(
        "marker=daita-managed-install-v1\n",
        encoding="utf-8",
    )
    (root / "current").symlink_to("generations/1.0.0-fixture-1")
    environment = {"DAITA_MANAGED_INSTALL_ROOT": str(root)}

    assert _is_trusted_managed_runtime(
        environ=environment,
        executable=runtime,
        home=home,
    )
    assert "https://daita-tech.io/install.sh" in MANAGED_REPAIR_GUIDANCE
    assert "--repair --no-onboard" in MANAGED_REPAIR_GUIDANCE
    assert "pipx reinstall daita-agents" in PIPX_REPAIR_GUIDANCE


def test_arbitrary_managed_environment_value_keeps_pipx_guidance(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()

    assert not _is_trusted_managed_runtime(
        environ={"DAITA_MANAGED_INSTALL_ROOT": str(tmp_path / "arbitrary")},
        executable=sys.executable,
        home=home,
    )


def test_package_cli_imports_and_headless_command_keep_integrations_lazy(tmp_path):
    script = """
import builtins
import sys

blocked = {
    "anthropic",
    "asyncpg",
    "google",
    "httpx",
    "keyring",
    "openai",
    "prompt_toolkit",
    "rich",
    "textual",
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
    daita.cli.main(
        [
            "--root",
            sys.argv[1],
            "--workspace",
            sys.argv[2],
            "create",
            "packaging-smoke",
        ]
    )
)
"""
    workspace = tmp_path.parent / f"{tmp_path.name}-packaging-workspace"
    workspace.mkdir()
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(workspace)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"name": "packaging-smoke"' in completed.stdout
