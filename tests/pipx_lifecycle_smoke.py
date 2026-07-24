"""Isolated local release smoke for the customer pipx lifecycle.

Run from the repository root:

    .venv/bin/python tests/pipx_lifecycle_smoke.py

The procedure performs the equivalent of ``python -m build``, ``pipx install``,
``daita --help``, ``pipx reinstall``, and ``pipx uninstall`` against only the
newly built local application artifact. Pip may read a configured package index
to resolve the wheel's declared dependencies; the procedure never changes an
index or uploads an artifact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENTRY_POINT = "daita.cli:main"


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _single_artifact(directory: Path, suffix: str) -> Path:
    artifacts = tuple(sorted(directory.glob(f"*{suffix}")))
    if len(artifacts) != 1:
        raise AssertionError(
            f"expected one {suffix} artifact, found {[item.name for item in artifacts]}"
        )
    return artifacts[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    pipx = shutil.which("pipx")
    if pipx is None:
        raise RuntimeError("pipx is required to run the isolated release smoke")
    if importlib.util.find_spec("build") is None:
        raise RuntimeError("the development environment is missing the build package")

    with tempfile.TemporaryDirectory(prefix="daita-pipx-smoke-") as temporary:
        workspace = Path(temporary).resolve()
        distribution = workspace / "dist"
        pipx_home = workspace / "pipx-home"
        pipx_bin = workspace / "pipx-bin"
        pipx_man = workspace / "pipx-man"
        pip_cache = workspace / "pip-cache"
        outside_checkout = workspace / "outside-checkout"
        separate_agent_home = workspace / "customer-agent-data"
        for directory in (
            distribution,
            pipx_home,
            pipx_bin,
            pipx_man,
            pip_cache,
            outside_checkout,
            separate_agent_home,
        ):
            directory.mkdir()

        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(distribution),
            ],
            cwd=ROOT,
        )
        wheel = _single_artifact(distribution, ".whl")
        sdist = _single_artifact(distribution, ".tar.gz")

        environment = os.environ.copy()
        environment.update(
            {
                "PIPX_HOME": str(pipx_home),
                "PIPX_BIN_DIR": str(pipx_bin),
                "PIPX_MAN_DIR": str(pipx_man),
                "PIPX_DEFAULT_PYTHON": sys.executable,
                "PIP_CACHE_DIR": str(pip_cache),
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PYTHONPATH": "",
            }
        )
        _run(
            [
                pipx,
                "install",
                "--skip-maintenance",
                str(wheel),
            ],
            cwd=outside_checkout,
            env=environment,
        )

        command = pipx_bin / "daita"
        help_result = _run(
            [str(command), "--help"],
            cwd=outside_checkout,
            env=environment,
        )
        if "usage: daita" not in help_result.stdout:
            raise AssertionError("installed daita --help did not render CLI usage")
        _run(
            [
                str(command),
                "--root",
                str(separate_agent_home),
                "create",
                "preservation-agent",
            ],
            cwd=outside_checkout,
            env=environment,
        )
        preserved_home = separate_agent_home / "agents" / "preservation-agent"
        preserved_paths = (
            preserved_home / "agent.toml",
            preserved_home / "state.db",
        )
        if not all(path.is_file() for path in preserved_paths):
            raise AssertionError("installed daita did not create a real agent home")
        preserved_hashes = {path.name: _sha256(path) for path in preserved_paths}

        installed_python = pipx_home / "venvs" / "daita-agents" / "bin" / "python"
        metadata_check = """
from importlib import metadata

distribution = metadata.distribution("daita-agents")
entry_points = {
    item.name: item.value
    for item in distribution.entry_points
    if item.group == "console_scripts"
}
assert distribution.version == "2.0.0a0"
assert entry_points == {"daita": "daita.cli:main"}
requirements = tuple(distribution.requires or ())
assert any(item.startswith("openai") for item in requirements)
assert any(item.startswith("asyncpg") for item in requirements)
assert any(item.startswith("prompt-toolkit") for item in requirements)
"""
        _run(
            [str(installed_python), "-I", "-c", metadata_check],
            cwd=outside_checkout,
            env=environment,
        )

        _run(
            [pipx, "reinstall", "--skip-maintenance", "daita-agents"],
            cwd=outside_checkout,
            env=environment,
        )
        _run(
            [str(command), "--help"],
            cwd=outside_checkout,
            env=environment,
        )
        if {path.name: _sha256(path) for path in preserved_paths} != preserved_hashes:
            raise AssertionError("pipx reinstall changed Daita-created agent state")
        _run(
            [pipx, "uninstall", "daita-agents"],
            cwd=outside_checkout,
            env=environment,
        )
        if command.exists():
            raise AssertionError("pipx uninstall left the daita command installed")
        if not all(path.is_file() for path in preserved_paths):
            raise AssertionError("pipx uninstall removed Daita-created agent state")
        if {path.name: _sha256(path) for path in preserved_paths} != preserved_hashes:
            raise AssertionError("pipx uninstall changed Daita-created agent state")

        print(f"wheel: {wheel.name}")
        print(f"sdist: {sdist.name}")
        print(f"entry point: daita = {EXPECTED_ENTRY_POINT}")
        print("pipx lifecycle: install, reinstall, uninstall")
        print("Daita-created agent.toml and state.db: preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
