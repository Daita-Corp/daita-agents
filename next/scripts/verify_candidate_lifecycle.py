#!/usr/bin/env python3
"""Build and prove the isolated candidate lifecycle without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile

_PROBE = r"""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from daita import Agent
from daita.llm import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import ActionProposal, Evidence, Observation


class TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        if tools:
            raise AssertionError("offline lifecycle context received tools")
        message = operation.trigger.payload["message"]
        if not isinstance(message, str):
            raise AssertionError("offline lifecycle trigger has no message")
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class TextDomain:
    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("offline lifecycle domain has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("offline lifecycle domain has no observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        return Readiness(
            allowed=True,
            code="ready",
            message="Offline lifecycle response is ready.",
            evaluated_at=datetime.now(timezone.utc),
        )


async def main() -> None:
    root = Path(sys.argv[1])
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="candidate lifecycle complete",
            ),
        ),
        provider_id="candidate:offline",
    )
    agent = await Agent.open(
        "atlas",
        root=root,
        model=provider,
        context_builder=TextContext(),
        domain=TextDomain(),
    )
    outcome = await agent.run("Complete the offline lifecycle check.")
    operation_id = outcome.operation_id
    if outcome.kind.value != "completed":
        raise AssertionError(f"unexpected loop exit: {outcome.kind.value}")
    await agent.close()

    reopened = await Agent.open("atlas", root=root)
    snapshot = await reopened.inspect(operation_id)
    if snapshot.operation.status.value != "succeeded":
        raise AssertionError(
            f"unexpected reopened status: {snapshot.operation.status.value}"
        )
    await reopened.close()
    print(
        json.dumps(
            {
                "operation_id": operation_id,
                "reopened_status": snapshot.operation.status.value,
            },
            sort_keys=True,
        )
    )


asyncio.run(main())
"""

_IGNORED_COPY_NAMES = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "*.egg-info",
    "*.pyc",
)
_SENSITIVE_ENV_SUFFIXES = (
    "_API_KEY",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v2 wheel/sdist and prove install, initialize, run, "
            "stop, reopen, uninstall, and state retention."
        )
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--python",
        dest="interpreters",
        action="append",
        required=True,
        help="Python 3.11/3.12 executable to verify; repeat for each interpreter",
    )
    return parser


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        stdout = completed.stdout[-4_096:]
        stderr = completed.stderr[-4_096:]
        raise RuntimeError(
            f"candidate command failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return completed


def _clean_environment(home: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "PYTHONPATH"
        and not key.endswith(_SENSITIVE_ENV_SUFFIXES)
        and key not in {"DATABASE_URL", "PGPASSWORD"}
    }
    environment["HOME"] = str(home)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _copy_candidate(source: Path, destination: Path) -> None:
    if not (source / "pyproject.toml").is_file():
        raise ValueError("candidate root must contain pyproject.toml")
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(*_IGNORED_COPY_NAMES),
    )
    symlinks = tuple(path for path in destination.rglob("*") if path.is_symlink())
    if symlinks:
        raise RuntimeError("candidate clean copy contains a symbolic link")


def _build(
    source: Path, output: Path, environment: dict[str, str]
) -> tuple[Path, Path]:
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=source,
        environment=environment,
    )
    wheels = tuple(output.glob("daita_agents-2.0.0a0-*.whl"))
    sdists = tuple(output.glob("daita_agents-2.0.0a0.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("candidate build must produce one wheel and one sdist")
    return wheels[0], sdists[0]


def _inspect_artifacts(wheel: Path, sdist: Path) -> dict[str, object]:
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = tuple(archive.namelist())
    if not wheel_names or any(
        name.startswith(("tests/", "examples/", "scripts/", "docs/", "next/"))
        or "/__pycache__/" in name
        or name.endswith((".pyc", ".db", ".sqlite", ".sqlite3"))
        for name in wheel_names
    ):
        raise RuntimeError("candidate wheel violates the production allowlist")
    if any(
        not name.startswith(("daita/", "daita_agents-2.0.0a0.dist-info/"))
        for name in wheel_names
    ):
        raise RuntimeError("candidate wheel contains an unexpected top-level path")

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = tuple(member.name for member in archive.getmembers())
    if not sdist_names or any(
        "/tests/" in f"/{name}/"
        or "/examples/" in f"/{name}/"
        or "/docs/" in f"/{name}/"
        or "/.github/" in f"/{name}/"
        or "/__pycache__/" in f"/{name}/"
        or name.endswith((".pyc", ".db", ".sqlite", ".sqlite3"))
        for name in sdist_names
    ):
        raise RuntimeError("candidate sdist violates the source allowlist")
    return {
        "sdist_entries": len(sdist_names),
        "sdist_sha256": _digest(sdist),
        "wheel_entries": len(wheel_names),
        "wheel_sha256": _digest(wheel),
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(128 * 1_024), b""):
            digest.update(block)
    return digest.hexdigest()


def _venv_python(environment_root: Path) -> Path:
    windows = environment_root / "Scripts" / "python.exe"
    return windows if windows.exists() else environment_root / "bin" / "python"


def _venv_cli(environment_root: Path) -> Path:
    windows = environment_root / "Scripts" / "daita.exe"
    return windows if windows.exists() else environment_root / "bin" / "daita"


def _verify_interpreter(
    interpreter: str,
    *,
    index: int,
    wheel: Path,
    workspace: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    environment_root = workspace / f"venv-{index}"
    state_root = (workspace / f"state-{index}").resolve()
    _run(
        [interpreter, "-m", "venv", str(environment_root)],
        cwd=workspace,
        environment=environment,
    )
    python = _venv_python(environment_root)
    cli = _venv_cli(environment_root)
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=workspace,
        environment=environment,
    )
    version = _run(
        [str(python), "-I", "-c", "import platform; print(platform.python_version())"],
        cwd=workspace,
        environment=environment,
    ).stdout.strip()
    if version.split(".")[:2] not in (["3", "11"], ["3", "12"]):
        raise RuntimeError(f"unsupported candidate interpreter: {version}")
    _run([str(cli), "--help"], cwd=workspace, environment=environment)
    _run(
        [
            str(cli),
            "--root",
            str(state_root),
            "agent",
            "init",
            "atlas",
            "--idempotency-key",
            f"candidate-init-{index}",
        ],
        cwd=workspace,
        environment=environment,
    )
    probe = _run(
        [str(python), "-I", "-c", _PROBE, str(state_root)],
        cwd=workspace,
        environment=environment,
    )
    probe_result = json.loads(probe.stdout)
    state_db = state_root / "agents" / "atlas" / "state.db"
    manifest = state_root / "agents" / "atlas" / "agent.toml"
    if not state_db.is_file() or not manifest.is_file():
        raise RuntimeError("candidate lifecycle did not persist the agent home")
    _run(
        [str(python), "-m", "pip", "uninstall", "-y", "daita-agents"],
        cwd=workspace,
        environment=environment,
    )
    if not state_db.is_file() or not manifest.is_file():
        raise RuntimeError("package uninstall removed retained local state")
    _run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('daita') is None",
        ],
        cwd=workspace,
        environment=environment,
    )
    return {
        "operation_status": probe_result["reopened_status"],
        "python": version,
        "state_retained_after_uninstall": True,
    }


def main() -> int:
    arguments = _parser().parse_args()
    candidate_root = arguments.candidate_root.resolve(strict=True)
    with TemporaryDirectory(prefix="daita-v2-candidate-lifecycle-") as temporary:
        workspace = Path(temporary).resolve()
        clean_home = workspace / "home"
        clean_home.mkdir(mode=0o700)
        source = workspace / "candidate"
        output = workspace / "dist"
        output.mkdir()
        _copy_candidate(candidate_root, source)
        environment = _clean_environment(clean_home)
        wheel, sdist = _build(source, output, environment)
        result = {
            "artifacts": _inspect_artifacts(wheel, sdist),
            "distribution": "daita-agents",
            "interpreters": [
                _verify_interpreter(
                    interpreter,
                    index=index,
                    wheel=wheel,
                    workspace=workspace,
                    environment=environment,
                )
                for index, interpreter in enumerate(arguments.interpreters)
            ],
            "schema_version": 1,
            "version": "2.0.0a0",
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
