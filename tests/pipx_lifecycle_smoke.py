"""Isolated local release and cross-version upgrade smoke.

Run from the repository root:

    .venv/bin/python tests/pipx_lifecycle_smoke.py

Version 1.0.0 has no predecessor, so pass its once-built candidate wheel and
exercise install, reinstall, and uninstall through ``pipx install``,
``pipx reinstall``, and ``pipx uninstall``:

    .venv/bin/python tests/pipx_lifecycle_smoke.py \
        --candidate-wheel /path/to/candidate.whl

With no arguments, the developer convenience path still builds one local
artifact with the equivalent of ``python -m build``. For every later release,
pass the immediately preceding wheel and the candidate wheel:

    .venv/bin/python tests/pipx_lifecycle_smoke.py \
        --baseline-wheel /path/to/previous.whl \
        --candidate-wheel /path/to/candidate.whl

The two-wheel procedure installs the baseline, creates real agent state, then
force-installs the candidate into the same isolated pipx environment and opens
the baseline-created state with it. Pip may read a configured package index to
resolve declared dependencies; the procedure never changes an index or uploads
an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENTRY_POINT = "daita.cli:main"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    arguments = parser.parse_args()
    if arguments.baseline_wheel is not None and arguments.candidate_wheel is None:
        parser.error("--baseline-wheel requires --candidate-wheel")
    return arguments


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


def _home_hashes(home: Path) -> dict[str, str]:
    paths = tuple(
        sorted(
            (
                path
                for path in home.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.relative_to(home).parts[0] != "run"
            ),
            key=lambda path: path.relative_to(home).as_posix(),
        )
    )
    return {path.relative_to(home).as_posix(): _sha256(path) for path in paths}


def _wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = tuple(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_names) != 1:
            raise AssertionError("wheel must contain exactly one METADATA document")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    versions = tuple(
        line.removeprefix("Version: ").strip()
        for line in metadata.splitlines()
        if line.startswith("Version: ")
    )
    if len(versions) != 1 or not versions[0]:
        raise AssertionError("wheel METADATA must contain exactly one version")
    return versions[0]


def main() -> int:
    arguments = _arguments()
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

        if arguments.candidate_wheel is None:
            _run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--outdir",
                    str(distribution),
                ],
                cwd=ROOT,
            )
            candidate_wheel = _single_artifact(distribution, ".whl")
            baseline_wheel = candidate_wheel
            sdist: Path | None = _single_artifact(distribution, ".tar.gz")
        elif arguments.baseline_wheel is None:
            candidate_wheel = arguments.candidate_wheel.resolve(strict=True)
            baseline_wheel = candidate_wheel
            sdist = None
        else:
            baseline_wheel = arguments.baseline_wheel.resolve(strict=True)
            candidate_wheel = arguments.candidate_wheel.resolve(strict=True)
            if baseline_wheel == candidate_wheel:
                raise ValueError(
                    "cross-version smoke requires distinct baseline and "
                    "candidate wheels"
                )
            sdist = None

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
                str(baseline_wheel),
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
        version_result = _run(
            [str(command), "--version"],
            cwd=outside_checkout,
            env=environment,
        )
        expected_version_output = f"daita {_wheel_version(baseline_wheel)}\n"
        if version_result.stdout != expected_version_output:
            raise AssertionError("installed daita --version did not match the wheel")
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
        installed_python = pipx_home / "venvs" / "daita-agents" / "bin" / "python"
        _run(
            [str(installed_python), "-m", "pip", "check"],
            cwd=outside_checkout,
            env=environment,
        )
        runtime_dependency_check = """
from io import BytesIO

import xlsxwriter

from daita.artifacts.renderers import _load_xlsxwriter


assert _load_xlsxwriter() is xlsxwriter
buffer = BytesIO()
workbook = xlsxwriter.Workbook(
    buffer,
    {
        "in_memory": True,
        "strings_to_formulas": False,
        "strings_to_urls": False,
    },
)
worksheet = workbook.add_worksheet("Data")
worksheet.write_string(0, 0, "isolated pipx XLSX smoke")
workbook.close()
content = buffer.getvalue()
assert content.startswith(b"PK")
assert len(content) > 0
"""
        _run(
            [str(installed_python), "-I", "-c", runtime_dependency_check],
            cwd=outside_checkout,
            env=environment,
        )
        seed_state = """
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

from daita import (
    Agent,
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
    SQLiteSource,
)
from daita.learning_candidates import (
    DocumentCandidateContent,
    LearningCandidate,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateStatus,
    LearningCandidateTarget,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider


async def main():
    root = Path(sys.argv[1])
    source_path = root / "upgrade-source.sqlite"
    with sqlite3.connect(source_path) as connection:
        connection.execute(
            "CREATE TABLE invoices(id INTEGER PRIMARY KEY, booked_at TEXT)"
        )
        connection.execute(
            "INSERT INTO invoices(id, booked_at) VALUES (1, '2026-01-15')"
        )

    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Baseline answer."),),
        provider_id="mock:upgrade-baseline",
    )
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32000,
        max_output_tokens=1000,
        supports_tools=True,
    )
    agent = await Agent.open(
        "preservation-agent",
        root=root,
        model=provider,
        model_profile=profile,
    )
    registration = await agent.attach(
        SQLiteSource(source_path, name="Upgrade source")
    )
    run = await agent.run("Remember the baseline upgrade run.")
    await agent.set_memory("Fiscal year begins in February.\\n")
    await agent.set_user_profile("Prefer concise upgrade reports.\\n")
    await agent.save_skill(
        "upgrade-check",
        "Verify preserved upgrade state.",
        "Inspect the registered source and compare the persisted baseline records.",
    )
    resource = (await agent.list_catalog_resources())[0]
    transcript = await agent.transcript(run.run_id)
    annotation = SemanticAnnotation(
        id="upgrade-booked-at",
        agent_id=agent.id,
        subject=SemanticSubject(
            source_ids=(registration.id,),
            resource_ids=(resource.id,),
            fields=(SemanticFieldReference(resource.id, "booked_at"),),
        ),
        kind=SemanticKind.TIME_SEMANTICS,
        statement="booked_at is the invoice booking date.",
        evidence=(
            SemanticEvidence(
                SemanticEvidenceKind.USER_ASSERTION,
                run.run_id,
                message_position=0,
            ),
        ),
        catalog_revisions=(
            ResourceRevisionBinding(resource.id, resource.current_revision),
        ),
        created_at=transcript.run.created_at,
        confirmed_at=transcript.run.created_at,
        confirmed_by="local-user",
    )
    await agent.save_semantic_annotation(annotation)

    digest = "1" * 64
    candidate = LearningCandidate(
        id="upgrade-candidate",
        agent_id=agent.id,
        target=LearningCandidateTarget.MEMORY,
        content=DocumentCandidateContent("Invoices use their booking date."),
        source_ids=(),
        reviewed_runs=(LearningCandidateRunReference(run.run_id, digest),),
        supporting_run_ids=(run.run_id,),
        review_fingerprint="2" * 64,
        artifact_state_sha256="3" * 64,
        catalog_revisions=(),
        candidate_fingerprint="4" * 64,
        status=LearningCandidateStatus.AWAITING_REVIEW,
        created_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )
    stamp = LearningCandidateReviewStamp(
        run_id=run.run_id,
        transcript_sha256=digest,
        artifact_state_sha256="3" * 64,
        catalog_state_sha256="5" * 64,
    )
    await agent._embedded._store.save_learning_candidate_review(
        agent.id,
        stamps=(stamp,),
        candidates=(candidate,),
    )
    expectations = {
        "agent_id": agent.id,
        "conversation_id": run.conversation_id,
        "resource_id": resource.id,
        "run_id": run.run_id,
        "source_id": registration.id,
    }
    await agent.close()

    validator = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="upgrade-validation",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id="ollama:llama3.1",
            ),
        ),
        provider_id="ollama:llama3.1",
    )
    agent = await Agent.open(
        "preservation-agent",
        root=root,
        model_validator=validator,
    )
    await agent.configure_model(
        provider="ollama",
        model="llama3.1",
        context_window_tokens=8192,
        max_output_tokens=1024,
    )
    await agent.close()
    (root / "upgrade-expectations.json").write_text(
        json.dumps(expectations, sort_keys=True),
        encoding="utf-8",
    )


asyncio.run(main())
"""
        _run(
            [
                str(installed_python),
                "-I",
                "-c",
                seed_state,
                str(separate_agent_home),
            ],
            cwd=outside_checkout,
            env=environment,
        )
        preserved_home = separate_agent_home / "agents" / "preservation-agent"
        preserved_paths = (
            preserved_home / "agent.toml",
            preserved_home / "state.db",
            preserved_home / "config.json",
            preserved_home / "MEMORY.md",
            preserved_home / "USER.md",
            preserved_home / "skills" / "upgrade-check" / "SKILL.md",
        )
        if not all(path.is_file() for path in preserved_paths):
            raise AssertionError("installed daita did not create a real agent home")
        preserved_hashes = _home_hashes(preserved_home)
        inspect_state = """
import asyncio
import json
from pathlib import Path
import sys

from daita import Agent


async def main():
    root = Path(sys.argv[1])
    expected = json.loads(
        (root / "upgrade-expectations.json").read_text(encoding="utf-8")
    )
    agent = await Agent.open("preservation-agent", root=root)
    sources = await agent.list_sources()
    summary = await agent.catalog_summary()
    transcript = await agent.transcript(expected["run_id"])
    runs = await agent.conversation_runs(expected["conversation_id"])
    skill = await agent.read_skill("upgrade-check")
    semantics = await agent.list_semantic_annotations()
    candidates = await agent.list_learning_candidates()
    active = await agent.active_source()
    projection = {
        "active_source_id": None if active is None else active.id,
        "agent_id": agent.id,
        "candidate_ids": [item.candidate.id for item in candidates],
        "catalog_relationship_count": summary.relationship_count,
        "catalog_resource_count": summary.resource_count,
        "conversation_run_ids": [item.transcript.run.id for item in runs],
        "memory": await agent.read_memory(),
        "model_provider_id": agent.model_route.candidates[0].provider_id,
        "run_answer": transcript.run.message,
        "run_result": runs[0].result.final_text,
        "semantic_ids": [item.annotation.id for item in semantics],
        "skill": None if skill is None else {
            "description": skill.description,
            "instructions": skill.instructions,
            "name": skill.name,
        },
        "source_ids": [item.id for item in sources],
        "user": await agent.read_user_profile(),
    }
    await agent.close()
    print(json.dumps(projection, ensure_ascii=True, sort_keys=True))


asyncio.run(main())
"""
        baseline_projection = _run(
            [
                str(installed_python),
                "-I",
                "-c",
                inspect_state,
                str(separate_agent_home),
            ],
            cwd=outside_checkout,
            env=environment,
        ).stdout
        inspected_hashes = _home_hashes(preserved_home)
        if inspected_hashes != preserved_hashes:
            changed = sorted(
                name
                for name in set(inspected_hashes) | set(preserved_hashes)
                if inspected_hashes.get(name) != preserved_hashes.get(name)
            )
            raise AssertionError(
                "baseline inspection changed Daita-created agent state: "
                + ", ".join(changed)
            )

        metadata_check = """
from importlib import metadata
import sys

distribution = metadata.distribution("daita-agents")
entry_points = {
    item.name: item.value
    for item in distribution.entry_points
    if item.group == "console_scripts"
}
assert distribution.version == sys.argv[1]
assert entry_points == {"daita": "daita.cli:main"}
requires_python = distribution.metadata["Requires-Python"]
assert requires_python is not None
assert {item.strip() for item in requires_python.split(",")} == {
    ">=3.11",
    "<3.13",
}
requirements = tuple(distribution.requires or ())
assert any(item.startswith("openai") for item in requirements)
assert any(item.startswith("anthropic") for item in requirements)
assert any(item.startswith("google-genai") for item in requirements)
assert any(item.startswith("asyncpg") for item in requirements)
assert any(item.startswith("sqlglot") for item in requirements)
assert any(item.startswith("keyring") for item in requirements)
assert any(item.startswith("prompt-toolkit") for item in requirements)
assert any(item.startswith("rich") for item in requirements)
assert any(item.startswith("XlsxWriter") for item in requirements)
"""
        _run(
            [
                str(installed_python),
                "-I",
                "-c",
                metadata_check,
                _wheel_version(baseline_wheel),
            ],
            cwd=outside_checkout,
            env=environment,
        )

        if baseline_wheel == candidate_wheel:
            _run(
                [pipx, "reinstall", "--skip-maintenance", "daita-agents"],
                cwd=outside_checkout,
                env=environment,
            )
        else:
            _run(
                [
                    pipx,
                    "install",
                    "--force",
                    "--skip-maintenance",
                    str(candidate_wheel),
                ],
                cwd=outside_checkout,
                env=environment,
            )
        _run(
            [str(command), "--help"],
            cwd=outside_checkout,
            env=environment,
        )
        _run(
            [str(installed_python), "-m", "pip", "check"],
            cwd=outside_checkout,
            env=environment,
        )
        _run(
            [str(installed_python), "-I", "-c", runtime_dependency_check],
            cwd=outside_checkout,
            env=environment,
        )
        _run(
            [
                str(command),
                "--root",
                str(separate_agent_home),
                "sources",
                "preservation-agent",
            ],
            cwd=outside_checkout,
            env=environment,
        )
        _run(
            [
                str(installed_python),
                "-I",
                "-c",
                metadata_check,
                _wheel_version(candidate_wheel),
            ],
            cwd=outside_checkout,
            env=environment,
        )
        candidate_projection = _run(
            [
                str(installed_python),
                "-I",
                "-c",
                inspect_state,
                str(separate_agent_home),
            ],
            cwd=outside_checkout,
            env=environment,
        ).stdout
        if candidate_projection != baseline_projection:
            raise AssertionError(
                "candidate did not preserve the baseline agent's logical state"
            )
        if _home_hashes(preserved_home) != preserved_hashes:
            raise AssertionError(
                "package replacement changed Daita-created agent state"
            )
        append_state = """
import asyncio
import json
from pathlib import Path
import sys

from daita import Agent
from daita.llm.models import FinishReason, ModelProfile, ModelResponse
from daita.llm.providers.mock import MockModelProvider


async def main():
    root = Path(sys.argv[1])
    expected = json.loads(
        (root / "upgrade-expectations.json").read_text(encoding="utf-8")
    )
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Candidate answer."),),
        provider_id="mock:upgrade-candidate",
    )
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32000,
        max_output_tokens=1000,
        supports_tools=True,
    )
    agent = await Agent.open(
        "preservation-agent",
        root=root,
        model=provider,
        model_profile=profile,
    )
    await agent.run(
        "Append after upgrade.",
        conversation_id=expected["conversation_id"],
    )
    runs = await agent.conversation_runs(expected["conversation_id"])
    await agent.close()
    assert len(runs) == 2
    assert runs[-1].result.final_text == "Candidate answer."


asyncio.run(main())
"""
        _run(
            [
                str(installed_python),
                "-I",
                "-c",
                append_state,
                str(separate_agent_home),
            ],
            cwd=outside_checkout,
            env=environment,
        )
        preserved_hashes = _home_hashes(preserved_home)
        _run(
            [pipx, "uninstall", "daita-agents"],
            cwd=outside_checkout,
            env=environment,
        )
        if command.exists():
            raise AssertionError("pipx uninstall left the daita command installed")
        if not all(path.is_file() for path in preserved_paths):
            raise AssertionError("pipx uninstall removed Daita-created agent state")
        if _home_hashes(preserved_home) != preserved_hashes:
            raise AssertionError("pipx uninstall changed Daita-created agent state")

        print(f"baseline wheel: {baseline_wheel.name}")
        print(f"candidate wheel: {candidate_wheel.name}")
        if sdist is not None:
            print(f"sdist: {sdist.name}")
        print(f"entry point: daita = {EXPECTED_ENTRY_POINT}")
        print("pipx lifecycle: install, replace/reinstall, uninstall")
        print("Complete Daita-created agent home: preserved and reopened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
