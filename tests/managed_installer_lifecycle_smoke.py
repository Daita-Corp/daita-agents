"""Managed-installer lifecycle smoke against one already-built candidate wheel.

The smoke renders deterministic immutable download fixtures into the canonical
``scripts/install.sh`` source. It does not claim public artifact, clean-machine,
or real-terminal evidence and it never contacts the public installer endpoint.

By default, uv and Python are deterministic local fixtures. Supplying the five
``--real-*`` arguments instead consumes an already-downloaded official uv
archive, uses it to download the exact uv-managed Python, and permits uv to
resolve the wheel's production dependencies from PyPI. The candidate wheel
remains a local exact artifact and is never uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from installer_fixtures import (
    InstallerFixture,
    create_installer_fixture,
    fixture_environment,
    sha256,
    wheel_metadata,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-wheel", type=Path, required=True)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--real-uv-archive", type=Path)
    parser.add_argument("--real-uv-version")
    parser.add_argument("--real-uv-member")
    parser.add_argument("--real-python-request")
    parser.add_argument("--real-python-identity")
    arguments = parser.parse_args()
    real_values = (
        arguments.real_uv_archive,
        arguments.real_uv_version,
        arguments.real_uv_member,
        arguments.real_python_request,
        arguments.real_python_identity,
    )
    if any(value is not None for value in real_values) and not all(
        value is not None for value in real_values
    ):
        parser.error("all five --real-* bootstrap arguments are required together")
    candidate = arguments.candidate_wheel.resolve(strict=True)
    _, candidate_version, _ = wheel_metadata(candidate)
    if candidate_version == "1.0.0":
        if arguments.baseline_wheel is not None:
            parser.error("Daita 1.0.0 is candidate-only; 0.x is not a valid baseline")
    elif arguments.baseline_wheel is None:
        parser.error("later 1.x candidates require --baseline-wheel")
    return arguments


def _create_fixture(
    directory: Path,
    *,
    wheel: Path,
    arguments: argparse.Namespace,
    installer_version: str = "1.0.0-fixture",
    release_sequence: int = 1,
) -> InstallerFixture:
    if arguments.real_uv_archive is None:
        return create_installer_fixture(
            directory,
            wheel=wheel,
            installer_version=installer_version,
            release_sequence=release_sequence,
        )
    assert isinstance(arguments.real_uv_version, str)
    assert isinstance(arguments.real_uv_member, str)
    assert isinstance(arguments.real_python_request, str)
    assert isinstance(arguments.real_python_identity, str)
    return create_installer_fixture(
        directory,
        wheel=wheel,
        installer_version=installer_version,
        release_sequence=release_sequence,
        bootstrap_uv_archive=arguments.real_uv_archive,
        bootstrap_uv_version=arguments.real_uv_version,
        bootstrap_uv_member=arguments.real_uv_member,
        bootstrap_python_request=arguments.real_python_request,
        bootstrap_python_identity=arguments.real_python_identity,
    )


def _run(
    command: list[str],
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and not path.is_symlink()
    }


def _current_generation(root: Path) -> Path:
    current = root / "current"
    if not current.is_symlink():
        raise AssertionError("managed current link is missing")
    generation = current.resolve(strict=True)
    if generation.parent != root / "generations":
        raise AssertionError("managed current link escaped generations")
    return generation


def _manifest(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise AssertionError("managed manifest is malformed")
        fields[key] = value
    return fields


def _assert_preserved(
    agent_root: Path,
    expected_agent_hashes: dict[str, str],
    sentinels: dict[Path, str],
) -> None:
    if _tree_hashes(agent_root) != expected_agent_hashes:
        raise AssertionError("managed lifecycle changed Daita application data")
    for path, expected in sentinels.items():
        if sha256(path) != expected:
            raise AssertionError(f"managed lifecycle changed sentinel: {path}")


def main() -> int:
    arguments = _arguments()
    candidate = arguments.candidate_wheel.resolve(strict=True)
    candidate_name, candidate_version, _ = wheel_metadata(candidate)
    if candidate_name != "daita-agents":
        raise AssertionError("candidate is not a daita-agents wheel")
    candidate_sha = sha256(candidate)

    with tempfile.TemporaryDirectory(prefix="daita-managed-smoke-") as temporary:
        workspace = Path(temporary).resolve()
        home = workspace / "home"
        home.mkdir()
        managed_root = home / ".local" / "share" / "daita"
        agent_root = home / ".daita"
        outside = workspace / "outside-sentinel"
        keychain = workspace / "keychain-boundary-sentinel"
        outside.write_text("preserve outside managed root", encoding="utf-8")
        keychain.write_text("no real keychain access", encoding="utf-8")
        sentinels = {outside: sha256(outside), keychain: sha256(keychain)}

        if arguments.baseline_wheel is None:
            active_fixture = _create_fixture(
                workspace / "candidate-fixture",
                wheel=candidate,
                arguments=arguments,
                release_sequence=1,
            )
            environment = fixture_environment(active_fixture, home)
            _run(
                [
                    "bash",
                    str(active_fixture.installer),
                    "--no-onboard",
                    "--no-modify-path",
                ],
                env=environment,
            )
        else:
            baseline = arguments.baseline_wheel.resolve(strict=True)
            baseline_name, baseline_version, _ = wheel_metadata(baseline)
            if baseline_name != "daita-agents":
                raise AssertionError("baseline is not a daita-agents wheel")
            if (
                baseline_version.startswith("0.")
                or baseline_version == candidate_version
            ):
                raise AssertionError(
                    "cross-version smoke requires a distinct supported 1.x baseline"
                )
            baseline_fixture = _create_fixture(
                workspace / "baseline-fixture",
                wheel=baseline,
                arguments=arguments,
                installer_version=f"{baseline_version}-fixture",
                release_sequence=1,
            )
            environment = fixture_environment(baseline_fixture, home)
            _run(
                [
                    "bash",
                    str(baseline_fixture.installer),
                    "--no-onboard",
                    "--no-modify-path",
                ],
                env=environment,
            )
            active_fixture = _create_fixture(
                workspace / "candidate-fixture",
                wheel=candidate,
                arguments=arguments,
                installer_version=f"{candidate_version}-fixture",
                release_sequence=2,
            )

        environment = fixture_environment(active_fixture, home)
        launcher = home / ".local" / "bin" / "daita"
        _run(
            [
                str(launcher),
                "--root",
                str(agent_root),
                "create",
                "preservation-agent",
            ],
            env=environment,
        )
        artifact = agent_root / "preserved-artifact.csv"
        artifact.write_text("id,value\n1,preserved\n", encoding="utf-8")
        expected_agent_hashes = _tree_hashes(agent_root)

        if arguments.baseline_wheel is not None:
            _run(
                [
                    "bash",
                    str(active_fixture.installer),
                    "--no-onboard",
                    "--no-modify-path",
                ],
                env=environment,
            )
            _assert_preserved(agent_root, expected_agent_hashes, sentinels)

        version = _run([str(launcher), "--version"], env=environment)
        if version.stdout != f"daita {candidate_version}\n":
            raise AssertionError("managed launcher version does not match candidate")
        help_result = _run([str(launcher), "--help"], env=environment)
        if "usage: daita" not in help_result.stdout:
            raise AssertionError("managed launcher help is unavailable")

        generation = _current_generation(managed_root)
        manifest = _manifest(generation / "manifest")
        python = generation / manifest["generation_python"]
        if manifest["wheel_sha256"] != candidate_sha:
            raise AssertionError(
                "managed installer did not consume the candidate wheel"
            )
        if manifest["wheel_filename"] != candidate.name:
            raise AssertionError("managed manifest does not name the candidate wheel")
        if arguments.real_uv_archive is not None:
            if manifest["uv_version"] != arguments.real_uv_version:
                raise AssertionError(
                    "managed manifest does not name the real uv version"
                )
            if manifest["python_identity"] != arguments.real_python_identity:
                raise AssertionError(
                    "managed manifest does not name the real Python identity"
                )
            real_dependency_check = """
import importlib
import platform
import sys

expected = sys.argv[1]
assert platform.python_version() == expected
for module in (
    "anthropic",
    "asyncpg",
    "google.genai",
    "keyring",
    "openai",
    "prompt_toolkit",
    "rich",
    "sqlglot",
    "xlsxwriter",
):
    importlib.import_module(module)
"""
            expected_python_version = arguments.real_python_identity.split("-")[1]
            _run(
                [
                    str(python),
                    "-I",
                    "-c",
                    real_dependency_check,
                    expected_python_version,
                ],
                env=environment,
            )

        before_verify = _tree_hashes(managed_root)
        _run(["bash", str(active_fixture.installer), "--verify"], env=environment)
        if _tree_hashes(managed_root) != before_verify:
            raise AssertionError("managed --verify mutated installer-owned files")

        first_target = os.readlink(managed_root / "current")
        first_generations = tuple((managed_root / "generations").iterdir())
        _run(
            [
                "bash",
                str(active_fixture.installer),
                "--no-onboard",
                "--no-modify-path",
            ],
            env=environment,
        )
        if os.readlink(managed_root / "current") != first_target:
            raise AssertionError("same-version repeat changed the active generation")
        if tuple((managed_root / "generations").iterdir()) != first_generations:
            raise AssertionError("same-version repeat created an extra generation")

        _run(
            [
                "bash",
                str(active_fixture.installer),
                "--repair",
                "--no-onboard",
                "--no-modify-path",
            ],
            env=environment,
        )
        repaired_target = os.readlink(managed_root / "current")
        if repaired_target == first_target:
            raise AssertionError("repair did not activate a fresh generation")
        if os.readlink(managed_root / "previous") != first_target:
            raise AssertionError("repair did not retain the verified prior generation")
        _assert_preserved(agent_root, expected_agent_hashes, sentinels)

        _run(["bash", str(active_fixture.installer), "--rollback"], env=environment)
        if os.readlink(managed_root / "current") != first_target:
            raise AssertionError(
                "rollback did not reactivate the recorded previous binary"
            )
        _assert_preserved(agent_root, expected_agent_hashes, sentinels)

        generation = _current_generation(managed_root)
        python = generation / _manifest(generation / "manifest")["generation_python"]
        provenance_environment = environment.copy()
        provenance_environment["DAITA_MANAGED_INSTALL_ROOT"] = str(managed_root)
        repair_guidance = _run(
            [
                str(python),
                "-I",
                "-c",
                "from daita._installation import repair_guidance; print(repair_guidance())",
            ],
            env=provenance_environment,
        )
        if "https://daita-tech.io/install.sh" not in repair_guidance.stdout:
            raise AssertionError(
                "managed runtime did not select managed repair guidance"
            )

        (generation / "bin" / "daita").write_text("damaged", encoding="utf-8")
        healthy_previous = os.readlink(managed_root / "previous")
        _run(
            [
                "bash",
                str(active_fixture.installer),
                "--repair",
                "--no-onboard",
                "--no-modify-path",
            ],
            env=environment,
        )
        if os.readlink(managed_root / "previous") != healthy_previous:
            raise AssertionError("repair recorded a damaged generation as previous")
        _assert_preserved(agent_root, expected_agent_hashes, sentinels)

        _run(["bash", str(active_fixture.installer), "--uninstall"], env=environment)
        if launcher.exists() or managed_root.exists():
            raise AssertionError(
                "managed uninstall left installer-owned application files"
            )
        _assert_preserved(agent_root, expected_agent_hashes, sentinels)

        print(f"candidate wheel: {candidate.name}")
        print(f"candidate sha256: {candidate_sha}")
        print("managed lifecycle: install, repeat, verify, repair, rollback, uninstall")
        print("application data, artifact, keychain boundary, and sentinels: preserved")
        if arguments.real_uv_archive is not None:
            print(
                f"real bootstrap: uv {arguments.real_uv_version}, "
                f"{arguments.real_python_identity}, production dependencies"
            )
        print("clean-machine and real-terminal support evidence: not claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
