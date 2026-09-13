from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.release_identity import (
    ReleaseIdentityError,
    compare_versions,
    parse_published_version_evidence,
    parse_version,
    prior_versions_for_mode,
    read_project_identity,
    require_newer,
)
from tests.support.paths import REPO_ROOT

ROOT = REPO_ROOT
HELPER = ROOT / "scripts" / "release_identity.py"
INSTALLER = ROOT / "scripts" / "install.sh"

ORDERED_CASES = (
    ("1.2.3", "1.2.3", "equal"),
    ("1.2.3", "1.2.4", "older"),
    ("1.2.9", "1.2.10", "older"),
    ("1.9.9", "1.10.0", "older"),
    ("9.9.9", "10.0.0", "older"),
    ("2.0.0", "1.99.99", "newer"),
)
INVALID_VERSIONS = (
    "",
    "1..3",
    ".2.3",
    "1.2.",
    "+1.2.3",
    "-1.2.3",
    " 1.2.3",
    "1.2.3 ",
    "01.2.3",
    "1.02.3",
    "1.2.03",
    "1.2.3-beta.1",
    "1.2.3+build.4",
    "v1.2.3",
    "1.2.3.4",
    "1.2.$(touch unsafe)",
    "1.2.3;touch unsafe",
    "1000000000.0.0",
    "0.1000000000.0",
    "0.0.1000000000",
    "١.٢.٣",
)


def _installer_function(name: str) -> str:
    lines = INSTALLER.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == "}")
    return "\n".join(lines[start : end + 1])


def _bash_compare(
    tmp_path: Path, left: str, right: str
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "compare.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        f"{_installer_function('validate_application_version')}\n"
        f"{_installer_function('compare_application_versions')}\n"
        'compare_application_versions "$1" "$2"\n',
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(script), left, right],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


@pytest.mark.parametrize(("left", "right", "expected"), ORDERED_CASES)
def test_python_and_bash_use_identical_canonical_semver_ordering(
    tmp_path: Path, left: str, right: str, expected: str
) -> None:
    assert compare_versions(left, right) == expected
    completed = _bash_compare(tmp_path, left, right)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == expected + "\n"


@pytest.mark.parametrize("value", INVALID_VERSIONS)
def test_python_and_bash_reject_the_same_malformed_or_unsafe_versions(
    tmp_path: Path, value: str
) -> None:
    sentinel = tmp_path / "unsafe"
    malicious = value.replace("unsafe", str(sentinel))
    with pytest.raises(ReleaseIdentityError):
        parse_version(malicious)
    completed = _bash_compare(tmp_path, malicious, "1.2.3")
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert not sentinel.exists()


def test_project_reader_and_cli_derive_one_exact_identity() -> None:
    identity = read_project_identity()
    version = subprocess.run(
        [sys.executable, str(HELPER), "project-version"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    tag = subprocess.run(
        [sys.executable, str(HELPER), "project-tag"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert identity.name == "daita-agents"
    assert version.stdout == identity.version + "\n"
    assert tag.stdout == identity.tag + "\n"

    compared = subprocess.run(
        [sys.executable, str(HELPER), "compare", "1.2.9", "1.2.10"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert compared.stdout == "older\n"


def test_project_reader_rejects_wrong_name_and_malformed_identity(
    tmp_path: Path,
) -> None:
    wrong_name = tmp_path / "wrong-name.toml"
    wrong_name.write_text(
        '[project]\nname = "foreign"\nversion = "1.2.3"\nrequires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    malformed = tmp_path / "malformed.toml"
    malformed.write_text(
        '[project]\nname = "daita-agents"\nversion = "1.2.03"\nrequires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    with pytest.raises(ReleaseIdentityError, match="project.name"):
        read_project_identity(wrong_name)
    with pytest.raises(ReleaseIdentityError, match="project.version"):
        read_project_identity(malformed)


def test_require_project_increase_uses_two_exact_project_files(tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    current = tmp_path / "current.toml"
    base.write_text(
        '[project]\nname = "daita-agents"\nversion = "1.2.9"\n'
        'requires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    current.write_text(
        '[project]\nname = "daita-agents"\nversion = "1.2.10"\n'
        'requires-python = ">=3.11"\n',
        encoding="utf-8",
    )

    increased = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "require-project-increase",
            "--base-project",
            str(base),
            "--project",
            str(current),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert increased.returncode == 0, increased.stderr
    refused = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "require-project-increase",
            "--base-project",
            str(current),
            "--project",
            str(base),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert refused.returncode == 2
    assert "project.version 1.2.9 must be newer than base version 1.2.10" in (
        refused.stderr
    )


def test_require_newer_uses_the_semantic_maximum_of_unordered_duplicates(
    tmp_path: Path,
) -> None:
    require_newer("2.0.0", ["1.9.9", "1.2.3", "1.10.0", "1.9.9"])
    previous = tmp_path / "previous.txt"
    previous.write_text("1.9.9\n1.2.3\n1.10.0\n1.9.9\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "require-newer",
            "--candidate",
            "1.10.0",
            "--previous-file",
            str(previous),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 2
    assert "equal to published maximum 1.10.0" in completed.stderr


def test_require_newer_accepts_the_first_release(tmp_path: Path) -> None:
    previous = tmp_path / "empty.txt"
    previous.write_bytes(b"")
    completed = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "require-newer",
            "--candidate",
            "1.2.3",
            "--previous-file",
            str(previous),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def _github_release(
    tag: str, *, draft: bool = False, prerelease: bool = False
) -> dict[str, object]:
    return {"tag_name": tag, "draft": draft, "prerelease": prerelease}


def test_published_evidence_unions_every_github_page_and_pypi_file_release() -> None:
    evidence = parse_published_version_evidence(
        [
            [_github_release("v1.2.9"), _github_release("v9.0.0", draft=True)],
            [_github_release("v1.10.0")],
            [_github_release("v1.2.9")],
        ],
        {
            "releases": {
                "1.2.10": [{"filename": "yanked.whl", "yanked": True}],
                "1.3.0": [],
            }
        },
    )

    assert evidence.github_versions == {"1.2.9", "1.10.0"}
    assert evidence.pypi_versions == {"1.2.10"}
    assert prior_versions_for_mode(
        evidence, candidate="2.0.0", tag="v2.0.0", mode="candidate"
    ) == ("1.2.9", "1.2.10", "1.10.0")


def test_candidate_and_protected_modes_apply_distinct_exact_equality_rules() -> None:
    evidence = parse_published_version_evidence(
        [[_github_release("v1.9.9")]],
        {
            "releases": {
                "2.0.0": [{"filename": "candidate.whl"}],
                "2.0.1": [],
            }
        },
    )
    with pytest.raises(ReleaseIdentityError, match="already has files on PyPI"):
        prior_versions_for_mode(
            evidence, candidate="2.0.0", tag="v2.0.0", mode="candidate"
        )
    assert prior_versions_for_mode(
        evidence, candidate="2.0.0", tag="v2.0.0", mode="protected"
    ) == ("1.9.9",)

    higher = parse_published_version_evidence(
        [[_github_release("v1.9.9")]],
        {
            "releases": {
                "2.0.0": [{"filename": "candidate.whl"}],
                "2.0.1": [{"filename": "higher.whl"}],
            }
        },
    )
    prior = prior_versions_for_mode(
        higher, candidate="2.0.0", tag="v2.0.0", mode="protected"
    )
    with pytest.raises(ReleaseIdentityError, match="older than published maximum"):
        require_newer("2.0.0", list(prior))


def test_release_evidence_fails_closed_on_existing_or_uncertain_state() -> None:
    existing = parse_published_version_evidence(
        [[_github_release("v2.0.0", draft=True)]],
        {"releases": {"2.0.0": [{"filename": "candidate.whl"}]}},
    )
    with pytest.raises(ReleaseIdentityError, match="already exists"):
        prior_versions_for_mode(
            existing, candidate="2.0.0", tag="v2.0.0", mode="protected"
        )
    with pytest.raises(ReleaseIdentityError, match="already exists"):
        prior_versions_for_mode(
            existing,
            candidate="2.0.0",
            tag="v2.0.0",
            mode="branch",
            tagged=True,
        )
    with pytest.raises(ReleaseIdentityError, match="missing from PyPI"):
        prior_versions_for_mode(
            parse_published_version_evidence([], {"releases": {}}),
            candidate="2.0.0",
            tag="v2.0.0",
            mode="protected",
        )
    with pytest.raises(ReleaseIdentityError, match="prerelease"):
        parse_published_version_evidence(
            [[_github_release("v2.0.0", prerelease=True)]], {"releases": {}}
        )
    with pytest.raises(ReleaseIdentityError, match="published GitHub release"):
        parse_published_version_evidence(
            [[_github_release("not-a-version")]], {"releases": {}}
        )
    with pytest.raises(ReleaseIdentityError, match="published PyPI release"):
        parse_published_version_evidence(
            [], {"releases": {"not-a-version": [{"filename": "bad.whl"}]}}
        )
    with pytest.raises(ReleaseIdentityError, match="pagination response"):
        parse_published_version_evidence([{"not": "a page"}], {"releases": {}})
