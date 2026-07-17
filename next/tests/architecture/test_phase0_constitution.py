from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

NEXT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = NEXT_ROOT.parent
BASELINE_COMMIT = "b87df31873d33fffbf50498f5dc4d8892115e8f8"

REQUIRED_ARTIFACTS = {
    "README.md",
    "STATUS.md",
    "PARITY_MATRIX.md",
    "QUALITY_GATES.md",
    "TEST_DISPOSITION.csv",
    "pyproject.toml",
    "decisions/README.md",
    "scripts/build_test_disposition.py",
    "scripts/capture_v1_oracles.py",
    "src/daita/__init__.py",
}


def test_required_phase0_artifacts_exist() -> None:
    missing = sorted(
        relative_path
        for relative_path in REQUIRED_ARTIFACTS
        if not (NEXT_ROOT / relative_path).is_file()
    )

    assert missing == []


def test_phase0_decisions_are_complete_and_accepted() -> None:
    decisions = sorted((NEXT_ROOT / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md"))

    assert [path.name[:4] for path in decisions] == [
        f"{number:04d}" for number in range(1, 15)
    ]
    assert all(
        "**Status:** Accepted" in path.read_text(encoding="utf-8") for path in decisions
    )


def test_isolated_distribution_metadata_matches_the_constitution() -> None:
    configuration = tomllib.loads(
        (NEXT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = configuration["project"]
    package_configuration = configuration["tool"]["setuptools"]

    assert project["name"] == "daita-agents"
    assert project["version"] == "2.0.0a0"
    assert project["requires-python"] == ">=3.11"
    assert project["dependencies"] == []
    assert package_configuration["package-dir"] == {"": "src"}
    assert package_configuration["packages"]["find"] == {
        "where": ["src"],
        "include": ["daita*"],
    }


def test_root_oracle_remains_identical_to_the_baseline() -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            BASELINE_COMMIT,
            "--",
            "daita",
            "tests",
            "pyproject.toml",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_phase10_remains_explicitly_excluded() -> None:
    for filename in ("README.md", "STATUS.md", "QUALITY_GATES.md"):
        text = (NEXT_ROOT / filename).read_text(encoding="utf-8")
        assert "Phase 10" in text
        assert any(
            term in text.lower() for term in ("excluded", "outside", "not authorize")
        )
