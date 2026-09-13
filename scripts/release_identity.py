#!/usr/bin/env python3
"""Read and compare the sole Daita release identity."""

from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = ROOT / "pyproject.toml"
_CANONICAL_VERSION = re.compile(
    r"(0|[1-9][0-9]{0,8})\." r"(0|[1-9][0-9]{0,8})\." r"(0|[1-9][0-9]{0,8})\Z"
)


class ReleaseIdentityError(ValueError):
    """Raised when release identity is missing, malformed, or non-increasing."""


@dataclass(frozen=True, slots=True)
class ProjectIdentity:
    name: str
    version: str
    requires_python: str

    @property
    def tag(self) -> str:
        return f"v{self.version}"


@dataclass(frozen=True, slots=True)
class PublishedVersionEvidence:
    github_tags: frozenset[str]
    github_versions: frozenset[str]
    pypi_versions: frozenset[str]

    @property
    def union(self) -> frozenset[str]:
        return self.github_versions | self.pypi_versions


def parse_version(value: object, *, label: str = "version") -> tuple[int, int, int]:
    """Return one canonical stable SemVer as bounded integer components."""

    if not isinstance(value, str) or _CANONICAL_VERSION.fullmatch(value) is None:
        raise ReleaseIdentityError(
            f"{label} must be canonical MAJOR.MINOR.PATCH with components from "
            "0 through 999999999"
        )
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def compare_versions(left: str, right: str) -> str:
    """Compare two admitted versions and return older, equal, or newer."""

    left_parts = parse_version(left, label="left version")
    right_parts = parse_version(right, label="right version")
    if left_parts < right_parts:
        return "older"
    if left_parts > right_parts:
        return "newer"
    return "equal"


def read_project_identity(path: Path = DEFAULT_PROJECT) -> ProjectIdentity:
    """Read the exact release-relevant project metadata."""

    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except tomllib.TOMLDecodeError as error:
        raise ReleaseIdentityError(f"project file is invalid TOML: {error}") from error
    project = document.get("project")
    if not isinstance(project, dict):
        raise ReleaseIdentityError("project file has no [project] table")
    name = project.get("name")
    version = project.get("version")
    requires_python = project.get("requires-python")
    if name != "daita-agents":
        raise ReleaseIdentityError("project.name must be daita-agents")
    parse_version(version, label="project.version")
    if not isinstance(requires_python, str) or not requires_python:
        raise ReleaseIdentityError("project.requires-python must be a non-empty string")
    assert isinstance(version, str)
    return ProjectIdentity(
        name=name,
        version=version,
        requires_python=requires_python,
    )


def parse_published_version_evidence(
    github_pages: object, pypi_document: object
) -> PublishedVersionEvidence:
    """Validate complete, already-fetched GitHub pages and the PyPI release map."""

    if not isinstance(github_pages, list) or not all(
        isinstance(page, list) for page in github_pages
    ):
        raise ReleaseIdentityError("GitHub pagination response is malformed")
    releases = [release for page in github_pages for release in page]
    if not all(isinstance(release, dict) for release in releases):
        raise ReleaseIdentityError("GitHub release response is malformed")
    github_tags: set[str] = set()
    github_versions: set[str] = set()
    for release in releases:
        tag = release.get("tag_name")
        draft = release.get("draft")
        prerelease = release.get("prerelease")
        if (
            not isinstance(tag, str)
            or not isinstance(draft, bool)
            or not isinstance(prerelease, bool)
        ):
            raise ReleaseIdentityError("GitHub release response is malformed")
        github_tags.add(tag)
        if draft:
            continue
        if prerelease:
            raise ReleaseIdentityError(
                f"published GitHub prerelease {tag} requires operator review"
            )
        value = tag[1:] if tag.startswith("v") else tag
        parse_version(value, label=f"published GitHub release {tag}")
        github_versions.add(value)

    if not isinstance(pypi_document, dict):
        raise ReleaseIdentityError("PyPI project response is malformed")
    release_map = pypi_document.get("releases")
    if not isinstance(release_map, dict):
        raise ReleaseIdentityError("PyPI project response has no release map")
    pypi_versions: set[str] = set()
    for value, files in release_map.items():
        if (
            not isinstance(value, str)
            or not isinstance(files, list)
            or not all(isinstance(file, dict) for file in files)
        ):
            raise ReleaseIdentityError("PyPI release map is malformed")
        if files:
            parse_version(value, label=f"published PyPI release {value}")
            pypi_versions.add(value)
    return PublishedVersionEvidence(
        github_tags=frozenset(github_tags),
        github_versions=frozenset(github_versions),
        pypi_versions=frozenset(pypi_versions),
    )


def prior_versions_for_mode(
    evidence: PublishedVersionEvidence,
    *,
    candidate: str,
    tag: str,
    mode: Literal["branch", "candidate", "protected"],
    tagged: bool = False,
) -> tuple[str, ...]:
    """Apply mode-specific equality rules and return the complete prior union."""

    parse_version(candidate, label="candidate version")
    if tag != f"v{candidate}":
        raise ReleaseIdentityError("candidate tag does not match candidate version")
    if mode not in {"branch", "candidate", "protected"}:
        raise ReleaseIdentityError(f"unsupported release workflow mode: {mode}")
    if (mode != "branch" or tagged) and tag in evidence.github_tags:
        raise ReleaseIdentityError(f"GitHub release {tag} already exists")
    if mode == "candidate" and candidate in evidence.pypi_versions:
        raise ReleaseIdentityError(f"candidate {candidate} already has files on PyPI")
    if mode == "protected" and candidate not in evidence.pypi_versions:
        raise ReleaseIdentityError(
            f"protected candidate {candidate} is missing from PyPI"
        )
    published_versions = set(evidence.union)
    if mode == "protected":
        published_versions.discard(candidate)
    return tuple(sorted(published_versions, key=parse_version))


def require_newer(candidate: str, previous: list[str]) -> None:
    """Require candidate to exceed every admitted version in previous."""

    candidate_parts = parse_version(candidate, label="candidate version")
    parsed_previous = [
        (parse_version(value, label=f"previous version on line {index}"), value)
        for index, value in enumerate(previous, start=1)
    ]
    if not parsed_previous:
        return
    maximum_parts, maximum = max(parsed_previous)
    if candidate_parts <= maximum_parts:
        relation = "equal to" if candidate_parts == maximum_parts else "older than"
        raise ReleaseIdentityError(
            f"candidate version {candidate} is {relation} published maximum {maximum}"
        )


def _prior_versions(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseIdentityError("previous-version file must be UTF-8") from error
    if not content:
        return []
    values = content.splitlines()
    if any(not value for value in values):
        raise ReleaseIdentityError(
            "previous-version file must contain one non-empty version per line"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("project-version", "project-tag"):
        command = subparsers.add_parser(name)
        command.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    compare = subparsers.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    increase = subparsers.add_parser("require-project-increase")
    increase.add_argument("--base-project", type=Path, required=True)
    increase.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    newer = subparsers.add_parser("require-newer")
    newer.add_argument("--candidate", required=True)
    newer.add_argument("--previous-file", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in {"project-version", "project-tag"}:
            identity = read_project_identity(arguments.project)
            print(
                identity.version
                if arguments.command == "project-version"
                else identity.tag
            )
        elif arguments.command == "compare":
            print(compare_versions(arguments.left, arguments.right))
        elif arguments.command == "require-project-increase":
            base = read_project_identity(arguments.base_project)
            current = read_project_identity(arguments.project)
            relation = compare_versions(current.version, base.version)
            if relation != "newer":
                raise ReleaseIdentityError(
                    f"project.version {current.version} must be newer than base "
                    f"version {base.version}; found {relation}"
                )
        elif arguments.command == "require-newer":
            require_newer(arguments.candidate, _prior_versions(arguments.previous_file))
        else:  # pragma: no cover - argparse owns command admission
            raise AssertionError("unreachable release identity command")
    except (OSError, ReleaseIdentityError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
