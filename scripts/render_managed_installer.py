#!/usr/bin/env python3
"""Render one reviewed, immutable Daita managed-installer release."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import stat
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "release" / "managed-installer.json"
DEFAULT_PROJECT = ROOT / "pyproject.toml"
DEFAULT_TEMPLATE = ROOT / "scripts" / "install.sh"

TARGET_PLACEHOLDERS = {
    "macos-arm64": "DARWIN_ARM64",
    "macos-x86_64": "DARWIN_X86_64",
    "linux-arm64-glibc": "LINUX_ARM64",
    "linux-x86_64-glibc": "LINUX_X86_64",
}
TARGET_RUNTIME_SHAPES = {
    "macos-arm64": (
        "uv-aarch64-apple-darwin.tar.gz",
        "macos-aarch64-none",
    ),
    "macos-x86_64": (
        "uv-x86_64-apple-darwin.tar.gz",
        "macos-x86_64-none",
    ),
    "linux-arm64-glibc": (
        "uv-aarch64-unknown-linux-gnu.tar.gz",
        "linux-aarch64-gnu",
    ),
    "linux-x86_64-glibc": (
        "uv-x86_64-unknown-linux-gnu.tar.gz",
        "linux-x86_64-gnu",
    ),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?\Z")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_PYTHON_REQUEST = re.compile(r"cpython-3\.12\.[0-9]+(?:\+[A-Za-z0-9._-]+)?\Z")
_PYTHON_IDENTITY = re.compile(
    r"cpython-3\.12\.[0-9]+-(?:macos|linux)-" r"(?:aarch64|x86_64)-(?:none|gnu)\Z"
)


class ReleaseInputError(ValueError):
    """Raised when release evidence is incomplete, mutable, or inconsistent."""


@dataclass(frozen=True, slots=True)
class WheelMetadata:
    filename: str
    version: str
    requires_python: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RenderedRelease:
    installer: str
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_object(value: object, *, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseInputError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ReleaseInputError(
            f"{label} fields do not match the release contract; "
            f"missing={missing}, extra={extra}"
        )
    return value


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseInputError(f"{label} must be a non-empty string")
    if any(character in value for character in ('"', "\\", "$", "`", "\n", "\r", "\t")):
        raise ReleaseInputError(f"{label} is unsafe for the shell installer")
    if "UNRESOLVED" in value:
        raise ReleaseInputError(f"{label} is unresolved")
    return value


def _safe_token(value: object, *, label: str) -> str:
    token = _required_string(value, label=label)
    if _SAFE_TOKEN.fullmatch(token) is None:
        raise ReleaseInputError(f"{label} contains unsupported characters")
    return token


def _immutable_url(value: object, *, label: str, expected_filename: str) -> str:
    url = _required_string(value, label=label)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ReleaseInputError(
            f"{label} must be an HTTPS artifact URL without credentials, "
            "a query, or a fragment"
        )
    segments = tuple(unquote(segment) for segment in parsed.path.split("/") if segment)
    if not segments or segments[-1] != expected_filename:
        raise ReleaseInputError(f"{label} must end with {expected_filename}")
    if any(segment.lower() == "latest" for segment in segments):
        raise ReleaseInputError(f"{label} contains a mutable latest segment")
    return url


def load_release_policy(path: Path) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReleaseInputError(f"release policy is invalid JSON: {error}") from error

    root = _required_object(
        policy,
        label="release policy",
        keys={"schema_version", "installer", "runtime"},
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ReleaseInputError("release policy schema_version must be 1")

    installer = _required_object(
        root["installer"],
        label="installer policy",
        keys={"version", "release_sequence"},
    )
    installer_version = _required_string(
        installer["version"], label="installer.version"
    )
    if _VERSION.fullmatch(installer_version) is None:
        raise ReleaseInputError("installer.version must be a semantic version")
    release_sequence = installer["release_sequence"]
    if not isinstance(release_sequence, int) or isinstance(release_sequence, bool):
        raise ReleaseInputError("installer.release_sequence must be an integer")
    if release_sequence < 1:
        raise ReleaseInputError("installer.release_sequence must be positive")

    runtime = _required_object(
        root["runtime"],
        label="runtime policy",
        keys={"uv_version", "python_request", "targets"},
    )
    uv_version = _safe_token(runtime["uv_version"], label="runtime.uv_version")
    if _VERSION.fullmatch(uv_version) is None:
        raise ReleaseInputError("runtime.uv_version must be a semantic version")
    python_request = _safe_token(
        runtime["python_request"], label="runtime.python_request"
    )
    if _PYTHON_REQUEST.fullmatch(python_request) is None:
        raise ReleaseInputError(
            "runtime.python_request must pin one exact CPython 3.12 patch release"
        )

    targets = _required_object(
        runtime["targets"],
        label="runtime.targets",
        keys=set(TARGET_PLACEHOLDERS),
    )
    normalized_targets: dict[str, dict[str, str]] = {}
    for target_name in TARGET_PLACEHOLDERS:
        target = _required_object(
            targets[target_name],
            label=f"runtime.targets.{target_name}",
            keys={
                "uv_archive",
                "uv_member",
                "uv_url",
                "uv_sha256",
                "python_identity",
            },
        )
        archive = _safe_token(target["uv_archive"], label=f"{target_name}.uv_archive")
        if not archive.endswith(".tar.gz"):
            raise ReleaseInputError(f"{target_name}.uv_archive must be a .tar.gz file")
        expected_archive, identity_suffix = TARGET_RUNTIME_SHAPES[target_name]
        if archive != expected_archive:
            raise ReleaseInputError(
                f"{target_name}.uv_archive must be the official {expected_archive} asset"
            )
        member = _required_string(target["uv_member"], label=f"{target_name}.uv_member")
        if member != f"{archive.removesuffix('.tar.gz')}/uv":
            raise ReleaseInputError(
                f"{target_name}.uv_member must select the uv binary in its archive"
            )
        url = _immutable_url(
            target["uv_url"],
            label=f"{target_name}.uv_url",
            expected_filename=archive,
        )
        official_url = (
            "https://github.com/astral-sh/uv/releases/download/"
            f"{uv_version}/{archive}"
        )
        if url != official_url:
            raise ReleaseInputError(
                f"{target_name}.uv_url must be the official versioned uv release URL"
            )
        sha256 = _required_string(target["uv_sha256"], label=f"{target_name}.uv_sha256")
        if _SHA256.fullmatch(sha256) is None:
            raise ReleaseInputError(f"{target_name}.uv_sha256 must be 64 lowercase hex")
        identity = _safe_token(
            target["python_identity"], label=f"{target_name}.python_identity"
        )
        if _PYTHON_IDENTITY.fullmatch(identity) is None:
            raise ReleaseInputError(
                f"{target_name}.python_identity must be an exact CPython 3.12 identity"
            )
        python_version = python_request.removeprefix("cpython-").split("+", 1)[0]
        expected_identity = f"cpython-{python_version}-{identity_suffix}"
        if identity != expected_identity:
            raise ReleaseInputError(
                f"{target_name}.python_identity must be {expected_identity}"
            )
        normalized_targets[target_name] = {
            "uv_archive": archive,
            "uv_member": member,
            "uv_url": url,
            "uv_sha256": sha256,
            "python_identity": identity,
        }

    return {
        "schema_version": 1,
        "installer": {
            "version": installer_version,
            "release_sequence": release_sequence,
        },
        "runtime": {
            "uv_version": uv_version,
            "python_request": python_request,
            "targets": normalized_targets,
        },
    }


def inspect_wheel(path: Path) -> WheelMetadata:
    if not path.is_file() or path.is_symlink():
        raise ReleaseInputError("candidate wheel must be one regular file")
    if "/" in path.name or not path.name.endswith(".whl"):
        raise ReleaseInputError("candidate wheel has an unsafe filename")

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ReleaseInputError("candidate wheel contains duplicate members")
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or "\\" in info.filename
                ):
                    raise ReleaseInputError("candidate wheel contains an unsafe path")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ReleaseInputError("candidate wheel contains a symbolic link")

            metadata_files = [
                info for info in infos if info.filename.endswith(".dist-info/METADATA")
            ]
            entry_files = [
                info
                for info in infos
                if info.filename.endswith(".dist-info/entry_points.txt")
            ]
            wheel_files = [
                info for info in infos if info.filename.endswith(".dist-info/WHEEL")
            ]
            if not (len(metadata_files) == len(entry_files) == len(wheel_files) == 1):
                raise ReleaseInputError(
                    "candidate wheel must contain one metadata, entry-point, and WHEEL file"
                )
            roots = {
                info.filename.split("/", 1)[0]
                for info in (*metadata_files, *entry_files, *wheel_files)
            }
            if len(roots) != 1:
                raise ReleaseInputError(
                    "candidate wheel metadata roots are inconsistent"
                )

            message = Parser().parsestr(archive.read(metadata_files[0]).decode("utf-8"))
            if message.get_all("Name") != ["daita-agents"]:
                raise ReleaseInputError(
                    "candidate wheel distribution is not daita-agents"
                )
            versions = message.get_all("Version")
            python_requirements = message.get_all("Requires-Python")
            if versions is None or len(versions) != 1:
                raise ReleaseInputError("candidate wheel must declare one version")
            if python_requirements is None or len(python_requirements) != 1:
                raise ReleaseInputError(
                    "candidate wheel must declare one Requires-Python"
                )
            version = versions[0]
            requires_python = _required_string(
                python_requirements[0], label="candidate wheel Requires-Python"
            )
            if _VERSION.fullmatch(version) is None:
                raise ReleaseInputError("candidate wheel version is not supported")
            expected_root = f"daita_agents-{version}.dist-info"
            if roots != {expected_root}:
                raise ReleaseInputError("candidate wheel metadata path is inconsistent")
            expected_filename = f"daita_agents-{version}-py3-none-any.whl"
            if path.name != expected_filename:
                raise ReleaseInputError(
                    f"candidate wheel filename must be {expected_filename}"
                )

            entries = configparser.ConfigParser(interpolation=None, strict=True)
            entries.read_string(archive.read(entry_files[0]).decode("utf-8"))
            if set(entries.sections()) != {"console_scripts"} or dict(
                entries.items("console_scripts")
            ) != {"daita": "daita.cli:main"}:
                raise ReleaseInputError(
                    "candidate wheel entry point is not daita.cli:main"
                )

            wheel_document = Parser().parsestr(
                archive.read(wheel_files[0]).decode("utf-8")
            )
            if wheel_document.get_all("Root-Is-Purelib") != ["true"]:
                raise ReleaseInputError("candidate wheel must be a pure-Python wheel")
            if wheel_document.get_all("Tag") != ["py3-none-any"]:
                raise ReleaseInputError("candidate wheel tag must be py3-none-any")
    except (UnicodeDecodeError, zipfile.BadZipFile, configparser.Error) as error:
        raise ReleaseInputError(f"candidate wheel is malformed: {error}") from error

    return WheelMetadata(
        filename=path.name,
        version=version,
        requires_python=requires_python,
        sha256=_sha256(path),
    )


def project_metadata(path: Path) -> tuple[str, str]:
    with path.open("rb") as source:
        project = tomllib.load(source).get("project")
    if not isinstance(project, dict):
        raise ReleaseInputError("pyproject.toml has no project table")
    version = project.get("version")
    requires_python = project.get("requires-python")
    if not isinstance(version, str) or not isinstance(requires_python, str):
        raise ReleaseInputError("project version and requires-python must be strings")
    return version, requires_python


def _normalized_requires_python(value: str) -> tuple[str, ...]:
    return tuple(sorted(part.strip() for part in value.split(",") if part.strip()))


def _replace_once(source: str, placeholder: str, value: str) -> str:
    count = source.count(placeholder)
    if count != 1:
        raise ReleaseInputError(
            f"installer template must contain {placeholder} exactly once; found {count}"
        )
    return source.replace(placeholder, value)


def render_managed_installer(
    *,
    policy: dict[str, Any],
    wheel: Path,
    wheel_url: str,
    template: Path = DEFAULT_TEMPLATE,
) -> RenderedRelease:
    metadata = inspect_wheel(wheel)
    checked_wheel_url = _immutable_url(
        wheel_url, label="wheel URL", expected_filename=metadata.filename
    )
    expected_tag_segment = f"v{metadata.version}"
    wheel_segments = tuple(
        unquote(segment)
        for segment in urlsplit(checked_wheel_url).path.split("/")
        if segment
    )
    if expected_tag_segment not in wheel_segments:
        raise ReleaseInputError(
            f"wheel URL must contain the immutable release tag {expected_tag_segment}"
        )

    installer = policy["installer"]
    runtime = policy["runtime"]
    replacements = {
        "UNRESOLVED_INSTALLER_VERSION": installer["version"],
        "UNRESOLVED_RELEASE_SEQUENCE": str(installer["release_sequence"]),
        "UNRESOLVED_DAITA_VERSION": metadata.version,
        "UNRESOLVED_WHEEL_FILENAME": metadata.filename,
        "UNRESOLVED_WHEEL_URL": checked_wheel_url,
        "UNRESOLVED_WHEEL_SHA256": metadata.sha256,
        "UNRESOLVED_WHEEL_REQUIRES_PYTHON": metadata.requires_python,
        "UNRESOLVED_UV_VERSION": runtime["uv_version"],
        "UNRESOLVED_PYTHON_REQUEST": runtime["python_request"],
    }
    for target_name, placeholder_name in TARGET_PLACEHOLDERS.items():
        target = runtime["targets"][target_name]
        replacements.update(
            {
                f"UNRESOLVED_UV_{placeholder_name}_ARCHIVE": target["uv_archive"],
                f"UNRESOLVED_UV_{placeholder_name}_MEMBER": target["uv_member"],
                f"UNRESOLVED_UV_{placeholder_name}_URL": target["uv_url"],
                f"UNRESOLVED_UV_{placeholder_name}_SHA256": target["uv_sha256"],
                f"UNRESOLVED_PYTHON_{placeholder_name}_IDENTITY": target[
                    "python_identity"
                ],
            }
        )

    rendered = template.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        rendered = _replace_once(rendered, placeholder, value)
    if "UNRESOLVED_" in rendered:
        raise ReleaseInputError("rendered installer retains unresolved literals")
    if not rendered.startswith("#!/usr/bin/env bash\n"):
        raise ReleaseInputError("rendered installer lost its Bash shebang")

    installer_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "installer": {
            "filename": "install.sh",
            "version": installer["version"],
            "release_sequence": installer["release_sequence"],
            "sha256": installer_sha256,
        },
        "application": {
            "version": metadata.version,
            "requires_python": metadata.requires_python,
        },
        "wheel": {
            "filename": metadata.filename,
            "url": checked_wheel_url,
            "sha256": metadata.sha256,
        },
        "runtime": runtime,
    }
    return RenderedRelease(installer=rendered, manifest=manifest)


def _write_atomic(path: Path, content: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_rendered_release(
    release: RenderedRelease, *, installer_output: Path, manifest_output: Path
) -> None:
    _write_atomic(installer_output, release.installer.encode("utf-8"), mode=0o755)
    manifest = (json.dumps(release.manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_atomic(manifest_output, manifest, mode=0o644)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a Daita installer from reviewed immutable release evidence."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-url", required=True)
    parser.add_argument("--installer-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        policy = load_release_policy(arguments.policy)
        metadata = inspect_wheel(arguments.wheel)
        version, requires_python = project_metadata(arguments.project)
        if metadata.version != version:
            raise ReleaseInputError(
                "candidate wheel version does not match pyproject.toml"
            )
        if _normalized_requires_python(
            metadata.requires_python
        ) != _normalized_requires_python(requires_python):
            raise ReleaseInputError(
                "candidate wheel Requires-Python does not match pyproject.toml"
            )
        authoritative_wheel_url = (
            "https://github.com/Daita-Corp/daita-agents/releases/download/"
            f"v{version}/{metadata.filename}"
        )
        if arguments.wheel_url != authoritative_wheel_url:
            raise ReleaseInputError(
                "wheel URL must be the authoritative versioned Daita GitHub release URL"
            )
        release = render_managed_installer(
            policy=policy,
            wheel=arguments.wheel,
            wheel_url=arguments.wheel_url,
            template=arguments.template,
        )
        write_rendered_release(
            release,
            installer_output=arguments.installer_output,
            manifest_output=arguments.manifest_output,
        )
    except (OSError, ReleaseInputError, tomllib.TOMLDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
