#!/usr/bin/env python3
"""Promote one verified Daita release into the Lightsail stable installer path."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPOSITORY = "Daita-Corp/daita-agents"
RELEASE_BASE_URL = f"https://github.com/{REPOSITORY}/releases/download"
DEFAULT_ROOT = Path("/srv/daita-installer")
ASSET_NAMES = ("install.sh", "release-manifest.json", "SHA256SUMS")
MAXIMUM_ASSET_BYTES = {
    "install.sh": 2 * 1024 * 1024,
    "release-manifest.json": 256 * 1024,
    "SHA256SUMS": 256 * 1024,
}
# The first stable release predates the current manifest schema. Its complete
# critical identity is pinned here solely so Lightsail can be seeded without a
# mutable compatibility rule. Every later release must use schema 2.
LEGACY_SCHEMA_ONE_RELEASES: Final[dict[str, tuple[str, str, str]]] = {
    "v1.0.1": (
        "1e27d90870b262665c9fe777f7a1f1d645a8a4aa86a9a2724314b300deb17bc9",
        "56ed3686b28e24bf4e7a0b96dbccfdb52a1f0021f8a1ea0eafe8206120c02d88",
        "2b46f0a440b722b86734d65bbf1fb86fb9e1c958bb99bcb1845d21dc806fafce",
    ),
}
_TAG = re.compile(
    r"v(0|[1-9][0-9]{0,8})\." r"(0|[1-9][0-9]{0,8})\." r"(0|[1-9][0-9]{0,8})\Z"
)
_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]{0,199})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORCED_PROMOTION = re.compile(
    r"promote (v(?:0|[1-9][0-9]{0,8})\.(?:0|[1-9][0-9]{0,8})\."
    r"(?:0|[1-9][0-9]{0,8}))\Z"
)


class InstallerPromotionError(RuntimeError):
    """Raised when stable installer promotion cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class ReleaseBundle:
    tag: str
    version: str
    directory: Path
    installer_sha256: str
    previous_tag: str | None = None


Download = Callable[[str, Path, int], None]
ValidateInstaller = Callable[[Path, str], None]


def parse_tag(tag: object) -> str:
    """Return the version from one canonical stable release tag."""

    if not isinstance(tag, str) or _TAG.fullmatch(tag) is None:
        raise InstallerPromotionError(
            "tag must be canonical vMAJOR.MINOR.PATCH with bounded integer components"
        )
    return tag[1:]


def compare_versions(left: str, right: str) -> str:
    """Compare two already-admitted canonical versions."""

    left_parts = tuple(int(part) for part in parse_tag(f"v{left}").split("."))
    right_parts = tuple(int(part) for part in parse_tag(f"v{right}").split("."))
    if left_parts < right_parts:
        return "older"
    if left_parts > right_parts:
        return "newer"
    return "equal"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, maximum_bytes: int) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise InstallerPromotionError(
            f"required release asset is missing: {path.name}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise InstallerPromotionError(
            f"release asset is not a regular file: {path.name}"
        )
    if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
        raise InstallerPromotionError(f"release asset has an invalid size: {path.name}")


def _require_mode(path: Path, expected: int) -> None:
    metadata = path.lstat()
    actual = stat.S_IMODE(metadata.st_mode)
    if actual != expected:
        raise InstallerPromotionError(
            f"installed release permissions are invalid: {path.name} is {actual:04o}"
        )
    if metadata.st_uid != os.geteuid():
        raise InstallerPromotionError(
            f"installed release ownership is invalid: {path.name}"
        )


def _read_json_object(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    _require_regular_file(path, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstallerPromotionError(f"{path.name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise InstallerPromotionError(f"{path.name} must contain one JSON object")
    return value


def _require_exact_object(
    value: object, *, fields: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise InstallerPromotionError(
            f"{label} fields do not match the release contract"
        )
    return value


def _parse_checksums(path: Path) -> dict[str, str]:
    _require_regular_file(path, maximum_bytes=MAXIMUM_ASSET_BYTES[path.name])
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise InstallerPromotionError("SHA256SUMS is not UTF-8") from error
    if not lines:
        raise InstallerPromotionError("SHA256SUMS is empty")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        matched = _CHECKSUM_LINE.fullmatch(line)
        if matched is None:
            raise InstallerPromotionError(
                f"SHA256SUMS line {line_number} is not canonical"
            )
        digest, filename = matched.groups()
        if filename in checksums:
            raise InstallerPromotionError(f"SHA256SUMS repeats {filename}")
        checksums[filename] = digest
    return checksums


def _metadata_path(directory: Path) -> Path:
    return directory / "promotion.json"


def validate_release_bundle(
    directory: Path, tag: str, *, require_metadata: bool = False
) -> ReleaseBundle:
    """Validate one downloaded or installed release directory without execution."""

    version = parse_tag(tag)
    installer = directory / "install.sh"
    manifest_path = directory / "release-manifest.json"
    checksums_path = directory / "SHA256SUMS"
    _require_regular_file(installer, maximum_bytes=MAXIMUM_ASSET_BYTES["install.sh"])
    manifest = _read_json_object(
        manifest_path, maximum_bytes=MAXIMUM_ASSET_BYTES["release-manifest.json"]
    )
    checksums = _parse_checksums(checksums_path)

    schema_version = manifest.get("schema_version")
    legacy_identity = LEGACY_SCHEMA_ONE_RELEASES.get(tag)
    if schema_version == 1:
        if legacy_identity is None:
            raise InstallerPromotionError(
                "release manifest schema 1 is not admitted for this tag"
            )
        if sha256_file(manifest_path) != legacy_identity[1]:
            raise InstallerPromotionError(
                "legacy release manifest does not match its pinned identity"
            )
    elif schema_version != 2:
        raise InstallerPromotionError("release manifest schema_version must be 2")
    else:
        _require_exact_object(
            manifest,
            fields={"schema_version", "application", "installer", "wheel", "runtime"},
            label="release manifest",
        )
        if not isinstance(manifest["runtime"], dict):
            raise InstallerPromotionError("release manifest runtime must be an object")
    application = _require_exact_object(
        manifest.get("application"),
        fields={"version", "requires_python"},
        label="release manifest application",
    )
    if schema_version == 1:
        installer_record = manifest.get("installer")
        if not isinstance(installer_record, dict) or not {
            "filename",
            "sha256",
        }.issubset(installer_record):
            raise InstallerPromotionError(
                "legacy release manifest installer fields are invalid"
            )
    else:
        installer_record = _require_exact_object(
            manifest.get("installer"),
            fields={"filename", "sha256"},
            label="release manifest installer",
        )
    wheel = _require_exact_object(
        manifest.get("wheel"),
        fields={"filename", "url", "sha256"},
        label="release manifest wheel",
    )
    if application.get("version") != version:
        raise InstallerPromotionError("release manifest version does not match the tag")
    if (
        not isinstance(application.get("requires_python"), str)
        or not application["requires_python"]
    ):
        raise InstallerPromotionError("release manifest requires_python is invalid")
    if installer_record.get("filename") != "install.sh":
        raise InstallerPromotionError("release manifest installer filename is invalid")
    expected_installer = installer_record.get("sha256")
    if (
        not isinstance(expected_installer, str)
        or _SHA256.fullmatch(expected_installer) is None
    ):
        raise InstallerPromotionError("release manifest installer checksum is invalid")

    wheel_filename = f"daita_agents-{version}-py3-none-any.whl"
    if wheel.get("filename") != wheel_filename:
        raise InstallerPromotionError("release manifest wheel filename is invalid")
    expected_wheel_url = f"{RELEASE_BASE_URL}/{tag}/{wheel_filename}"
    if wheel.get("url") != expected_wheel_url:
        raise InstallerPromotionError("release manifest wheel URL is invalid")
    wheel_sha256 = wheel.get("sha256")
    if not isinstance(wheel_sha256, str) or _SHA256.fullmatch(wheel_sha256) is None:
        raise InstallerPromotionError("release manifest wheel checksum is invalid")
    expected_checksum_names = {"install.sh", "release-manifest.json", wheel_filename}
    if schema_version == 2:
        expected_checksum_names.add("agent-home-contract.json")
    if set(checksums) != expected_checksum_names:
        raise InstallerPromotionError(
            "SHA256SUMS entries do not match the exact release contract"
        )

    installer_sha256 = sha256_file(installer)
    if (
        schema_version == 1
        and legacy_identity is not None
        and (
            installer_sha256 != legacy_identity[0] or wheel_sha256 != legacy_identity[2]
        )
    ):
        raise InstallerPromotionError(
            "legacy release assets do not match their pinned identity"
        )
    if installer_sha256 != expected_installer:
        raise InstallerPromotionError("installer does not match the release manifest")
    if checksums.get("install.sh") != installer_sha256:
        raise InstallerPromotionError("installer does not match SHA256SUMS")
    if checksums.get("release-manifest.json") != sha256_file(manifest_path):
        raise InstallerPromotionError("release manifest does not match SHA256SUMS")
    if checksums.get(wheel_filename) != wheel_sha256:
        raise InstallerPromotionError(
            "wheel manifest evidence does not match SHA256SUMS"
        )

    previous_tag: str | None = None
    metadata_path = _metadata_path(directory)
    if require_metadata:
        metadata = _read_json_object(metadata_path, maximum_bytes=64 * 1024)
        if set(metadata) != {
            "schema_version",
            "tag",
            "version",
            "installer_sha256",
            "previous_tag",
        }:
            raise InstallerPromotionError(
                "installed promotion metadata fields are invalid"
            )
        if (
            metadata.get("schema_version") != 1
            or metadata.get("tag") != tag
            or metadata.get("version") != version
            or metadata.get("installer_sha256") != installer_sha256
        ):
            raise InstallerPromotionError(
                "installed promotion metadata is inconsistent"
            )
        candidate_previous = metadata.get("previous_tag")
        if candidate_previous is not None:
            parse_tag(candidate_previous)
            if compare_versions(candidate_previous[1:], version) != "older":
                raise InstallerPromotionError("installed previous tag is not older")
            previous_tag = candidate_previous
        _require_mode(directory, 0o555)
        _require_mode(installer, 0o555)
        _require_mode(manifest_path, 0o444)
        _require_mode(checksums_path, 0o444)
        _require_mode(metadata_path, 0o444)

    return ReleaseBundle(
        tag=tag,
        version=version,
        directory=directory,
        installer_sha256=installer_sha256,
        previous_tag=previous_tag,
    )


def curl_download(url: str, destination: Path, maximum_bytes: int) -> None:
    """Download one fixed release asset over HTTPS and enforce its byte ceiling."""

    try:
        completed = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--tlsv1.2",
                "--connect-timeout",
                "15",
                "--max-time",
                "120",
                "--max-filesize",
                str(maximum_bytes),
                "--retry",
                "5",
                "--retry-all-errors",
                "--output",
                str(destination),
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=150,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C"},
        )
    except subprocess.TimeoutExpired as error:
        raise InstallerPromotionError("release asset download timed out") from error
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"curl exited {completed.returncode}"
        raise InstallerPromotionError(f"release asset download failed: {message}")
    _require_regular_file(destination, maximum_bytes=maximum_bytes)


def validate_installer_syntax(installer: Path, version: str) -> None:
    """Parse downloaded installer bytes without executing the release script."""

    del version
    try:
        completed = subprocess.run(
            ["bash", "-n", str(installer)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C"},
        )
    except subprocess.TimeoutExpired as error:
        raise InstallerPromotionError(
            "installer syntax validation timed out"
        ) from error
    if completed.returncode != 0:
        raise InstallerPromotionError(
            f"installer syntax validation failed with exit {completed.returncode}"
        )


def validate_installer_commands(installer: Path, version: str) -> None:
    """Exercise the rendered installer's non-mutating surfaces off the host."""

    with tempfile.TemporaryDirectory(prefix="daita-installer-validation-") as temporary:
        home = Path(temporary)
        environment = {
            "HOME": str(home),
            "LANG": "C",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        }
        commands = (
            (["bash", "-n", str(installer)], "syntax"),
            (["bash", str(installer), "--version"], "version"),
            (
                [
                    "bash",
                    str(installer),
                    "--dry-run",
                    "--no-onboard",
                    "--no-modify-path",
                ],
                "dry-run",
            ),
        )
        for command, label in commands:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                raise InstallerPromotionError(
                    f"installer {label} validation timed out"
                ) from error
            if completed.returncode != 0:
                raise InstallerPromotionError(
                    f"installer {label} validation failed with exit "
                    f"{completed.returncode}"
                )
            if label in {"version", "dry-run"} and version not in completed.stdout:
                raise InstallerPromotionError(
                    f"installer {label} output does not contain the promoted version"
                )


class ManagedInstallerHost:
    """Own immutable host releases and the one atomic stable pointer."""

    def __init__(
        self,
        root: Path = DEFAULT_ROOT,
        *,
        download: Download = curl_download,
        validate_installer: ValidateInstaller = validate_installer_syntax,
    ) -> None:
        self.root = root
        self.releases = root / "releases"
        self.current = root / "current"
        self.lock = root / ".promotion.lock"
        self._download = download
        self._validate_installer = validate_installer

    def _admit_root(self) -> None:
        if not self.root.is_absolute():
            raise InstallerPromotionError("installer root must be absolute")
        try:
            metadata = self.root.lstat()
        except FileNotFoundError as error:
            raise InstallerPromotionError(
                f"installer root does not exist: {self.root}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise InstallerPromotionError("installer root must be a real directory")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o755:
            raise InstallerPromotionError(
                "installer root ownership or permissions are invalid"
            )
        if self.releases.exists() or self.releases.is_symlink():
            release_metadata = self.releases.lstat()
            if not stat.S_ISDIR(release_metadata.st_mode):
                raise InstallerPromotionError(
                    "installer host releases path must be a real directory"
                )
            if (
                release_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(release_metadata.st_mode) != 0o755
            ):
                raise InstallerPromotionError(
                    "installer releases ownership or permissions are invalid"
                )
        else:
            self.releases.mkdir(mode=0o755)
            os.chmod(self.releases, 0o755)

    @contextmanager
    def _promotion_lock(self) -> Iterator[None]:
        try:
            descriptor = os.open(
                self.lock,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except OSError as error:
            raise InstallerPromotionError("promotion lock is unsafe") from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise InstallerPromotionError("promotion lock is unsafe")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _current_bundle(self) -> ReleaseBundle | None:
        if not self.current.exists() and not self.current.is_symlink():
            return None
        if not self.current.is_symlink():
            raise InstallerPromotionError("current installer pointer is not a symlink")
        target = os.readlink(self.current)
        parts = Path(target).parts
        if len(parts) != 2 or parts[0] != "releases":
            raise InstallerPromotionError(
                "current installer pointer has an invalid target"
            )
        tag = parts[1]
        parse_tag(tag)
        if target != f"releases/{tag}":
            raise InstallerPromotionError(
                "current installer pointer target is not canonical"
            )
        directory = self.releases / tag
        if not directory.exists() or directory.is_symlink() or not directory.is_dir():
            raise InstallerPromotionError(
                "current installer release is missing or unsafe"
            )
        return validate_release_bundle(directory, tag, require_metadata=True)

    def _installed_bundle(self, tag: str) -> ReleaseBundle | None:
        directory = self.releases / tag
        if not directory.exists() and not directory.is_symlink():
            return None
        if directory.is_symlink() or not directory.is_dir():
            raise InstallerPromotionError("installed release path is unsafe")
        return validate_release_bundle(directory, tag, require_metadata=True)

    def _download_bundle(self, tag: str, directory: Path) -> ReleaseBundle:
        for name in ASSET_NAMES:
            self._download(
                f"{RELEASE_BASE_URL}/{tag}/{name}",
                directory / name,
                MAXIMUM_ASSET_BYTES[name],
            )
        return validate_release_bundle(directory, tag)

    def _write_metadata(
        self, directory: Path, bundle: ReleaseBundle, previous_tag: str | None
    ) -> None:
        document = {
            "schema_version": 1,
            "tag": bundle.tag,
            "version": bundle.version,
            "installer_sha256": bundle.installer_sha256,
            "previous_tag": previous_tag,
        }
        path = _metadata_path(directory)
        with path.open("x", encoding="utf-8") as destination:
            json.dump(document, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(path, 0o644)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _switch_current(self, tag: str) -> None:
        temporary = self.root / f".current-{os.getpid()}-{secrets.token_hex(8)}"
        if temporary.exists() or temporary.is_symlink():
            raise InstallerPromotionError("temporary current pointer already exists")
        try:
            os.symlink(f"releases/{tag}", temporary)
            os.replace(temporary, self.current)
            self._fsync_directory(self.root)
        finally:
            if temporary.is_symlink():
                temporary.unlink()

    def promote(self, tag: str) -> ReleaseBundle:
        """Download, validate, and atomically publish a forward release."""

        version = parse_tag(tag)
        self._admit_root()
        with self._promotion_lock():
            current = self._current_bundle()
            if current is not None:
                relation = compare_versions(version, current.version)
                if relation == "older":
                    raise InstallerPromotionError(
                        f"Daita {version} cannot replace newer stable Daita {current.version}"
                    )

            stage = Path(tempfile.mkdtemp(prefix=f".{tag}-", dir=self.releases))
            published = False
            try:
                candidate = self._download_bundle(tag, stage)
                self._validate_installer(stage / "install.sh", version)
                installed = self._installed_bundle(tag)
                if installed is not None:
                    if installed.installer_sha256 != candidate.installer_sha256:
                        raise InstallerPromotionError(
                            "installed release bytes conflict with the downloaded release"
                        )
                    selected = installed
                else:
                    previous_tag = None if current is None else current.tag
                    self._write_metadata(stage, candidate, previous_tag)
                    os.chmod(stage / "install.sh", 0o555)
                    for name in ("release-manifest.json", "SHA256SUMS"):
                        os.chmod(stage / name, 0o444)
                    os.chmod(_metadata_path(stage), 0o444)
                    os.chmod(stage, 0o555)
                    self._fsync_directory(stage)
                    destination = self.releases / tag
                    os.replace(stage, destination)
                    published = True
                    self._fsync_directory(destination)
                    self._fsync_directory(self.releases)
                    selected = validate_release_bundle(
                        destination, tag, require_metadata=True
                    )
                self._switch_current(tag)
                return selected
            finally:
                if not published and stage.exists():
                    os.chmod(stage, 0o700)
                    shutil.rmtree(stage)

    def rollback(self) -> ReleaseBundle:
        """Atomically restore the predecessor recorded by the current release."""

        self._admit_root()
        with self._promotion_lock():
            current = self._current_bundle()
            if current is None:
                raise InstallerPromotionError("no stable installer is published")
            if current.previous_tag is None:
                raise InstallerPromotionError(
                    "the current stable installer has no recorded predecessor"
                )
            previous = self._installed_bundle(current.previous_tag)
            if previous is None:
                raise InstallerPromotionError("the recorded predecessor is missing")
            self._validate_installer(
                previous.directory / "install.sh", previous.version
            )
            self._switch_current(previous.tag)
            return previous


def forced_promotion_tag(environment: dict[str, str] | None = None) -> str:
    """Admit exactly one command supplied by an OpenSSH forced-command key."""

    values = os.environ if environment is None else environment
    original = values.get("SSH_ORIGINAL_COMMAND", "")
    matched = _FORCED_PROMOTION.fullmatch(original)
    if matched is None:
        raise InstallerPromotionError(
            "forced deployment key accepts only: promote vMAJOR.MINOR.PATCH"
        )
    tag = matched.group(1)
    parse_tag(tag)
    return tag


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("tag")
    subparsers.add_parser("forced-promote")
    subparsers.add_parser("rollback")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        host = ManagedInstallerHost(arguments.root)
        if arguments.command == "promote":
            bundle = host.promote(arguments.tag)
            print(f"promoted {bundle.tag} {bundle.installer_sha256}")
        elif arguments.command == "forced-promote":
            bundle = host.promote(forced_promotion_tag())
            print(f"promoted {bundle.tag} {bundle.installer_sha256}")
        elif arguments.command == "rollback":
            bundle = host.rollback()
            print(f"rolled back to {bundle.tag} {bundle.installer_sha256}")
        else:  # pragma: no cover - argparse owns command admission
            raise AssertionError("unreachable command")
    except (InstallerPromotionError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
