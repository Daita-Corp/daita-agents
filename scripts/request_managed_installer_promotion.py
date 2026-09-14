#!/usr/bin/env python3
"""Request one constrained Lightsail promotion and verify the public endpoint."""

from __future__ import annotations

import argparse
import base64
import binascii
import ipaddress
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts.managed_installer_host import (
    InstallerPromotionError,
    ReleaseBundle,
    sha256_file,
    validate_installer_commands,
    validate_release_bundle,
)

STABLE_INSTALLER_URL = "https://daita-tech.io/install.sh"
_SSH_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}\Z")
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


def validate_ssh_destination(user: object, host: object) -> tuple[str, str]:
    if not isinstance(user, str) or _SSH_USER.fullmatch(user) is None:
        raise InstallerPromotionError("deployment SSH user is invalid")
    if not isinstance(host, str) or not host or len(host) > 253:
        raise InstallerPromotionError("deployment SSH host is invalid")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if all(label.isdigit() for label in labels) or any(
            _DNS_LABEL.fullmatch(label) is None for label in labels
        ):
            raise InstallerPromotionError("deployment SSH host is invalid") from None
    else:
        if address.version != 4:
            raise InstallerPromotionError("deployment SSH host must be IPv4 or DNS")
    return user, host


def _require_ssh_file(path: Path, *, private: bool) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise InstallerPromotionError(f"SSH file is missing: {path.name}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise InstallerPromotionError(
            f"SSH file is not a non-empty regular file: {path.name}"
        )
    if private and metadata.st_mode & 0o077:
        raise InstallerPromotionError(
            "deployment private key permissions are too broad"
        )


def validate_known_hosts(path: Path, host: str) -> None:
    """Require one exact unhashed Ed25519 key for the selected destination."""

    _require_ssh_file(path, private=False)
    if path.stat().st_size > 16 * 1024:
        raise InstallerPromotionError("SSH known-hosts file is too large")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as error:
        raise InstallerPromotionError("SSH known-hosts file is not ASCII") from error
    if len(lines) != 1:
        raise InstallerPromotionError("SSH known-hosts must contain exactly one key")
    fields = lines[0].split(" ")
    if len(fields) != 3 or fields[0] != host or fields[1] != "ssh-ed25519":
        raise InstallerPromotionError(
            "SSH known-hosts must pin the exact destination Ed25519 key"
        )
    try:
        decoded = base64.b64decode(fields[2], validate=True)
    except (ValueError, binascii.Error) as error:
        raise InstallerPromotionError("SSH known-hosts key is invalid") from error
    prefix = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"
    if len(decoded) != len(prefix) + 32 or not decoded.startswith(prefix):
        raise InstallerPromotionError("SSH known-hosts key is not Ed25519")


def request_remote_promotion(
    bundle: ReleaseBundle,
    *,
    user: str,
    host: str,
    identity_file: Path,
    known_hosts_file: Path,
) -> None:
    user, host = validate_ssh_destination(user, host)
    _require_ssh_file(identity_file, private=True)
    validate_known_hosts(known_hosts_file, host)
    completed = subprocess.run(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            str(identity_file),
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "HostKeyAlgorithms=ssh-ed25519",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "RequestTTY=no",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts_file}",
            f"{user}@{host}",
            f"promote {bundle.tag}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C"},
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"ssh exited {completed.returncode}"
        raise InstallerPromotionError(f"remote promotion failed: {message}")
    expected = f"promoted {bundle.tag} {bundle.installer_sha256}"
    if completed.stdout.strip() != expected:
        raise InstallerPromotionError("remote promotion receipt is missing or invalid")


def download_stable_installer(destination: Path) -> None:
    try:
        completed = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--proto",
                "=https",
                "--tlsv1.2",
                "--connect-timeout",
                "15",
                "--max-time",
                "60",
                "--max-filesize",
                str(2 * 1024 * 1024),
                "--output",
                str(destination),
                "--write-out",
                "%{http_code}\n%{url_effective}\n",
                STABLE_INSTALLER_URL,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C"},
        )
    except subprocess.TimeoutExpired as error:
        raise InstallerPromotionError("stable endpoint download timed out") from error
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"curl exited {completed.returncode}"
        raise InstallerPromotionError(f"stable endpoint download failed: {message}")
    if completed.stdout.splitlines() != ["200", STABLE_INSTALLER_URL]:
        raise InstallerPromotionError(
            "stable endpoint redirected or returned a non-200 response"
        )


FetchStable = Callable[[Path], None]
Pause = Callable[[float], None]


def verify_public_promotion(
    bundle: ReleaseBundle,
    *,
    fetch: FetchStable = download_stable_installer,
    pause: Pause = time.sleep,
    attempts: int = 6,
    delay_seconds: float = 5.0,
) -> None:
    if attempts < 1 or attempts > 20:
        raise InstallerPromotionError("public verification attempts are outside bounds")
    last_error = "public installer did not match"
    with tempfile.TemporaryDirectory(prefix="daita-stable-verify-") as temporary:
        destination = Path(temporary) / "install.sh"
        for attempt in range(1, attempts + 1):
            try:
                fetch(destination)
                if sha256_file(destination) != bundle.installer_sha256:
                    raise InstallerPromotionError(
                        "public installer checksum does not match the promoted release"
                    )
                validate_installer_commands(destination, bundle.version)
                return
            except (InstallerPromotionError, OSError) as error:
                last_error = str(error)
                if attempt < attempts:
                    pause(delay_seconds)
    raise InstallerPromotionError(
        f"public promotion verification failed after {attempts} attempts: {last_error}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "promote", "verify"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--tag", required=True)
        subparser.add_argument("--artifacts", type=Path, required=True)
        if command == "promote":
            subparser.add_argument("--host", required=True)
            subparser.add_argument("--user", required=True)
            subparser.add_argument("--identity-file", type=Path, required=True)
            subparser.add_argument("--known-hosts-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        bundle = validate_release_bundle(arguments.artifacts, arguments.tag)
        if arguments.command == "validate":
            validate_installer_commands(
                arguments.artifacts / "install.sh", bundle.version
            )
            print(f"validated installer {bundle.tag} {bundle.installer_sha256}")
        elif arguments.command == "promote":
            request_remote_promotion(
                bundle,
                user=arguments.user,
                host=arguments.host,
                identity_file=arguments.identity_file,
                known_hosts_file=arguments.known_hosts_file,
            )
            print(f"requested promotion {bundle.tag} {bundle.installer_sha256}")
        elif arguments.command == "verify":
            verify_public_promotion(bundle)
            print(f"verified stable installer {bundle.tag} {bundle.installer_sha256}")
        else:  # pragma: no cover - argparse owns command admission
            raise AssertionError("unreachable command")
    except (InstallerPromotionError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
