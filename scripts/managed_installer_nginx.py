#!/usr/bin/env python3
"""Safely add or remove Daita's exact managed-installer Nginx location."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

CONFIG_PATH = Path("/opt/bitnami/nginx/conf/server_blocks/daita-tech-server-block.conf")
BACKUP_PATH = Path("/etc/daita-installer-host/nginx-before-installer.conf")
SNIPPET_PATH = Path(
    "/usr/local/libexec/daita-installer/daita-installer-location.nginx.conf"
)
NGINX = Path("/opt/bitnami/nginx/sbin/nginx")
HTTPS_ANCHOR = "    # Reverse proxy to your app running on port 3000\n"
MAXIMUM_CONFIG_BYTES = 1024 * 1024


class NginxCutoverError(RuntimeError):
    """Raised when the exact Nginx cutover cannot be proven safe."""


RunCommand = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


def run_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C"},
    )


def _read_regular(
    path: Path,
    *,
    maximum_bytes: int = MAXIMUM_CONFIG_BYTES,
    expected_mode: int | None = None,
) -> str:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise NginxCutoverError(f"required file is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise NginxCutoverError(f"required path is not a regular file: {path}")
    if metadata.st_uid != os.geteuid():
        raise NginxCutoverError(f"required file has invalid ownership: {path}")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise NginxCutoverError(f"required file has invalid permissions: {path}")
    if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
        raise NginxCutoverError(f"required file has an invalid size: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise NginxCutoverError(f"required file is not UTF-8: {path}") from error


def _require_owned_directory(path: Path, expected_mode: int) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != expected_mode
    ):
        raise NginxCutoverError(
            f"required directory ownership or permissions are invalid: {path}"
        )


def render_cutover(current: str, snippet: str) -> str:
    """Return the only accepted installer-location insertion."""

    normalized = snippet.rstrip() + "\n"
    if "server {" in normalized or "proxy_pass" in normalized:
        raise NginxCutoverError("installer location snippet has unsafe directives")
    if current.count("listen 443 ssl;") != 1:
        raise NginxCutoverError("expected exactly one HTTPS server")
    if current.count("server_name daita-tech.io www.daita-tech.io;") != 2:
        raise NginxCutoverError("expected the exact HTTP and HTTPS server names")
    if current.count(HTTPS_ANCHOR) != 1 or current.count("    location / {") != 1:
        raise NginxCutoverError("website reverse-proxy anchor is missing or ambiguous")
    if "/install.sh" in current:
        raise NginxCutoverError("an unrecognized installer route already exists")
    anchor_index = current.index(HTTPS_ANCHOR)
    https_index = current.index("listen 443 ssl;")
    proxy_index = current.index("    location / {")
    if not https_index < anchor_index < proxy_index:
        raise NginxCutoverError("website HTTPS structure is not the admitted layout")
    return current.replace(HTTPS_ANCHOR, normalized + "\n" + HTTPS_ANCHOR, 1)


def _write_new(path: Path, content: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class NginxInstallerCutover:
    """Own one exact, recoverable edit to the existing website server block."""

    def __init__(
        self,
        *,
        config: Path = CONFIG_PATH,
        backup: Path = BACKUP_PATH,
        snippet: Path = SNIPPET_PATH,
        nginx: Path = NGINX,
        run: RunCommand = run_command,
    ) -> None:
        self.config = config
        self.backup = backup
        self.snippet = snippet
        self.nginx = nginx
        self._run = run

    def _nginx(self, *arguments: str) -> None:
        completed = self._run((str(self.nginx), *arguments))
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            detail = detail[:1000] if detail else f"exit {completed.returncode}"
            raise NginxCutoverError(f"Nginx {' '.join(arguments)} failed: {detail}")

    def _activate(self, candidate: str, previous: str) -> None:
        _write_atomic(self.config, candidate)
        try:
            self._nginx("-t")
            self._nginx("-s", "reload")
        except (NginxCutoverError, OSError, subprocess.TimeoutExpired) as error:
            _write_atomic(self.config, previous)
            try:
                self._nginx("-t")
                self._nginx("-s", "reload")
            except (NginxCutoverError, OSError, subprocess.TimeoutExpired) as recovery:
                raise NginxCutoverError(
                    f"Nginx activation failed and recovery also failed: {recovery}"
                ) from error
            raise NginxCutoverError(
                "Nginx activation failed; the prior configuration was restored"
            ) from error

    def cutover(self) -> bool:
        current = _read_regular(self.config, expected_mode=0o644)
        snippet = _read_regular(
            self.snippet, maximum_bytes=64 * 1024, expected_mode=0o444
        )
        normalized = snippet.rstrip() + "\n"

        if self.backup.exists() or self.backup.is_symlink():
            backup = _read_regular(self.backup, expected_mode=0o600)
            expected = render_cutover(backup, snippet)
            if current == expected:
                self._nginx("-t")
                self._nginx("-s", "reload")
                return False
            if current != backup:
                raise NginxCutoverError(
                    "current Nginx config diverges from the retained cutover backup"
                )
        else:
            backup = current
            expected = render_cutover(backup, snippet)
            self.backup.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            _require_owned_directory(self.backup.parent, 0o755)
            _write_new(self.backup, backup)
            _fsync_directory(self.backup.parent)

        if expected.count(normalized) != 1:
            raise NginxCutoverError("candidate does not contain one exact location")
        self._activate(expected, backup)
        return True

    def restore(self) -> bool:
        current = _read_regular(self.config, expected_mode=0o644)
        backup = _read_regular(self.backup, expected_mode=0o600)
        snippet = _read_regular(
            self.snippet, maximum_bytes=64 * 1024, expected_mode=0o444
        )
        expected = render_cutover(backup, snippet)
        if current == backup:
            self._nginx("-t")
            self._nginx("-s", "reload")
            return False
        if current != expected:
            raise NginxCutoverError(
                "current Nginx config has unrelated drift; refusing to overwrite it"
            )
        self._activate(backup, current)
        return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cutover", "restore"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if os.geteuid() != 0:
        parser.exit(2, "error: Nginx cutover must run as root\n")
    try:
        cutover = NginxInstallerCutover()
        changed = (
            cutover.cutover() if arguments.command == "cutover" else cutover.restore()
        )
        state = "changed" if changed else "already in the requested state"
        print(f"Nginx installer route {state}")
    except (NginxCutoverError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
