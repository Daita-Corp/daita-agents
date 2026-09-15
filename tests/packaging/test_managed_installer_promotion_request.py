from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.managed_installer_host import InstallerPromotionError, ReleaseBundle
from scripts.request_managed_installer_promotion import (
    main,
    request_remote_promotion,
    validate_known_hosts,
    validate_ssh_destination,
    verify_public_promotion,
)
from tests.support.paths import REPO_ROOT

REQUEST_PROGRAM = REPO_ROOT / "scripts" / "request_managed_installer_promotion.py"
TEST_HOST = "192.0.2.10"


def _bundle(tmp_path: Path, content: bytes = b"verified installer") -> ReleaseBundle:
    return ReleaseBundle(
        tag="v1.2.3",
        version="1.2.3",
        directory=tmp_path,
        installer_sha256=hashlib.sha256(content).hexdigest(),
    )


def _known_host(host: str = TEST_HOST) -> str:
    blob = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + b"x" * 32
    return f"{host} ssh-ed25519 {base64.b64encode(blob).decode()}\n"


def test_request_program_is_runnable_by_its_workflow_path() -> None:
    completed = subprocess.run(
        [sys.executable, str(REQUEST_PROGRAM), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "{validate,promote,verify}" in completed.stdout


@pytest.mark.parametrize(
    ("user", "host"),
    (
        ("-oProxyCommand=id", TEST_HOST),
        ("root;id", TEST_HOST),
        ("deploy", "-oProxyCommand=id"),
        ("deploy", "999.207.21.166"),
        ("deploy", "host..example.com"),
        ("deploy", "host name"),
        ("deploy", "example.com;id"),
    ),
)
def test_ssh_destination_rejects_option_and_shell_injection(
    user: str, host: str
) -> None:
    with pytest.raises(InstallerPromotionError):
        validate_ssh_destination(user, host)


def test_remote_request_uses_only_pinned_noninteractive_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    identity = tmp_path / "identity"
    identity.write_text("private", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(_known_host(), encoding="utf-8")
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": f"promoted {bundle.tag} {bundle.installer_sha256}\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(
        "scripts.request_managed_installer_promotion.subprocess.run", run
    )

    request_remote_promotion(
        bundle,
        user="deploy",
        host=TEST_HOST,
        identity_file=identity,
        known_hosts_file=known_hosts,
    )

    assert len(calls) == 1
    command = calls[0]
    assert "GlobalKnownHostsFile=/dev/null" in command
    assert "HostKeyAlgorithms=ssh-ed25519" in command
    assert "KbdInteractiveAuthentication=no" in command
    assert "PreferredAuthentications=publickey" in command
    assert "RequestTTY=no" in command
    assert command[-2:] == [f"deploy@{TEST_HOST}", "promote v1.2.3"]


@pytest.mark.parametrize(
    "content",
    (
        _known_host("wrong.example.com"),
        _known_host() + _known_host(),
        f"{TEST_HOST} ssh-rsa AAAA\n",
        "*.example.com ssh-ed25519 AAAA\n",
    ),
)
def test_known_hosts_requires_one_exact_destination_ed25519_key(
    tmp_path: Path, content: str
) -> None:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(content, encoding="utf-8")

    with pytest.raises(InstallerPromotionError):
        validate_known_hosts(known_hosts, TEST_HOST)


def test_public_verification_retries_then_accepts_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"verified installer"
    bundle = _bundle(tmp_path, expected)
    attempts = 0
    pauses: list[float] = []

    def fetch(destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        destination.write_bytes(b"stale" if attempts == 1 else expected)

    monkeypatch.setattr(
        "scripts.request_managed_installer_promotion.validate_installer_commands",
        lambda installer, version: None,
    )

    verify_public_promotion(
        bundle, fetch=fetch, pause=pauses.append, attempts=2, delay_seconds=0.25
    )

    assert attempts == 2
    assert pauses == [0.25]


def test_cli_separates_validation_promotion_and_public_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        "scripts.request_managed_installer_promotion.validate_release_bundle",
        lambda directory, tag: bundle,
    )
    monkeypatch.setattr(
        "scripts.request_managed_installer_promotion.validate_installer_commands",
        lambda installer, version: calls.append("commands"),
    )
    monkeypatch.setattr(
        "scripts.request_managed_installer_promotion.request_remote_promotion",
        lambda *args, **kwargs: calls.append("request"),
    )
    monkeypatch.setattr(
        "scripts.request_managed_installer_promotion.verify_public_promotion",
        lambda release: calls.append("public"),
    )

    common = ["--tag", bundle.tag, "--artifacts", str(tmp_path)]
    assert main(["validate", *common]) == 0
    assert (
        main(
            [
                "promote",
                *common,
                "--host",
                TEST_HOST,
                "--user",
                "deploy",
                "--identity-file",
                str(tmp_path / "identity"),
                "--known-hosts-file",
                str(tmp_path / "known_hosts"),
            ]
        )
        == 0
    )
    assert main(["verify", *common]) == 0
    assert calls == ["commands", "request", "public"]


def test_public_verification_never_accepts_wrong_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    validations: list[Path] = []

    def fetch(destination: Path) -> None:
        destination.write_bytes(b"wrong")

    monkeypatch.setattr(
        "scripts.request_managed_installer_promotion.validate_installer_commands",
        lambda installer, version: validations.append(installer),
    )

    with pytest.raises(InstallerPromotionError, match="after 3 attempts"):
        verify_public_promotion(
            bundle, fetch=fetch, pause=lambda delay: None, attempts=3
        )

    assert validations == []


@pytest.mark.parametrize("attempts", (0, 21))
def test_public_verification_attempt_count_is_bounded(
    tmp_path: Path, attempts: int
) -> None:
    with pytest.raises(InstallerPromotionError, match="outside bounds"):
        verify_public_promotion(_bundle(tmp_path), attempts=attempts)
