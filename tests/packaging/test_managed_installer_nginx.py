from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.managed_installer_nginx import (
    HTTPS_ANCHOR,
    NginxCutoverError,
    NginxInstallerCutover,
)

ORIGINAL = f"""# HTTP server
server {{
    listen 80 default_server;
    server_name daita-tech.io www.daita-tech.io;
    return 301 https://$host$request_uri;
}}

# HTTPS server
server {{
    listen 443 ssl;
    server_name daita-tech.io www.daita-tech.io;

{HTTPS_ANCHOR}    location / {{
      proxy_pass http://localhost:3000;
    }}
}}
"""

SNIPPET = """    # Daita managed installer; the agent release workflow owns these bytes.
    location = /install.sh {
        alias /srv/daita-installer/current/install.sh;
        default_type text/x-shellscript;
        add_header Cache-Control "no-store" always;
        add_header X-Content-Type-Options "nosniff" always;
    }
"""


class Commands:
    def __init__(self, returncodes: list[int] | None = None) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.returncodes = [] if returncodes is None else returncodes

    def __call__(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        returncode = self.returncodes.pop(0) if self.returncodes else 0
        return subprocess.CompletedProcess(command, returncode, "", "synthetic error")


def _cutover(tmp_path: Path, commands: Commands) -> NginxInstallerCutover:
    config = tmp_path / "server.conf"
    config.write_text(ORIGINAL, encoding="utf-8")
    config.chmod(0o644)
    snippet = tmp_path / "snippet.conf"
    snippet.write_text(SNIPPET, encoding="utf-8")
    snippet.chmod(0o444)
    return NginxInstallerCutover(
        config=config,
        backup=tmp_path / "outside-server-blocks" / "before.conf",
        snippet=snippet,
        nginx=Path("/trusted/nginx"),
        run=commands,
    )


def test_cutover_changes_only_the_exact_route_and_is_idempotent(tmp_path: Path) -> None:
    commands = Commands()
    cutover = _cutover(tmp_path, commands)

    assert cutover.cutover() is True
    changed = cutover.config.read_text(encoding="utf-8")
    assert changed == ORIGINAL.replace(HTTPS_ANCHOR, SNIPPET + "\n" + HTTPS_ANCHOR)
    assert cutover.backup.read_text(encoding="utf-8") == ORIGINAL
    assert cutover.backup.stat().st_mode & 0o777 == 0o600
    assert cutover.config.stat().st_mode & 0o777 == 0o644
    assert commands.commands == [
        ("/trusted/nginx", "-t"),
        ("/trusted/nginx", "-s", "reload"),
    ]

    assert cutover.cutover() is False
    assert commands.commands[-2:] == [
        ("/trusted/nginx", "-t"),
        ("/trusted/nginx", "-s", "reload"),
    ]


def test_failed_nginx_validation_restores_the_prior_configuration(
    tmp_path: Path,
) -> None:
    commands = Commands([1, 0, 0])
    cutover = _cutover(tmp_path, commands)

    with pytest.raises(NginxCutoverError, match="prior configuration was restored"):
        cutover.cutover()

    assert cutover.config.read_text(encoding="utf-8") == ORIGINAL
    assert commands.commands == [
        ("/trusted/nginx", "-t"),
        ("/trusted/nginx", "-t"),
        ("/trusted/nginx", "-s", "reload"),
    ]


def test_restore_refuses_to_overwrite_unrelated_later_drift(tmp_path: Path) -> None:
    cutover = _cutover(tmp_path, Commands())
    assert cutover.cutover() is True
    cutover.config.write_text(
        cutover.config.read_text(encoding="utf-8") + "# later change\n",
        encoding="utf-8",
    )

    with pytest.raises(NginxCutoverError, match="unrelated drift"):
        cutover.restore()


def test_ambiguous_website_layout_is_never_changed(tmp_path: Path) -> None:
    cutover = _cutover(tmp_path, Commands())
    cutover.config.write_text(ORIGINAL + HTTPS_ANCHOR, encoding="utf-8")

    with pytest.raises(NginxCutoverError, match="missing or ambiguous"):
        cutover.cutover()

    assert not cutover.backup.exists()


def test_cutover_rejects_permission_drift_before_changing_configuration(
    tmp_path: Path,
) -> None:
    cutover = _cutover(tmp_path, Commands())
    cutover.snippet.chmod(0o644)

    with pytest.raises(NginxCutoverError, match="invalid permissions"):
        cutover.cutover()

    assert cutover.config.read_text(encoding="utf-8") == ORIGINAL
    assert not cutover.backup.exists()
