"""The login service definitions: generated content and file placement only.

No test runs launchctl or systemctl; install and uninstall get a recording runner.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from daita.host import service
from daita.host.paths import HostPaths

pytestmark = pytest.mark.unit

# Characters that need XML or systemd escaping, in the paths users can have.
ROOT = Path("/Users/R&D <team>/100% $daita")
INTERPRETER = '/Users/R&D <team>/pipx/venvs/daita "agents"/bin/python'
SEARCH_PATH = "/Users/a/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"


def test_launchd_plist_runs_this_interpreter_with_the_host_log() -> None:
    paths = HostPaths(ROOT)

    plist = plistlib.loads(
        service.launchd_plist(paths, interpreter=INTERPRETER, search_path=SEARCH_PATH)
    )

    assert plist == {
        "Label": "ai.daita.host",
        "ProgramArguments": [
            INTERPRETER,
            "-m",
            "daita",
            "--root",
            str(ROOT),
            "host",
            "serve",
            "--auto",
        ],
        "EnvironmentVariables": {"PATH": SEARCH_PATH},
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Standard",
        "StandardOutPath": str(ROOT / "logs" / "host.log"),
        "StandardErrorPath": str(ROOT / "logs" / "host.log"),
    }


def test_systemd_unit_quotes_the_command_and_appends_to_the_host_log() -> None:
    paths = HostPaths(ROOT)

    unit = service.systemd_unit(paths, interpreter=INTERPRETER, search_path=SEARCH_PATH)

    lines = unit.splitlines()
    assert lines[:4] == ["[Unit]", "Description=Daita agent host", "", "[Service]"]
    assert (
        "ExecStart="
        '"/Users/R&D <team>/pipx/venvs/daita \\"agents\\"/bin/python" "-m" "daita" '
        '"--root" "/Users/R&D <team>/100%% $$daita" "host" "serve" "--auto"'
    ) in lines
    assert f'Environment="PATH={SEARCH_PATH}"' in lines
    log = "/Users/R&D <team>/100%% $daita/logs/host.log"
    assert f"StandardOutput=append:{log}" in lines
    assert f"StandardError=append:{log}" in lines
    assert {"Restart=on-failure", "RestartSec=3", "WantedBy=default.target"} <= set(
        lines
    )


def test_a_value_spanning_lines_cannot_reach_a_unit() -> None:
    with pytest.raises(ValueError, match="single line"):
        service.systemd_unit(
            HostPaths(Path("/tmp/a\nExecStartPre=/bin/evil")),
            interpreter=INTERPRETER,
            search_path=SEARCH_PATH,
        )


@pytest.mark.parametrize(
    ("platform", "relative"),
    [
        ("darwin", "Library/LaunchAgents/ai.daita.host.plist"),
        ("linux", ".config/systemd/user/daita-host.service"),
    ],
)
def test_install_writes_the_definition_and_uninstall_removes_it(
    tmp_path: Path, platform: str, relative: str
) -> None:
    paths = HostPaths(tmp_path / "state")
    commands: list[tuple[list[str], bool]] = []

    def run(command: list[str], check: bool) -> None:
        commands.append((command, check))

    assert not service.installed(platform, tmp_path)
    notes = service.install(
        paths, start=False, run=run, platform=platform, home=tmp_path
    )

    path = tmp_path / relative
    assert path.stat().st_mode & 0o777 == 0o644
    assert service.installed(platform, tmp_path)
    assert str(paths.log) in path.read_text(encoding="utf-8")
    assert notes[0] == f"Daita host service installed: {path}"
    assert "It starts at your next login." in notes
    if platform == "linux":
        assert service.LINGER_NOTE in notes
        assert commands == [
            (["systemctl", "--user", "daemon-reload"], True),
            (["systemctl", "--user", "enable", "daita-host.service"], True),
        ]
    else:
        assert commands == []

    commands.clear()
    assert service.uninstall(paths, run=run, platform=platform, home=tmp_path)
    assert not path.exists()
    assert commands[0][0][:2] == (
        ["launchctl", "bootout"] if platform == "darwin" else ["systemctl", "--user"]
    )
    assert not service.uninstall(paths, run=run, platform=platform, home=tmp_path)


def test_other_platforms_have_no_service() -> None:
    assert not service.installed("win32")
    with pytest.raises(ValueError, match="macOS and Linux only"):
        service.service_path("win32")
