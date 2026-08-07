from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Mapping
import zipfile

import pytest

from installer_fixtures import (
    INSTALLER_SOURCE,
    create_installer_fixture,
    fixture_environment,
    sha256,
)


def _bash_executable() -> str:
    executable = shutil.which("bash")
    if executable is None:
        raise RuntimeError("bash is required for managed-installer tests")
    return executable


_BASH = _bash_executable()


def _run(
    installer: Path,
    *arguments: str,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_BASH, str(installer), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=None if env is None else dict(env),
    )


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        if path.is_symlink():
            digest.update(b"link\0" + os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"file\0" + path.read_bytes())
        else:
            digest.update(b"dir\0")
    return digest.hexdigest()


def _managed_root(home: Path) -> Path:
    return home / ".local" / "share" / "daita"


def _restricted_commands(directory: Path, names: tuple[str, ...]) -> Path:
    directory.mkdir()
    for name in names:
        executable = shutil.which(name)
        if executable is None:
            continue
        (directory / name).symlink_to(executable)
    return directory


def _current_target(home: Path) -> str:
    return os.readlink(_managed_root(home) / "current")


def _install(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    fixture = create_installer_fixture(tmp_path / "fixture")
    home = tmp_path / "home"
    home.mkdir()
    environment = fixture_environment(fixture, home)
    completed = _run(
        fixture.installer,
        "--no-onboard",
        "--no-modify-path",
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return fixture.installer, environment, home


def test_canonical_installer_syntax_and_release_literals_fail_closed(tmp_path: Path):
    syntax = subprocess.run(
        ["bash", "-n", str(INSTALLER_SOURCE)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    before = _tree_digest(home)
    completed = _run(INSTALLER_SOURCE, "--no-onboard", env=environment)

    assert completed.returncode == 1
    assert "not release-ready" in completed.stderr
    assert _tree_digest(home) == before


def test_help_version_and_parser_errors_are_non_mutating(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment["HOME"] = str(home)

    version = _run(INSTALLER_SOURCE, "--version", env=environment)
    help_result = _run(INSTALLER_SOURCE, "--help", env=environment)
    unknown = _run(INSTALLER_SOURCE, "--wat", env=environment)
    conflict = _run(
        INSTALLER_SOURCE,
        "--verify",
        "--repair",
        env=environment,
    )
    combined_help = _run(INSTALLER_SOURCE, "--help", "--dry-run", env=environment)

    assert version.returncode == 0
    assert "Daita 1.0.0" in version.stdout
    assert help_result.returncode == 0
    for option in (
        "--dry-run",
        "--verify",
        "--repair",
        "--rollback",
        "--uninstall",
        "--no-onboard",
        "--no-modify-path",
    ):
        assert option in help_result.stdout
    assert unknown.returncode == 2
    assert conflict.returncode == 2
    assert combined_help.returncode == 2
    assert tuple(home.iterdir()) == ()


@pytest.mark.parametrize(
    "action",
    ((), ("--verify",), ("--repair",), ("--rollback",), ("--uninstall",)),
)
def test_dry_run_for_every_action_reports_exact_intent_without_mutation(
    tmp_path: Path,
    action: tuple[str, ...],
):
    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment.update({"HOME": str(home), "SHELL": "/bin/zsh"})
    before = _tree_digest(home)

    completed = _run(INSTALLER_SOURCE, *action, "--dry-run", env=environment)

    assert completed.returncode == 0, completed.stderr
    assert "dry-run; no downloads or writes" in completed.stdout
    assert "Wheel URL: UNRESOLVED_WHEEL_URL" in completed.stdout
    assert f"Managed root: {home}/.local/share/daita" in completed.stdout
    assert "PATH:" in completed.stdout
    assert _tree_digest(home) == before


@pytest.mark.parametrize(
    ("kernel", "machine", "libc", "expected"),
    (
        ("Darwin", "arm64", "", "macos-arm64"),
        ("Darwin", "x86_64", "", "macos-x86_64"),
        ("Linux", "aarch64", "glibc 2.35", "linux-arm64-glibc"),
        ("Linux", "x86_64", "glibc 2.35", "linux-x86_64-glibc"),
    ),
)
def test_supported_os_architecture_selection_is_exact(
    tmp_path: Path,
    kernel: str,
    machine: str,
    libc: str,
    expected: str,
):
    home = tmp_path / "home"
    home.mkdir()
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uname").write_text(
        "#!/usr/bin/env bash\n"
        f"[[ $1 == -s ]] && printf '{kernel}\\n' || printf '{machine}\\n'\n",
        encoding="utf-8",
    )
    (fake / "getconf").write_text(
        f"#!/usr/bin/env bash\nprintf '{libc}\\n'\n",
        encoding="utf-8",
    )
    (fake / "ldd").write_text(
        f"#!/usr/bin/env bash\nprintf '{libc}\\n'\n",
        encoding="utf-8",
    )
    for path in fake.iterdir():
        path.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {"HOME": str(home), "PATH": f"{fake}{os.pathsep}{environment['PATH']}"}
    )

    completed = _run(INSTALLER_SOURCE, "--dry-run", env=environment)

    assert completed.returncode == 0, completed.stderr
    assert f"Target: {expected}" in completed.stdout
    assert tuple(home.iterdir()) == ()


@pytest.mark.parametrize(
    ("kernel", "machine", "libc", "message"),
    (
        ("Plan9", "x86_64", "", "unsupported installer target"),
        ("Linux", "riscv64", "glibc 2.35", "unsupported installer target"),
        ("Linux", "x86_64", "musl libc", "requires glibc"),
    ),
)
def test_unsupported_platforms_fail_before_mutation(
    tmp_path: Path,
    kernel: str,
    machine: str,
    libc: str,
    message: str,
):
    home = tmp_path / "home"
    home.mkdir()
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uname").write_text(
        "#!/usr/bin/env bash\n"
        f"[[ $1 == -s ]] && printf '{kernel}\\n' || printf '{machine}\\n'\n",
        encoding="utf-8",
    )
    for name in ("getconf", "ldd"):
        path = fake / name
        path.write_text(f"#!/usr/bin/env bash\nprintf '{libc}\\n'\n", encoding="utf-8")
    for path in fake.iterdir():
        path.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {"HOME": str(home), "PATH": f"{fake}{os.pathsep}{environment['PATH']}"}
    )

    completed = _run(INSTALLER_SOURCE, "--dry-run", env=environment)

    assert completed.returncode == 1
    assert message in completed.stderr
    assert tuple(home.iterdir()) == ()


def test_checksum_and_wheel_metadata_rejection_publish_nothing(tmp_path: Path):
    fixture = create_installer_fixture(tmp_path / "fixture")
    home = tmp_path / "home"
    home.mkdir()
    environment = fixture_environment(fixture, home)
    fixture.wheel.write_bytes(fixture.wheel.read_bytes() + b"corrupt")

    checksum = _run(
        fixture.installer,
        "--no-onboard",
        "--no-modify-path",
        env=environment,
    )

    assert checksum.returncode == 1
    assert "SHA-256 mismatch" in checksum.stderr
    assert not (home / ".local" / "bin" / "daita").exists()
    assert not (_managed_root(home) / "current").exists()
    assert not (_managed_root(home) / "installer").exists()
    assert not (_managed_root(home) / "python").exists()

    fixture = create_installer_fixture(tmp_path / "metadata-fixture")
    home2 = tmp_path / "home2"
    home2.mkdir()
    environment2 = fixture_environment(fixture, home2)
    rewritten_wheel = fixture.wheel.with_suffix(".rewritten.whl")
    with zipfile.ZipFile(fixture.wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        members = {name: archive.read(name) for name in archive.namelist()}
    members[metadata_name] = members[metadata_name].replace(
        b"Name: daita-agents", b"Name: foreign"
    )
    with zipfile.ZipFile(
        rewritten_wheel,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    fixture.wheel.write_bytes(rewritten_wheel.read_bytes())
    old_sha = fixture.wheel_sha256
    new_sha = sha256(fixture.wheel)
    fixture.installer.write_text(
        fixture.installer.read_text(encoding="utf-8").replace(old_sha, new_sha),
        encoding="utf-8",
    )

    metadata_result = _run(
        fixture.installer,
        "--no-onboard",
        "--no-modify-path",
        env=environment2,
    )

    assert metadata_result.returncode == 1
    assert "wheel distribution name" in metadata_result.stderr
    assert not (home2 / ".local" / "bin" / "daita").exists()
    assert not (_managed_root(home2) / "current").exists()
    assert not (_managed_root(home2) / "installer").exists()
    assert not (_managed_root(home2) / "python").exists()


def test_missing_downloader_hash_utility_and_failed_tls_publish_nothing(
    tmp_path: Path,
):
    fixture = create_installer_fixture(tmp_path / "fixture")
    required_for_platform = ("uname", "getconf", "ldd")
    for name, commands, expected in (
        ("downloader", required_for_platform, "required command is unavailable: curl"),
        (
            "hash",
            required_for_platform
            + (
                "curl",
                "tar",
                "awk",
                "grep",
                "cp",
                "mv",
                "rm",
                "mkdir",
                "chmod",
                "readlink",
                "env",
            ),
            "SHA-256 utility",
        ),
    ):
        home = tmp_path / f"{name}-home"
        home.mkdir()
        restricted = _restricted_commands(tmp_path / f"{name}-bin", commands)
        environment = fixture_environment(fixture, home)
        environment["PATH"] = str(restricted)

        completed = _run(
            fixture.installer,
            "--no-onboard",
            "--no-modify-path",
            env=environment,
        )

        assert completed.returncode == 1
        assert expected in completed.stderr
        assert not (home / ".local" / "bin" / "daita").exists()
        assert not (_managed_root(home) / "current").exists()

    failed_home = tmp_path / "tls-home"
    failed_home.mkdir()
    failed_environment = fixture_environment(fixture, failed_home)
    failed_environment["DAITA_FAKE_CURL_FAIL_MATCH"] = "uv-fixture"
    failed = _run(
        fixture.installer,
        "--no-onboard",
        "--no-modify-path",
        env=failed_environment,
    )
    assert failed.returncode != 0
    assert not (failed_home / ".local" / "bin" / "daita").exists()
    assert not (_managed_root(failed_home) / "current").exists()


def test_unsafe_home_and_managed_path_are_rejected_before_mutation(tmp_path: Path):
    fixture = create_installer_fixture(tmp_path / "fixture")
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    linked_environment = fixture_environment(fixture, linked_home)

    linked = _run(fixture.installer, "--no-onboard", env=linked_environment)

    assert linked.returncode == 1
    assert "symlink HOME" in linked.stderr
    assert tuple(real_home.iterdir()) == ()

    ordinary_home = tmp_path / "ordinary-home"
    ordinary_home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (ordinary_home / ".local").symlink_to(outside, target_is_directory=True)
    path_environment = fixture_environment(fixture, ordinary_home)

    escaped = _run(fixture.installer, "--no-onboard", env=path_environment)

    assert escaped.returncode == 1
    assert "must not be a symlink" in escaped.stderr
    assert tuple(outside.iterdir()) == ()


def test_unsafe_uv_archive_member_is_rejected_before_execution(tmp_path: Path):
    fixture = create_installer_fixture(tmp_path / "fixture")
    unsafe = tmp_path / "unsafe.tar.gz"
    target = tmp_path / "target"
    target.write_text("not executable", encoding="utf-8")
    with tarfile.open(unsafe, "w:gz") as archive:
        info = archive.gettarinfo(target, arcname="uv-fixture/uv")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        archive.addfile(info)
    fixture.uv_archive.write_bytes(unsafe.read_bytes())
    old_sha = sha256(unsafe)  # archive bytes now match this digest
    source = fixture.installer.read_text(encoding="utf-8")
    current_archive_sha = next(
        line.split('"')[1]
        for line in source.splitlines()
        if line.startswith("readonly UV_DARWIN_ARM64_SHA256=")
    )
    fixture.installer.write_text(
        source.replace(current_archive_sha, old_sha),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    environment = fixture_environment(fixture, home)

    completed = _run(
        fixture.installer,
        "--no-onboard",
        "--no-modify-path",
        env=environment,
    )

    assert completed.returncode == 1
    assert "not a regular file" in completed.stderr
    assert not (home / ".local" / "bin" / "daita").exists()
    assert not (_managed_root(home) / "current").exists()


def test_install_repeat_verify_repair_rollback_and_uninstall_preserve_data(
    tmp_path: Path,
):
    fixture = create_installer_fixture(tmp_path / "fixture")
    home = tmp_path / "home"
    home.mkdir()
    app_data = home / ".daita"
    artifact = app_data / "agents" / "atlas" / "artifacts" / "report.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("sensitive,customer,data\n", encoding="utf-8")
    external = home / "unrelated-sentinel"
    external.write_text("preserve me", encoding="utf-8")
    keychain_sentinel = tmp_path / "keychain-sentinel"
    keychain_sentinel.write_text("external keychain boundary", encoding="utf-8")
    preserved = {
        "data": _tree_digest(app_data),
        "external": sha256(external),
        "keychain": sha256(keychain_sentinel),
    }
    environment = fixture_environment(fixture, home)
    environment["DAITA_FAKE_UV_LOG"] = str(tmp_path / "uv.log")

    installed = _run(fixture.installer, "--no-onboard", env=environment)
    assert installed.returncode == 0, installed.stderr
    launcher = home / ".local" / "bin" / "daita"
    assert launcher.is_file() and not launcher.is_symlink()
    first_target = _current_target(home)
    first_generations = tuple((_managed_root(home) / "generations").iterdir())
    shell_before_repeat = (home / ".zshrc").read_bytes()

    repeated = _run(fixture.installer, "--no-onboard", env=environment)
    assert repeated.returncode == 0, repeated.stderr
    assert _current_target(home) == first_target
    assert tuple((_managed_root(home) / "generations").iterdir()) == first_generations
    assert (home / ".zshrc").read_bytes() == shell_before_repeat
    assert (home / ".zshrc").read_text().count("# >>> Daita managed PATH >>>") == 1

    before_verify = _tree_digest(_managed_root(home))
    verified = _run(fixture.installer, "--verify", env=environment)
    assert verified.returncode == 0, verified.stderr
    assert _tree_digest(_managed_root(home)) == before_verify

    repaired = _run(
        fixture.installer,
        "--repair",
        "--no-onboard",
        env=environment,
    )
    assert repaired.returncode == 0, repaired.stderr
    repaired_target = _current_target(home)
    assert repaired_target != first_target, repaired.stdout
    assert os.readlink(_managed_root(home) / "previous") == first_target

    rolled_back = _run(fixture.installer, "--rollback", env=environment)
    assert rolled_back.returncode == 0, rolled_back.stderr
    assert _current_target(home) == first_target
    assert os.readlink(_managed_root(home) / "previous") == repaired_target
    assert "Application data was not changed" in rolled_back.stdout

    for key, digest in preserved.items():
        path = {"data": app_data, "external": external, "keychain": keychain_sentinel}[
            key
        ]
        actual = _tree_digest(path) if path.is_dir() else sha256(path)
        assert actual == digest

    uninstalled = _run(fixture.installer, "--uninstall", env=environment)
    assert uninstalled.returncode == 0, uninstalled.stderr
    assert not launcher.exists()
    assert not _managed_root(home).exists()
    assert (home / ".zshrc").read_bytes() == b""
    assert _tree_digest(app_data) == preserved["data"]
    assert sha256(external) == preserved["external"]
    assert sha256(keychain_sentinel) == preserved["keychain"]

    uv_log = (tmp_path / "uv.log").read_text(encoding="utf-8")
    for variable in (
        "UV_TOOL_DIR=",
        "UV_TOOL_BIN_DIR=",
        "UV_PYTHON_INSTALL_DIR=",
        "UV_CACHE_DIR=",
        "UV_PYTHON_PREFERENCE=only-managed",
        "UV_PYTHON_DOWNLOADS=manual",
        "UV_DEFAULT_INDEX=https://pypi.org/simple",
        "UV_NO_BUILD=1",
        "UV_PRERELEASE=disallow",
        "PIP_CONFIG_FILE=/dev/null",
    ):
        assert variable in uv_log


def test_repair_recovers_a_damaged_current_without_recording_it_as_previous(
    tmp_path: Path,
):
    installer, environment, home = _install(tmp_path)
    root = _managed_root(home)
    first = _current_target(home)
    initial_repair = _run(
        installer,
        "--repair",
        "--no-onboard",
        "--no-modify-path",
        env=environment,
    )
    assert initial_repair.returncode == 0, initial_repair.stderr
    healthy_previous = os.readlink(root / "previous")
    assert healthy_previous == first
    damaged = root / _current_target(home) / "bin" / "daita"
    damaged.write_text("damaged", encoding="utf-8")

    repaired = _run(
        installer,
        "--repair",
        "--no-onboard",
        "--no-modify-path",
        env=environment,
    )

    assert repaired.returncode == 0, repaired.stderr
    assert os.readlink(root / "previous") == healthy_previous
    assert _run(installer, "--verify", env=environment).returncode == 0


@pytest.mark.parametrize(
    "failpoint",
    (
        "after-lock",
        "after-uv",
        "after-python",
        "after-wheel",
        "after-bootstrap",
        "after-tool-install",
        "after-staged-checks",
        "after-manifest",
        "after-current-switch",
        "after-previous-switch",
        "after-launcher",
    ),
)
def test_failure_at_each_transaction_boundary_preserves_the_previous_command(
    tmp_path: Path,
    failpoint: str,
):
    installer, environment, home = _install(tmp_path)
    launcher = home / ".local" / "bin" / "daita"
    previous_launcher = launcher.read_bytes()
    previous_target = _current_target(home)
    data = home / ".daita" / "sentinel"
    data.parent.mkdir()
    data.write_text("unchanged", encoding="utf-8")
    environment["DAITA_INSTALLER_TEST_FAILPOINT"] = failpoint

    failed = _run(
        installer,
        "--repair",
        "--no-onboard",
        "--no-modify-path",
        env=environment,
    )

    assert failed.returncode == 1
    assert failpoint in failed.stderr
    assert launcher.read_bytes() == previous_launcher
    assert _current_target(home) == previous_target
    assert data.read_text(encoding="utf-8") == "unchanged"
    assert not (_managed_root(home) / "install-state" / "mutation.lock").exists()
    assert tuple((_managed_root(home) / "staging").iterdir()) == ()


def test_signal_status_and_cleanup_are_deterministic(tmp_path: Path):
    fixture = create_installer_fixture(tmp_path / "fixture")
    home = tmp_path / "home"
    home.mkdir()
    environment = fixture_environment(fixture, home)
    environment.update(
        {
            "DAITA_FAKE_UV_SIGNAL": "SIGINT",
            "DAITA_FAKE_UV_SIGNAL_AT": "python install",
        }
    )

    interrupted = _run(
        fixture.installer,
        "--no-onboard",
        "--no-modify-path",
        env=environment,
    )

    assert interrupted.returncode == 130
    assert not (home / ".local" / "bin" / "daita").exists()
    assert not (_managed_root(home) / "current").exists()
    assert tuple((_managed_root(home) / "staging").iterdir()) == ()


def test_foreign_and_pipx_style_launcher_collisions_are_preserved(tmp_path: Path):
    fixture = create_installer_fixture(tmp_path / "fixture")
    for name, symbolic in (("foreign", False), ("pipx", True)):
        home = tmp_path / name
        target_bin = home / ".local" / "bin"
        target_bin.mkdir(parents=True)
        foreign = target_bin / "foreign-daita"
        foreign.write_text("foreign bytes", encoding="utf-8")
        launcher = target_bin / "daita"
        if symbolic:
            launcher.symlink_to(foreign)
        else:
            launcher.write_bytes(foreign.read_bytes())
        before = foreign.read_bytes()
        environment = fixture_environment(fixture, home)

        completed = _run(
            fixture.installer,
            "--no-onboard",
            "--no-modify-path",
            env=environment,
        )

        assert completed.returncode == 1
        assert "foreign or pipx-owned" in completed.stderr
        assert foreign.read_bytes() == before
        if symbolic:
            assert launcher.is_symlink() and launcher.resolve() == foreign
        else:
            assert launcher.read_bytes() == before
        assert not _managed_root(home).exists()


def test_changed_path_block_is_warned_and_preserved_on_repeat_and_uninstall(
    tmp_path: Path,
):
    fixture = create_installer_fixture(tmp_path / "fixture")
    home = tmp_path / "home"
    home.mkdir()
    environment = fixture_environment(fixture, home)
    installed = _run(fixture.installer, "--no-onboard", env=environment)
    assert installed.returncode == 0, installed.stderr
    shell_file = home / ".zshrc"
    shell_file.write_text(
        shell_file.read_text(encoding="utf-8") + "# user changed this file\n",
        encoding="utf-8",
    )
    changed = shell_file.read_bytes()

    repeated = _run(fixture.installer, "--no-onboard", env=environment)
    assert repeated.returncode == 0, repeated.stderr
    assert "PATH block changed" in repeated.stderr
    assert shell_file.read_bytes() == changed

    uninstalled = _run(fixture.installer, "--uninstall", env=environment)
    assert uninstalled.returncode == 0, uninstalled.stderr
    assert "PATH block changed" in uninstalled.stderr
    assert shell_file.read_bytes() == changed


def test_noninteractive_install_never_launches_onboarding(tmp_path: Path):
    fixture = create_installer_fixture(tmp_path / "fixture")
    home = tmp_path / "home"
    home.mkdir()
    environment = fixture_environment(fixture, home)

    completed = _run(
        fixture.installer,
        "--no-modify-path",
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Onboarding not launched without a controlling terminal" in completed.stdout
    assert f"Run later: {home}/.local/bin/daita" in completed.stdout


def test_installer_never_names_application_data_as_an_owned_mutation_target():
    source = INSTALLER_SOURCE.read_text(encoding="utf-8")

    assert "\nsudo " not in source
    assert "pipx uninstall" not in source
    assert "git clone" not in source
    assert "uv installer" not in source.lower()
    assert "$HOME/.daita" not in source
    assert "DAITA_MANAGED_INSTALL_ROOT" in source
    assert "</dev/tty >/dev/tty 2>/dev/tty" in source
    assert "--retry 3" in source
    assert "--connect-timeout 10 --max-time 300" in source
    assert "UV_NO_CONFIG=1" in source
    assert "UV_NO_BUILD=1" in source
