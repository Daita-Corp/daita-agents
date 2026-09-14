from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.managed_installer_host as host_module
from scripts.managed_installer_host import (
    InstallerPromotionError,
    ManagedInstallerHost,
    forced_promotion_tag,
    parse_tag,
    validate_installer_syntax,
    validate_release_bundle,
)
from tests.support.paths import REPO_ROOT

BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_managed_installer_host.sh"
HOST_PROGRAM = REPO_ROOT / "scripts" / "managed_installer_host.py"
NGINX_LOCATION = REPO_ROOT / "release" / "daita-installer-location.nginx.conf"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _release_assets(version: str, *, installer_suffix: str = "") -> dict[str, bytes]:
    tag = f"v{version}"
    wheel = f"daita_agents-{version}-py3-none-any.whl"
    installer = (
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f'VERSION="{version}"\n'
        'case "${1:-}" in\n'
        '  --version) printf "Daita installer for Daita %s\\n" "$VERSION" ;;\n'
        '  --dry-run) printf "Daita version: %s\\n" "$VERSION" ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n"
        f"# {installer_suffix}\n"
    ).encode()
    manifest = {
        "schema_version": 2,
        "installer": {"filename": "install.sh", "sha256": _sha256(installer)},
        "application": {"version": version, "requires_python": ">=3.11,<3.13"},
        "wheel": {
            "filename": wheel,
            "url": (
                "https://github.com/Daita-Corp/daita-agents/releases/download/"
                f"{tag}/{wheel}"
            ),
            "sha256": "a" * 64,
        },
        "runtime": {"not_interpreted_by_host": True},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    checksums = (
        f"{'a' * 64}  {wheel}\n"
        f"{_sha256(installer)}  install.sh\n"
        f"{_sha256(manifest_bytes)}  release-manifest.json\n"
        f"{'b' * 64}  agent-home-contract.json\n"
    ).encode()
    return {
        "install.sh": installer,
        "release-manifest.json": manifest_bytes,
        "SHA256SUMS": checksums,
    }


class Repository:
    def __init__(self, releases: dict[str, dict[str, bytes]]) -> None:
        self.releases = releases
        self.requests: list[str] = []

    def download(self, url: str, destination: Path, maximum_bytes: int) -> None:
        self.requests.append(url)
        marker = "/releases/download/"
        tail = url.split(marker, maxsplit=1)[1]
        tag, name = tail.split("/", maxsplit=1)
        content = self.releases[tag][name]
        assert len(content) <= maximum_bytes
        destination.write_bytes(content)


def _host(
    root: Path, repository: Repository, validations: list[tuple[str, str]]
) -> ManagedInstallerHost:
    root.mkdir()

    def validate(installer: Path, version: str) -> None:
        validations.append((installer.read_text(encoding="utf-8"), version))

    return ManagedInstallerHost(
        root,
        download=repository.download,
        validate_installer=validate,
    )


@pytest.mark.parametrize(
    "tag",
    (
        "1.2.3",
        "v1.2",
        "v01.2.3",
        "v1.02.3",
        "v1.2.03",
        "v1.2.3-rc1",
        "v1.2.3/../../escape",
        "v1000000000.0.0",
    ),
)
def test_tag_parser_rejects_noncanonical_or_unsafe_values(tag: str) -> None:
    with pytest.raises(InstallerPromotionError):
        parse_tag(tag)


def test_release_bundle_binds_manifest_checksums_and_fixed_wheel_url(
    tmp_path: Path,
) -> None:
    assets = _release_assets("1.2.3")
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)

    bundle = validate_release_bundle(tmp_path, "v1.2.3")

    assert bundle.version == "1.2.3"
    assert bundle.installer_sha256 == _sha256(assets["install.sh"])


def test_schema_two_manifest_and_checksum_inventory_are_exact(tmp_path: Path) -> None:
    assets = _release_assets("1.2.3")
    manifest = json.loads(assets["release-manifest.json"])
    manifest["unexpected"] = "ignored fields are forbidden"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    assets["release-manifest.json"] = manifest_bytes
    assets["SHA256SUMS"] = (
        f"{'a' * 64}  daita_agents-1.2.3-py3-none-any.whl\n"
        f"{_sha256(assets['install.sh'])}  install.sh\n"
        f"{_sha256(manifest_bytes)}  release-manifest.json\n"
        f"{'b' * 64}  agent-home-contract.json\n"
    ).encode()
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)

    with pytest.raises(InstallerPromotionError, match="fields do not match"):
        validate_release_bundle(tmp_path, "v1.2.3")

    del manifest["unexpected"]
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (tmp_path / "release-manifest.json").write_bytes(manifest_bytes)
    with (tmp_path / "SHA256SUMS").open("w", encoding="utf-8") as checksums:
        checksums.write(f"{'a' * 64}  daita_agents-1.2.3-py3-none-any.whl\n")
        checksums.write(f"{_sha256(assets['install.sh'])}  install.sh\n")
        checksums.write(f"{_sha256(manifest_bytes)}  release-manifest.json\n")
        checksums.write(f"{'b' * 64}  agent-home-contract.json\n")
        checksums.write(f"{'c' * 64}  unexpected.bin\n")

    with pytest.raises(InstallerPromotionError, match="exact release contract"):
        validate_release_bundle(tmp_path, "v1.2.3")


def test_schema_one_is_accepted_only_with_a_complete_pinned_legacy_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _release_assets("1.2.3")
    manifest = json.loads(assets["release-manifest.json"])
    manifest["schema_version"] = 1
    manifest["installer"]["historical_field"] = "ignored-but-hash-bound"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    assets["release-manifest.json"] = manifest_bytes
    assets["SHA256SUMS"] = (
        f"{'a' * 64}  daita_agents-1.2.3-py3-none-any.whl\n"
        f"{_sha256(assets['install.sh'])}  install.sh\n"
        f"{_sha256(manifest_bytes)}  release-manifest.json\n"
    ).encode()
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)

    with pytest.raises(InstallerPromotionError, match="not admitted"):
        validate_release_bundle(tmp_path, "v1.2.3")

    monkeypatch.setattr(
        host_module,
        "LEGACY_SCHEMA_ONE_RELEASES",
        {
            "v1.2.3": (
                _sha256(assets["install.sh"]),
                _sha256(manifest_bytes),
                "a" * 64,
            )
        },
    )
    assert validate_release_bundle(tmp_path, "v1.2.3").version == "1.2.3"

    (tmp_path / "release-manifest.json").write_bytes(manifest_bytes + b" ")
    with pytest.raises(InstallerPromotionError, match="pinned identity"):
        validate_release_bundle(tmp_path, "v1.2.3")


@pytest.mark.parametrize("corrupt", ("install.sh", "release-manifest.json"))
def test_release_bundle_rejects_corrupt_downloads(tmp_path: Path, corrupt: str) -> None:
    assets = _release_assets("1.2.3")
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)
    with (tmp_path / corrupt).open("ab") as destination:
        destination.write(b"corrupt")

    with pytest.raises(InstallerPromotionError, match="does not match|not valid"):
        validate_release_bundle(tmp_path, "v1.2.3")


def test_forward_promotion_is_immutable_atomic_and_records_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Repository(
        {"v1.2.9": _release_assets("1.2.9"), "v1.2.10": _release_assets("1.2.10")}
    )
    validations: list[tuple[str, str]] = []
    host = _host(tmp_path / "host", repository, validations)
    release_modes_at_rename: list[int] = []
    real_replace = os.replace

    def replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.parent == host.releases:
            release_modes_at_rename.append(stat.S_IMODE(source_path.stat().st_mode))
        real_replace(source, destination)

    monkeypatch.setattr(host_module.os, "replace", replace)

    first = host.promote("v1.2.9")
    second = host.promote("v1.2.10")

    assert first.previous_tag is None
    assert second.previous_tag == "v1.2.9"
    assert os.readlink(host.current) == "releases/v1.2.10"
    assert (host.releases / "v1.2.10").stat().st_mode & 0o777 == 0o555
    assert (host.current / "install.sh").stat().st_mode & 0o777 == 0o555
    assert (host.current / "promotion.json").stat().st_mode & 0o777 == 0o444
    assert (host.releases / "v1.2.9" / "install.sh").read_bytes() == (
        repository.releases["v1.2.9"]["install.sh"]
    )
    assert release_modes_at_rename == [0o555, 0o555]
    assert [version for _, version in validations] == ["1.2.9", "1.2.10"]


def test_downgrade_is_refused_before_any_download(tmp_path: Path) -> None:
    repository = Repository(
        {"v1.2.9": _release_assets("1.2.9"), "v1.2.10": _release_assets("1.2.10")}
    )
    validations: list[tuple[str, str]] = []
    host = _host(tmp_path / "host", repository, validations)
    host.promote("v1.2.10")
    requests_before = list(repository.requests)

    with pytest.raises(InstallerPromotionError, match="cannot replace newer"):
        host.promote("v1.2.9")

    assert repository.requests == requests_before
    assert os.readlink(host.current) == "releases/v1.2.10"


def test_host_refuses_permissive_root_and_symlink_lock(tmp_path: Path) -> None:
    repository = Repository({"v1.0.1": _release_assets("1.0.1")})
    root = tmp_path / "host"
    host = _host(root, repository, [])
    root.chmod(0o777)
    with pytest.raises(InstallerPromotionError, match="root ownership or permissions"):
        host.promote("v1.0.1")
    assert repository.requests == []

    root.chmod(0o755)
    foreign = tmp_path / "foreign-lock"
    foreign.write_text("do not follow", encoding="utf-8")
    host.lock.symlink_to(foreign)
    with pytest.raises(InstallerPromotionError, match="promotion lock is unsafe"):
        host.promote("v1.0.1")
    assert foreign.read_text(encoding="utf-8") == "do not follow"
    assert repository.requests == []


def test_validation_failure_leaves_current_bytes_untouched(tmp_path: Path) -> None:
    repository = Repository(
        {"v1.0.0": _release_assets("1.0.0"), "v1.0.1": _release_assets("1.0.1")}
    )
    root = tmp_path / "host"
    root.mkdir()

    def validate(installer: Path, version: str) -> None:
        if version == "1.0.1":
            raise InstallerPromotionError("synthetic validation failure")

    host = ManagedInstallerHost(
        root, download=repository.download, validate_installer=validate
    )
    first = host.promote("v1.0.0")

    with pytest.raises(InstallerPromotionError, match="synthetic"):
        host.promote("v1.0.1")

    assert os.readlink(host.current) == "releases/v1.0.0"
    assert sha256_file(host.current / "install.sh") == first.installer_sha256
    assert not (host.releases / "v1.0.1").exists()


def test_repeated_promotion_requires_identical_immutable_bytes(tmp_path: Path) -> None:
    repository = Repository({"v1.0.1": _release_assets("1.0.1")})
    validations: list[tuple[str, str]] = []
    host = _host(tmp_path / "host", repository, validations)
    first = host.promote("v1.0.1")
    second = host.promote("v1.0.1")
    assert first.installer_sha256 == second.installer_sha256

    repository.releases["v1.0.1"] = _release_assets(
        "1.0.1", installer_suffix="different"
    )
    with pytest.raises(InstallerPromotionError, match="conflict"):
        host.promote("v1.0.1")
    assert os.readlink(host.current) == "releases/v1.0.1"


def test_installed_release_permission_drift_is_rejected(tmp_path: Path) -> None:
    repository = Repository({"v1.0.1": _release_assets("1.0.1")})
    host = _host(tmp_path / "host", repository, [])
    host.promote("v1.0.1")
    (host.releases / "v1.0.1").chmod(0o755)

    with pytest.raises(InstallerPromotionError, match="permissions are invalid"):
        host.promote("v1.0.1")


def test_current_pointer_must_use_the_canonical_relative_target(tmp_path: Path) -> None:
    repository = Repository({"v1.0.1": _release_assets("1.0.1")})
    host = _host(tmp_path / "host", repository, [])
    host.promote("v1.0.1")
    host.current.unlink()
    host.current.symlink_to("releases/./v1.0.1")

    with pytest.raises(InstallerPromotionError, match="not canonical"):
        host.promote("v1.0.1")


def test_local_rollback_restores_only_the_recorded_predecessor(tmp_path: Path) -> None:
    repository = Repository(
        {"v1.0.0": _release_assets("1.0.0"), "v1.0.1": _release_assets("1.0.1")}
    )
    validations: list[tuple[str, str]] = []
    host = _host(tmp_path / "host", repository, validations)
    host.promote("v1.0.0")
    host.promote("v1.0.1")

    restored = host.rollback()

    assert restored.tag == "v1.0.0"
    assert os.readlink(host.current) == "releases/v1.0.0"
    with pytest.raises(InstallerPromotionError, match="no recorded predecessor"):
        host.rollback()


@pytest.mark.parametrize(
    "command",
    (
        "",
        "rollback",
        "promote v1.2.3 extra",
        "promote v1.2.3; id",
        "promote v1.2.3\nrollback",
        " promote v1.2.3",
    ),
)
def test_forced_key_rejects_every_command_except_one_exact_promotion(
    command: str,
) -> None:
    with pytest.raises(InstallerPromotionError):
        forced_promotion_tag({"SSH_ORIGINAL_COMMAND": command})


def test_forced_key_accepts_one_exact_canonical_promotion() -> None:
    assert (
        forced_promotion_tag({"SSH_ORIGINAL_COMMAND": "promote v123.45.6"})
        == "v123.45.6"
    )


def test_host_program_is_self_contained_when_installed(tmp_path: Path) -> None:
    program = tmp_path / HOST_PROGRAM.name
    shutil.copyfile(HOST_PROGRAM, program)

    completed = subprocess.run(
        [sys.executable, str(program), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert "forced-promote" in completed.stdout


def test_host_syntax_validation_never_executes_release_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = tmp_path / "install.sh"
    installer.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(host_module.subprocess, "run", run)

    validate_installer_syntax(installer, "1.2.3")

    assert calls == [["bash", "-n", str(installer)]]


def test_bootstrap_installs_one_forced_key_without_a_general_shell() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert 'readonly DEPLOY_USER="daita-installer-deploy"' in source
    assert 'readonly HOST_ROOT="/srv/daita-installer"' in source
    assert '[[ "$key_type" == "ssh-ed25519"' in source
    assert 'restrict,command="%s"' in source
    assert "forced-promote" in source
    assert "managed_installer_nginx.py" in source
    assert "daita-installer-location.nginx.conf" in source
    assert "mktemp" in source
    assert "daita-installer-release" in source
    assert 'chown root:root "$DEPLOY_HOME"' in source
    assert 'chown root:root "$AUTHORIZED_KEYS_TEMP"' in source
    assert "managed path must not be a symlink" in source
    assert "refusing to replace an existing unmanaged path" in source
    assert "private-key" not in source.lower()
    assert "LIGHTSAIL_SSH_KEY" not in source


def test_nginx_location_changes_only_the_exact_stable_installer_path() -> None:
    source = NGINX_LOCATION.read_text(encoding="utf-8")

    assert "location = /install.sh" in source
    assert "alias /srv/daita-installer/current/install.sh;" in source
    assert 'add_header Cache-Control "no-store" always;' in source
    assert "proxy_pass" not in source
    assert "server {" not in source


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
