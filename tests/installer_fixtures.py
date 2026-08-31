"""Deterministic local artifacts for the reviewed managed-installer source."""

from __future__ import annotations

import gzip
import hashlib
import sys
import tarfile
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path

from scripts.render_managed_installer import render_managed_installer

ROOT = Path(__file__).resolve().parents[1]
INSTALLER_SOURCE = ROOT / "scripts" / "install.sh"
FIXTURE_PYTHON_IDENTITY = "cpython-3.12.99-linux-x86_64-gnu"
FIXTURE_PYTHON_REQUEST = "cpython-3.12.99"
FIXTURE_UV_VERSION = "0.0.0-fixture"


@dataclass(frozen=True, slots=True)
class InstallerFixture:
    installer: Path
    downloads: Path
    fake_bin: Path
    uv_archive: Path
    wheel: Path
    wheel_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_metadata(path: Path) -> tuple[str, str, str]:
    with zipfile.ZipFile(path) as archive:
        names = tuple(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        if len(names) != 1:
            raise AssertionError("fixture wheel must contain exactly one METADATA file")
        metadata = archive.read(names[0]).decode("utf-8")

    def one(label: str) -> str:
        values = tuple(
            line.removeprefix(f"{label}: ")
            for line in metadata.splitlines()
            if line.startswith(f"{label}: ")
        )
        if len(values) != 1:
            raise AssertionError(f"fixture wheel must contain exactly one {label}")
        return values[0]

    return one("Name"), one("Version"), one("Requires-Python")


def build_minimal_wheel(directory: Path, *, version: str = "1.0.0") -> Path:
    wheel = directory / f"daita_agents-{version}-py3-none-any.whl"
    dist_info = f"daita_agents-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("daita/__init__.py", f'__version__ = "{version}"\n')
        archive.writestr(
            "daita/cli.py",
            textwrap.dedent(f"""\
                import argparse

                def main(argv=None):
                    parser = argparse.ArgumentParser(prog="daita")
                    parser.add_argument("--version", action="version", version="daita {version}")
                    parser.parse_args(argv)
                    return 0

                if __name__ == "__main__":
                    raise SystemExit(main())
                """),
        )
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.4\n"
            "Name: daita-agents\n"
            f"Version: {version}\n"
            "Requires-Python: >=3.11,<3.13\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: daita-installer-fixture\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        archive.writestr(
            f"{dist_info}/entry_points.txt",
            "[console_scripts]\n" "daita = daita.cli:main\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel


def create_installer_fixture(
    directory: Path,
    *,
    wheel: Path | None = None,
    installer_version: str = "1.0.0-fixture",
    release_sequence: int = 1,
    bootstrap_uv_archive: Path | None = None,
    bootstrap_uv_version: str = FIXTURE_UV_VERSION,
    bootstrap_uv_member: str | None = None,
    bootstrap_python_request: str = FIXTURE_PYTHON_REQUEST,
    bootstrap_python_identity: str = FIXTURE_PYTHON_IDENTITY,
) -> InstallerFixture:
    directory.mkdir(parents=True, exist_ok=True)
    downloads = directory / "downloads"
    fake_bin = directory / "fake-bin"
    downloads.mkdir()
    fake_bin.mkdir()
    source_wheel = build_minimal_wheel(directory) if wheel is None else wheel.resolve()
    name, version, _requires_python = wheel_metadata(source_wheel)
    if name != "daita-agents":
        raise AssertionError(f"unexpected wheel distribution: {name}")
    wheel_copy = downloads / source_wheel.name
    wheel_copy.write_bytes(source_wheel.read_bytes())

    if bootstrap_uv_archive is None:
        if bootstrap_uv_member is not None:
            raise AssertionError("a uv member requires a real bootstrap archive")
        fake_uv = directory / "uv"
        fake_uv.write_text(_fake_uv_source(), encoding="utf-8")
        fake_uv.chmod(0o755)
        archive_name = "uv-fixture.tar.gz"
        archive_member = "uv-fixture/uv"
        uv_archive = downloads / archive_name
        with (
            uv_archive.open("wb") as raw_archive,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_archive, mtime=0
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            info = archive.gettarinfo(str(fake_uv), arcname=archive_member)
            info.mode = 0o755
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with fake_uv.open("rb") as source:
                archive.addfile(info, source)
    else:
        source_uv_archive = bootstrap_uv_archive.resolve(strict=True)
        if bootstrap_uv_member is None:
            raise AssertionError("a real bootstrap archive requires its uv member")
        archive_name = source_uv_archive.name
        archive_member = bootstrap_uv_member
        if "/" in archive_name or not archive_name.endswith(".tar.gz"):
            raise AssertionError("real bootstrap archive has an unsafe filename")
        with tarfile.open(source_uv_archive, mode="r:gz") as archive:
            members = tuple(
                member
                for member in archive.getmembers()
                if member.name == archive_member
            )
        if len(members) != 1 or not members[0].isfile():
            raise AssertionError(
                "real bootstrap archive must contain one regular selected uv member"
            )
        uv_archive = downloads / archive_name
        uv_archive.write_bytes(source_uv_archive.read_bytes())

    curl = fake_bin / "curl"
    curl.write_text(_fake_curl_source(), encoding="utf-8")
    curl.chmod(0o755)

    target = {
        "uv_archive": archive_name,
        "uv_member": archive_member,
        "uv_url": f"https://fixtures.invalid/releases/{archive_name}",
        "uv_sha256": sha256(uv_archive),
        "python_identity": bootstrap_python_identity,
    }
    policy = {
        "schema_version": 1,
        "installer": {
            "version": installer_version,
            "release_sequence": release_sequence,
        },
        "runtime": {
            "uv_version": bootstrap_uv_version,
            "python_request": bootstrap_python_request,
            "targets": {
                "macos-arm64": dict(target),
                "macos-x86_64": dict(target),
                "linux-arm64-glibc": dict(target),
                "linux-x86_64-glibc": dict(target),
            },
        },
    }
    rendered = render_managed_installer(
        policy=policy,
        wheel=wheel_copy,
        wheel_url=(
            f"https://fixtures.invalid/releases/download/v{version}/{wheel_copy.name}"
        ),
    ).installer
    installer = directory / "install.sh"
    installer.write_text(rendered, encoding="utf-8")
    installer.chmod(0o755)
    return InstallerFixture(
        installer=installer,
        downloads=downloads,
        fake_bin=fake_bin,
        uv_archive=uv_archive,
        wheel=wheel_copy,
        wheel_sha256=sha256(wheel_copy),
    )


def fixture_environment(fixture: InstallerFixture, home: Path) -> dict[str, str]:
    import os

    environment = os.environ.copy()
    environment.update(
        {
            "DAITA_FAKE_BASE_PYTHON": sys.executable,
            "DAITA_FAKE_PYTHON_IDENTITY": FIXTURE_PYTHON_IDENTITY,
            "DAITA_FIXTURE_DOWNLOAD_DIR": str(fixture.downloads),
            "HOME": str(home),
            "PATH": f"{fixture.fake_bin}{os.pathsep}{environment['PATH']}",
            "SHELL": "/bin/zsh",
        }
    )
    return environment


def _fake_curl_source() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        output=""
        url=""
        while (($#)); do
            case "$1" in
                --output)
                    output="$2"
                    shift 2
                    ;;
                http*)
                    url="$1"
                    shift
                    ;;
                *) shift ;;
            esac
        done
        [[ -n "$output" && -n "$url" ]]
        if [[ -n "${DAITA_FAKE_CURL_SIGNAL:-}" ]]; then
            kill "-${DAITA_FAKE_CURL_SIGNAL}" "$PPID"
            exit 99
        fi
        if [[ -n "${DAITA_FAKE_CURL_FAIL_MATCH:-}" && "$url" == *"$DAITA_FAKE_CURL_FAIL_MATCH"* ]]; then
            exit 22
        fi
        source_path="$DAITA_FIXTURE_DOWNLOAD_DIR/${url##*/}"
        cp "$source_path" "$output"
        """)


def _fake_uv_source() -> str:
    return textwrap.dedent(r"""
        #!__PYTHON__
        from __future__ import annotations

        import os
        from pathlib import Path
        import shutil
        import signal
        import subprocess
        import sys
        import venv

        args = sys.argv[1:]
        key = " ".join(args[:2])
        log = os.environ.get("DAITA_FAKE_UV_LOG")
        if log:
            with Path(log).open("a", encoding="utf-8") as output:
                output.write(key + "\n")
                for name in (
                    "UV_TOOL_DIR",
                    "UV_TOOL_BIN_DIR",
                    "UV_PYTHON_INSTALL_DIR",
                    "UV_CACHE_DIR",
                    "UV_PYTHON_PREFERENCE",
                    "UV_PYTHON_DOWNLOADS",
                    "UV_DEFAULT_INDEX",
                    "UV_NO_BUILD",
                    "UV_PRERELEASE",
                    "PIP_CONFIG_FILE",
                ):
                    output.write(f"{name}={os.environ.get(name, '')}\n")
        if os.environ.get("DAITA_FAKE_UV_FAIL") == key:
            raise SystemExit(91)
        requested_signal = os.environ.get("DAITA_FAKE_UV_SIGNAL")
        if requested_signal and os.environ.get("DAITA_FAKE_UV_SIGNAL_AT") == key:
            os.kill(os.getppid(), getattr(signal, requested_signal))
            raise SystemExit(92)

        base_python = Path(os.environ["DAITA_FAKE_BASE_PYTHON"])
        if args[:2] == ["python", "install"]:
            identity = os.environ["DAITA_FAKE_PYTHON_IDENTITY"]
            target = Path(os.environ["UV_PYTHON_INSTALL_DIR"]) / identity / "bin"
            target.mkdir(parents=True, exist_ok=True)
            wrapper = target / "python3.12"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ ${1:-} == -I && ${2:-} == -c && ${3:-} == *platform.python_version* ]]; then\n"
                "  printf 'cpython-3.12.99\\n'\n"
                "  exit 0\n"
                "fi\n"
                f"exec {str(base_python)!r} \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            raise SystemExit(0)

        if args[:2] == ["tool", "install"]:
            wheel = Path(args[-1])
            tool = Path(os.environ["UV_TOOL_DIR"]) / "daita-agents"
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(tool)
            python = tool / "bin" / "python"
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--force-reinstall",
                    str(wheel),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            public_bin = Path(os.environ["UV_TOOL_BIN_DIR"])
            public_bin.mkdir(parents=True, exist_ok=True)
            destination = public_bin / "daita"
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(tool / "bin" / "daita")
            raise SystemExit(0)

        if args[:2] == ["pip", "check"]:
            if "--python" not in args:
                raise SystemExit("fake uv pip check requires --python")
            raise SystemExit(0)

        raise SystemExit(f"unsupported fake uv command: {args}")
        """).lstrip().replace("__PYTHON__", sys.executable)
