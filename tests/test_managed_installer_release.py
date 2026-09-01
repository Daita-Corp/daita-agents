from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from installer_fixtures import build_minimal_wheel

from scripts.render_managed_installer import (
    DEFAULT_POLICY,
    DEFAULT_TEMPLATE,
    ReleaseInputError,
    load_release_policy,
    render_managed_installer,
    write_rendered_release,
)

ROOT = Path(__file__).parents[1]
RENDERER = ROOT / "scripts" / "render_managed_installer.py"
RELEASE_VERSION = "1.0.1"
WHEEL_URL = (
    f"https://github.com/Daita-Corp/daita-agents/releases/download/v{RELEASE_VERSION}/"
    f"daita_agents-{RELEASE_VERSION}-py3-none-any.whl"
)


def _policy_document() -> dict[str, Any]:
    return json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))


def _write_policy(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_reviewed_policy_renders_one_deterministic_release(tmp_path: Path):
    wheel = build_minimal_wheel(tmp_path, version=RELEASE_VERSION)
    policy = load_release_policy(DEFAULT_POLICY)

    first = render_managed_installer(
        policy=policy,
        wheel=wheel,
        wheel_url=WHEEL_URL,
    )
    second = render_managed_installer(
        policy=policy,
        wheel=wheel,
        wheel_url=WHEEL_URL,
    )

    assert first == second
    assert "UNRESOLVED_" not in first.installer
    assert f'readonly DAITA_VERSION="{RELEASE_VERSION}"' in first.installer
    assert 'readonly UV_VERSION="0.12.7"' in first.installer
    assert 'readonly PYTHON_REQUEST="cpython-3.12.14"' in first.installer
    assert first.manifest["installer"]["release_sequence"] == 2
    assert first.manifest["wheel"]["url"] == WHEEL_URL
    assert (
        first.manifest["installer"]["sha256"]
        == hashlib.sha256(first.installer.encode("utf-8")).hexdigest()
    )

    installer = tmp_path / "release" / "install.sh"
    manifest = tmp_path / "release" / "release-manifest.json"
    write_rendered_release(first, installer_output=installer, manifest_output=manifest)
    syntax = subprocess.run(
        ["bash", "-n", str(installer)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert syntax.returncode == 0, syntax.stderr
    assert installer.stat().st_mode & 0o777 == 0o755
    assert manifest.stat().st_mode & 0o777 == 0o644
    assert json.loads(manifest.read_text(encoding="utf-8")) == first.manifest


def test_release_build_backend_is_exactly_pinned():
    with (ROOT / "pyproject.toml").open("rb") as source:
        build_system = tomllib.load(source)["build-system"]

    assert build_system == {
        "requires": ["setuptools==82.0.1", "wheel==0.47.0"],
        "build-backend": "setuptools.build_meta",
    }


def test_policy_requires_exact_target_coverage_and_checksums(tmp_path: Path):
    document = _policy_document()
    targets = document["runtime"]["targets"]
    del targets["macos-x86_64"]

    with pytest.raises(ReleaseInputError, match="fields do not match"):
        load_release_policy(_write_policy(tmp_path / "missing.json", document))

    document = _policy_document()
    document["runtime"]["targets"]["linux-x86_64-glibc"]["uv_sha256"] = "0" * 63
    with pytest.raises(ReleaseInputError, match="64 lowercase hex"):
        load_release_policy(_write_policy(tmp_path / "checksum.json", document))

    document = _policy_document()
    document["runtime"]["targets"]["macos-arm64"][
        "python_identity"
    ] = "cpython-3.12.14-linux-aarch64-gnu"
    with pytest.raises(ReleaseInputError, match="macos-aarch64-none"):
        load_release_policy(_write_policy(tmp_path / "identity.json", document))

    document = _policy_document()
    document["runtime"]["targets"]["macos-arm64"]["uv_url"] = (
        "https://artifacts.example.test/releases/download/0.12.7/"
        "uv-aarch64-apple-darwin.tar.gz"
    )
    with pytest.raises(ReleaseInputError, match="official versioned uv release URL"):
        load_release_policy(_write_policy(tmp_path / "host.json", document))


@pytest.mark.parametrize(
    "url",
    (
        "http://github.com/astral-sh/uv/releases/download/0.12.7/uv-aarch64-apple-darwin.tar.gz",
        "https://github.com/astral-sh/uv/releases/latest/uv-aarch64-apple-darwin.tar.gz",
        "https://github.com/astral-sh/uv/releases/download/0.12.7/uv-aarch64-apple-darwin.tar.gz?mutable=1",
    ),
)
def test_policy_rejects_mutable_or_untrusted_transport_shape(tmp_path: Path, url: str):
    document = _policy_document()
    document["runtime"]["targets"]["macos-arm64"]["uv_url"] = url

    with pytest.raises(ReleaseInputError):
        load_release_policy(_write_policy(tmp_path / "mutable.json", document))


def test_renderer_rejects_mutable_or_mistagged_wheel_urls(tmp_path: Path):
    wheel = build_minimal_wheel(tmp_path, version=RELEASE_VERSION)
    policy = load_release_policy(DEFAULT_POLICY)

    with pytest.raises(ReleaseInputError, match="mutable latest"):
        render_managed_installer(
            policy=policy,
            wheel=wheel,
            wheel_url=(
                "https://github.com/Daita-Corp/daita-agents/releases/latest/"
                f"{wheel.name}"
            ),
        )
    with pytest.raises(
        ReleaseInputError,
        match=f"immutable release tag v{RELEASE_VERSION}",
    ):
        render_managed_installer(
            policy=policy,
            wheel=wheel,
            wheel_url=(
                "https://github.com/Daita-Corp/daita-agents/releases/download/v2.0.0/"
                f"{wheel.name}"
            ),
        )


def test_cli_binds_wheel_metadata_to_the_project(tmp_path: Path):
    wheel = build_minimal_wheel(tmp_path, version="2.0.0")
    completed = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--wheel",
            str(wheel),
            "--wheel-url",
            (
                "https://github.com/Daita-Corp/daita-agents/releases/download/v2.0.0/"
                f"{wheel.name}"
            ),
            "--installer-output",
            str(tmp_path / "install.sh"),
            "--manifest-output",
            str(tmp_path / "release-manifest.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "wheel version does not match pyproject.toml" in completed.stderr
    assert not (tmp_path / "install.sh").exists()
    assert not (tmp_path / "release-manifest.json").exists()


def test_template_exposes_every_release_value_as_a_fail_closed_sentinel():
    source = DEFAULT_TEMPLATE.read_text(encoding="utf-8")

    for placeholder in (
        "UNRESOLVED_INSTALLER_VERSION",
        "UNRESOLVED_RELEASE_SEQUENCE",
        "UNRESOLVED_DAITA_VERSION",
        "UNRESOLVED_WHEEL_FILENAME",
        "UNRESOLVED_WHEEL_URL",
        "UNRESOLVED_WHEEL_SHA256",
        "UNRESOLVED_WHEEL_REQUIRES_PYTHON",
        "UNRESOLVED_UV_VERSION",
        "UNRESOLVED_PYTHON_REQUEST",
    ):
        assert source.count(placeholder) == 1


def test_release_workflow_covers_every_reviewed_target_before_publication():
    workflow = (ROOT / ".github/workflows/managed-release.yml").read_text(
        encoding="utf-8"
    )
    policy = load_release_policy(DEFAULT_POLICY)

    assert workflow.count("python -m build --wheel") == 1
    assert "build==1.5.1" in workflow
    assert "group: managed-release\n" in workflow
    assert "workflow_dispatch:" in workflow
    assert "publish:" in workflow
    assert "default: false" in workflow
    assert 'tags:\n      - "v*"' in workflow
    assert "managed-installer-release" in workflow
    assert "needs:\n      - build\n      - native-installer-smoke" in workflow
    assert "Refuse an existing mutable release" in workflow
    assert "Require the exact public PyPI wheel" in workflow
    assert "Require a forward-only release sequence" in workflow
    assert "current <= previous" in workflow
    assert "inputs.publish == true" in workflow
    assert 'test "$GITHUB_REF_TYPE" = "tag"' in workflow
    assert "actions/attest-build-provenance@v3" in workflow
    assert "gh release create" in workflow
    assert "Verify published bytes" in workflow
    assert 'cmp "release-artifacts/$artifact"' in workflow
    assert workflow.index("Require the exact public PyPI wheel") < workflow.index(
        "gh release create"
    )
    assert "sha256sum --check SHA256SUMS" in workflow
    assert 'len(files) != 1 or files[0].get("filename") != wheel' in workflow
    assert 'files[0].get("digests", {}).get("sha256")' in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "environment:\n      name: pypi" not in workflow
    assert "PYPI_API_KEY" not in workflow
    assert "Resolve the reviewed target policy" in workflow
    assert 'policy = json.load(open("release/managed-installer.json"' in workflow
    assert "steps.runtime.outputs.uv_sha256" in workflow
    for runner in (
        "macos-15",
        "macos-15-intel",
        "ubuntu-24.04-arm",
        "ubuntu-24.04",
    ):
        assert f"runner: {runner}" in workflow
    for target_name, target in policy["runtime"]["targets"].items():
        assert f"target: {target_name}" in workflow
        assert target["uv_archive"] not in workflow
        assert target["uv_sha256"] not in workflow
        assert target["python_identity"] not in workflow
