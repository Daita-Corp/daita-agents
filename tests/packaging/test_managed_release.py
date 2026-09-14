from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from scripts.release_identity import read_project_identity
from scripts.render_managed_installer import (
    DEFAULT_POLICY,
    DEFAULT_TEMPLATE,
    ReleaseInputError,
    load_release_policy,
    render_managed_installer,
    write_rendered_release,
)
from tests.support.installer import build_minimal_wheel
from tests.support.paths import REPO_ROOT

ROOT = REPO_ROOT
RENDERER = ROOT / "scripts" / "render_managed_installer.py"
RELEASE_VERSION = read_project_identity().version
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
    assert first.manifest == {
        "schema_version": 2,
        "application": {
            "version": RELEASE_VERSION,
            "requires_python": ">=3.11,<3.13",
        },
        "wheel": {
            "filename": wheel.name,
            "url": WHEEL_URL,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
        "installer": {
            "filename": "install.sh",
            "sha256": hashlib.sha256(first.installer.encode("utf-8")).hexdigest(),
        },
        "runtime": policy["runtime"],
    }
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
    assert set(_policy_document()) == {"runtime"}

    document = _policy_document()
    document["schema_version"] = 2
    with pytest.raises(ReleaseInputError, match="fields do not match"):
        load_release_policy(_write_policy(tmp_path / "extra-root.json", document))

    document = _policy_document()
    document["installer"] = {}
    with pytest.raises(ReleaseInputError, match="fields do not match"):
        load_release_policy(_write_policy(tmp_path / "retired-root.json", document))

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
    major, minor, patch = (int(part) for part in RELEASE_VERSION.split("."))
    mismatch = f"{major}.{minor}.{patch + 1}"
    wheel = build_minimal_wheel(tmp_path, version=mismatch)
    completed = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--wheel",
            str(wheel),
            "--wheel-url",
            (
                f"https://github.com/Daita-Corp/daita-agents/releases/download/v{mismatch}/"
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
        "UNRESOLVED_DAITA_VERSION",
        "UNRESOLVED_TEST_FAILPOINTS",
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
    assert workflow.count("persist-credentials: false") == workflow.count(
        "actions/checkout@"
    )
    assert "group: managed-release\n" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "workflow_dispatch:" in workflow
    assert "publish:" in workflow
    assert "default: false" in workflow
    assert 'tags:\n      - "v*"' in workflow
    assert "managed-installer-release" in workflow
    assert (
        "needs:\n      - build\n      - deterministic-and-static\n"
        "      - native-installer-smoke" in workflow
    )
    assert "Collect complete GitHub and PyPI published-version evidence" in workflow
    assert "Repeat complete protected published-version admission" in workflow
    assert workflow.count("gh api --paginate --slurp") == 2
    assert workflow.count('"https://pypi.org/pypi/daita-agents/json"') == 2
    assert workflow.count("parse_published_version_evidence") == 4
    assert workflow.count("prior_versions_for_mode") == 4
    assert 'mode="protected"' in workflow
    assert '"$GITHUB_EVENT_NAME" == "push"' in workflow
    assert (
        "steps.identity.outputs.mode != 'branch' || github.ref_type == 'tag'"
        in workflow
    )
    assert "Require the exact public PyPI wheel" in workflow
    assert "require-newer" in workflow
    assert "inputs.publish == true" in workflow
    assert 'test "$GITHUB_REF_TYPE" = "tag"' in workflow
    assert 'test "$(git cat-file -t "$GITHUB_REF_NAME")" = "tag"' in workflow
    assert (
        "actions/attest-build-provenance@62fc1d596301d0ab9914e1fec14dc5c8d93f65cd"
        in workflow
    )
    assert "gh release create" in workflow
    assert "--draft" in workflow
    assert "Verify draft assets before immutable publication" in workflow
    assert "Revalidate the remote annotated tag before draft creation" in workflow
    assert workflow.count("git ls-remote --refs --exit-code origin") == 2
    assert 'gh release edit "$GITHUB_REF_NAME"' in workflow
    assert "--draft=false --verify-tag" in workflow
    assert "Verify published bytes" in workflow
    assert "Verify published artifact provenance" in workflow
    assert 'gh release verify "$GITHUB_REF_NAME"' in workflow
    assert "--signer-workflow" in workflow
    assert '--source-digest "$GITHUB_SHA"' in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "Validate installer before admitting deployment credentials" in workflow
    assert "Request atomic stable installer promotion" in workflow
    assert "Verify stable installer from public endpoint" in workflow
    assert workflow.count("python scripts/request_managed_installer_promotion.py") == 3
    assert "request_managed_installer_promotion.py validate" in workflow
    assert "request_managed_installer_promotion.py promote" in workflow
    assert "request_managed_installer_promotion.py verify" in workflow
    assert "unset DEPLOY_PRIVATE_KEY DEPLOY_KNOWN_HOSTS" in workflow
    assert "MANAGED_INSTALLER_SSH_PRIVATE_KEY" in workflow
    assert "MANAGED_INSTALLER_SSH_KNOWN_HOSTS" in workflow
    assert "group: managed-installer-stable" in workflow
    assert 'cmp "release-artifacts/$artifact"' in workflow
    assert (
        workflow.index("Repeat complete protected published-version admission")
        < workflow.index("Require the exact public PyPI wheel")
        < workflow.index("gh release create")
        < workflow.index("Verify draft assets before immutable publication")
        < workflow.index('gh release edit "$GITHUB_REF_NAME"')
        < workflow.index("Verify published bytes")
        < workflow.index("Verify published artifact provenance")
        < workflow.index("Validate installer before admitting deployment credentials")
        < workflow.index("Request atomic stable installer promotion")
        < workflow.index("Verify stable installer from public endpoint")
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
    assert (
        workflow.index("Validate project, tag, and checkout identity")
        < workflow.index("Collect complete GitHub and PyPI published-version evidence")
        < workflow.index("Admit candidate before building")
        < workflow.index("Build and inspect the candidate wheel once")
    )
    assert "deterministic-and-static:" in workflow
    release_gates = workflow[
        workflow.index("  deterministic-and-static:") : workflow.index(
            "  native-installer-smoke:"
        )
    ]
    assert "needs: build" in release_gates
    assert (
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
        in release_gates
    )
    assert "${{ needs.build.outputs.wheel }}[dev]" in release_gates
    assert "DAITA_TEST_CANDIDATE_WHEEL:" in release_gates
    assert "python -m build --wheel" not in release_gates
    assert "python -m pytest tests/" in release_gates
    assert "Validate GitHub Actions workflow syntax" in release_gates
    assert (
        "rhysd/actionlint@sha256:887a259a5a534f3c4f36cb02dca341673c6089431057242cdc931e9f133147e9"
        in release_gates
    )
    assert "Scan the release source and complete history" in release_gates
    assert "zricethezav/gitleaks@sha256:" in release_gates
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


def test_ci_builds_one_wheel_and_lifecycle_jobs_only_consume_that_artifact():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert workflow.count("python -m build --wheel") == 1
    assert "permissions:\n  contents: read\n" in workflow
    assert workflow.count("persist-credentials: false") == workflow.count(
        "actions/checkout@"
    )
    assert "Validate GitHub Actions workflow syntax" in workflow
    assert (
        "rhysd/actionlint@sha256:887a259a5a534f3c4f36cb02dca341673c6089431057242cdc931e9f133147e9"
        in workflow
    )
    assert "release-artifact:" in workflow
    assert "pipx-lifecycle:" in workflow
    assert "managed-lifecycle:" in workflow
    assert workflow.count("name: release-artifact") >= 3
    assert (
        workflow.count(
            "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
        )
        == 3
    )
    producer = workflow.index("Build and inspect the candidate wheel once")
    first_render = workflow.index("python scripts/render_managed_installer.py")
    second_render = workflow.index(
        "python scripts/render_managed_installer.py", first_render + 1
    )
    assert producer < first_render < second_render
    assert (
        workflow.index("Require a changed pull-request version to increase") < producer
    )
    consumers = workflow[workflow.index("  pipx-lifecycle:") :]
    assert "python -m build --wheel" not in consumers
    assert "--candidate-wheel" in consumers


def test_recovery_workflow_is_protected_exact_and_forward_only():
    workflow = (ROOT / ".github/workflows/promote-managed-installer.yml").read_text(
        encoding="utf-8"
    )

    assert "environment: managed-installer-release" in workflow
    assert workflow.count("persist-credentials: false") == workflow.count(
        "actions/checkout@"
    )
    assert "group: managed-installer-stable" in workflow
    assert "github.event.repository.default_branch" in workflow
    assert "persist-credentials: false" in workflow
    assert 'gh release view "$TAG"' in workflow
    assert '"isPrerelease": False' in workflow
    assert workflow.count("--pattern") == 3
    assert "validate_release_bundle" in workflow
    assert 'gh release verify "$TAG"' in workflow
    assert '[[ "$TAG" == "v1.0.1" ]]' in workflow
    assert '--source-ref "refs/tags/$TAG"' in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert workflow.count("python scripts/request_managed_installer_promotion.py") == 3
    assert "request_managed_installer_promotion.py validate" in workflow
    assert "request_managed_installer_promotion.py promote" in workflow
    assert "request_managed_installer_promotion.py verify" in workflow
    assert "unset DEPLOY_PRIVATE_KEY DEPLOY_KNOWN_HOSTS" in workflow
    assert (
        workflow.index("Verify release immutability and artifact provenance")
        < workflow.index("Validate installer before admitting deployment credentials")
        < workflow.index("Request atomic stable installer promotion")
        < workflow.index("Verify stable installer from public endpoint")
    )
    assert "rollback" not in workflow.lower()


def test_all_release_workflow_actions_are_pinned_to_full_commit_shas():
    import re

    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        for reference in re.findall(
            r"^\s*-?\s*uses:\s*([^\s#]+)", source, re.MULTILINE
        ):
            assert re.fullmatch(r"[^/@\s]+/[^/@\s]+@[0-9a-f]{40}", reference), (
                path,
                reference,
            )
