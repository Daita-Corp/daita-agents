# Managed installer

`scripts/install.sh` is the repository implementation of Daita's managed
application delivery contract. It installs the canonical `daita-agents` wheel
into isolated generations under `~/.local/share/daita`, publishes
`~/.local/bin/daita`, and leaves application data under `~/.daita` separately
owned.

## Release boundary

The checked-in `scripts/install.sh` remains a fail-closed source template. Its
release literals contain explicit `UNRESOLVED_*` sentinels for:

- installer version and release sequence;
- the immutable `daita-agents` wheel URL and SHA-256;
- the official `uv` version; and
- each packaged target's `uv` archive URL, checksum, and managed CPython 3.12
  identity.

Installation and repair stop before mutation while any sentinel remains.
`--help`, `--version`, and `--dry-run` remain available for template review.
Never serve this source template directly.

`release/managed-installer.json` is the reviewed release policy. It pins one
installer sequence, uv 0.12.7, exact official uv archives and SHA-256 digests,
and the CPython 3.12.14 identity for each supported target:

- Apple Silicon macOS;
- Intel macOS;
- ARM64 Linux with glibc; and
- x86-64 Linux with glibc.

`scripts/render_managed_installer.py` validates the policy, the candidate
wheel archive, its distribution metadata, and its immutable versioned URL. It
then renders `install.sh` and `release-manifest.json` atomically. The manifest
records the exact wheel, installer, runtime, target, and checksum evidence.
The deterministic installer fixtures use this renderer too; there is no
second test-only substitution implementation.

The public `https://daita-tech.io/install.sh` endpoint has not been promoted
yet. Keep the customer-facing quick start on pipx until that endpoint serves
the exact reviewed release asset and passes the post-promotion checks below.

## Candidate verification

The managed lifecycle smoke can exercise an unpublished local wheel with an
already downloaded official `uv` archive, a real uv-managed Python, and
production dependencies resolved from PyPI. Supply all verification arguments
together:

```bash
.venv/bin/python tests/managed_installer_lifecycle_smoke.py \
  --candidate-wheel /absolute/path/to/the-once-built-candidate.whl \
  --real-uv-archive /absolute/path/to/the-verified-official-uv.tar.gz \
  --real-uv-version <version> \
  --real-uv-member <archive-directory>/uv \
  --real-python-request <exact-request> \
  --real-python-identity <exact-resolved-identity>
```

The smoke copies the candidate wheel and `uv` archive into a temporary fixture
transport; it does not upload them. The installer verifies both checksums and
the wheel metadata. The selected `uv` binary downloads the managed Python and
resolves the wheel's declared production dependencies from PyPI.

Run the same once-built candidate wheel through
`tests/pipx_lifecycle_smoke.py`. Release verification also includes syntax,
shellcheck, deterministic tests, architecture checks, formatting, typing, and
clean-machine tests on every platform for which support will be claimed.

## Automated release

`.github/workflows/managed-release.yml` is the publication control plane. A
manual run with **Publish** disabled builds and verifies a release candidate
without publishing it. Pushing a tag matching the project version, such as
`v1.0.1`, runs the same verification without publishing. To publish, manually
run the workflow on that exact tag with **Publish** enabled. The GitHub
publication job names the `managed-installer-release` environment, and the
PyPI job names the separate `pypi` environment. Configure both with required
reviewers before the first joint release.

The workflow:

1. validates that the tag, project version, and reviewed installer policy
   agree;
2. builds the wheel once and passes those exact bytes to every later job;
3. renders the installer twice and requires byte-for-byte deterministic output;
4. runs the managed and pipx lifecycles against the once-built wheel;
5. downloads and verifies the pinned official uv archive, installs the exact
   managed Python, and runs the real managed lifecycle on all four native
   target runners;
6. records `SHA256SUMS` and GitHub artifact attestations;
7. requires an explicit publish run, enforces a forward-only sequence, and
   refuses to replace an existing release; and
8. refuses a version already present on PyPI, creates the versioned GitHub
   release, and downloads every public asset again to prove that the published
   bytes match the verified bytes;
9. publishes the exact once-built wheel through PyPI Trusted Publishing with
   no stored API token; and
10. reads the version-specific PyPI JSON API and verifies the public wheel's
    filename and SHA-256 against the candidate artifact.

Before the first PyPI run, configure the existing `daita-agents` project with
this GitHub Actions Trusted Publisher:

```text
Owner: Daita-Corp
Repository: daita-agents
Workflow: managed-release.yml
Environment: pypi
```

The protected manual publish run is the only PyPI upload path. A tag push or a
candidate run with **Publish** disabled cannot upload. The PyPI job receives
only `id-token: write`, downloads the already verified workflow artifact, and
uploads only the wheel staged in `pypi-dist`. It does not rebuild or tolerate
an existing version. If the upload job fails after the GitHub release succeeds,
fix the publisher configuration and use GitHub's **Re-run failed jobs** action;
the successful build and GitHub publication jobs remain unchanged.

Before tagging a later release, increment `installer.release_sequence` in the
reviewed policy. An older installer refuses to replace a newer installed
sequence.

## Stable endpoint promotion

The release workflow intentionally does not mutate the marketing deployment.
After the versioned GitHub release succeeds, deploy its exact `install.sh`
asset to `/install.sh` on `daita-tech.io` without templating, redirects, or
runtime substitution. Record the release-manifest checksum in the deployment
change and verify the public bytes:

```bash
curl -fsSL --proto '=https' --tlsv1.2 \
  https://daita-tech.io/install.sh -o /tmp/daita-install.sh
shasum -a 256 /tmp/daita-install.sh
bash /tmp/daita-install.sh --version
bash /tmp/daita-install.sh --dry-run --no-onboard --no-modify-path
```

The SHA-256 must equal `installer.sha256` in the versioned
`release-manifest.json`. Only then should the website expose the pipe-to-shell
command. Stable rollback means redeploying an earlier reviewed `install.sh`;
it never changes application data or OS-keychain entries.
