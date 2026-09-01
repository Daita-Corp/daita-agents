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
`v1.0.1`, runs the same verification without publishing. The successful tag
run retains the exact verified wheel in its `managed-release` workflow
artifact. Download and publish that wheel to PyPI locally, then manually run
the workflow on the same tag with **Publish** enabled. The GitHub publication
job names the `managed-installer-release` environment; configure it with the
required reviewers.

The workflow:

1. validates that the tag, project version, and reviewed installer policy
   agree;
2. builds the wheel once and passes those exact bytes to every later job;
3. renders the installer twice and requires byte-for-byte deterministic output;
4. runs the managed and pipx lifecycles against the once-built wheel;
5. downloads and verifies the pinned official uv archive, installs the exact
   managed Python, and runs the real managed lifecycle on all four native
   target runners;
6. records `SHA256SUMS` and retains the verified release artifacts;
7. requires the exact once-built wheel to be published locally to PyPI before
   the protected GitHub publication run can proceed;
8. reads the version-specific PyPI JSON API and verifies that it contains only
   the expected wheel with the candidate artifact's SHA-256;
9. refuses to replace an existing GitHub release and enforces a forward-only
   release sequence; and
10. attests the exact artifacts, creates the versioned GitHub release, and
    downloads every public asset again to prove that the published bytes match
    the verified bytes.

## Local PyPI publication

PyPI publication intentionally uses the project API key on a release operator's
machine. The workflow stores no PyPI credential, requests no PyPI OIDC token,
and does not require a `pypi` GitHub environment or Trusted Publisher. Keep the
key only in the ignored repository-root `.env` file:

```text
PYPI_API_KEY=pypi-...
```

Use this order for every release:

1. merge the reviewed version and installer-sequence change;
2. create and push the annotated `vX.Y.Z` tag;
3. wait for the tag-triggered **Daita managed release** workflow to pass on all
   four native targets;
4. download and extract that run's `managed-release` artifact into a clean
   local directory;
5. verify `SHA256SUMS`, then upload only its wheel with Twine and the local API
   key; and
6. manually run **Daita managed release** on the same tag with **Publish**
   enabled and approve the `managed-installer-release` environment.

From the repository root, with the downloaded files in
`/absolute/path/to/managed-release`, run:

```bash
cd /absolute/path/to/managed-release
shasum -a 256 -c SHA256SUMS

(
  set -eu
  PYPI_API_KEY="$(sed -n 's/^PYPI_API_KEY=//p' /absolute/path/to/daita-agents/.env)"
  test -n "$PYPI_API_KEY"
  TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_KEY" \
    /absolute/path/to/daita-agents/.venv/bin/python -m twine upload \
    --non-interactive --disable-progress-bar \
    daita_agents-X.Y.Z-py3-none-any.whl
)
```

The final workflow run fails closed unless PyPI returns exactly that wheel
filename and SHA-256. PyPI versions are immutable, so never rebuild or retry
with different bytes under the same version. Rotate the API key immediately if
the local `.env` file or release machine is exposed.

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
