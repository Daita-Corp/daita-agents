# Managed installer and release procedure

`scripts/install.sh` is the fail-closed source template for Daita's managed
application delivery. A rendered installer creates isolated binary generations
under `~/.local/share/daita`, publishes `~/.local/bin/daita`, and never owns or
changes application data under `~/.daita` or OS-keychain entries.

## One application release identity

The sole authored Daita release version is `[project].version` in
`pyproject.toml`. The build backend writes it into the `daita-agents` wheel;
the release renderer inspects that metadata and derives the installer identity,
public manifest identity, wheel filename, immutable URL, and expected `vX.Y.Z`
tag. Runtime `daita.__version__`, the CLI, TUI, and MCP client identity read the
installed distribution metadata.

For an ordinary release, edit only `project.version`. Never copy the release
value into source, policy, workflow configuration, tests, or documentation.
The agent-home revision is independent and changes only through a reviewed,
append-only durable-format migration.

An editable installation stores a metadata snapshot. After changing
`project.version` or checking out a commit with another value, refresh it before
importing Daita or running tests:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

## Reviewed runtime policy and rendered evidence

`release/managed-installer.json` contains only the reviewed external runtime:
the exact `uv` version, managed CPython request, and all four target-specific
archive names, members, immutable URLs, checksums, and Python identities. These
runtime pins change only when deliberately reviewed; they do not advance for
every Daita release.

The checked-in installer template contains fail-closed `UNRESOLVED_*` values
for the Daita version inspected from the wheel and for every immutable artifact
fact. Never serve the template directly. `scripts/render_managed_installer.py`
validates the runtime policy and untrusted wheel archive, requires the wheel
metadata to equal `pyproject.toml`, validates the authoritative versioned URL,
and atomically writes `install.sh` plus schema-2 `release-manifest.json`.
Identical inputs must produce byte-identical outputs.

The public manifest records one explicit `application.version`, the wheel and
installer evidence, and the complete runtime policy. `installer.sha256` is the
checksum of the rendered bytes, not a second version.

## Installation ordering and recovery

The rendered installer accepts only stable canonical `MAJOR.MINOR.PATCH`
versions whose three components are at most nine decimal digits. Its Bash
comparator uses bounded base-10 integer components and works on the Bash shipped
with every supported macOS and Linux target.

Before install or repair stages anything, the installer contains the active
generation path, reads its manifest, and independently asks that generation's
Python for exactly one installed `daita-agents` distribution version without
importing Daita. A valid matching pair supplies the active ordering identity.
A missing, malformed, or disagreeing manifest blocks normal installation;
explicit repair may use the independently recovered metadata version, but only
after the normal downgrade check. If metadata is missing, ambiguous, malformed,
or inaccessible, both install and repair fail without mutation.

Normal install accepts a fresh install or a newer candidate. An identical
version and wheel checksum is verified idempotently. Different wheel bytes under
the same version are refused unless explicit repair reinstalls the exact artifact
embedded by that versioned installer. Repair never permits an older candidate.
`--rollback` is the only downgrade operation: it verifies and swaps the recorded
current and previous binary generations and never changes application data.

Test failpoints are enabled only by an explicit in-process renderer argument
used by deterministic fixtures. The public renderer CLI has no such option and
always emits a disabled gate, so the test environment variable cannot affect
published installer bytes.

## Publication controls

Production release safety depends on repository controls as well as workflow
code. Configure all of the following before publishing:

- a GitHub ruleset protecting every `v*` tag from update or deletion, without a
  routine release-actor bypass;
- the `managed-installer-release` protected environment with required reviewers;
- the managed release workflow as the only supported GitHub release publisher;
  and
- the repository-wide `managed-release` concurrency group with
  `cancel-in-progress: false`.

Manual tag moves or deletion, manual GitHub release creation, and administrator
bypass are unsupported emergency actions requiring incident review.

The managed workflow has three modes:

| Mode | Required ref and registry state | Publication |
| --- | --- | --- |
| Branch/manual verification | A branch may verify an already published matching project version; a manually selected tag must still pass every tagged-ref identity and forward-ordering gate | Never |
| Tag candidate | Exact derived tag; candidate absent from PyPI and newer than the complete registry union | Never |
| Protected publication | Exact derived tag; exact candidate present on PyPI, no GitHub release, and candidate newer than every other registry version | After approval |

For candidate and protected modes, the workflow exhaustively paginates
published non-draft GitHub releases and reads the complete PyPI release map.
Every release key with at least one file counts, including releases whose files
are all yanked. Malformed, prerelease, truncated, unauthenticated, rate-limited,
or otherwise uncertain evidence blocks the run. The protected run repeats this
collection immediately before publication and permits only the expected equal
PyPI candidate; every higher version still blocks it.

Each CI or managed-release workflow run builds one wheel. All managed, pipx,
and four native lifecycle jobs download and consume those exact bytes without
rebuilding. The managed release retains syntax and shellcheck gates, deterministic
double rendering, exact runtime downloads, checksums, all-platform native smoke,
four-artifact provenance attestation, GitHub release immutability, exact PyPI
filename and SHA-256 verification, and post-publication downloads compared byte
for byte.

## Release procedure

1. Confirm the protected-tag ruleset, exclusive workflow publisher, protected
   environment, and global concurrency control are active.
2. Query both registries and set `project.version` to the next unused patch
   release after their semantic maximum. Refresh the editable environment.
3. Merge only after ordinary CI and release-identity checks pass.
4. Run `python scripts/release_identity.py project-tag` and create that exact
   annotated tag.
5. Push the tag and wait for the candidate workflow and all four native smokes.
6. Download that run's `managed-release` artifact and verify `SHA256SUMS`.
7. Upload only its exact wheel to PyPI.
8. Run the protected workflow on the same tag with **Publish** enabled and
   approve `managed-installer-release`.
9. Confirm it re-read both registries, verified the exact PyPI wheel, attested
   all four artifacts, created the immutable release, and compared downloaded
   public bytes.
10. Promote the exact versioned `install.sh` bytes to the stable website endpoint
    and verify its SHA-256 against `release-manifest.json`.

PyPI upload remains a deliberate local operator step using the ignored
repository-root `.env` value `PYPI_API_KEY`. The workflow stores no PyPI secret
and requests no PyPI publishing token. From the clean downloaded artifact
directory:

```bash
shasum -a 256 -c SHA256SUMS
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_KEY" \
  /absolute/path/to/daita-agents/.venv/bin/python -m twine upload \
  --non-interactive --disable-progress-bar \
  daita_agents-X.Y.Z-py3-none-any.whl
```

Never rebuild or upload different bytes under an existing version.

## Stable endpoint promotion

The release workflow does not mutate the marketing deployment. After the
versioned GitHub release succeeds, deploy its exact `install.sh` asset to
`https://daita-tech.io/install.sh` without templating, redirects, or runtime
substitution, then verify the public bytes:

```bash
curl -fsSL --proto '=https' --tlsv1.2 \
  https://daita-tech.io/install.sh -o /tmp/daita-install.sh
shasum -a 256 /tmp/daita-install.sh
bash /tmp/daita-install.sh --version
bash /tmp/daita-install.sh --dry-run --no-onboard --no-modify-path
```

The checksum must equal `installer.sha256` in the versioned public manifest.

## Accepted operational limits

- An installer-only correction consumes a new Daita patch version and wheel.
- The stable-only globally increasing contract has no beta channel and cannot
  publish a maintenance release below an already published higher version.
- Normal installation cannot select an arbitrary older release; rollback is
  limited to the recorded verified previous generation.
- Binary rollback may be unable to open a home advanced by a forward-only home
  migration; persistence admission continues to fail closed.
- Release verification is intentionally expensive even though every consumer
  reuses one wheel.
- Registry reachability and complete trustworthy responses are mandatory.
  Deleted registry history cannot be reconstructed, so versions must never be
  deleted or reused.

The stable public endpoint has not yet been promoted. Keep the customer quick
start on pipx until the exact reviewed asset passes the promotion checks above.
