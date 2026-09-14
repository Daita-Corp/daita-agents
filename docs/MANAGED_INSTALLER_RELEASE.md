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
- repository release immutability, which makes future published assets and their
  associated tags unchangeable;
- the `managed-installer-release` protected environment with required reviewers,
  self-review disabled, and deployment patterns admitting only the default `main`
  branch plus `v*` tags (publication uses a tag; recovery uses `main`);
- the default-branch ruleset with CODEOWNER review required for workflow, release,
  script, documentation, and test changes;
- the managed release workflow as the only supported GitHub release publisher;
- the repository-wide `managed-release` build/publish concurrency group; and
- the cross-workflow `managed-installer-stable` promotion concurrency group,
  both with `cancel-in-progress: false`.

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
five-artifact provenance attestation, GitHub release immutability, exact PyPI
filename and SHA-256 verification, and post-publication downloads compared byte
for byte. Actionlint validates all workflow syntax before release gates continue;
its sole ignored diagnostic is its outdated runner catalog rejecting GitHub's
documented `macos-15-intel` hosted-runner label. The fifth artifact is the verified
`agent-home-contract.json` snapshot;
it records release compatibility evidence without participating in runtime home
admission. All third-party Actions plus the Gitleaks and Actionlint containers are
pinned by full commit or image digest. CI and the managed-release publication gate scan both a
clean current-source archive and complete Git history with redacted findings.

## Release procedure

1. Confirm the protected-tag ruleset, exclusive workflow publisher, protected
   environment, and global concurrency control are active.
2. Query both registries and set `project.version` to the next unused patch
   release after their semantic maximum. Refresh the editable environment.
3. Refresh `release/agent-home-contract.json`; if it differs from the latest
   tagged snapshot, include the candidate home migration and golden fixture.
4. Merge only after ordinary CI, home-contract, and release-identity checks pass.
5. Run `python scripts/release_identity.py project-tag` and create that exact
   annotated tag.
6. Push the tag and wait for the candidate workflow and all four native smokes.
7. Download that run's `managed-release` artifact and verify `SHA256SUMS`.
8. Upload only its exact wheel to PyPI.
9. Run the protected workflow on the same tag with **Publish** enabled and
   approve `managed-installer-release`.
10. Confirm it re-read both registries, verified the exact PyPI wheel, attested
    all five artifacts, revalidated the exact remote annotated tag, created and
    byte-verified a draft, revalidated the tag again, published it as an immutable
    release, compared downloaded public bytes, verified provenance, atomically
    promoted `install.sh`, and verified the stable public endpoint.

If a run stops after draft creation but before publication, inspect and delete only
that workflow-created draft before retrying the same protected run. Never replace
or delete a published release.

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

The Daita-agents repository owns the stable installer. The marketing application
does not contain or deploy a copy. Nginx on Lightsail serves only the exact
`/install.sh` route from `/srv/daita-installer/current/install.sh`; every other
request remains with the existing website application.

The protected publication job performs the normal promotion automatically. It
verifies the GitHub-hosted provenance of the exact published bytes, connects with
a dedicated SSH key, and can send only `promote vMAJOR.MINOR.PATCH`. The server's
forced command downloads `install.sh`, `release-manifest.json`, and `SHA256SUMS`
from the fixed `Daita-Corp/daita-agents` release URL. It validates their exact
identity and parses the installer with `bash -n` without executing release bytes,
then atomically moves an immutable release directory into place and switches one
relative `current` symlink. The ephemeral GitHub runner performs the version and
non-mutating dry-run checks immediately before requesting promotion and again from
the public endpoint. A lower version is refused before any download.

The workflow requires GitHub's immutable-release attestation after publication,
in addition to the build-provenance attestations it creates. The one-time v1.0.1
seed predates those build attestations, so recovery admits it only after verifying
its immutable-release attestation and the host's three hard-coded artifact
identities. That exception cannot admit any other tag.

The runner then downloads `https://daita-tech.io/install.sh` without following
redirects, requires HTTP 200 and the exact final URL, checks the expected SHA-256,
and repeats the non-mutating installer checks. A published GitHub release remains
available if promotion fails; run `Recover managed installer promotion` on the
default branch with that exact existing tag after correcting the deployment
problem. The recovery workflow uses the same protected environment, provenance
policy, server command, and public verification and cannot create or replace a
release.

Configure these protected-environment values once:

- variable `MANAGED_INSTALLER_SSH_HOST`: the Lightsail host or fixed public IPv4;
- secret `MANAGED_INSTALLER_SSH_PRIVATE_KEY`: the dedicated unencrypted Ed25519
  private key used only by this forced command; and
- secret `MANAGED_INSTALLER_SSH_KNOWN_HOSTS`: a reviewed pinned host-key line for
  that exact destination.

The corresponding private key must never be stored in the repository, website,
server checkout, workflow artifact, or runner log.
The workflow executes installer validation in a step that receives no deployment
secrets. Its next step writes the key and known-host pin into one mode-700 temporary
directory, unsets their environment variables before starting Python, requests only
the forced promotion, and removes the directory on exit. Public installer execution
runs in the following secret-free step.

### One-time Lightsail bootstrap and cutover

Generate a new dedicated Ed25519 key outside the repository. From a reviewed
checkout copied to Lightsail, run the bootstrap as root with only its public key:

```bash
sudo bash scripts/bootstrap_managed_installer_host.sh \
  /absolute/path/to/daita-installer-release.pub
```

The bootstrap creates the locked `daita-installer-deploy` system account, a
single restrictive root-owned `authorized_keys` entry in a root-owned home,
root-owned host code under `/usr/local/libexec/daita-installer`, and the deployment tree under
`/srv/daita-installer`. The key grants no general shell, forwarding, PTY, agent
forwarding, or rollback command. The installed host command is self-contained and
the bootstrap requires the server's `/usr/bin/python3` to be version 3.10 or newer.

Seed the current public v1.0.1 release through the same host implementation:

```bash
sudo -u daita-installer-deploy /usr/bin/python3 \
  /usr/local/libexec/daita-installer/managed_installer_host.py \
  --root /srv/daita-installer promote v1.0.1
```

The host accepts that one pre-schema-2 release only by its three pinned SHA-256
identities. All later releases require manifest schema 2.

The bootstrap also installs [the reviewed Nginx location](../release/daita-installer-location.nginx.conf)
and a root-only cutover helper. After confirming the Nginx worker can read
`/srv/daita-installer/current/install.sh`, run:

```bash
sudo /usr/bin/python3 \
  /usr/local/libexec/daita-installer/managed_installer_nginx.py cutover
```

The helper admits only the observed `daita-tech.io` two-server layout, retains one
mode-600 backup outside Nginx's include directory, inserts only the exact-match
location before the existing catch-all, runs the real Nginx configuration test,
and performs a graceful reload. A failed test or reload automatically restores,
retests, and reloads the prior configuration. It never replaces the server block
or the website catch-all location. Then verify:

```bash
curl -fsSL --proto '=https' --tlsv1.2 \
  https://daita-tech.io/install.sh -o /tmp/daita-install.sh
shasum -a 256 /tmp/daita-install.sh
bash /tmp/daita-install.sh --version
bash /tmp/daita-install.sh --dry-run --no-onboard --no-modify-path
```

The checksum must equal `installer.sha256` in the versioned public manifest.
Only after that check succeeds should the website repository's historical
`public/install.sh` be removed.

If the initial public cutover check fails, restore the pre-cutover Nginx file with
the same drift protection:

```bash
sudo /usr/bin/python3 \
  /usr/local/libexec/daita-installer/managed_installer_nginx.py restore
```

### Emergency rollback

The CI deployment key deliberately cannot roll back. An authorized Lightsail
administrator can atomically restore only the predecessor recorded by the current
release:

```bash
sudo -u daita-installer-deploy /usr/bin/python3 \
  /usr/local/libexec/daita-installer/managed_installer_host.py \
  --root /srv/daita-installer rollback
```

The host revalidates the predecessor's identity and syntax before switching the symlink.
Repeat the public checksum and non-mutating checks immediately afterward. A
rollback changes only the served installer bytes; it does not change existing
Daita installations or their agent homes.

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

There is no scheduled live smoke test. Normal publication and manual recovery
both perform release-time public verification; a user-reported or observed outage
is handled through the protected recovery or rollback procedures above.
