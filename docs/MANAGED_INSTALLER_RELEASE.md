# Managed installer

`scripts/install.sh` is the repository implementation of Daita's managed
application delivery contract. It installs the canonical `daita-agents` wheel
into isolated generations under `~/.local/share/daita`, publishes
`~/.local/bin/daita`, and leaves application data under `~/.daita` separately
owned.

## Current availability

The checked-in installer is not publishable. Its release literals contain
explicit `UNRESOLVED_*` sentinels for:

- installer version and release sequence;
- the immutable `daita-agents` wheel URL and SHA-256;
- the official `uv` version; and
- each packaged target's `uv` archive URL, checksum, and managed CPython 3.12
  identity.

Installation and repair stop before mutation while any sentinel remains.
`--help`, `--version`, and `--dry-run` remain available for review. The public
`https://daita-tech.io/install.sh` endpoint is not live, and support is not
claimed for any managed-installer target.

Deterministic tests render fixture-pinned copies of the same script. Those
tests validate installer mechanics but do not establish public artifact,
clean-machine, operating-system, or terminal support.

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

## Publication responsibilities

Publishing is an external release operation. The release process must:

1. build the candidate wheel once in an isolated environment;
2. verify those exact bytes with both lifecycle smoke scripts on each claimed
   target;
3. publish the same wheel bytes to the canonical package host;
4. record the immutable wheel URL and SHA-256;
5. replace every installer sentinel with reviewed official artifact values;
6. rerun the complete verification set against the resolved installer; and
7. publish the reviewed versioned installer before updating the stable
   endpoint.

The repository installer cannot publish, upload, or promote artifacts. Binary
rollback changes only the active verified generation; it never rolls back
application data or OS-keychain entries.
