# Managed installer release status

`scripts/install.sh` is the sole repository-owned implementation of Daita's
managed application delivery contract. It installs the canonical
`daita-agents` wheel into isolated generations beneath
`~/.local/share/daita`, publishes `~/.local/bin/daita`, and leaves application
data under `~/.daita` separately owned.

The checked-in installer is intentionally not publishable. Its release
literals remain explicit `UNRESOLVED_*` sentinels for:

- installer version and release sequence;
- the immutable `daita-agents` wheel URL and SHA-256;
- the official `uv` version; and
- each intended target's official `uv` archive URL/SHA-256 and exact managed
  CPython 3.12 build identity.

Normal installation and repair fail before mutation while any sentinel
remains. `--help`, `--version`, and `--dry-run` remain available for review.
Deterministic tests render local fixture-pinned copies of the same script;
those fixture artifacts are not release evidence.

## Intended, not yet claimed, targets

| Target | Intended baseline | Status |
| --- | --- | --- |
| macOS arm64 | macOS 13+, Terminal.app | Real bootstrap lifecycle passed on macOS 26.4.1 arm64 in a temporary home; clean-machine and real-terminal evidence remain unverified, so support is not claimed |
| macOS x86_64 | macOS 13+, Terminal.app | Unverified; not claimed |
| Linux x86_64 glibc | Ubuntu 22.04/24.04, reviewed local terminal | Unverified; not claimed |
| Linux arm64 glibc | Ubuntu 24.04, GNOME Terminal | Unverified; not claimed |
| WSL2 | Deferred | Not claimed |
| Native Windows | Deferred | Not implemented or claimed |

Automated lifecycle results prove installer mechanics only. Every claimed
entry still requires the specified clean ordinary-user machine and real local
terminal evidence. Other Linux distributions, musl/Alpine, SSH, tmux, VS
Code, and third-party terminals remain unverified.

## Pre-publication candidate certification

The managed lifecycle smoke can exercise an unpublished local wheel with an
already-downloaded official `uv` archive, a real uv-managed Python, and
production dependencies resolved from PyPI. Supply all five `--real-*`
arguments together:

```bash
.venv/bin/python tests/managed_installer_lifecycle_smoke.py \
  --candidate-wheel /absolute/path/to/the-once-built-candidate.whl \
  --real-uv-archive /absolute/path/to/the-verified-official-uv.tar.gz \
  --real-uv-version <version> \
  --real-uv-member <archive-directory>/uv \
  --real-python-request <exact-request> \
  --real-python-identity <exact-resolved-identity>
```

The candidate wheel and `uv` archive are copied into a temporary fixture
transport; they are not uploaded. The canonical installer still verifies both
checksums and wheel metadata. The selected real `uv` binary downloads the
managed Python and resolves the wheel's declared production dependencies from
PyPI. This proves the selected host/bootstrap combination but does not prove
public artifact delivery, another target, a clean machine, or real-terminal
behavior.

On 2026-08-06, the exact 1.0.0 candidate wheel with SHA-256
`4c485f4587f179fd4a632e0d2c80da4511aa9a4a0cfb475f4da77936b2b0bfc5`
passed this lifecycle on macOS 26.4.1 arm64 with official `uv` 0.12.2 archive
SHA-256
`fa909fea3bc06f460db79017030a221fdbc43ec4478f089cb554d8335c090817`
and `cpython-3.12.13-macos-aarch64-none`. Install, repeat, read-only verify,
repair, rollback, damaged-generation repair, uninstall, production-dependency
imports, and application-data/sentinel preservation all passed. This is local
candidate evidence, not a final release pin or support claim.

## Publication gate

The release owner must close Daita, then:

1. build the candidate wheel exactly once in an isolated environment;
2. pass that absolute wheel path to both lifecycle smoke scripts, including
   the real-bootstrap managed mode on each intended target before publication;
3. publish those exact wheel bytes to the canonical package host;
4. record the immutable wheel URL and published SHA-256;
5. replace every installer sentinel with reviewed official artifact evidence;
6. run syntax, shellcheck, deterministic, managed, pipx, architecture,
   formatting, typing, clean-machine, and real-terminal gates;
7. publish the reviewed installer bytes and checksum at the immutable
   versioned endpoint; and
8. only then promote those same bytes to
   `https://daita-tech.io/install.sh`.

No step in this repository change publishes, uploads, promotes, or modifies
that endpoint. Binary rollback only changes the active verified generation; it
does not roll back application data.
