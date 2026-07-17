# ADR 0001: Replacement isolation and baseline

- **Status:** Accepted
- **Date:** 2026-07-16

## Context

The replacement needs its final `daita` import name while v1 remains runnable
as a behavioral oracle. Allowing either package to import the other would turn
the replacement into a permanent compatibility layer.

## Decision

- Build only under `next/`, with production source at `next/src/daita/`.
- Keep the final distribution/import names `daita-agents` and `daita`; do not
  create `daita.v2` or `daita_next`.
- Record v1 baseline commit
  `b87df31873d33fffbf50498f5dc4d8892115e8f8` on branch `next`. No baseline tag
  exists; the immutable commit is the Phase 0 oracle. A final v1 tag is a
  Phase 10 precondition and is not authorized now.
- Root `daita/` is frozen except for a separately authorized security or
  production-critical fix. Every such fix must be evaluated for a v2
  regression test.
- V2 production never imports, executes, symlinks, or falls back to v1.
- Compare implementations only through subprocess output or neutral serialized
  fixtures. Never install both editable packages in one interpreter.
- Keep `next/` outside the root distribution until cutover.
- Anchor the ignored local plan by SHA-256
  `403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935`.

## Consequences

Phase 0 must provide import, source-scan, symlink, and package-content checks.
Leaf behavior may be ported only after assigning its v2 owner and adding a v2
regression test.
