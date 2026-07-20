# Fresh state and migration

## Candidate policy

Daita 2.0 starts with a new state root. Before cutover, the default is
`~/.daita-next/`; callers and the CLI may select another root explicitly.
Each agent owns a private directory containing an `agent.toml` bootstrap
manifest, authoritative `state.db`, and content-addressed blobs.

V2 does not open, copy, or lazily interpret v1 state. Agent bootstrap rejects
a v1 root, a descendant of a v1 root, identity mismatches, incompatible schema
versions, symlink aliases, and competing writers. Opening a current v2 agent
runs only ordered v2 migrations, with a SQLite backup before a schema change.

## Upgrade procedure

1. Stop every v1 and v2 process that could own the selected agent.
2. Back up the existing v1 state and any attached source data independently.
3. Install the v2 candidate in a clean environment.
4. Choose a new v2 root and initialize a new agent there.
5. Reattach sources using secret references and least-privilege credentials.
6. Recreate only reviewed memories, skills, and monitors; do not copy database
   rows or internal files between versions.
7. Run read-only validation before enabling a controlled write recipe.
8. Keep the v1 backup until Phase 10 cutover and rollback approval are complete.

Uninstalling the Python package does not remove an agent root. State deletion
is a separate, explicit operator action and is not performed by v2 tooling.

## Current v2 schema upgrades

Existing v2 Agent Homes upgrade only through the ordered package-owned SQLite
migrations, with a verified backup before the first pending migration. Phase
9.5 adds exact task/evidence read authority, reconstructable model routes, and
configured extension bindings. Legacy rows remain readable where safe, but no
migration invents authority, a route, a secret, or an extension configuration:

- pre-authority tasks/evidence retain schema-zero facts and fail closed when a
  current safety decision requires exact source/resource revisions;
- profile-only homes remain inspection-only until an operator explicitly sets
  a reconstructable route;
- homes without an extension binding gain no implicit configured extensions;
  and
- resolved secret values are never migration input or output.

Migration 15's multi-source columns are authoritative for comparisons. The
older migration-13 singular source column remains only a verified SQLite
compatibility projection when two source IDs are stored; it is removed during
record decoding and never narrows or replaces the exact multi-source facts.
Migration/reopen and cross-source comparison regressions pass in P9.5-Q08.

## No implicit migration

There is no general v1-to-v2 state migration command. The lifecycle models and
trust boundaries differ enough that a permanent dual reader would hide unsafe
or unsupported mappings. A future one-time importer would require a separate,
backup-first design with explicit source versions, checksums, validation, and
auditable per-record dispositions; it may never become a normal runtime path.
