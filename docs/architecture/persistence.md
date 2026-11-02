# Persistence, agent-home revisions and upgrades

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

Where durable state lives, how its format is versioned, and the release contract. See also `docs/LOCAL_STATE_UPGRADES.md`.

## Current home revision

The current agent-home format is **revision 4**, extending released revision 3
with bounded foreground analytical evidence, retained artifact computation
provenance and agent-owned declared catalog links. Released revisions 1 through 3 and their migrations remain immutable. Revision 4 is the
sole current runnable format; legacy decoders remain migration-only.

## State boundary, migrations and releases

`daita.storage.sql.SQLStateStore` owns shared state operations; `sql_jobs`
and `sql_routines` hold the job and routine transitions it runs inside its
transactions. The default SQLite adapter implements the supported
`SQLDatabase` connection boundary. Custom adapters reuse these operations and
strict codecs, with no copied state-transition logic. `run_sql_transaction` is
the shared cancellation-safe transaction owner for adapter administrative writes.
Adapters own physical admission, connection lifetime and error normalization.
The caller must drain operations before closing its borrowed database resources.

The canonical schema and existing home registry remain the single source of
record formats and data transformations. Adapters may project the schema and
execute those transformations using their own physical storage mechanics. Schema
projection alone does not upgrade stored data. Adapter deployment/security/layout
checks belong to the consuming application; do not add their implementation
fingerprints or dependencies to the framework's release gate.

`daita.storage.sqlite.SQLiteStateStore` is the default local admission wrapper.
`daita.storage.home_migrations` is the sole persistence-compatibility
authority for the complete agent home. One monotonic home revision covers the
database, persisted records, model configuration, memory, user profile, skills,
artifacts, and other durable files that must change together. It is independent
of the package version and Git tag.

Production home revisions 1 through 3 are frozen. The registry is ordered and append-only;
released migration IDs, checksums, implementations, target schemas, historical
decoders, and golden fixtures never change. `CURRENT_HOME_REVISION` derives from
the last registry entry. A format change appends one owner-local home migration
and declares every affected relative path. Fresh homes are created directly at
the current complete format rather than replaying history.

Current runtime serializers accept only the current shape and contain no
per-record version discriminators or compatibility branches. Historical parsing
belongs only to the immutable migration that consumes that shape. The revision-1
bridge admits only the exact three observed preproduction shapes; it is not a
general legacy framework.

Open owns the existing agent-home writer lock across inspection, crash recovery,
upgrade, full-home validation, and runtime composition. Upgrades preflight disk
space, stage and back up all affected files, use SQLite's backup API, apply the
known suffix in order, validate the complete staged home, and publish `state.db`
last. Hash-bound journals recover staging, partial commit, and rollback states.
Publication failure restores the prior home, and one rollback bundle is retained.
Newer, unsupported, reordered, checksum-edited, or damaged homes fail closed
without a state rewrite.

Every supported production revision has one immutable whole-home golden fixture.
Tests cover every supported-to-current path, durable-data preservation, failure
and crash boundaries, recovery, rollback, and downgrade refusal. The inclusive
support window is an explicit registry policy; changing it is a release decision.
See `docs/LOCAL_STATE_UPGRADES.md` for the operational contract.

Release compatibility is maintained between tagged releases, not arbitrary
development commits. `release/agent-home-contract.json` is the committed
semantic snapshot of the current physical SQLite schema, strict stored-record
field contracts, owned home-file layouts, and migration registry. Before a pull
request or tag, refresh and verify it with:

```bash
python scripts/check_home_release_contract.py write
python scripts/check_home_release_contract.py check
```

CI compares the candidate with the latest earlier tag containing a snapshot. A
durable-contract change requires `CURRENT_HOME_REVISION` to exceed that tagged
revision and requires the matching migration and golden fixture. A release with
no persistence change retains the current home revision. Every managed GitHub
release publishes its exact snapshot even when it matches the prior release.

One not-yet-released next migration may be amended across development commits;
update its candidate checksum, fixture, and snapshot together. Compatibility
between homes created by unreleased commits is not promised. Once the first Git
tag containing that revision is created, its migration ID, checksum,
implementation, target schema, historical decoder, and golden fixture are
immutable. The snapshot is release evidence, not a second runtime compatibility
authority. A semantic durable-format change that the generator cannot infer
still requires an explicit new home revision.

The exact pre-registry revision-3 development home from commit `91e0f8ce` is an
explicit migration input. Its original checksum, released prefix, and former
schema must all match before a staged same-revision conversion. The conversion
preserves caller and MCP records, backfills artifact metadata, and updates the
revision-3 ledger checksum. It uses the ordinary staged upgrade and retains one
rollback copy. Unknown development histories still fail closed; there is no
legacy runtime path.

Source read authority exists only in `source_read_scopes`. Native relational write
authority exists only in `relational_write_scopes`. Connection JSON never
owns either permission. Reconstruction fails closed, refresh preserves exact
scopes, and detach revokes both scope families atomically.

All state mutation must be atomic and cancellation-safe, and there is never a
second writer around an agent's SQLite state (invariants in `AGENTS.md`). Today
there is no event sourcing, replay projection or second state abstraction; adding
one is an architecture change. A job is one row in the `jobs` table: its
checklist, delegations and answers live in the record, which is current state
compared-and-set by revision, not an event log, and it stays behind
`SQLiteStateStore`. Revision 4 replaced the task-graph tables with it; the
frozen graph records and `storage/sqlite_graph.py` remain only because the
released revision-2 conversion imports them.
