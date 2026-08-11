# Local state upgrades

Daita package upgrades and Daita local-state upgrades are separate concerns.
The package can be replaced by pipx or the managed installer without touching
agent data. The first subsequent command that opens an agent admits its local
SQLite database and, when necessary, upgrades its explicit state format.

## User contract

For a supported upgrade, keep using the same command and agent home:

```bash
pipx upgrade daita-agents
daita
```

Daita automatically preserves:

- agent identity and manifest;
- model route, settings, and secret references;
- source registrations, active-source selection, and current catalogs;
- complete run inputs, transcripts, terminal results, and conversation order;
- artifact references and committed artifact files;
- `MEMORY.md`, `USER.md`, and authorized skills;
- semantic annotations and learning candidates/review state; and
- database-write receipts and every other table or file owned by the supported
  agent-home format.

Secret values are not migration payloads. Environment references continue to
name the same environment variables, keychain references continue to name the
same OS-keychain entries, and neither value is copied into SQLite.

A backup is optional disaster recovery. A routine supported upgrade neither
requires nor restores one.

## Admission outcomes

| Existing state | Behavior |
| --- | --- |
| Current format | Validate and open without a schema write. |
| Supported immediately preceding format | Upgrade atomically, validate, commit, and open. |
| Unversioned known 1.0 candidate shape | Identify by exact schema, stamp or upgrade atomically, and open. |
| Newer format | Refuse the downgrade without changing state; install the same or a newer Daita package. |
| Recognizable pre-1.0 framework | Refuse without creating mixed current state in that root. |
| Damaged or unexpected schema | Refuse without changing database contents. |
| Migration failure | Roll back the SQLite transaction; no partial schema is admitted. |
| Caller cancellation | Wait for the in-flight transaction to settle, then surface cancellation; state is wholly old or wholly upgraded. |

Headless CLI commands return a structured error with `code`, `state_path`,
`found_format`, `current_format`, and `state_changed`. Interactive startup
renders the equivalent facts as a human-readable terminal diagnostic. The
stable error codes are `state_format_newer`, `state_format_legacy`,
`state_database_damaged`, and `state_migration_failed`.

## Ownership and safety

`SQLiteStateStore` is the only state-format and migration owner. Admission runs
after `EmbeddedAgent` acquires the existing process-level writer lock, so two
package processes cannot migrate one home concurrently. The migration uses
`BEGIN IMMEDIATE`, checks the exact source schema, runs SQLite integrity and
foreign-key checks, follows the ordered SQLite-owned migration ledger to the
current format, validates every intermediate target, and commits once. DDL,
format markers, and data changes roll back together on failure.

The direct agent loop, capability registry, data runtime, catalog, adapters,
artifact store, memory and skill stores, and configuration owners do not gain a
second compatibility path. Files outside SQLite are kept in place and are not
rewritten merely because a database format changes.

The currently certified finite steps are:

- format 1 to 2 adds the immutable database-write receipt table without
  changing prior rows; and
- format 2 to 3 makes the fail-closed PostgreSQL admission default explicit by
  adding `write_access: false` only where an older source registration omitted
  it. Existing `true` and `false` values, source identity, credential
  references, active-source selection, catalogs, conversations, and all other
  state remain unchanged.

The ledger is the release history for state changes. Package releases that do
not change the durable contract do not add a format or migration. A release
that does change it adds one reviewed `from_version -> to_version` entry with
its exact source schema, target schema, transform, and target validation. The
admission path contains no first/previous-version branches; it follows and
validates the available chain automatically, including when a user skips a
package release whose migration remains supported.

Source refresh is not source reattachment or a configuration mutation. It
reopens the exact persisted registration for discovery and atomically replaces
catalog truth only after discovery succeeds; it cannot change write admission,
credentials, identity, or attachment lifecycle.

## Release certification

Every state-format release must retain a schema fixture generated from the
actual immediately preceding wheel and add its explicit migration step. The
deterministic contract suite proves schema identification, row preservation,
rollback, cancellation, downgrade refusal, legacy/damage classification,
diagnostics, lock release, credential-reference preservation, and receipts.

Before publishing, run the isolated two-wheel lifecycle with once-built
artifacts:

```bash
.venv/bin/python tests/pipx_lifecycle_smoke.py \
  --baseline-wheel /path/to/immediately-preceding.whl \
  --candidate-wheel /path/to/candidate.whl
```

That test installs the real prior wheel, creates an agent with conversations,
an artifact, memory, user profile, skill, semantics, learning state, model
configuration, source registration, current catalog, and active source;
replaces the package; opens and upgrades the same home; compares every prior
database row and every non-database file; appends a conversation; and verifies
that package uninstall still leaves the complete agent home intact.

If a release skips more than one state format, certify every required explicit
step or require sequential package upgrades. Never guess at an unknown schema,
silently initialize over it, or use a backup restore as the normal upgrade
algorithm.
