# Local state upgrades

Daita package replacement and local-state evolution are separate concerns. Pipx
and the managed installer replace application code without touching agent data.
The first command that subsequently opens an agent validates its SQLite state
and applies any supported durable revisions automatically.

## User contract

For a supported upgrade, keep using the same agent home:

```bash
pipx upgrade daita-agents
daita
```

No separate state command, import, restore, or backup is required. A backup is
optional disaster recovery and is not a compatibility mechanism. Secret values
remain behind their environment or OS-keychain references; migrations never
copy them into SQLite.

## Resolved design decisions

- `SourceRegistration.configuration["write_access"]` remains available as a
  computed public projection; it is not durable connection or admission
  authority.
- Compatibility diagnostics use `current_revision` and `found_revision` once,
  with no parallel numeric-format payload.
- The supported preledger floor is the exact pre-receipt and receipt-era wheel
  schemas listed below; speculative partial shapes are rejected.
- No `state doctor` or manual `state upgrade` command is added. Automatic
  first-open admission plus the structured CLI/TUI failure already provides the
  required operational path without creating a second public surface.
- Each published durable change has one immutable migration file. The current
  ledger therefore has separate receipt-table and PostgreSQL-admission entries.

## Admission outcomes and diagnostics

| Existing state | Behavior |
| --- | --- |
| Exact current journal | Validate the journal, schema, integrity, foreign keys, and open without a database write. |
| Known journal prefix | Apply each missing migration in order in one transaction, validate each target, stamp its checksum, and open. |
| Supported preledger shape | Identify the exact historical schema in the isolated bridge, create the journal, apply the remaining migrations, and open. |
| Unknown, reordered, gapped, or checksum-mismatched journal | Refuse without changing state. |
| State created by a newer release | Refuse the downgrade without changing state; install the same or a newer Daita package. |
| Recognizable pre-1.0 framework | Refuse without creating mixed current state in that root. |
| Damaged or unexpected schema | Refuse without changing database contents. |
| Upgrade failure | Roll back the SQLite transaction; no partial revision is admitted. |
| Caller cancellation | Wait for the transaction to settle, then surface cancellation; state is wholly old or wholly current. |

Headless CLI diagnostics expose `code`, `state_path`, `found_revision`,
`current_revision`, and `state_changed`. Interactive startup renders the same
classification in descriptive language and does not expose historical numeric
markers. Stable error codes are `state_revision_newer`, `state_legacy`,
`state_database_damaged`, `state_revision_unsupported`, and
`state_upgrade_failed`.

## Ownership and ledger rules

`SQLiteStateStore` is the only owner of local-state admission and migration.
Admission starts only after `EmbeddedAgent` acquires the existing per-agent
process writer lock. The store uses `BEGIN IMMEDIATE`; validates the exact
source schema, database integrity, and foreign keys; applies the known suffix;
validates every target; and commits once. DDL, data changes, and journal rows
therefore roll back together.

`state_migrations` is an immutable, ordered, checksummed ledger. The rows in an
existing database must be an exact prefix of the migration definitions shipped
by the running package. A durable change gets one owner-local migration file
with a stable ID, immutable definition/checksum, exact source and target
schemas, one transform, and target validation. Existing IDs are never edited,
reordered, or reused. Releases with no durable change add no ledger row.
Checksum material canonically binds the ordinal, migration ID, definition,
exact source and target schemas, transform source, and target-validator source;
editing any part of that historical contract changes its ledger identity.

Fresh databases are created directly at the current schema and stamped with the
complete journal. Current databases are only read and validated on open. The
agent loop, runtime, catalog, adapters, artifacts, memory, skills, and hosting
layers do not implement alternate compatibility paths.

## Persisted-record codecs

Record-shape evolution is separate from database migration. SQLite persistence
uses explicit codecs for identity/identifiers, sources, receipts, catalog
syncs/snapshots, artifact references, transcript/run records, semantic
annotations, and learning candidates/review stamps. Each codec declares its
required fields, additive defaults, nested records, enums, datetimes, decimals,
and unknown-field policy. Stored class-name text never selects a Python type.

An additive payload default that leaves the physical schema and invariants
unchanged is a codec change, not a ledger entry. A physical table, index,
foreign-key, ownership, or cross-record invariant change is a migration. This
keeps package releases from rewriting an entire database merely because one
serialized record gained an optional field.

## Bounded preledger bridge

The historical marker reader exists only in
`daita.storage.sqlite_migrations.preledger`. It admits these exact combinations:

- the pre-receipt table shape with marker 0 or 1; and
- the receipt-era table shape with marker 0, 2, or 3.

The bridge never guesses from a partial schema. It rejects newer markers,
recognizable pre-1.0 framework tables, and all other shapes. Remove this unit
only after the minimum supported Daita release is guaranteed to have created
`state_migrations` and product support for every preledger home has been
deliberately ended. Normal journaled opens never read the historical marker.

## PostgreSQL write-admission cutover

PostgreSQL write admission is control-plane authority, not connection
configuration. Its sole durable owner is `postgresql_write_admissions`, keyed
by `(agent_id, source_id)` with a foreign key to the source registration.

During the preledger cutover, an attached PostgreSQL source with historical
`write_access: true` receives one admission row. False or missing values receive
no row, invalid values fail and roll back, and detached sources are not
admitted. The migration removes `write_access` from persisted source JSON.
Public source registrations retain a computed Boolean projection for API/CLI/TUI
compatibility; connection reconstruction always remains fail-closed. Refresh
cannot change admission, and detach deletes it atomically.

## Preservation matrix

| Durable item | Upgrade treatment |
| --- | --- |
| `metadata` | Preserve agent identity rows byte-for-byte. |
| `sources` | Preserve identity, lifecycle, connection configuration, and secret references; only remove the obsolete admission key during cutover. |
| `syncs`, `snapshots` | Preserve current catalog syncs and snapshots byte-for-byte. |
| `runs`, `messages` | Preserve inputs, complete ordered transcripts, conversation order, and terminal results byte-for-byte. |
| `semantic_annotations` | Preserve every encoded annotation byte-for-byte. |
| `learning_candidates` | Preserve candidates and review state byte-for-byte. |
| `database_write_receipts` | Preserve every existing immutable receipt; create the empty table only when absent. |
| `postgresql_write_admissions` | Create from valid active historical admissions; preserve existing current rows. |
| `state_migrations` | Validate the exact existing prefix and append only the known missing suffix. |
| `agent.toml` | Preserve bytes; it remains the manifest cross-check for SQLite identity. |
| `config.json` | Preserve model route, settings, and secret references byte-for-byte. |
| `MEMORY.md`, `USER.md` | Preserve approved advisory documents byte-for-byte. |
| `skills/*/SKILL.md` | Preserve every authorized skill file byte-for-byte. |
| `artifacts/**`, including `delivery-config.json` | Preserve committed artifact bytes, metadata files, and delivery configuration byte-for-byte. |
| `run/host.lock` | Use as the existing writer boundary; it is ephemeral rather than durable migration data, and each owner writes its current PID while holding the lock. |
| User-selected export destinations outside the agent home | Leave untouched; only the persisted destination reference is preserved. |

Source refresh is not reattachment or configuration mutation. It reopens the
persisted registration for discovery and atomically replaces catalog truth only
after discovery succeeds. It cannot change admission, credentials, identity, or
attachment lifecycle.

## Release certification

Every durable revision adds focused fixture and migration contracts. The suite
must prove exact journal validation, fresh stamping, current no-write open,
known-prefix traversal, preledger admission, row/file preservation, rollback,
cancellation, downgrade refusal, legacy/damage classification, diagnostics,
lock release, credential-reference preservation, codec compatibility, and
write-admission cutover.

Before publishing, build the baseline and candidate wheels once, then run the
isolated two-wheel lifecycle:

```bash
.venv/bin/python tests/pipx_lifecycle_smoke.py \
  --baseline-wheel /path/to/immediately-preceding.whl \
  --candidate-wheel /path/to/candidate.whl

.venv/bin/python tests/managed_installer_lifecycle_smoke.py \
  --baseline-wheel /path/to/immediately-preceding.whl \
  --candidate-wheel /path/to/candidate.whl
```

The smokes install the real prior wheel, create realistic state including an
enabled PostgreSQL admission, replace the package, open the same home, compare
the complete logical projection and all non-database files, verify every
unchanged prior table, verify the intentional source/admission cutover, and
check the exact journal. Never guess at unknown state, silently initialize over
it, or use backup restore as the normal upgrade algorithm.
