# Local state upgrades

Daita package replacement and local state evolution are separate concerns. Pipx
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

- Public source registrations contain connection and lifecycle data only;
  permissions are exact dedicated scope records.
- Compatibility diagnostics use `current_revision` and `found_revision` once,
  with no parallel numeric-format payload.
- The supported preledger floor is the exact pre-receipt and receipt-era wheel
  schemas listed below; speculative partial shapes are rejected.
- No `state doctor` or manual `state upgrade` command is added. Automatic
  first-open admission plus the structured CLI/TUI failure already provides the
  required operational path without creating a second public surface.
- Each published durable change has one immutable migration file. The current
  ledger therefore has separate receipt-table, historical PostgreSQL-admission,
  and scoped-source-permission entries.

## Ownership and ledger rules

`SQLiteStateStore` is the only owner of local state admission and migration.
Admission starts only after `EmbeddedAgent` acquires the existing per agent
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

Record shape evolution is separate from database migration. SQLite persistence
uses explicit codecs for identity/identifiers, sources, source-permission
scopes, receipts, catalog syncs/snapshots, artifact references, transcript/run
records, semantic annotations, and learning candidates/review stamps. Each
codec declares its required fields, additive defaults, nested records, enums,
datetimes, decimals, and unknown-field policy. Stored class-name text never
selects a Python type.

An additive payload default that leaves the physical schema and invariants
unchanged is a codec change, not a ledger entry. A physical table, index,
foreign key, ownership, or cross record invariant change is a migration. This
keeps package releases from rewriting an entire database merely because one
serialized record gained an optional field.

## Historical PostgreSQL admission and scoped-permission cutover

The immutable historical cutover first moved the legacy Boolean out of source
connection JSON. The following scoped-permissions migration converts every
legacy state to zero update scopes, creates one explicit `all` read scope for
each active source, and removes the legacy table from the current schema.
Connection reconstruction remains fail-closed. Refresh preserves exact scopes,
and detach deletes both scope families atomically.

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
| `source_read_scopes` | Create one explicit `all` scope for every active source. |
| `postgresql_update_scopes` | Create empty; historical broad admission grants no exact table scope. |
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
after discovery succeeds. It cannot change permission scopes, credentials,
identity, or attachment lifecycle.

## Release certification

Every durable revision adds focused fixture and migration contracts. The suite
must prove exact journal validation, fresh stamping, current no-write open,
known-prefix traversal, preledger admission, row/file preservation, rollback,
cancellation, downgrade refusal, legacy/damage classification, diagnostics,
lock release, credential-reference preservation, codec compatibility, and the
historical-admission/scoped-permission cutover.

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
