# Local state during development

Daita has not frozen its first production state format. Until the North Star
work is complete and the first production release is explicitly approved,
local development state has no backward-compatibility guarantee.

## Current development contract

- SQLite has one current physical schema.
- Each persisted record family has one current shape.
- A codec may carry a version discriminator, but its only supported development
  value is `1`.
- Schema and record-shape changes update the current implementation directly.
- No migration, legacy decoder, pre-ledger bridge, or compatibility fixture is
  added for state produced only by unreleased code.
- A development agent home may need to be deleted and recreated after a state
  change.

`SQLiteStateStore` remains the sole state owner. Fresh databases are created at
the complete current schema and stamped with one checksummed
`development_baseline` row. That baseline is intentionally mutable before the
production freeze, so a changed checksum rejects an older development database
without modifying it.

The generic migration engine remains in place and is tested with isolated
synthetic revisions. It is not used to preserve intermediate development
formats.

## First production freeze

The first production release will explicitly freeze:

- the complete SQLite schema;
- each codec-v1 record shape;
- the first immutable journal baseline and checksum; and
- exact release fixtures used by future upgrade tests.

Only durable changes after that freeze will add owner-local migrations or new
supported codec versions. Released migration IDs and checksums will then be
immutable.

## Future production upgrade contract

After the production baseline exists, the intended user flow remains:

```bash
pipx upgrade daita-agents
daita
```

The first normal open will validate a supported journal prefix under the
existing agent-home writer lock. If migrations are required, Daita will clone a
verified backup into a same-directory staged database, apply the known suffix
in one transaction, validate the complete target, retain one rollback point,
and atomically activate the stage. Failure before activation will leave the
active database unchanged.

There is no separate state-upgrade command or parallel migration owner.
