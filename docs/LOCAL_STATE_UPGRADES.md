# Local state compatibility

Daita has not frozen its first production state format. Local development
state therefore has no backward-compatibility guarantee.

## Current development contract

- SQLite has one current physical schema.
- Each persisted record family has one current shape.
- A codec may carry a version discriminator, but the only supported value is
  `1`.
- Schema and record-shape changes update the current implementation directly.
- State produced only by unreleased code receives no migration, historical
  decoder, bridge, or compatibility fixture.
- A development agent home may need to be deleted and recreated after a state
  change.

`SQLiteStateStore` is the sole state boundary. A new database is created at the
complete current schema and stamped with one checksummed
`development_baseline` row. The baseline remains mutable until the first
production state freeze. A checksum change rejects an older development
database without modifying it.

The migration engine is retained and tested with isolated synthetic revisions.
It does not preserve intermediate development formats.

## Production baseline

The first production state freeze makes the complete SQLite schema, every
codec-v1 record shape, journal baseline, checksum, and release fixture
immutable. Durable changes after that point use new owner-local migrations or
additional supported codec versions. A released migration ID or checksum does
not change.

## Upgrade behavior

The normal upgrade flow is:

```bash
pipx upgrade daita-agents
daita
```

On open, Daita validates the migration journal under the existing agent-home
writer lock. When a migration is required, it clones a verified backup into a
same-directory temporary database, applies the known journal suffix in one
transaction, validates the complete target, retains one rollback point, and
atomically replaces the active database. A failure before replacement leaves
the active database unchanged.

There is no separate state-upgrade command or parallel migration component.
