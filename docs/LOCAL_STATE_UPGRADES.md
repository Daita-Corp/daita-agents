# Agent-home compatibility and upgrades

Daita's production persistence contract is the revision of the complete agent
home, not the package version, a Git tag, the SQLite schema version, or a set of
independent record-codec versions. The current production home revision is `2`
and the minimum automatically supported production revision is `1`.

One home revision covers every durable component whose shapes must remain
compatible together, including:

- `state.db` and its persisted records;
- `config.json`;
- `agent.toml`, `MEMORY.md`, and `USER.md` when a transition affects them;
- retained skills; and
- artifact and delivery metadata and files.

`daita state status NAME` reports the installed release's current revision, the
home's detected revision, the minimum supported revision, whether an upgrade is
required, and whether crash recovery is pending. It is read-only.

## One compatibility authority

`src/daita/storage/home_migrations/registry.py` is the sole compatibility
authority. It contains one ordered, append-only sequence of immutable
migrations and derives `CURRENT_HOME_REVISION` from the last entry. Each
migration declares:

- one contiguous integer revision;
- one stable migration ID;
- the complete target SQLite schema;
- every home-relative path it may change; and
- one checksum bound to its definition, implementation source, target schema,
  and other declared implementation material.

The committed `agent_home_migrations` ledger in `state.db` must be an exact
prefix of that registry. Missing, reordered, unknown, or checksum-edited
entries are rejected. A valid longer prefix is recognized as a home written by
a newer release and is also rejected without modification.

Current runtime serializers accept exactly the current record shape. They do
not contain per-record compatibility branches or version discriminators.
Historical parsing and translation live only in the immutable migration that
needs them. This keeps ordinary reads and writes independent of how old a home
once was.

Git tags and the Python package version identify and distribute a release. They
are useful provenance and rollback coordinates, but they do not participate in
runtime persistence admission. No tag-to-schema table or package-version
comparison is required for a home to open. Multiple application releases may
use the same home revision when persistence has not changed.

The application's sole authored release identity is `project.version` in
`pyproject.toml`; runtime displays read installed distribution metadata. Neither
fact changes this registry or authorizes a home-format transition.

## Open and upgrade behavior

The normal operator flow remains:

```bash
pipx upgrade daita-agents
daita
```

Opening an agent acquires that home's existing writer lock before inspecting,
recovering, or upgrading it. A current home is fully validated before its
runtime composition opens. A supported older home is upgraded automatically by
this sequence:

1. Inspect the home and exact migration prefix without writing.
2. Recover any authenticated unfinished upgrade journal.
3. Preflight regular-file containment and free space for a staged copy and
   backup, plus safety headroom.
4. Copy every affected file beneath `.home-upgrade/`; SQLite databases are
   copied with SQLite's online backup API.
5. Apply every required migration in order to the staged home and append its
   ledger row only after that transition succeeds.
6. Validate the complete staged target: database health and exact schema,
   identity, all current records and transcripts, model configuration, memory,
   user profile, skills, artifacts, and delivery configuration.
7. Publish affected non-database files first and `state.db` last, checking
   journaled hashes before and after every replacement.
8. Validate the complete active home, retain one previous-home rollback bundle
   beneath `.home-rollbacks/`, and remove the work journal.

If staging or validation fails, the active home remains byte-for-byte
unchanged. If publication fails, the coordinator restores the verified backup.
After process interruption, the durable journal deterministically discards an
unfinished staging area, finishes a prepared/partially committed upgrade, or
finishes a recorded rollback before normal admission continues. Daita never
opens a mixed-revision home.

Corrupt current records, an invalid ledger, an unsupported historical shape,
or a newer revision fail closed with machine-readable diagnostics. These
refusals do not rewrite or reset the home.

## Production revision policy

Revision `1` is frozen. Released migration files, IDs, checksums, historical
decoders, and golden fixtures are immutable. A durable format change requires a
new revision even when it changes only a non-SQLite file. Fresh homes are built
directly at the latest complete revision; they do not replay historical
migrations.

Revision `2` adds the required caller-owned target posture to retained run
inputs. Revision `1` runs are translated to the conservative `single_target`
posture during the staged whole-home upgrade.

To add revision `N`:

1. Add one owner-local `revision_NNNN.py` transition and declare every affected
   path.
2. Append it to `HOME_MIGRATIONS`; never edit an earlier transition.
3. Commit its literal released checksum.
4. Add a complete golden agent-home fixture for revision `N`.
5. Test every supported prior production revision through the current revision,
   including crash boundaries, rollback, durable-data preservation, and
   downgrade refusal.
6. Change `MINIMUM_SUPPORTED_HOME_REVISION` only as an explicit release-policy
   decision, with the support window documented before release.

There is deliberately no independently growing codec registry, SQLite-only
migration sequence, spreadsheet, or Git-tag map that must line up with this
revision. The append-only home registry is the one persistence version source.

## First production bridge

Revision `1` contains a one-time, immutable bridge for the three complete
preproduction home shapes observed immediately before the freeze. It translates
their SQLite layout, removes redundant record `version` fields, and converts
saved model configuration to the stable identifier-and-limits form. The bridge
accepts only exact known shapes and preserves identities, sources, scopes,
memory, user profile, skills, conversations, and other supported durable state.

This bridge is not a general legacy framework. Daita `0.19.0` and earlier are a
different product family and remain outside the automatic support window.
