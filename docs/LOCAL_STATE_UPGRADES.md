# Agent-home compatibility and upgrades

Daita versions its entire agent home as one unit. The home revision is separate
from the package version, Git tag, and SQLite schema version. Current revision
`4` ships with 1.3.0; supported homes from revision `1` onward upgrade automatically.

Revision `4` adds bounded analytical evidence, artifact computation provenance,
declared catalog links and per-resource catalog storage; it converts each stored
source snapshot into per-resource catalog rows.

Revision `3`, released in 1.2, adds caller provenance, MCP service contracts, and an independent
artifact registry. Upgrades preserve existing artifact bytes. Homes from the
original revision-3 MCP development format also convert automatically, preserving
caller and MCP records. Unrecognized development formats are rejected; contributor
guidance in [`AGENTS.md`](../AGENTS.md) defines the exact supported source.

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

## Revisions and what each upgrade changes

An upgrade applies every later revision in order, so a revision-1 home opened by
1.3.0 goes through revisions 2, 3 and 4. Each entry below is taken from that
revision's migration in `src/daita/storage/home_migrations/`. "Touches" lists the
migration's `affected_paths`: the files it may rewrite. Every upgrade stages and
backs up those files first (see *Open and upgrade behavior*).

| Revision | First release | Touches |
| --- | --- | --- |
| 1 | 1.1.0 | `state.db`, `config.json` |
| 2 | 1.1.0 | `state.db` |
| 3 | 1.2.0 | `state.db` (reads `artifacts/*/*/manifest.json`, never writes them) |
| 4 | 1.3.0 | `state.db`, `config.json` |

Releases 1.0.x wrote a home with an earlier development journal
(`development_baseline`); it has no home revision. Releases 1.1.1, 1.1.2, 1.2.1 and
1.2.2 did not change the revision.

### Revision 1 (1.1.0)

Admits a home written before the revision ledger existed, and only when its
database matches one of three exact recognized shapes; any other home is refused
unchanged. This bridge is not a general legacy framework: Daita `0.19.0` and
earlier are a different product family and are not upgraded automatically.

- The development journal (`state_migrations`) is replaced by the revision ledger.
  Every other table is copied as it is, and stored records lose their redundant
  `version` field.
- The upgrade is refused, not lossy, when the old home holds rows in a table the
  new format has no place for: database write receipts, PostgreSQL update scopes,
  or the old conversation inbox.
- `config.json`: each model candidate's `profile` becomes `profile_limits`. For a
  model Daita already knows, the saved context and output limits are dropped and
  Daita's own limits apply. The retry policy's `attempts` becomes a per-candidate
  limit with a total of at most 25 attempts, and a missing model call policy gets
  the default timeouts.

### Revision 2 (1.1.0)

Replaces single jobs and autonomous follow-ups with the durable task graph.

- Succeeded, failed, cancelled and needs-attention jobs keep their state and
  evidence. A job with a cancellation requested becomes cancelled.
- An unfinished job stays runnable only if it is a data-profile job run by Daita
  that has not passed its deadline, has fewer than 3 attempts and no recorded
  external activity. Every other unfinished job, and every job with a follow-up
  that was not delivered, becomes needs attention and does not run.
- Follow-ups are removed. A delivered follow-up's delivery moves to its job.
- Daita does not fall back to the previous executor. Revision `1` decoders stay
  inside this migration and are not part of current execution.

### Revision 3 (1.2.0)

Adds caller provenance, MCP service contracts and the artifact registry. No
record is dropped.

- Each stored run records who called it: the run's scope principal, or the agent
  for a foreground run.
- Each MCP binding is marked as owned by the agent with no personal connection,
  and each MCP tool's raw input schema is recorded as unknown; the admitted
  schema and its digest are kept.
- An artifact is registered when a run's tool result, a job task result, a
  delivery or an unfinished job attempt refers to it. Its bytes stay where they
  are. An artifact file nothing refers to is not registered.
- A home from the exact pre-registry development build of revision 3 is also
  converted; its caller and MCP records are kept as they are.

### Revision 4 (1.3.0)

Replaces the task graph with checklist jobs, stores the catalog per resource and
simplifies routines. This revision drops or resets data:

- **Jobs.** Each graph job becomes one checklist job with the same job ID. A
  finished job keeps its state and accepted result; a cancellation in progress
  becomes cancelled. An unfinished job fails with "Created by an older version of
  Daita; start it again." (`job_upgraded`) and shows as Needs you. Converted jobs
  start with an empty checklist, no runs counted, a limit of 20 runs and
  1,000,000 tokens. Result deliveries and receipts keep their job ID. Graph
  attention notices are deleted, and so is a second result delivery for the same
  job and target. The graph tables, including tasks, attempts, results,
  checkpoints, comments, controls, events and budget ledgers, are dropped.
- **Catalog.** Each source's stored snapshot becomes one row per table,
  relationship and search entry, with unchanged resource IDs and revisions, and
  every table counts as fully described. The snapshot table is dropped. Catalog
  change history starts empty. Every past sync is marked complete.
- **SQLite sources.** A table name with uppercase letters gets the lowercase
  identity SQLite itself uses, so its resource ID changes. Read scopes, semantic
  notes and the catalog move to the new IDs; runs, messages, artifacts and
  evidence keep the old ones as history. A SQLite catalog with two tables whose
  names differ only by case stops the upgrade.
- **Write scopes.** `update` stays `update`, `upsert` becomes `insert` and
  `update`, and delete is never granted. Key and identity columns are dropped. A
  scope whose table left the catalog or changed since it was granted is deleted.
- **Routines.** A routine keeps its title, prompt, schedule, sources, counters and
  state. Its frozen access, grants, model routes, sensitivity, budgets, expiry,
  outcome contract, prechecks, skills and provenance are dropped, and its runs ask
  before writes. An occurrence drops its precheck, reservations and copy of the
  run's scope; one waiting on a precheck becomes claimed, and one skipped as
  unchanged becomes completed.
- **MCP.** Tool bindings drop unattended eligibility; every MCP action asks at the
  time.
- **Learning.** Learning candidates and their review stamps are deleted. Approved
  memory, skills and semantic notes are kept.
- **Other records.** Artifacts gain empty computation evidence. A semantic note
  whose table is still at the revision it was saved against gains the native
  types of the fields it names.
- **`config.json`.** Each saved model route candidate gains
  `reasoning_effort: null`, which keeps the provider's default. A candidate with
  any other set of fields stops the upgrade.

## One compatibility authority

`src/daita/storage/home_migrations/registry.py` is the sole compatibility
authority. It contains one ordered, append-only sequence of immutable
migrations and derives `CURRENT_HOME_REVISION` from the last entry. Each
migration declares:

- one contiguous integer revision;
- one stable migration ID;
- the complete target SQLite schema;
- every home-relative path it may change and any bounded read-only inputs; and
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

## Release contract gate

Compatibility is a release-to-release promise. Development commits may evolve
one candidate next migration; Daita does not promise that a home written by one
unreleased commit can be opened by another unreleased commit.

`release/agent-home-contract.json` is deterministic release evidence generated
from the current persistence owners. It records:

- the normalized complete SQLite target schema;
- every literal strict `record_fields` contract in `sqlite_codecs`;
- the model configuration, agent manifest, retained-skill, artifact, and
  delivery-configuration layouts; and
- the current migration definitions, affected paths, and checksums.

Refresh and verify it from the repository root with the configured development
environment:

```bash
python scripts/check_home_release_contract.py write
python scripts/check_home_release_contract.py check
```

Pull-request and tag CI compare the candidate with the latest earlier tagged
snapshot. A changed durable contract with the same or an older home revision is
rejected. A persistence-changing release therefore appends one migration and
advances the revision; a release without persistence changes reuses the current
revision. The first release containing this mechanism establishes the baseline
when no earlier tag has a snapshot.

The managed release attaches `agent-home-contract.json` to every GitHub release
and includes it in release checksums and provenance. This records the exact
contract shipped by each tag without introducing a tag-to-schema runtime map.

The generator detects structural changes, but it cannot prove that unchanged
field names retain unchanged meaning. A developer changing durable semantics,
authorization, classification, receipts, or cross-file invariants must append a
home migration even when the generated structural snapshot would be unchanged.
The migration registry remains the sole runtime compatibility authority.

## Open and upgrade behavior

For a managed installation, close running Daita hosts and install the current
release. Opening the agent then checks its home and applies any supported
upgrade:

```bash
curl -fsSL https://daita-tech.io/install.sh | bash
daita
```

If Daita was installed with pipx, continue to upgrade it with
`pipx upgrade daita-agents`.

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
5. Apply every required migration in order to the staged home and record each
   completed transition in its ledger.
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

Revisions `1` and `2` are frozen. Released migration files, IDs, checksums,
historical decoders, and golden fixtures are immutable. A durable format change
requires a new revision even when it changes only a non-SQLite file. Fresh
homes are built directly at the latest complete revision; they do not replay
historical migrations.

A new revision is a mutable release candidate only until its first Git tag. It
may be refined across development commits by updating its implementation,
candidate checksum, fixture, and release-contract snapshot together. After that
tag, it is released and falls under the immutability rule above.

To add revision `N`:

1. Add one owner-local `revision_NNNN.py` transition and declare every affected
   path.
2. Append it to `HOME_MIGRATIONS`; never edit an earlier transition.
3. Commit its literal released checksum.
4. Add a complete golden agent-home fixture for revision `N`, add its database
   SQL to `revision_database`, and publish the `N-1` to `N` state-directory
   upgrade fixture for custom storage adapters.
5. Test every supported prior production revision through the current revision,
   including crash boundaries, rollback, durable-data preservation, and
   downgrade refusal.
6. Change `MINIMUM_SUPPORTED_HOME_REVISION` only as an explicit release-policy
   decision, with the support window documented before release.

There is deliberately no independently growing codec registry, SQLite-only
migration sequence, spreadsheet, or Git-tag map that must line up with this
revision. The append-only home registry is the one persistence version source.
