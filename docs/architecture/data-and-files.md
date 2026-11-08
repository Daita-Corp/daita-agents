# Data, catalog, sources, local files and artifacts

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

Source types, the catalog, data queries and writes, folders and artifacts.

## Data, catalog and adapters

The source answers; the catalog advises. `daita.catalog` stores a normalized
entry for each resource of each source and implements catalog search and
inspection. Data code reads the source directly. No read, export or scope check
waits on a catalog entry or revision, and a missing, stale or unreadable catalog
never grants or denies access. The model sees source display names and table
names, never source, resource or relationship IDs: `CapabilityRegistry.resolve_call`
turns names into IDs where a call enters the runtime, and `model_tool_result`
renames the IDs in a result or drops those the run's scope does not name.

A `CatalogRelationship` belongs to one source. A `CatalogLink` is an
agent-owned declared reference between fields of two resources, possibly in
different sources, managed only through `Agent.declare_catalog_link`,
`remove_catalog_link` and `list_catalog_links`; no model tool declares links.
Declaration requires both resources and every paired field in the current
catalog. Links survive detach and reattach because source and resource IDs are
stable. Currency is computed, never stored: a link is current while both
sources are active and every paired field exists in a current tabular facet;
otherwise it is stale as `source_inactive`, `resource_missing` or
`field_missing`. Unrelated column changes keep a link current. The catalog
service merges current links with source relationships in one edge set, labelled
`provenance: "declared"` with `cross_source`, for search neighbours, schema
slices and traversal; sources joined by current links form one join group, so
`cross_source_unsupported` remains only for seeds with no linking group. Links
are read fresh for each projection, outside the per-source index cache. Readable
and run-scope filters apply to both link endpoints, so a projection never
crosses a resource outside scope. No model tool traverses the catalog.
`catalog_describe` lists each current link leaving the
table it describes, by name, when the run may read the far endpoint. Each
relationship carries `from_row_estimate` and `to_row_estimate` when the catalog
has them. For a table with key-like columns (a name ending in `_id`, `_key`, `_code` or
`_uuid`, or a unique key other than `id`), `catalog_describe` also lists up to
eight `link_candidates`: columns of readable tables in other sources with the
same name or a matching `<name>_id` to `id` pairing and a compatible type, not
already linked, with `status: "unconfirmed"` and the far table's
`to_row_estimate`. A candidate is a hint. It grants no read authority and is
never declared or stored.

`catalog_search` always asks both the catalog and the live source, through
one bounded lookup of table and column names (`pg_class` and `pg_attribute` in the
configured schemas, `sqlite_schema`, a file listing), and merges compact hits marked
`seen_live`. `catalog_describe` takes one name and reads the live source: native
column types, enum values, keys, comments, a partitioned table as one table with a
child count, and row estimates. A schema or folder name lists what it holds, capped
at 100. A cached description answers only when the source cannot be reached, and says
so; a table the live source no longer has is unknown even if the catalog remembers
it. A failing catalog leaves both tools answering from the source.

Both tools label look-alike tables (`catalog/labels.py`): within one source, tables whose
core name tokens are similar and whose columns overlap get an advisory `status`
(`canonical`, `likely_stale`, `deprecated` or `unknown`), short `evidence` and
`look_alikes`. Evidence is row estimates, inbound foreign keys, and a table comment or a
current single-table note that matches a deprecation phrase, which marks the table
`deprecated`; the last analyze is shown but never decisive. Facts come live from
`adapters/postgresql_query.py`, so tables past a truncated catalog are labelled too.
`catalog_describe` runs one extra live name search to find look-alikes and answers
unchanged if it fails. Nothing is hidden or reordered, no record shape stores a label,
query history is not used, and labels appear only in tool results, never in the prompt.
For live PostgreSQL foreign keys, `catalog_describe` relationships also carry
`cardinality` (`one_to_one` or `many_to_one`) from the unique indexes.

PostgreSQL discovery lists every relation kind from `pg_class`: tables, partitioned
tables and foreign tables as `TABLE`, views and materialized views as `VIEW`. A
partition folds under its parent when the role can read an ancestor in the source's
configured schemas; otherwise it is listed as its own table, so a role granted only a
child partition can still find it. It keeps every column whatever its type, with its
`format_type` and comment. A composite primary key shows only when the role can read
every column of it; a readable unique key still appears.

Attach never fails on size. `ResourceAdapter.reconcile` is names first: it yields
one INVENTORY batch listing every object, up to `DiscoveryRequest.max_inventory`
(1,000,000), then DETAIL batches that describe at most `max_resources` (1,000)
tables, 512 columns and 256 indexes per table, and 2,000 relationships; PostgreSQL
stops reading table detail after 60 seconds. A listed table without detail is
cataloged by name only, still readable and describable through the live tools, and
`CatalogSync.truncated` means the source's catalog lacks some detail (or the
inventory hit its cap): after a commit it is also true while any row of the source
still has only a name, so a fill pass that has not finished leaves it set.
A primary key shows whole or not at all, and an index or foreign key that names a
cut column is dropped. `source_directory` adds `catalog_partial` for a truncated
source: the catalog lists every object but describes only some, so the model reads
the rest live. A table with only a name cannot take a note.

The catalog stores one row per resource (`catalog_resources`, with `detail` marking
inventory or complete), one per relationship, a `catalog_search` row per table
(its normalized words in one column matched by `LIKE`; ranking stays in Python, so
SQLite and PostgreSQL rank alike), a change log, and one state row per source. A
pass commits in one transaction, so a reader sees the whole old catalog or the
whole new one. An unchanged table is never rewritten: a re-read writes only its sync
and state rows. A table missing from a complete inventory is removed; a truncated
inventory removes nothing, and a failed pass commits nothing. Syncs are pruned to
the newest 20 per source, the change log to 90 days or 10,000 rows per source.

PostgreSQL discovery sends five tagged statements however many tables there are:
resources, bulk columns, bulk indexes, relationships and estimates. The two bulk
statements read through a server-side cursor in batches of `_BULK_FETCH_ROWS`, and
rows are grouped by table in Python so only one table's raw rows are held. Each
batch is a round trip, so the batch size sets how latency scales with catalog size.
They share their SQL fragments with the single-table `COLUMNS_SQL` and `INDEXES_SQL`
that live describe uses, so the two cannot drift. Each bulk read runs in a savepoint
with the time left in the discovery budget as its timeout, rather than the 8-second
command timeout, because one statement now reads what a thousand did. The column read
stops with a quarter of the budget left, so the tables whose columns arrived can still
get their indexes. When the budget ends a read, the table in progress is dropped and
the tables before it are kept with both their columns and indexes, so a kept table
never shows a falsely empty index list.

The catalog is kept current by the keeper (`hosting/keeper.py`), one task per open
agent. The catalog is read on attach and on an explicit `Agent.refresh_source`; after
that the keeper probes each source cheaply (`SourceOperation.CHANGE_PROBE`: a token per
scope) and reads only the scopes whose token moved, committing the new tokens in the
same transaction as the read, so an interrupted read is simply read again. The probe
interval starts at the type's `min_probe_seconds`, doubles after each unchanged probe
up to `max_probe_seconds` and resets on a change or a use; two failed probes mark the
source unreachable and one success recovers it. Due times are wall-clock times, so
sleep and restart catch up. A run start (every origin), `Agent.catalog_status(check=True)`,
a lookup miss or a missing-object error, damaged catalog rows and the permissions
screen (which waits at most 2 s) are triggers; a type with no probe is only fully
rechecked on its `full_check_seconds`. While a source has resources without detail the
keeper describes more of them in passes of 1,000, and the daily full check records
`missed`, the changes no probe flagged. `Agent.set_keep_current(source_id, False)`
stops all of it for a source; an explicit `Agent.refresh_source` still works.

Keeper reads take no execution lease and run under the source management lock, with
`_source_commit_lock` for the commit only. A person's attach, edit, detach, refresh
or permission change first cancels a keeper read in flight. Upkeep is never background
work for the host's idle policy and never sets a wake time. An unchanged probe
rewrites the stored source state at most every 10 minutes. The probe is one bounded
read-only statement or stat walk that reports names and hashes only, ignores the read
scope and fails with a stable error code. PostgreSQL has one token per configured
schema the role can use, covering what a revision stores and what the role may read,
so a grant or revoke moves it but row counts and statistics do not. SQLite has one
token from `PRAGMA schema_version` and the file's identity. A folder has one token from
a stat walk over the files discovery would catalog. `Agent.catalog_status` reports
`current`, `updating`, `unreachable`, `paused` or `not_checked`, with the described and
total counts; `Agent.catalog_changes` lists recent changes; `daita sources` and the
TUI sources screen show them, with a Keep current switch and Check now.

A refresh runs under the source management lock. `_source_commit_lock` covers
only the commit, so a revoke does not wait for discovery. Permission apply
revalidates current facts rather than a catalog generation, so a sync between
preview and apply does not force a new preview.

`Agent.check_source(source_id)` asks the live source and returns a `SourceHealth`:
whether it is reachable, the role Daita connects as, `read_only`, and a warning
for each broad grant (superuser, row-level-security bypass, role or database
creation, membership in broad built-in roles, tables the role can change, create
rights in a configured schema). SQLite and folder sources report only
`read_only`. It
takes no lock and no lease, and an unreachable source is an unhealthy result with
an error code, not a raise. The result is not model-visible and not stored.
`/sources` shows it for the selected source, attach shows it, and the host
serves it at `POST /sources/{source_id}/check`.

`Agent.check_link(from_source_id, from_table, from_columns, to_source_id,
to_table, to_columns)` and `Agent.check_catalog_link(link_id)` return a
`LinkCheck`: how well two key columns join, from a bounded read of each side's
distinct keys (`domains/data/key_read.py`). Each side goes through the normal
read path, with the same scope and read policy as `data_query`, and takes the
source's admission permit; no catalog entry is needed. `match_rate` is the share
of distinct non-null `from` keys found on the `to` side, with both key counts,
the matched counts and the key types read (so `"1"` against `1` shows). A side
that hits its bound (50,000 keys, 16 MiB, 10 seconds) sets `from_truncated` or
`to_truncated` and makes the result `partial`. An unreachable source, an
unknown table or a refused read is a result with an `error_code`; only an
unknown or detached source, or an unknown link id, raises `ValueError`. It is
on demand, not a model tool, and the host serves it at `POST /links/check` and
`POST /links/{link_id}/check`.

Opening a home decodes no catalog rows. A read skips a catalog row that cannot be
decoded as if it were absent, so reads and approved writes continue; when a
the keeper reads each source whose row was skipped in full on its next pass, and
that read rewrites the row. A lookup that finds the catalog unreadable asks the keeper
the same way. A `semantic_save` on a table with no catalog
entry, or with only a name, is refused with `semantic_unknown_resource`, which says
reads are unaffected.

`daita.adapters` implements source admission, containment, discovery,
freshness checks, and I/O. One composed `SourceTypeRegistry` per agent maps each
`adapter_id` to its factory, credential fields, read capacity, SQL dialect and
operation backends, and its `SourceFreshness` (probe interval bounds and the full-check period). `SourceOperation` names those backends: `SQL_READ`,
`TABULAR_READ`, `TABULAR_EXPORT`, `PROFILE`, `RELATIONAL_WRITE`, `FILE_SEARCH`, `TEXT_READ`, `PATH_QUERY`, `LIVE_LOOKUP` and `CHANGE_PROBE`;
every operation except `TABULAR_READ`, `FILE_SEARCH`, `TEXT_READ`, `LIVE_LOOKUP` and
`CHANGE_PROBE`
requires an SQL dialect the read policy in `domains/data/sql` supports, checked at registration.
A stored source whose type is not registered stays listed
but unavailable: it gets no tools or source capacity, and refresh or edit fails.
Pass the composing `AgentConfig` to `Agent.delete` so credentials of its source
types are removed too. PostgreSQL read diagnostics, update readiness and
keychain password storage remain PostgreSQL-only public conveniences. SQLite paths must be absolute and bounded. Local
workspace access uses only admitted relative logical paths and rejects
symlink, traversal, secret-file, and special-file access. PostgreSQL and MCP
credentials remain secret references and are resolved only at the integration
boundary.

The read policy belongs in `daita.domains.data.sql`; connector guardrails still
apply during execution. The data domain calls `validate_read`
(`sql/read_validation.py`, parsing in `sql/analysis.py`) once per statement, before
any source I/O, and the backend receives the result as a `CheckedRead`, runs its
canonical SQL and never validates again. It does recheck the source's stored read
scope at dispatch, because the runtime prepares a read before it takes the source
permit, so revoked permission still stops a queued read. SQL that cannot be parsed is refused with
its position and never run. Policy allows one `SELECT` or `WITH` statement, with
no data-modifying CTE, `SELECT INTO` or locking clause. A source read in full
(`unrestricted_source_ids`) admits any relation the engine resolves in its
configured schemas, and system catalogs. A SELECTED or narrowed scope folds each
name, hashes it forward with `catalog_resource_id` and tests the listed IDs, with
no catalog lookup, and refuses system catalogs. A relation in another schema is
refused naming the schema. A refused name returns `resource_read_not_allowed`, with
close names from the catalog when it has them. A result's classification comes from its source's
registration, never from catalog resources. The source reports everything else:
a missing table or column comes back as `query_invalid` with the server's message,
hint and position. Other failures cross by SQLSTATE category only, because their
text can echo a value (`postgresql_query.py`, `sqlite_query.py`).

PostgreSQL reads run in a read-only transaction with `search_path` pinned to the
source's configured schemas; `pg_catalog` stays implicitly first. With one schema
an unqualified name means that schema, and with several it is refused with
`unqualified_relation`, listing candidates in the run's scope from a live lookup.
SQLite tables have one identity per name: `native_identity` is the folded
`main.<name>`, and `name` keeps the declared case. Functions in `*_to_xml*`,
`cursor_to_*`, `ts_stat` and `ts_rewrite` are refused in every scope, because they
run SQL the parser cannot see. For the rest, one tagged `pg_proc` lookup in the
read transaction allows a called name only when it is in `pg_catalog`, no overload
of it is volatile (except `random`, `clock_timestamp`, `timeofday`,
`gen_random_uuid` and `timezone`) and, for an unqualified call, no schema on the
path has a namesake. The same lookup covers `x.name` and `(x).name`, which
PostgreSQL resolves to `name(x)` when `x` has no such column: such a read is refused
if `name` takes a row and is outside `pg_catalog` or volatile, even when a real
column has that name. SQLite keeps its `mode=ro` connection and authorizer, which
also refuses `load_extension`. Do not duplicate any of this in a generic policy
layer.

A single-statement `data_query` read may return advisory `warnings`
(`sql/query_checks.py`): a fan-out (`SUM`, `AVG` or `COUNT(col)` over a one-to-many
join, with keys from a live describe of the joined tables, at most 4), duplicate
primary-key values in a joined, non-aggregated result, and an explanation of an empty
result from one bounded probe (min and max of a range-filtered column, or up to 10
distinct values of an equality-filtered column) that runs through the normal read
policy. Checks run only when triggered, share a 1-second budget inside the read's
deadline, and any failure is dropped; they never change or block a result.
Multi-statement, combined and folder reads are unchanged. The fan-out check shares
`select_equalities` in `sql/duckdb_read.py` with the combined-query join checks below.

`data_query` with `inputs` routes to `data.query_combined`, a foreground-only
read that joins two to sixteen sources. The model names each source by display
name; the runtime resolves it before the data domain sees the call. Each SQL input
`{alias, source, sql}` takes the single-source path unchanged: run scope, the read
policy for the source type's `sql_dialect` and that type's registry backend. A
folder input, which must be an attached folder, adds `path_pattern` and takes the
folder route below. A tabular input `{alias, source, table, columns, filters}`
names one table of a type that declares `TABULAR_READ`; it takes the same
run-scope and stored-permission checks; the catalog only advises, so an unlisted
table or column is the source's to serve or refuse (`tabular_columns_unavailable` when it returns other columns), and
filters are only AND-combined `eq`, `in`, `lt`,
`lte`, `gt` and `gte` on scalar values. Its backend receives one
`ReadTabularRequest` and returns a `BoundedResultProjection`.
It serves types without SQL; the built-in types read through SQL inputs instead,
which can aggregate before the join. A type with `TABULAR_READ` but no `SQL_READ` is usable
only as a combined-query input; single-source `data_query` is refused for it,
and `data_query` is offered only while a source with `SQL_READ` is in scope.
A run withholds the `data_query` routes no in-scope source can serve: export
needs a `TABULAR_EXPORT` source and `path_pattern` needs a folder source.
A tool definition describes only the routes its run admits.
Inputs share one budget (`COMBINED_QUERY_TOTAL_MAX_ROWS`/`_BYTES`) enforced
while they read: sources read concurrently, one input at a time within a source
(the call already holds one admission permit per source), each read granted an
equal share of what is left. An input that overflows its share is read again
alone with what the finished reads left; one that still does not fit refuses the
whole call with `combined_input_truncated`, so no join ever sees incomplete
input; an input with no rows still joins and aggregates. The first failing input
cancels the rest and its own error is returned, with the alias in its details
as `input_alias`. The budget is 50,000 rows and 32 MiB for all inputs together;
SQLite stops fetching at its byte share and PostgreSQL prefetches at most 256
rows, so a read does not hold much past its share. Inputs are read separately
(`rows`, `read_at` and `read_ms` per input, `read_consistency` in the result),
not as one snapshot. The result also carries the worker's `join_checks` and
advisory `warnings` (unmatched keys, key types of different families, an
unfinished check). After the result is safe, the worker measures each
`a.x = b.y` equality in the combining query: both key types, both row counts and
the rows with no match on the other side and how many of those would match
ignoring case and surrounding spaces, within one five-second budget
(`equality_joins` in `sql/duckdb_read.py`, `_send_join_checks` in
`adapters/local_file_query.py`). A check that fails or runs out of time leaves a
gap and the status `partial`; it never changes or blocks the result. The
combining `sql` is one DuckDB SELECT or UNION validated before execution to
read every alias and nothing else: no table functions, file, HTTP or extension
access. It runs in the same private DuckDB worker as folder-source SQL, one
query at a time; a concurrent call waits for the worker. The result carries the union of input tables, the highest input
sensitivity and per-input lineage (`source`, `read_at`, and `canonical_sql`/`sql_fingerprint` for SQL inputs or
`columns`/`filters` for tabular inputs). Declared catalog links
inform discovery and join paths only; they grant no read authority.

PostgreSQL metadata shows only what the source's role can read. `pg_class` and
`pg_attribute` list everything, so discovery, the schema probe and live lookups all build on
the shared `READABLE_*` SQL in `adapters/postgresql.py`. A name the role cannot use only
misleads the model; the role remains the boundary either way.

Data capabilities are reads except for `data_write`, backed by PostgreSQL: one
`INSERT`, `UPDATE` or `DELETE` on one table with a write scope. Four rules hold for every
source type: run what was checked, ask the source, verify the effect, and refuse by
principle. The refusals are the fixed list in `docs/RELATIONAL_WRITES.md`; when a
statement misbehaves, ask which rule failed instead of adding a refusal.

The flow and the seam are shared, and a source's language stays in its adapter.
`domains/data` owns approval, receipts, scopes and comparing evidence, and never reads
SQL. A source implements `RelationalWriteBackend`:

| Method | Returns |
| --- | --- |
| `describe` | `WriteFacts`: target, operation, columns, read names, and an opaque `statement` only that adapter reads. Parses for early errors, before any I/O |
| `preview` | `WritePreview`: count, identity digest, sample, warnings and `verification` |
| `apply` | `VerifiedEffect`: what happened, with commit certainty |

PostgreSQL is the one implementation. `adapters/postgresql_write_sql.py` owns its
language: early refusals, the `RETURNING` Daita adds, and the `SELECT` derived from the
statement. A write is one call: `mode` defaults to `apply`, whose side-effect preflight
previews and builds what the approval card shows. After approval the runtime runs the
preflight again and refuses a drifted preview, then hands the approved preflight to the
executor (`ToolExecution.approved_preflight`). `mode=preview` routes to the read-only
`data.preview_write`.

Both preview and apply first run `EXPLAIN (FORMAT JSON, VERBOSE)` on the exact text plus
`RETURNING`, and refuse unless every `ModifyTable` names the one approved operation and
table. That takes `ROW EXCLUSIVE` on the table even when read-only, so it runs in a
savepoint that releases the lock before the derived `SELECT`.

In the write transaction the backend (`adapters/postgresql_write.py`) re-reads the one
target table, locks existing target rows, reruns the derived `SELECT` and compares the
row digest (`ctid`, `xmin` and, when nothing varies, new values), structure digest and
count with the approved fingerprint. It then runs the statement and compares what
`RETURNING` reports with the locked read, using an order-independent digest, and rolls
back on any difference. A row's identity is `ctid` for a `DELETE`, or the first unique
not-null key the statement does not assign and, for an `INSERT`, lists; without one the
level is `count`. `xmax = 0` tells inserts from updates, and only on an `INSERT`: on a
plain `UPDATE` it means nothing. A statement varies when the function lookup finds a
non-immutable function or the parser finds a SQL time keyword; its sample then shows new
values as computed, since the runtime rechecks the whole preflight after approval. Receipt reservation and
finalization stay in `CapabilityRuntime`; an uncertain commit stays uncertain, and
nothing is retried or replayed.

The derived `SELECT` is rendered from a parse tree, so it is a claim, and a wrong one
fails preview or fails verification. sqlglot cannot render `E'...'` faithfully, so those
literals are kept out of the tree. Every other statement Daita sends is the model's exact
text. The function rule also checks each name written before `(`, because the tree renames
some calls, hides others and drops the case of a quoted one; reads keep their name checks from the parser, since `EXPLAIN`
expands views to their base tables.

`RelationalWriteScope` binds a table name, explicit `insert`, `update` and `delete`
operations, insert and update columns and a row limit, and reads no catalog.
Receipts from the removed tools (`data.update_rows`, `data.upsert_rows`) still decode, list
and resolve.
A native write in a scheduled or job run asks a person at the time, exactly as a
foreground write does; the card carries the routine or job title. Full request
sensitivity must fit the source's classification. DDL, `TRUNCATE`, `COPY`, `MERGE`,
a statement's own `RETURNING`, chunking, automatic retry and replay are not supported.

## Local computer files and artifacts

Local files reach the agent only through the `folder` source type
(`adapters/folder.py`), which wraps the descriptor-contained
`LocalWorkspaceBackend` and its private DuckDB worker. There are two kinds of
folder source:

- An **attached folder** (`Agent.attach(FolderSource(root, subpaths=...))`) is a
  durable, cataloged source like any other. `reconcile` catalogs only its data
  files (csv, tsv, json-records, ndjson and parquet) as FILE resources, the
  FOLDER resources that hold them with `CONTAINS` edges, a `FileFacet` (format,
  media type, size, stat-based physical revision) and a `TabularFacet` read
  through the worker. Other files stay reachable only live, through the session
  folder. It lists every data file up to the inventory limit and describes up to
  1,000 within a fixed walk and schema budget; a pass that reaches either commits
  with `CatalogSync.truncated`. Attached folders are always contained to their root;
  hosted compositions do not register the folder type. Like any source, they
  are in scope for scheduled and job runs; the session folder never is.
- The **session folder source** is built by the composition root from the
  session's `LocalWorkspace`. Its `source:workspace:sha256:` ID is stable for the agent
  and workspace, sorts after every stored source, and is never written to the
  agent home: it is not a registration and is never listed. It is not cataloged;
  its operations run live. Plain local CLI/TUI composition uses computer access:
  its frozen working directory is the default, while absolute and `~/` paths can
  address other OS-permitted locations. Typed `LocalWorkspace` callers remain
  contained by default unless they explicitly select computer access. Only a
  foreground user run receives it: hosted compositions have none, and scope
  resolution drops it for machine-originated runs even when a frozen scope names
  it. Explicit `source_scope_ids` narrow catalog sources and never drop it.

Private Daita state is rejected when targeted, pruned from broader searches and
refused as an attached root. `file_search` and `file_read` take a folder source's
display name as `source`; `file_search` supports one root or a bounded multi-root set with one
shared budget, qualified matches, and per-root coverage, and `file_read` returns
bounded chunks, `position=end` tail windows and opaque cursors. `data_query` with
`source`, `path_pattern` and `sql` queries one homogeneous dataset matched
against a folder's current files and exposes only the relation `data`; live and
attached folders share this one route. An attached folder answers only for data files within its `subpaths` that the run may
read, decided from each file itself: search skips any other file, read refuses it
before any file content is read, and a `path_pattern` that matches one is refused
whole, naming it. A changed file is
read as it is now; only a read cursor stays bound to the bytes it started on and
fails with `file_changed`. Every result carries the
source's sensitivity and its binding provenance. A combined `data_query` uses
the same worker and validator, binding typed copies of its complete inputs in
private scratch and exposing only their aliases; homes without local scratch
(`Agent.from_storage`) do not offer it.

`artifact_edit_text` names an earlier session-folder `file_read` of this run by
its `result_ref` or path; the artifact domain takes that read's authenticated
binding from the transcript, so the model never copies it. That capability commits a complete replacement artifact
without changing the local file. `artifact_save_local` requires exact approval,
revalidates the unchanged bound file, and atomically publishes the artifact.
Drift requires a fresh read and edit. Attached folders are read-only sources:
their reads carry no edit binding.

The artifact store is the sole storage boundary for committed artifact bytes
and manifests. `ArtifactByteStorage` supplies only exclusive publication,
bounded reads and idempotent deletion; the existing registry and lifecycle owner
retain identity, quotas, validation and recovery. Consumers own concrete clients,
credentials, storage layout and resource lifetime. See `docs/ARTIFACT_STORAGE.md`.
The shared `artifacts` table owns current identity, caller ownership,
and `creating`/`ready`/`deleting` lifecycle state. Reads and listings query ready rows;
transcripts, accepted job results and deliveries are historical evidence, not a
live inventory. Creation reserves a row before filesystem publication. Deletion
removes bytes before removing its pending row. Startup reconciles interrupted
creation and deletion; there is no separate cleanup queue or permanent tombstone.
Clearing transcripts preserves registered artifacts and their ownership.
Active job artifact reservations block deletion until their attempt finishes.
Job recovery can publish only under a currently active reservation.
`artifact_list` returns bounded safe metadata only for the
current conversation. `artifact_read` returns a bounded preview for an exact
known artifact ID owned by the current agent home. `artifact_create` with
`from="artifact"` supports only a verified Daita-generated XLSX `Data` snapshot converted to CSV and
records the parent artifact. The typed owner API `Agent.list_artifacts` supplies
bounded pages across conversations for human management in the TUI's `/artifacts`
screen; hosted listings retain caller isolation. Model inventory remains
conversation-scoped. There is no hidden current-file pointer, raw model
path/byte interface, or alternate artifact execution path.

`artifact_create` routes by `from` to four unchanged capabilities: `table`,
`document`, `tool_result` (canonical JSON snapshot) and `artifact` (XLSX to CSV
conversion). Arguments that belong to another route are rejected before
dispatch. `from="table"` creates one bounded model-authored CSV, XLSX, or
HTML table from earlier successful results in the current run, cited by their
visible `result_ref` values in `evidence_refs`. The runtime resolves each ref to
its call, and the artifact domain authenticates each result against the immutable registry and
persisted transcript, rejects failed, stale, reordered, cross-run, or tampered
lineage, inherits the highest result sensitivity, and records
the relational tables each result read, per input for a combined query. The artifact remains explicitly
derived analysis rather than exact or complete source data. Exact complete
relational export remains the separate `data.export_tabular` capability, reached
through `data_query` with `output="csv"`, `"xlsx"` or `"arrow"`.
