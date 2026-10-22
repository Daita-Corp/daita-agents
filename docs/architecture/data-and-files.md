# Data, catalog, sources, local files and artifacts

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

Source types, catalog truth, data queries and writes, folders and artifacts.

## Data, catalog and adapters

`daita.catalog` holds normalized structural truth and implements catalog
search, inspection, and traversal. Data code consumes catalog contracts rather
than building another schema graph or querying source clients for planning
facts.

A `CatalogRelationship` belongs to one source snapshot. A `CatalogLink` is an
agent-owned declared reference between fields of two resources, possibly in
different sources, managed only through `Agent.declare_catalog_link`,
`remove_catalog_link` and `list_catalog_links`; no model tool declares links.
Declaration requires both resources and every paired field in the current
catalog. Links survive detach and reattach because source and resource IDs are
stable. Currency is computed, never stored: a link is current while both
sources are active and every paired field exists in a current tabular facet;
otherwise it is stale as `source_inactive`, `resource_missing` or
`field_missing`. Unrelated column changes keep a link current. Search
neighbours, traversal and schema join selection merge current links with
source relationships in one edge set, labelled `provenance: "declared"` with
`cross_source`; sources joined by current links form one join group, so
`cross_source_unsupported` remains only for seeds with no linking group. Links
are read fresh for each projection, outside the per-source index cache. Readable
and run-scope filters apply to both link endpoints, so traversal never crosses
a resource outside scope. A `catalog_describe` narrowed by `source_id` still
lists each current link leaving that source, with its far endpoint unselected.

`daita.adapters` implements source admission, containment, discovery,
freshness checks, and I/O. One composed `SourceTypeRegistry` per agent maps each
`adapter_id` to its factory, credential fields, read capacity, SQL dialect and
operation backends. `SourceOperation` names those backends: `SQL_READ`,
`TABULAR_READ`, `TABULAR_EXPORT`, `PROFILE`, `RELATIONAL_UPDATE`,
`RELATIONAL_UPSERT`, `FILE_SEARCH`, `TEXT_READ` and `PATH_QUERY`; every
operation except `TABULAR_READ`, `FILE_SEARCH` and `TEXT_READ` requires an SQL
dialect the validators in `domains/data/sql` support, checked at registration.
A stored source whose type is not registered stays listed
but unavailable: it gets no tools or source capacity, and refresh or edit fails.
Pass the composing `AgentConfig` to `Agent.delete` so credentials of its source
types are removed too. PostgreSQL read diagnostics, update readiness and
keychain password storage remain PostgreSQL-only public conveniences. SQLite paths must be absolute and bounded. Local
workspace access uses only admitted relative logical paths and rejects
symlink, traversal, secret-file, and special-file access. PostgreSQL and MCP
credentials remain secret references and are resolved only at the integration
boundary.

SQL validation belongs in `daita.domains.data.sql`; connector guardrails still
apply during execution. Do not duplicate either mechanism in a generic policy
layer.

`data_query` with `inputs` routes to `data.query_combined`, a foreground-only
read that joins two to four sources. Each SQL input `{alias, source_id,
resource_ids, sql}` takes the single-source path unchanged: run scope, catalog
validation by the source type's `sql_dialect`, bound resource revisions and that
type's registry backend. A tabular input `{alias, source_id, resource_id,
columns, filters}` names one resource of a type that declares `TABULAR_READ`; it
takes the same scope checks, every column and filter column must be in the
current catalog schema, and filters are only AND-combined `eq`, `in`, `lt`,
`lte`, `gt` and `gte` on scalar values. Its backend receives one
`ReadTabularRequest` bound to the validated resource revision, refuses a stale
revision with `query_binding_stale`, and returns a `BoundedResultProjection`.
It serves types without SQL; the built-in types read through SQL inputs instead,
which can aggregate before the join. A type with `TABULAR_READ` but no `SQL_READ` is usable
only as a combined-query input; single-source `data_query` is refused for it,
and `data_query` is offered only while a source with `SQL_READ` is in scope.
A run withholds the `data_query` routes no in-scope source can serve: export
needs a `TABULAR_EXPORT` source and `path_pattern` needs the session folder.
A tool definition describes only the routes its run admits, so scheduled and
graph runs never read about combined or live-folder queries.
Inputs read at most an internal, model-invisible
ceiling (`COMBINED_QUERY_INPUT_MAX_ROWS`/`_BYTES`); an input that reaches it
refuses the whole call with `combined_input_truncated`, so no join ever sees
incomplete input; an input with no rows still joins and aggregates. The
combining `sql` is one DuckDB SELECT or UNION validated before execution to
read every alias and nothing else: no table functions, file, HTTP or extension
access. It runs in the same private DuckDB worker as folder-source SQL, one
query at a time; a concurrent call waits for the worker. The result carries the union of input resource
revisions, the highest input sensitivity and per-input lineage (`source_id`,
`source_revision`, and `canonical_sql`/`sql_fingerprint` for SQL inputs or
`columns`/`filters` for tabular inputs). Declared catalog links
inform discovery and join paths only; they grant no read authority.

Data capabilities are reads except for explicitly admitted `data_update_rows` and
`data_upsert_rows`, initially backed by PostgreSQL. Their read-only counterparts are
`data_preview_update_rows` and `data_preview_upsert_rows`. Execution requires an
authenticated current-run preview and matching intent. Update retains its exact
selection, target-count, drift, rollback and receipt safeguards.

`RelationalWriteScope` binds exact structural revision, explicit update/upsert
operations, keys, insert/update columns, admitted identity generation and row limits.
Update permission never implies insertion authority. Upsert uses one bounded uniform
scalar batch, supported non-null unique-key equality, explicit omitted/null semantics,
and narrowly admitted identity generation. Its transaction acquires EXCLUSIVE on the
exact table before the authoritative scan, rebuilds the preview, rejects drift,
inserts missing rows, updates changed rows, skips unchanged rows, and verifies exact
counts summing to the input count. Mismatches roll back the entire batch. Sequence
allocations may leave gaps after rollback; receipts describe table-row effects.

The data domain normalizes native standing grants and permits at most one native
write capability per routine, with one invocation per occurrence and exact ceilings.
An unchanged batch consumes its reservation and produces verified zero-mutation
evidence. Full request sensitivity must fit the current target classification.
Research lineage remains model-derived claims, distinct from transaction facts.
Receipt reservation/finalization stays in CapabilityRuntime. No arbitrary SQL,
insert-only tool, delete, DDL, chunking, automatic retry or replay is supported.
Native graph release is limited to these same PostgreSQL update/upsert contracts.

## Local computer files and artifacts

Local files reach the agent only through the `folder` source type
(`adapters/folder.py`), which wraps the descriptor-contained
`LocalWorkspaceBackend` and its private DuckDB worker. There are two kinds of
folder source:

- An **attached folder** (`Agent.attach(FolderSource(root, subpaths=...))`) is a
  durable, cataloged source like any other. Discovery catalogs only its data
  files (csv, tsv, json-records, ndjson and parquet) as FILE resources, the
  FOLDER resources that hold them with `CONTAINS` edges, a `FileFacet` (format,
  media type, size, stat-based physical revision) and a `TabularFacet` read
  through the worker. Other files stay reachable only live, through the session
  folder. Discovery is bounded by `max_resources` and a fixed walk and schema
  budget; a discovery that reaches either commits a bounded snapshot with
  `CatalogSync.truncated`. Attached folders are always contained to their root;
  hosted compositions do not register the folder type. Like any source, they
  can be granted to scheduled routines and graph jobs.
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
refused as an attached root. `file_search` and `file_read` take a folder
`source_id`; `file_search` supports one root or a bounded multi-root set with one
shared budget, qualified matches, and per-root coverage, and `file_read` returns
bounded chunks, `position=end` tail windows and opaque cursors. `data_query` with
`source_id`, `path_pattern` and `sql` queries one homogeneous dataset matched
live in the session folder and exposes only the relation `data`. An attached
folder answers only for its readable cataloged files at their cataloged
physical revision: search skips any other file and read refuses it before any
file content is read, `data_query` takes their `resource_ids` as the one
relation `data`, combined queries accept them as SQL inputs, and a changed
file is refused with `query_binding_stale` until the source is refreshed. Every result carries the
source's sensitivity and its binding provenance. A combined `data_query` uses
the same worker and validator, binding typed copies of its complete inputs in
private scratch and exposing only their aliases; homes without local scratch
(`Agent.from_storage`) do not offer it.

An authenticated current-run session-folder `file_read` binding can feed
`artifact_edit_text`. That capability commits a complete replacement artifact
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
lineage, inherits the highest result sensitivity, and preserves current
relational resource revisions where present, per input for a combined query. The artifact remains explicitly
derived analysis rather than exact or complete source data. Exact complete
relational export remains the separate `data.export_tabular` capability, reached
through `data_query` with `output="csv"`, `"xlsx"` or `"arrow"`.
