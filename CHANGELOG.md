# Changelog

## 1.3.0 (unreleased)

Daita's agent now works with your data where it lives. The model names sources and
tables, never IDs. Reads go straight to the source, and the catalog only advises.
One tool, `data_write`, makes every PostgreSQL write. Large sources attach instead of
failing, and a combined query can join up to 16 sources. Routines and background jobs
are simpler: a routine is a title, a prompt and a schedule, and a job is an agent run
with a checklist. Every effect is approved by a person at the time, so standing grants
are gone.

### Upgrading from 1.2

An existing home migrates automatically to home revision 4 the first time it opens.

- **Write scopes.** Stored scopes convert on upgrade: `update` stays `update`, `upsert`
  becomes `insert` and `update`, and delete is never granted. A scope whose table is
  gone from the catalog or changed since it was granted was already refused as stale,
  and it is dropped rather than revived. `Agent.preview_source_permissions(
  relational_write_scopes=...)` is now keyed by full table name, and a scope no longer
  has `key_columns` or `generated_identity_columns`.
- **Renamed or removed tools and APIs.** Update any caller, stored routine or prompt
  that names one:
  - `data_update_rows` and `data_upsert_rows` are replaced by `data_write`.
  - `data_query` takes `source`, the source's display name, instead of `source_id`,
    and no longer takes `resource_ids`. `catalog_search` and `catalog_describe` take
    `source` the same way.
  - `Agent.relational_update_readiness` and `Agent.relational_upsert_readiness` are
    replaced by `Agent.relational_write_readiness(source_id, table)`,
    `RelationalUpdateReadiness` by `RelationalWriteReadiness`, and
    `daita relational-update-readiness` by `daita relational-write-readiness`.
  - Jobs and routines: see the lists of removed tools, methods and commands in
    "Routines are a title, a prompt and a schedule" and "Background jobs are checklist
    runs" below.
  - Old transcripts that hold results of the removed write tools still replay, and
    their receipts (`data.update_rows`, `data.upsert_rows`) still decode, list and
    resolve.
- **Source type and catalog APIs.** For integrators:
  - `ResourceAdapter.reconcile(request)` replaces `discover`. It yields one INVENTORY
    `ResourceBatch` of every resource, then DETAIL batches, and `DiscoveryResult` is
    removed.
  - `DiscoveryRequest.scopes` replaces `refresh_schemas`, `previous` is the source's
    stored `CatalogEntry` tuple, and `max_inventory` is new; `max_resources` counts
    detailed resources and may be up to 100,000.
  - `CatalogStore` drops `commit_snapshot`, `list_current_snapshot_refs`,
    `load_current_snapshot` and `list_unreadable_snapshot_source_ids`. It gains
    `commit_catalog`, `current_syncs`, `load_source_state`, `list_catalog_changes`,
    `take_unreadable_catalog_sources`, `load_entry`, `load_entries`, `list_entries`
    and `search_catalog`.
- **Large sources attach truncated.** Attaching a source over the discovery limits no
  longer raises `DiscoveryLimitError`, which is removed. The source attaches with a
  truncated catalog. The model's source directory says the catalog holds only part of
  the source (`catalog_partial`), and tables outside it stay readable and describable
  live. The TUI permission pickers still list only cataloged tables, so a table past
  the limits cannot be granted there, and a granted table that falls past them can be
  dropped when someone re-picks the list.
- **Limits.**
  - *Discovery:* every table listed (up to 1,000,000); 1,000 described, with 512
    columns and 256 indexes per table, and 2,000 relationships. PostgreSQL stops reading table structure after 60 seconds.
  - *`data_query`:* up to 8 statements per call; 64 tables per call (16 in a job run);
    5 seconds per statement and 15 seconds in total; 100 rows and 16 KB shared by all
    statements; SQL up to 65,536 characters; up to 128 parameters. `output=csv`,
    `xlsx` or `arrow` exports one statement in full, up to 100,000 rows and 60 seconds.
  - *Combined queries:* 2 to 16 inputs; the inputs share 50,000 rows and 32 MiB within
    the 15-second deadline; the joining SQL is one DuckDB statement of at most 8 KiB.
  - *`data_write`:* one statement of at most 64 KiB on one table; an `INSERT ...
    VALUES` inserts at most 1,000 rows; the approval card shows up to 5 sample rows.
  - *`catalog_search`:* up to 50 hits (default 12) for a query of up to 1,024
    characters. *`catalog_describe`:* up to 256 columns, and a schema or folder lists
    at most 100 tables.
  - *Link checks:* each check reads at most 50,000 keys and 16 MiB, with 10 seconds
    per read.
- **Stored shapes.** Home revision 4 also changes what a user may notice:
  - SQLite table names with uppercase letters now have one identity per name, as
    SQLite itself does, so their resource IDs change. The migration rewrites every
    stored reference that must keep matching.
  - A stored routine, job or MCP tool is converted as described in its section below.
  - Old records keep decoding; a record that no longer needs a catalog revision simply
    stops requiring it.
  - Each saved model route candidate in `config.json` gains `reasoning_effort: null`,
    which keeps the provider's default.

### The model works with names, not IDs

The model refers to a source by its display name and to a table by its name. It never
sees or copies a source ID, resource ID or handle.

- **Tool arguments.** `data_query`, `data_write`, `catalog_search` and
  `catalog_describe` take `source`, the display name shown in the source directory.
  `data_query` has no `source_id` or `resource_ids`.
- **Duplicate names.** Two sources in one run with the same display name refuse the
  call with `source_ambiguous`, listing the candidates by display name and adapter. A
  name that matches none refuses with `source_unknown` and lists the run's source
  names.
- **Results.** A result shows names where it once showed IDs: a field `x_source_id`
  becomes `x_source` and `x_resource_ids` becomes `x_tables`. An ID outside the run's
  scope shows as null, or as "a table outside this run" inside text.

### Reads go to the source

The source answers and the catalog advises. A stale, empty or damaged catalog can no
longer fail a read.

- **One read policy, before any source I/O.** A read is exactly one `SELECT`, set
  operation (`UNION`, `INTERSECT`, `EXCEPT`) or `WITH` statement, including `WITH
  RECURSIVE`, on PostgreSQL and SQLite. A read the policy refuses returns
  `sql_validation_failed`, whose `details.issues` lists every problem by code:
  `multiple_statements`, `mutation_not_allowed`, `read_statement_required`,
  `explain_not_allowed`, `sql_too_large`, `function_not_allowed`,
  `operator_not_allowed` and the parameter checks. SQL that fails to parse is
  refused with the issue `sql_parse_error` and never runs.
- **PostgreSQL functions.** Two checks apply. Before the read, a function that
  cannot be resolved, `*_to_xml`, `cursor_to_*`, `ts_stat` and `ts_rewrite` are
  issues `function_not_allowed`, and an explicit operator outside `pg_catalog` is
  `operator_not_allowed`. Then the server says what each called name resolves to
  on the pinned search path (`pg_proc`), and a refusal there is the error
  `function_not_allowed`. A `pg_catalog` function is allowed only when no overload
  of its name is volatile; `random`, `clock_timestamp`, `timeofday`,
  `gen_random_uuid` and `timezone` are the exceptions and are allowed. A function
  shadowed by another schema is refused. On SQLite, table-valued functions such as
  `pragma_table_info` are issues `function_not_allowed`.
- **Name scope without the catalog.** Daita folds each table name and hashes it to the
  ID the stored permission lists, so a table the catalog has never seen is readable if
  it is granted. A read refused only for scope (issues `resource_out_of_scope` for a
  table outside the run, `schema_not_configured` for a schema not attached to the
  source) returns `resource_read_not_allowed`, naming the tables and close readable
  names.
- **Several schemas.** When a PostgreSQL source has several configured schemas, an
  unqualified table name is refused at execution with the error `unqualified_relation`,
  which lists the readable schema-qualified candidates. With one schema, unqualified
  names resolve in it.
- **Introspection SQL only for a full read scope.** Reading `pg_catalog`,
  `information_schema` or `sqlite_master` needs a run that may read the whole source;
  otherwise it is a scope refusal (`resource_read_not_allowed`) that points to
  `catalog_search` and `catalog_describe`.
- **Errors by SQLSTATE category.** Timeout (57014) is `postgresql_query_timeout`;
  insufficient privilege (42501) is `not_permitted`; other class 42 errors are
  `query_invalid`, with the server's hint and position; a read-only violation (25006)
  is `read_only_violation`; class 22 is `query_data_error`; class 08 is
  `postgresql_connection_failed`; the rest are `postgresql_query_failed` with the
  SQLSTATE. Only safe text crosses to the model.
- **`catalog_search` and `catalog_describe` ask the live source.** Every search checks
  the live source as well as the catalog and says whether each hit was seen live.
  `catalog_describe` reports `answered_from` (live or cache).
- **SQLite resource IDs fold case.** `Orders` and `orders` are one table, as SQLite
  treats them.
- **Look-alike tables get an advisory label.** When `catalog_search` or
  `catalog_describe` shows tables in one source with similar names and overlapping
  columns, each carries a `status` (`canonical`, `likely_stale`, `deprecated` or
  `unknown`), short `evidence` (row estimates, inbound foreign keys, a comment or
  current note that says deprecated; the last analyze is shown but never decisive) and
  its `look_alikes`. Nothing is hidden or reordered, and labels appear only in tool
  results, never in the prompt. PostgreSQL facts are read live, so tables past a
  truncated catalog get them too. `catalog_describe` makes one extra name search for
  this; if it fails the answer is unchanged.
- **Relationships carry `cardinality`.** For live PostgreSQL foreign keys,
  `catalog_describe` marks each `one_to_one` or `many_to_one` from the unique indexes.
- **`data_query` warns about wrong-looking results.** A single-statement read may return
  advisory `warnings`: a fan-out (`SUM`, `AVG` or `COUNT(col)` over a one-to-many join,
  at most 4), duplicate primary-key values in a joined, non-aggregated result, and, for
  an empty result, what the filtered column actually holds (min and max of a range
  filter, or up to 10 distinct values of an equality filter) from one bounded probe
  through the normal read policy. Checks run only when triggered, share 1 second within
  the read's deadline, and a failed check is dropped; a result is never changed or
  blocked. Multi-statement, combined and folder reads are unchanged.
- **Benchmarks.** The live schema baseline has a hard tier with planted traps
  (`DAITA_SCHEMA_BASELINE_TIER`), and a labelled corpus measures fan-out detection
  (37 of 40 caught, 0 of 44 false alarms).

### One write tool: `data_write`

The model writes to PostgreSQL with one SQL statement (`INSERT`, `UPDATE` or `DELETE`)
through `data_write`, previewed and approved as before. Daita builds the preview,
shows it on the approval card, re-checks it for drift after approval and writes once
with a receipt, in foreground, routine, job and sub-runs alike. `mode=preview` only
shows the preview.

- **Write scopes** name a table, `insert`, `update` and `delete` operations, columns and
  a row limit.
- **Verified writes.** PostgreSQL receives each read and write exactly as written, so
  `E'\\d+'` now works. Before a write, PostgreSQL reports what the statement targets;
  before commit, Daita compares the rows it touched with the rows on the card and rolls
  back if they differ. The card says whether it checks `rows` or only the `count`.
  Refusals are a fixed list of six. `UPDATE` and `DELETE` without `WHERE` are a card
  warning, as are time, randomness and a row matched twice. A statement that cannot be
  previewed fails with `write_preview_failed` and PostgreSQL's own message. A preview
  takes a brief table lock, so it can time out behind a schema change.
- **The catalog no longer gates writes.** A table with no catalog entry can be enabled
  and written, and unrelated DDL no longer fails a write.
- **Triggers and cascading foreign keys** do not refuse a table; the approval card
  names them and says their effects are not previewed.
- Generated identities are not returned, since a statement cannot have its own
  `RETURNING`.

### Attach never fails on size, and the catalog keeps up

A large source now attaches and lists every table (up to 1,000,000), then describes
up to 1,000 of them with 512 columns and 256 indexes per table and 2,000
relationships; on PostgreSQL it stops reading table structure after 60 seconds. A table
without a description is still found by name and stays reachable through
`catalog_search`, `catalog_describe` and queries, and the source is marked truncated;
the model is told the catalog lists every object but describes only some.
`DiscoveryLimitError` is gone.

- **The catalog refreshes without rewriting unchanged tables.** It stores one row per
  table, so a refresh writes only what changed, readers see the whole old catalog or
  the whole new one, and a failed refresh changes nothing. It keeps the last 20
  syncs and 90 days of changes per source.
- **Folder source edits work.** Editing an attached folder source no longer fails for
  want of its agent paths.

- **PostgreSQL discovery no longer slows with table count.** It takes five statements
  however many tables there are, instead of two per table, so a source over a slow
  network keeps all its tables rather than running out of time. Catalog contents and
  every limit are unchanged. Attaching 869 tables at 50 ms latency went from about 61 s,
  truncated at 540 tables, to about 2 s with all of them.
- **The catalog keeps itself current.** After attach, a keeper (one task per open agent)
  probes each source and re-reads only what changed; see "The catalog stays current
  without being asked" below. The five-minute limit on automatic refreshes is gone.
- **Revoking a source no longer waits for a refresh.**
- **No re-preview after a sync.** A catalog sync between previewing and applying source
  permissions no longer forces a new preview.
- **Connection check.** `Agent.check_source(source_id)` asks the live source whether it
  is reachable, which role Daita connects as, whether that role can only read, and
  warns about broad grants such as a superuser role. `/sources` shows it for the
  selected source, attaching a source shows it, and `daita attach` prints it as
  `connection`. The host serves it at `POST /sources/{source_id}/check`.
- **PostgreSQL shows only what the role can read.** A composite primary key appears only
  if the role can read every column of it. A partition is listed on its own unless a
  readable parent in the configured schemas holds it, so a role granted only a child
  partition can find it.
- **A damaged catalog entry no longer stops anything.** Reads and approved writes
  continue, the home still opens, and the keeper rebuilds the entry from the source.
- **`semantic_save` on a table with no catalog entry, or only a name,** is refused with
  `semantic_unknown_resource` and says reads are unaffected, instead of
  `resource_read_not_allowed`.

### The catalog stays current without being asked

A keeper, one task per open agent, checks each source cheaply and reads again only
what changed, so a new table or column shows up without a refresh. It never delays a
run, read or revoke, takes no execution lease, does not count as background work for
the host's idle policy, and sets no wake time.

- **A probe per source type.** It is one bounded read-only statement (or one stat
  walk) that reports names and hashes only and ignores the read scope. PostgreSQL gives
  one token per configured schema the role can use; it moves with tables, columns
  (including column comments), indexes, keys, and grants, roles or memberships that
  change what the role can read, and not with rows, statistics, table comments or enum
  labels. SQLite gives one token from
  `PRAGMA schema_version` and the file's identity, so data writes do not move it.
  A folder gives one token from a stat walk over the files that discovery would catalog.
- **Adaptive intervals.** A probe starts at the type's minimum (PostgreSQL 20 s,
  SQLite and folders 5 s), doubles after each unchanged probe up to its maximum
  (PostgreSQL 15 minutes, SQLite and folders 5 minutes), and starts over after a change
  or when the source is used. A type with no probe is only fully rechecked on its
  schedule. Two failed probes in a row mark the source unreachable; the next success
  recovers it. A probe times out after 5 seconds. Due times are wall-clock times, so
  sleep and restart catch up.
- **What starts a probe.** A run start (foreground, routine or job) if the last probe
  is older than 30 seconds, `Agent.catalog_status(check=True)`, a lookup miss or a
  missing-table or column error if the last probe is older than 5 seconds, and the
  permissions screen, which waits at most 2 seconds. The 300-second limit on automatic
  refreshes is removed.
- **Scoped reads.** A change in some PostgreSQL schemas re-reads only those schemas.
  A changed folder describes only its changed files. A pass still lists everything, so
  removals stay correct.
- **Resumable.** The new tokens are committed in the same transaction as the read that
  matched them, so an interrupted pass is read again on the next probe.
- **Background fill.** A catalog with tables that have only a name gets them described
  in passes of 1,000 until none remain. Progress is the stored detail, so closing and
  reopening resumes it.
- **Daily full check.** Once a day a full read records `missed`, the changes no probe
  flagged, and logs a warning naming only the source type and kinds of change.
- **People come first.** A person's attach, edit, detach, refresh or permission change
  cancels a keeper read in flight and runs first. Close stays prompt.
- **An unchanged probe writes the source's state row at most every 10 minutes.**
- **Keep current.** `Agent.set_keep_current(source_id, False)` stops all probes, fills
  and checks for a source. An explicit `Agent.refresh_source` still works.
- **Status.** New `Agent.catalog_status(source_id, check=False)` returns a
  `CatalogStatus` (`current`, `updating`, `unreachable`, `paused` or `not_checked`, the
  last check, change and full check times, resources, described, whether the inventory
  is truncated, and a stable error code). `Agent.catalog_changes(source_id, limit=50)`
  lists recent changes. `Agent.set_keep_current` returns the status. The host serves
  them at `GET /sources/{source_id}/catalog/status`,
  `GET /sources/{source_id}/catalog/changes` and
  `POST /sources/{source_id}/catalog/keep-current`.
- **`daita sources`** adds `status`, `checked_at`, `described` and `total` for each
  active source.
- **`/sources` in the TUI** shows a status line under each connection block (for example
  "Up to date · checked 2 min ago · 1,500 tables"), a "Today:" line of recent changes,
  a Keep current switch and Check now.
- **Stored sync `truncated`** now means the source's catalog lacks some detail: the
  pass was cut short, or any table still has only a name after the commit.
- **No stored shape changes.** The keeper uses the existing source state row.
- **For integrators.** `SourceOperation.CHANGE_PROBE` and `ChangeProbeBackend` (one
  `probe(agent_id, source_id, cursor)` returning a `SourceProbe` of tokens per scope);
  `SourceType.freshness` is a `SourceFreshness` (minimum and maximum probe interval, and
  the full-check period); `DiscoveryRequest.fill` keeps every described `previous`
  entry that is still listed and describes up to `max_resources` that lack detail.
  `CatalogStore` gains `save_source_state`, `catalog_detail_counts`, and
  `commit_catalog(probe_tokens=, probed_at=)`.
- **Measured.** A probe at 1,500 tables takes about 150 ms on PostgreSQL (one statement,
  best case on a quiet laptop; the 100 ms target was not met), about 1 ms on SQLite and
  about 133 ms on a folder (median). A change reached the catalog in a median of 16 s
  on PostgreSQL, 4 s on SQLite and 5 s on a folder. Over 108, 112 and 110 scripted
  changes the probes missed none.

### Cross-source joins

A combined `data_query` now joins 2 to 16 sources, up from 4, and tells the model how
well the join matched.

- **Limits.** All inputs share 50,000 rows and 32 MiB, enforced while they read, within
  one 15-second deadline. Sources read at the same time and the inputs of one source one
  at a time. An input that overflows its share is read again alone with what is left;
  one that still does not fit refuses the call with `combined_input_truncated`. A failing
  input cancels the others and returns its own error code, with the alias in its details
  as `input_alias`.
- **Join checks.** Each result reports `rows`, `read_at` and `read_ms` per input,
  `read_consistency` (inputs are read separately, not as one snapshot), `join_checks`
  (key types, row counts, rows with no match, and how many of those would match
  ignoring case and surrounding spaces, for each equality in the join) and
  `warnings`. The checks run after the result is safe, never block it, and report
  `partial` if they cannot finish.
- **The catalog no longer gates tabular inputs.** An unlisted table or column is the
  source's to serve or refuse.
- **Smaller reads.** SQLite stops fetching at its byte share, and PostgreSQL prefetches
  at most 256 rows.
- **Link hints.** `catalog_describe` relationships carry `from_row_estimate` and
  `to_row_estimate`, and it lists `link_candidates`, unconfirmed joins to other sources.
  A candidate grants no read access and is never declared.
- **Link check.** `Agent.check_link(...)` and `Agent.check_catalog_link(link_id)` read
  the key columns of both sides once, within bounds, and return a `LinkCheck` with the
  match rate. A read that hits its bound makes the result `partial`. The host serves
  both. They are not model tools.
- **Removed:** the tabular-input error codes `sql_resource_source_mismatch`,
  `catalog_provenance_missing` and `tabular_column_unknown`, which no longer occur.
  `daita.domains.data.COMBINED_QUERY_INPUT_MAX_ROWS` and `_BYTES` are replaced by
  `COMBINED_QUERY_TOTAL_MAX_ROWS` and `_BYTES`, now totals for the whole call.

### Routines are a title, a prompt and a schedule

A routine now holds what a person asked for: a title, a prompt, a schedule
(once, interval or calendar, including relative timing) and, optionally, the
sources to use. Each run uses the agent's current model and normal run limits,
delivers to the conversation's Inbox, and asks a person before any write.
Missed slots are skipped, a one-time routine ends after it runs, and a
recurring routine pauses as Needs you after 3 consecutive failures.

- **Python API.** `Agent.create_routine(intent, *, conversation_id=None,
  confirmation_handler=None)` and `Agent.update_routine(routine_id, intent, *,
  expected_revision=None, confirmation_handler=None)` take a `RoutineIntent`
  (`title`, `prompt`, `schedule`, `sources`, `run_immediately`). On update,
  omitted fields keep their values and the revision defaults to the current one.
  Without a conversation, a routine delivers to a new conversation of its own.
- **Removed from the public API:** `ScheduledRoutineDraft`,
  `Agent.propose_routine`, `Agent.promote_routine`,
  `Agent.distribution_destinations`, `DistributionDestination`,
  `MisfirePolicy`, `ReportingMode`, `RequestedCapabilityGrant` and
  `ResourceRevisionPrecheck`. `ScheduledRoutine` no longer carries access
  ceilings, grants, model routes, a sensitivity ceiling, budgets, expiry, an
  outcome contract, prechecks, skills or promotion evidence; its source list
  is `source_ids`.
- **CLI.** `daita routines create <agent> --title … --schedule … --prompt …`
  (or `--spec file.json`), with optional `--source` and `--run-immediately`;
  `daita routines update <agent> <routine_id>` takes the same flags.
  `daita routines promote` and `daita routines destinations` are removed.
- **Model tools.** `routine_save` takes `title`, `prompt`, `schedule`,
  `sources` and `run_immediately` (plus `routine_id` to update). The
  `distribution_destination_list` tool is removed.
- **Host.** The routine routes and `RemoteAgent` follow the Python API; the
  proposal, promotion and destination routes are removed.
- **Existing homes.** Upgrading rewrites stored routines to the new shape; they
  keep their schedule, prompt, named sources and state, keep running, and their
  writes now ask.

### Background jobs are checklist runs

A background job is now an ordinary agent run that keeps going in the host. It
plans with a checklist, hands independent pieces to sub-runs (several at once),
asks you when it needs you, and delivers its final answer to the
conversation's Inbox. Writes ask a person at the time. A job survives restarts
by starting a new run from its saved checklist, and also continues in a new
run when its context fills up or you answer its question. It stops as failed
after 20 runs or ten times the per-run token limit in total; answering raises
the limits and continues it. This replaces the task-graph job system.

- **Python API.** `Agent.start_job(objective, sources=None, *,
  conversation_id=None)`, `list_jobs`, `inspect_job` (now a `Job`),
  `answer_job(job_id, text)` and `cancel_job`. `Job`, `JobState`,
  `ChecklistItem`, `ChecklistStatus`, `Delegation` and `DelegationState` are
  exported. `HostCapacityLimits.graph_parallelism` is now `job_parallelism`
  (sub-runs at once per agent, default 2).
- **Removed from the public API:** `GraphJob`, `GraphState`,
  `GraphInspection`, `TaskResult` and the graph job methods
  (`list_job_tasks`, `inspect_job_task`, `list_job_dependencies`,
  `list_task_attempts`, `read_task_result`, `list_task_checkpoints`,
  `list_task_artifacts`, `list_task_controls`, `job_timeline`, `job_board`,
  `read_job_result`, `graph_blockers`, `answer_task_input`,
  `accept_task_review`, `request_task_review_changes`, `retry_task_control`,
  `reject_task_control`, `cancel_graph_job`, `replace_graph_task_by_policy`,
  `start_authorized_replacement_job`).
- **CLI.** `daita jobs list | inspect | answer | cancel`. The graph and review
  job commands (`tasks`, `dependencies`, `attempts`, `result`, `checkpoints`,
  `artifacts`, `timeline`, `board`, `blockers`, `accept-review`,
  `request-changes`, `retry-control`, `reject`, `cancel-graph`,
  `replace-task`) are removed.
- **Model tools.** `job_start`, `job_list`, `job_inspect` and `job_cancel` in
  conversations; `checklist_write`, `delegate` and `ask_user` in a job's own
  run. `start_graph_job`, `job_read_results` (folded into `job_inspect`) and the
  graph task tools are removed, and so is the internal data-profile job.
- **Host.** The job routes and `RemoteAgent` follow the Python API; the graph
  task and control routes are removed.
- **Existing homes.** Upgrading converts graph jobs: finished ones keep their
  accepted result, unfinished ones (including profile jobs) fail with "Created
  by an older version of Daita; start it again.", result deliveries and
  receipts keep their job, graph attention notices are dropped, and the graph
  tables are removed.

### Anthropic prompt caching

Repeated steps and repeated questions on Anthropic models now reuse the provider's
prompt cache, which lowers cost and latency. Each request's prefix (tools, system
text, then messages) is identical across the steps of a run and across runs of one
agent, except the current run's first user turn.

- **Start context moves to the question.** The run-start time, the unresolved-effect
  notice and question-specific procedures (SQL dates, Python requested reads, export
  requests) are computed once per run and shown as a block after the question on the start
  message, in the request only. The transcript and stored message are unchanged.
- **Python guidance is steady.** It shows whenever `python_execute` is callable, not
  chosen from the question's words or earlier results.
- **Rounded object count.** The source directory's approximate object count is
  rounded to two significant figures.
- **Provider cache.** The Anthropic adapter marks the system block and the
  conversation tail for caching, whatever the model profile's
  `supports_prompt_caching` says. OpenAI and Gemini are unchanged.
- **A tool load no longer fails the run.** On the preserved-thinking Claude models
  (claude-fable-5-1, claude-mythos-5-1, claude-opus-5-5, claude-sonnet-5-5,
  claude-haiku-5-5) the adapter asks the provider to drop a thinking block that no longer
  matches the longer tool list, so every account loses one step's reasoning instead of
  newer accounts failing the run with a 400.
- **Limits.** `toolbox_load` changes the system text and rewrites the cache. Once
  history is full, the sliding window drops the oldest turn and shifts the cached
  history. Changes to memory, the user profile, the skill index or the directory also
  change the system text.

### Choose a model's reasoning level

Each agent can pin how hard its model thinks. Until now every run used the provider's
default: `medium` on Claude Opus 5.5 and `high` on Claude Fable 5.1, Sonnet 5 and
Opus 4.8.

- **Where.** `/model` in the TUI asks for a level after the model, with "Provider
  default" first. `Agent.configure_model(..., reasoning_effort="high")` does the same
  from Python, and `daita run --no-host --model ... --reasoning-effort high` sets it
  for one run.
- **Levels** are the provider's own: Anthropic `low`, `medium`, `high`, `xhigh`,
  `max` (`output_config.effort`), none on Claude Haiku 4.5; OpenAI `none`, `low`,
  `medium`, `high`, `xhigh`, `max` (`reasoning.effort`), without `none` on GPT-6
  Astra; Gemini `minimal`, `low`, `medium`, `high` (`thinking_config.thinking_level`).
  Subscription and custom providers don't accept one. Saving checks the level with
  the model, so a level the model rejects is refused before it is saved.
- **Fixed for a run.** The level is set when the provider is built, so every request
  in a run uses it and Anthropic's cached messages stay valid. A change applies when
  the agent next reopens. It is part of the route's machine-run binding; routes with
  no level keep the binding they had.

### Smoother interactive tools

Tools no longer make the model copy back values Daita already holds, and a call
with several argument mistakes gets all of them back at once.

- **Argument errors.** Every structural problem in a call's arguments comes
  back in one `invalid_arguments` error whose `details.issues` lists up to 32
  problems (`code`, `path`, `name`, `problem`), with `issue_count` for the
  total. For a `oneOf`, the issues are those of the closest shape. The
  per-problem codes (`missing_arguments`, `unexpected_arguments`,
  `invalid_argument_type`, `invalid_argument_value`) now appear inside
  `issues` instead of as the error code.
- **No copied values.** `python_execute` drops `expected_state`; a result says
  `state_reset: true` when the interpreter was replaced. `semantic_save` drops
  `catalog_revisions` and `expected_sha256`, and `semantic_save` with
  `action=delete` drops `expected_sha256`: Daita binds the current catalog
  revisions, and a replacement, supersession or deletion needs an earlier
  `semantic_view` in the run, whose version Daita binds. `skill_save` drops
  `expected_sha256`; replacing needs an earlier `skill_view` in the run.
  `artifact_edit_text` names the earlier `file_read` by `result_ref` or `path`
  instead of copying its `binding`, and `expected_occurrences` defaults to 1.
  `routine_control` makes `expected_revision` optional (default: the current
  revision).
- **Defaults are visible.** A tool that routes by a selector argument (`mode`,
  `action`, `output`) now states the selector's default in its schema.
- **Approval cards** show every value the action will use, including bound
  and defaulted ones (for example a new annotation's id and the routine
  revision being controlled).
- **Python API.** Custom executors receive the preflight a person approved as
  `ToolExecution.approved_preflight`.

### Removed: standing grants and the resident host

Every effect is now approved by a person at the time, so the pre-authorization
machinery is gone.

- **Removed grant API.** `AutomationEligibility`, `AutomationGrantPolicy`,
  `AutomationScopeProposal`, `Capability.automation_eligibility`,
  `Capability.automation_grant_policy`, `CapabilityRuntime.prepare_automation_grant`
  and every domain's `prepare_automation_grant`, and
  `SideEffectPlan.capability_grant_digest`. Scheduled and job runs may use
  every capability except routine and job management.
- **MCP.** `MCPToolSelection` and `MCPToolBinding` drop
  `automation_eligibility`, and the TUI's MCP screen drops its eligibility
  choice. Every MCP action asks at the time, in any run.
- **Toolbox.** `toolbox_search` results drop `automation_eligibility`,
  `requires_automation_grant`, `automation_contract` and
  `automation_contract_omitted`; contracts drop `grant_policy`.
- **Removed resident host.** `daita host --agent`, `daita.ResidentReady` and
  `daita.run_resident_host` are removed; use `daita host serve` or
  `daita.connect()`. The ignored `daita chat --model` flag is removed.
- **Existing homes.** Upgrading drops each stored MCP tool's eligibility and
  each routine occurrence's copy of its run's scope. Grants recorded by older
  runs stay readable as history and are never honoured.
