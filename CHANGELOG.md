# Changelog

## 1.3.0 (unreleased)

### One write tool: `data_write`

The model writes to PostgreSQL with one SQL statement (`INSERT`, `UPDATE` or `DELETE`)
through `data_write`, previewed and approved as before. It replaces `data_update_rows`
and `data_upsert_rows`.

- **Write scopes** name a table, `insert`, `update` and `delete` operations, columns and
  a row limit. Upgrading converts stored scopes: `update` stays `update`, `upsert`
  becomes `insert` and `update`, and delete is never granted.
  A scope already refused as stale is dropped rather than revived.
  `Agent.preview_source_permissions(relational_write_scopes=...)` is now keyed by full
  table name, without `key_columns` or `generated_identity_columns`.
- **Verified writes.** PostgreSQL receives each read and write exactly as written, so
  `E'\\d+'` now works. Before a write, PostgreSQL reports what the statement targets;
  before commit, Daita compares the rows it touched with the rows on the card and rolls
  back if they differ. The card says whether it checks `rows` or only the `count`.
  Refusals are a fixed list of six. `UPDATE` and `DELETE` without `WHERE` are now a card
  warning, as are time, randomness and a row matched twice. A statement that cannot be
  previewed fails with `write_preview_failed` and PostgreSQL's own message. A preview now
  takes a brief table lock, so it can time out behind a schema change.
- **The catalog no longer gates writes.** A table with no catalog entry can be enabled
  and written, and unrelated DDL no longer fails a write.
- **Triggers and cascading foreign keys** no longer refuse a table; the approval card
  names them and says their effects are not previewed.
- **Removed:** `Agent.relational_update_readiness` and
  `Agent.relational_upsert_readiness` become `Agent.relational_write_readiness(source_id,
  table)`, `RelationalUpdateReadiness` becomes `RelationalWriteReadiness`, and
  `daita relational-update-readiness` becomes `daita relational-write-readiness`.
  Generated identities are no longer returned, since a statement cannot have its own
  `RETURNING`.

### Attach never fails on size, and the catalog keeps up

A large source now attaches. Discovery reads up to 1,000 tables, 512 columns and 256
indexes per table and 2,000 relationships, and on PostgreSQL stops reading table
structure after 60 seconds. Whatever is cut stays reachable through `catalog_search`,
`catalog_describe` and queries, and the source is marked truncated; the model is told
the catalog holds only part of it. `DiscoveryLimitError` is gone.

- **The catalog refreshes itself.** It refreshes on attach, on `Agent.refresh_source`,
  when a search or describe finds a table the catalog lacks or has wrong, and when a
  query fails because a table or column does not exist. Automatic refreshes run in the
  background, never delay a read, and run at most once per source every five minutes.
  On PostgreSQL a miss on a schema-qualified name re-reads only that schema.
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
- **A damaged catalog row no longer stops work.** While the agent is open, reads and
  approved writes continue; opening a home with a damaged row still fails.
- **`semantic_save` on a table with no catalog entry** is refused with
  `semantic_unknown_resource` and says reads are unaffected, instead of
  `resource_read_not_allowed`.

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

### Smoother interactive tools

Tools no longer make the model copy back values Daita already holds, a
relational write is one call and one approval, and a call with several
argument mistakes gets all of them back at once.

- **Argument errors.** Every structural problem in a call's arguments comes
  back in one `invalid_arguments` error whose `details.issues` lists up to 32
  problems (`code`, `path`, `name`, `problem`), with `issue_count` for the
  total. For a `oneOf`, the issues are those of the closest shape. The
  per-problem codes (`missing_arguments`, `unexpected_arguments`,
  `invalid_argument_type`, `invalid_argument_value`) now appear inside
  `issues` instead of as the error code.
- **One-call writes.** `data_update_rows` and `data_upsert_rows` default to
  `mode=apply`: Daita builds the preview, shows it on the approval card,
  re-checks it for drift after approval and writes once with a receipt, in
  foreground, routine, job and sub-runs alike. `mode=preview` still only
  shows the preview. `preview_fingerprint` and `expected_affected_rows` are
  removed from the arguments and from preview results.
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
