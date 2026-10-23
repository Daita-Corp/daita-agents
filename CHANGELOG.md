# Changelog

## 1.3.0 (unreleased)

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
