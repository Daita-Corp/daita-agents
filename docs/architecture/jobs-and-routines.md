# Durable jobs, scheduled routines and deliveries

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

Machine-originated work: background jobs, time-triggered routines, outcomes and
deliveries.

## Checklist jobs

A background job is an ordinary agent run that keeps going in the host. It
plans with a checklist, hands independent pieces to sub-runs and asks a person
when it needs one. These boundaries hold; changing one is an architecture
change:

- one durable `Job` record per job, in the `jobs` table (strict codec, one
  row): objective, optional `source_ids`, state (`running`, `waiting_for_you`,
  `done`, `failed`, `cancelled`), the model's checklist (at most 64 items with
  `pending`/`in_progress`/`done`/`cancelled` status and optional `after` ids),
  delegations (sub-runs and their bounded summaries), the open question and the
  person's answers, run and token counts and limits, and the final summary;
- `JobOwner` is the one writer of job records in a process. It serializes
  changes and compare-and-sets the record's revision, so a receipt that holds a
  job for a person is never overwritten. `JobSupervisor` runs one main run at a
  time per job and the sub-runs it delegates;
- every run is an ordinary `AgentLoop` run through the one `CapabilityRuntime`.
  A job continues by starting a **new** main run from its saved record; no run
  is ever resumed;
- model-written text (checklist titles and notes, goals, summaries, questions)
  is stored sanitized and bounded, is labeled untrusted when shown to a model,
  and grants nothing. The record carries a sensitivity floor: the highest
  sensitivity any of its runs or sub-runs saw, applied to later runs and to
  tool outputs that repeat its text.

### Tools

Foreground runs get `job_start(objective, sources?)` (on demand, no confirm
card: the job shows on the Work board at once and can be cancelled),
`job_list`, `job_inspect` (checklist, delegations, question and final result)
and `job_cancel` (on demand). A job's main run, and only it, gets:

- `checklist_write(items, merge?)`: replace the checklist or merge by id;
  returns the full list. Starting or finishing an item before every item it
  comes `after` is `done` is refused, naming them; unknown `after` ids and
  cycles are refused; a cancelled item does not satisfy its dependants;
- `delegate(goal, context, item?, sources?)`: start a sub-run with only that
  goal and context, wait for it and return its bounded summary. Several calls in
  one step run in parallel. With `item`, code marks the item `in_progress`, then
  `done`, or `pending` again if the sub-run fails. `sources` narrow the main
  run's fixed scope and can never widen it;
- `ask_user(question)`: must be the only call in its step; the run ends with a
  machine directive, the job waits as **Needs you**, and the answer starts a new
  run that sees it.

Sub-runs (`JOB_SUBRUN`) are flat: they have none of the job tools. Routine runs
have neither job-run tools nor the foreground job tools; the one exclusion list
is in `hosting/embedded.py`.

### Runs and resuming

A main run (`RunOrigin.JOB`) is a machine run: its scope is fixed at start from
the job's sources (or every current source) and every capability except the
exclusions, its limits are the agent's normal run limits, and writes ask a
person at the time. Its start is a code-owned instruction plus an untrusted
payload: the objective, checklist, delegation summaries, answers, the writes
already committed (from receipts) and the previous run's last assistant
message, shrunk oldest-first to fit its bound. Main runs live in a conversation
named after the job; each sub-run has its own. Neither imports conversation
history.

A job continues in a new run when the host restarts while it is running, when a
main run's last request passed 60% of the model's input window (the run ends
after that step as `context_threshold_reached`), when the person answers its
question, and after a run stopped at one of its own step, time, token or
tool-call limits. Sub-runs that were running at a restart are marked failed
with "Interrupted by restart."; the model sees that and can delegate again. A
run that ends with final text finishes the job: that text is its result,
delivered to the originating conversation's Inbox as a Done card. An
unanswered approval interrupts the run and holds the job as **Needs you**; any
other run failure fails the job.

### Limits, writes and cancellation

A job stops as `failed` after 20 runs or ten times the per-run token limit in
total (sub-runs count), whichever comes first; answering a job stopped at a
limit raises both by the same amount and continues it. Main runs take a
`WorkloadClass.JOB` execution lease from `RunAdmissionCoordinator`. A run counts
against the job only once it holds the lease, and its time limit starts then; a
wait that times out leaves the job running to try again. Sub-runs run
inside their main run's lease and deadline, at most `HostCapacityLimits.
job_parallelism` (default 2) per agent at once. A provider quota rejection
holds the job for five minutes instead of failing it.

A receipt is reserved only after approval, and a job receipt requires its job
to be running. A write waiting for approval at a restart was never dispatched:
the resumed run sees it is not committed and asks again. An uncertain receipt
holds the job as **Needs you** until a person resolves it, and resolving the
job's last uncertain write lets it continue.

`job_cancel` stops the main run and its sub-runs at once; a cancelled job never
restarts. Agent identity is the job authorization boundary; the originating
conversation is provenance, not an access gate.

Revision 4 converted the task graph: finished graph jobs became `done`,
`failed` or `cancelled` jobs keeping their accepted result, unfinished ones
became `failed` with "Created by an older version of Daita; start it again.",
result deliveries and receipts kept their job id, graph attention notices were
dropped, and the graph tables went. The revision-2 conversion still imports the
frozen graph records (`jobs/graph/models.py`, `validation.py`), their codec and
`storage/sqlite_graph.py`; nothing at runtime uses them.

## Scheduled routines and deliveries

A routine is a title, a prompt and a schedule, optionally limited to named
sources. `RoutineIntent` is its one shape for the `routine_save` tool, the CLI,
`Agent.create_routine`/`update_routine` and the host routes, and
`RoutineIntent.from_mapping` is the one parser. A save shows one confirm card
(title, schedule in local time, prompt, sources); it is the commitment to
future runs and model spend. An update keeps omitted fields, and its expected
revision defaults to the current one.

`ScheduledRoutine` and `RoutineOccurrence` are the only scheduled-work records.
A routine keeps its identity, conversation, title, prompt
(`authorized_instruction`), schedule, optional `source_ids`, state, revision,
counters and timestamps; it freezes no access, grants, routes, sensitivity,
budgets, expiry or outcome. `RoutineOwner` validates saves (resolving source
names to ids and relative timing to instants) and owns list, inspect, update,
pause, resume, run-now and disable. `RoutineSupervisor` computes due slots,
conditionally claims occurrences, fences stale work, fixes each run's scope at
start, invokes `AgentLoop`, and atomically finalizes the occurrence and its
delivery. Mutation tools return a compact identity, revision, state and
schedule receipt; inspection shows the full record.

Routines support exact one-time, anchored interval, and IANA-timezone calendar
schedules with explicit daylight-saving gap/overlap behavior; relative timing
(`after_seconds`, `anchor_after_seconds`, the next named local weekday) is
resolved when the routine is saved. Raw prompt text never determines whether a
time slot is due. Fixed code defaults, with no options:

- delivery to the routine's conversation Inbox (a routine created outside a
  conversation gets a new one of its own);
- missed slots are skipped: a single late slot still runs, an accumulated
  backlog does not;
- each run has the agent's current model and normal run limits, and a routine
  has no sensitivity ceiling of its own; its prompt counts as internal;
- a recurring routine runs until paused or disabled, and pauses as Needs you
  after 3 consecutive failures; resuming clears the count;
- a one-time routine ends after it runs.

Scheduled runs build their scope when they start: the named sources (or every
current source, never the session folder), their readable resources, and every
capability except the machine-run exclusions in `hosting/embedded.py` (local
computer files, and creating, changing or running routines and jobs). Reads
just run; every write or external action asks a person at the time through the
host's interactions, is receipted before dispatch and is never replayed. An
uncertain receipt pauses the routine until a person resolves it.

Routine occurrences validate committed artifacts against one fixed code
outcome (any artifacts within the shared delivery bounds). `DistributionPlan` freezes ordered target bindings. `Delivery` is the sole
durable representation of a logical delivery; the Inbox is a bounded product
view over deliveries. The only distribution destination is the originating
conversation inbox. Producer finalization validates committed artifacts and
the current destination, constructs immutable references, and commits the
producer outcome and unique deliveries atomically before any UI wake.

The Daita host (`daita host serve`, `src/daita/host/`) opens every agent that
has background work (an active or due routine, an unfinished job, an
unacknowledged delivery) as an ordinary `EmbeddedAgent`, so routines and jobs
progress while the host runs, whether or not a TUI is open. It reuses the
existing runtime and state store and is each home's only writer while it holds
the agent. Clients start it on demand; `daita host install` registers it as a
login service (launchd or systemd) so it restarts after a crash and at login.
`daita host --agent <name>` is deprecated: it prints a notice and runs
`daita host serve`.
