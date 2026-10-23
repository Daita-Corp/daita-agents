# Durable jobs, scheduled routines and deliveries

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

Machine-originated work: graph jobs, time-triggered routines, outcomes and deliveries.

## Job graph boundaries

The job graph, introduced in home revision 2, keeps these boundaries. Changing
one is an architecture change:

- the adaptive task graph is owned inside `jobs`; `JobOwner` owns commands and one
  graph-aware `JobSupervisor` owns graph-local selection, claims and recovery;
- there is still one reentrant `AgentLoop` semantic implementation and one
  `CapabilityRuntime`; each model task attempt has a fresh isolated `RunSession`,
  exact transcript and one `RunSessionWriter`;
- `RunAdmissionCoordinator` owns host workload admission, lifecycle drain and keyed
  provider/source/MCP/SQLite/effect permits; it does not own graph transitions,
  model routing or capability semantics;
- root authority, outcome, distribution, deadlines and total budgets are immutable;
  planner-created task scope is a validated subset and model text never grants
  authority;
- graph effects are narrowly released only for preview-bound PostgreSQL update/upsert
  and specifically admitted synchronous MCP actions under exact code-owned grants;
  all other graph work remains effect-free and the existing receipt boundary remains
  the only effect truth;
- graph transactions are short indexed SQLite CAS operations. Provider, source, MCP,
  artifact and effect I/O never occurs beneath a graph transaction or broad host
  lock;
- normalized tasks, edges, attempts, accepted results, controls, mutations, budget
  ledgers and bounded audit events are current graph state/evidence. Events are never
  event-sourced authority;
- routines remain the separate time-triggered occurrence system; a routine is not a
  graph task and no routine-to-graph bridge is part of graph V1;
- revision 2 is one atomic conversion/cutover. Current runtime codecs contain only
  the graph shape afterward; revision-1 decoders remain migration-only; the old
  single-job, connected-executor and autonomous-follow-up paths are deleted at the
  specified gates rather than retained as a selectable compatibility engine.

## Durable graph jobs

`GraphJob`, `JobGraph`, normalized tasks, dependencies, attempts, accepted results,
controls, mutations, bounded events and budget ledgers are the sole current durable
job records. `JobOwner` owns admission and bounded lifecycle commands. One
graph-aware `JobSupervisor` owns graph-local selection, claims, fencing, recovery,
fairness and finalization; all capability work still executes through the one
`CapabilityRuntime`, and every model task uses the one `AgentLoop` implementation.

No model tool starts a profile job: models profile data with `data_query`
aggregates. The static effect-free profile graph keeps its internal executor and
finalizer (`jobs.data_profile.execute`/`.finalize`) so migrated revision-2 profile
jobs still run. `start_graph_job`
admits one code-resolved graph-eligible initial task and internal finalizer when a
model route is configured. Graph work is bounded by per-task steps, the host
per-run token ceiling, wall time and `work_units` attempts; graph contracts and
graph-task scopes carry no dollar ceiling, and only graph-task scopes may omit
one. Planner-created work is a validated immutable-authority subset. Unmappable
migrated work is `needs_attention`; no legacy executor can run it.

An effectful initial task additionally requires foreground approval and one
code-normalized exact grant. Native grants bind the exact PostgreSQL source,
resource revision, operation, columns, keys, identity generation and row ceiling;
the task must execute its frozen preview before applying it once. MCP grants bind
the exact active binding revision, remote tool and fixed nested JSON, with only
explicit top-level scalar variables, `AUTOMATION_DIRECT`, `DIRECT_RESULT`, no
required asynchronous task support and one call. Effect execution holds the global
`effect:v1:global` lane plus `relwrite:<source>:<resource>` or
`mcp:<agent>:<binding>`.

Receipt reservation atomically authenticates the live job, task, attempt, claim,
fencing epoch, task-spec digest and grant before dispatch. A task result may retain
only its authenticated successful receipt reference. Any reserved operation that
cannot be joined to an accepted task result blocks the task and descendants through
a typed `effect_uncertain` control; startup and cancellation reconciliation never
replay it. Human receipt resolution remains immutable evidence and does not retry,
compensate or mark the task successful.

Agent identity is the job authorization boundary. The originating
conversation and run are immutable provenance, not access gates. Bounded list,
inspect, result-read, and cancellation operations can address any job owned by
the agent. Cross-agent lookup fails without exposing metadata. Work progresses only
while a process holds the agent open; normally that is the Daita host.

There is no current connected-executor mode, embedded single-job attempt/result
state, autonomous-follow-up driver or selectable legacy supervisor. Finalizers
authenticate accepted task results and publish through the existing delivery
boundary. Revision-1 terminal follow-up/delivery provenance is preserved only by
migration-owned conversion.

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
outcome (any artifacts within the shared delivery bounds); graph jobs keep
their `OutcomeContract`. `DistributionPlan` freezes ordered target bindings. `Delivery` is the sole
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
