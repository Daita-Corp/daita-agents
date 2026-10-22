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

Routine mutation tools return compact identity/revision/state, schedule and reservation
receipts; full contracts remain in inspection and enforcement. Stopped model runs
retain completed tool evidence and do not roll back committed routines. Terminal
notices and CLI summaries project that evidence without creating another durable
outcome or synthesizing a model answer.

`ScheduledRoutine` and `RoutineOccurrence` are the only scheduled-work records.
`RoutineOwner` admits exact foreground-authorized, self-contained instructions
and implements bounded create, list, inspect, update, pause, resume, run-now,
disable, expiry, budget, and lifecycle transitions. `RoutineSupervisor`
computes due slots, conditionally claims occurrences, fences stale work,
reserves one run, invokes `AgentLoop`, and atomically finalizes the occurrence
and its logical deliveries.

Routines support exact one-time, anchored interval, and IANA-timezone calendar
schedules with explicit daylight-saving gap/overlap behavior and bounded
`skip` or `latest_only` misfire handling. A routine freezes its exact source,
resource, MCP binding, capability, model-route, sensitivity, outcome,
distribution, budget, expiry, and optional retained skill-content contracts.
Raw prompt text never determines whether a time slot is due.

Scheduled execution permits statically declared `automation_direct` capabilities.
Native effects additionally require exact data-owned grants and runtime receipts;
effect-free operations retain read/none access. It can
create only these artifacts:

- `artifact.create_document`;
- `data.export_tabular`;
- `artifact.snapshot_result`;
- `analysis.execute` in the existing local macOS ARM64 composition, using the
  exact routine read scope and the same bounded native worker and artifact store.
  Its brokered reads own source permits; the interpreter holds no source permit.
  Scheduled analysis cannot use local computer files or retained artifact inputs.

`artifact.snapshot_result` produces bounded canonical `application/json` from
an exact earlier successful result in the same run. It performs no source I/O
or format projection.

Scheduled runs can perform one explicitly granted native update/upsert and admitted
MCP actions within their exact per-occurrence grants. They
cannot start or cancel jobs, manage routines, call
unadmitted remote actions, publish local files, deliver through external distribution destinations, run shell
commands, or submit workflows or execution graphs. An exact resource-revision
precheck may complete an unchanged occurrence without a model call.

`OutcomeContract` validates terminal conclusions and required artifacts.
`DistributionPlan` freezes ordered target bindings. `Delivery` is the sole
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
