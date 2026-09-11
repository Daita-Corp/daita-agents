# Scheduled assignments and outcomes

Daita scheduled routines perform bounded assignments once or on a recurring schedule. A routine freezes
one exact self-contained instruction and executes each admitted occurrence
through the ordinary `AgentLoop` and `CapabilityRuntime`. Each terminal
occurrence converges atomically with one immutable logical `Delivery` in the
originating conversation's durable inbox.

## Supported schedules

A routine uses one typed schedule:

- `once` at one exact UTC instant;
- `interval` anchored to one exact instant, with intervals from 60 seconds up
  to five years; or
- `calendar` at an exact local hour and minute in an IANA timezone.

Calendar definitions reject ambiguous abbreviations such as `CST`. They retain
an explicit daylight-saving gap policy (`skip` or `next_valid`) and overlap
policy (`first` or `second`). Missed schedules use bounded `skip` or
`latest_only` handling; Daita never replays an unbounded backlog. Scheduled and
manual run-now slots have stable occurrence identities, so duplicate ticks and
host restarts converge on the existing occurrence.

## Frozen authority

`toolbox_search` exposes exact capability IDs, automation eligibility, and the
domain-owned `automation_contract` for tools requiring grants: constraints schemas,
connector references, and evidence bases. `requires_automation_grant` refers to
scheduled execution; foreground actions request exact approval when invoked.
Load an execution tool when its exact argument schema is needed to author fixed
or variable grant arguments. The generic grant schema alone does not define an
MCP action's argument names or types. If a contract exceeds the bounded discovery
page, `automation_contract_omitted` directs the model to `toolbox_load` for its
complete declaration. Load also returns these contracts for selected tools.
These declarations support proposal authoring; they do not grant execution
authority. Reads need capability inclusion, without
an effect grant. The routine tool declares the finite once, interval, and calendar
schedule shapes, including calendar day selectors and daylight-saving policies.
Cron and RRULE strings are not accepted schedule alternatives. Resource prechecks
require exact capability, contract, source, and resource references; an MCP binding
revision is not a resource precheck.

`always` reporting requires omitting the precheck; `changes_only` requires one.
Prechecks track structural revisions, not changing row values, and cannot skip
effectful or MCP assignments. Creation and revision validate these rules before
grant preparation or execution-contract binding and return actionable errors.
Admission rechecks the same rules; drafts and persisted records share their
reporting/precheck invariant.

Per-run token and estimated-cost ceilings must not exceed their corresponding
cumulative ceilings. Drafts and stored routines share that invariant. The routine
owner rejects inconsistent create/revision proposals with `routine_budget_invalid`
before grant preparation, connector readiness, contract binding, approval or
mutation. The error names both fields so the next ordinary model step can submit
the user's exact authorized budgets. Daita never repairs or enlarges them. The
remaining foreground allowance is distinct from a future routine's budgets.

Admission validates and retains the exact agent, conversation, source,
resource, MCP binding, capability contract, model route, sensitivity, budget,
expiration, typed outcome contract, and immutable inbox distribution plan. Only
capabilities statically declared
`automation_direct` can enter the ceiling. Effectful capabilities additionally
require a domain-normalized `CapabilityGrant`, an exact call ceiling, a receipt
policy and an `EffectRequirement`. Routine-management capabilities remain
interactive-only. Source-free assignments use empty source/resource/binding
ceilings; those empty machine ceilings never mean all currently admitted sources.
Native update/upsert and explicitly admitted direct-result MCP actions have concrete
domain grant contracts; the common routine contract does not confer connector
authority. MCP grants fix exact top-level arguments and allow only named scalar
variables, with 1–256 reserved calls per occurrence further narrowed by run limits.
MCP completion evidence is server-reported invocation only. See
[MCP admission and actions](MCP_CONNECTIVITY.md) for the supported subset.
If an artifact is required, admission also proves that at least one allowed
producer can satisfy its media type, authorship, exact-source, sensitivity, and
byte bounds. Impossible contracts fail before the routine is created.

The certified scheduled artifact surface is deliberately small:

- `artifact.create_document` for model-authored text or Markdown;
- `data.export_tabular` for exact source-data CSV or XLSX;
- `artifact.snapshot_result` for canonical JSON from an exact earlier
  successful result in the current scheduled run.

Artifact inventory, reading, conversion, editing, local publication, and
export-location capabilities remain interactive-only.

An optional `changes_only` resource-revision precheck runs through the ordinary
trusted runtime request. It is restricted to one exact structural catalog/resource
revision and the supported catalog/report tools. It cannot skip row-value research,
effects, MCP calls, source-free work or broader resource assignments. When its canonical observation is unchanged, the
occurrence advances with zero model calls and one no-change Delivery. `always`
routines do not use that precheck.

Failures before a model run starts release the reserved occurrence budget and
advance the routine without inventing a run record. When consecutive failures
reach the configured threshold, the routine moves to `needs_attention` and the
same atomic finalization inserts one escalation in the conversation inbox. The
item explicitly reports that no model run started. Terminal-run failures use
their existing conclusion as the one escalation, so an occurrence never
creates separate report and escalation deliveries.

Finalization records all known consumption, even when it exceeds a reservation.
Incomplete usage conservatively consumes at least the full reservation, while the
run retains its partial or unavailable estimate. Such a run cannot produce a
successful outcome. A routine without enough cumulative allowance for another run
moves to `needs_attention`; resume and run-now cannot bypass that budget. There is
no extra model call after the run reaches a step, token, or cost limit.

Skills are optional and exact. Admission copies the current validated
`SKILL.md` bytes into the existing SkillStore's bounded content-addressed
retention area. Later edits or deletion of the current skill do not change the
routine; missing or digest-mismatched retained bytes fail closed. Skill text,
source values, MCP metadata and output, precheck observations, and prior
transcript content remain untrusted data and cannot expand authority.

Scheduled execution cannot start or cancel a durable job, manage another routine,
publish local files, deliver through an external Distribution destination, run
shell commands, or submit a graph. An admitted effect capability uses its existing
domain and the common runtime, within an exact standing grant.

## Immediate execution and effect outcomes

For an interval or calendar schedule, `run_immediately=true` creates one immediate
occurrence in the same SQLite transaction as the assignment. That occurrence uses
the ordinary supervisor, run budget and action limits. The original calendar or
interval anchor retains its next future slot. Approval covers this one assignment;
there is no separate foreground write and no second immediate occurrence on a
retry with the same creation identity. Revisions use explicit run-now instead.

Model-authored creation must explicitly supply `run_immediately`: `false` means
scheduled-only (and is required for a one-time schedule). Omitting the choice
fails validation. Model-authored revisions may omit it or supply `false`; `true`
is rejected. Typed Python owner inputs keep their default of `false`. Review the
actual saved schedule and immediate choice; explicit syntax alone cannot verify
that the model understood the requested timing.

Every permitted effect has one completion requirement. A positive minimum requires
that many unique successful, validated invocations; zero explicitly permits no
action. Approval retains the accepted evidence bases and rejects stronger evidence
than the capability can produce. Server-reported invocation evidence does not
establish a downstream business result. A persuasive answer or artifact cannot
substitute for a required action.

The finalizer loads authoritative receipts for the exact run, occurrence, grant
and capability contract and authenticates their successful tool results. It keeps
valid partial artifacts if another action or artifact minimum fails. An uncertain
or started effect forces failure even on an optional path. The inbox then shows a
code-authored failure notice with receipt and artifact references, rather than
presenting the model's completion claim as success.

Uncertainty immediately pauses the originating routine. Resume, run-now, changed
arguments, revised grants and new effectful clones cannot bypass its durable block.
Previously approved unrelated routines and read-only work can continue. Exact
foreground recovery records a separate immutable resolution and never dispatches
an action: close-without-retry disables the routine; allow-future-work leaves it
paused until explicitly resumed. A delayed finalizer preserves that stop decision.
See [Effect receipts and recovery](EFFECT_RECEIPTS.md).

## Creation receipts and stopped model runs

The model-facing create, update, and control tools return a compact, validated
receipt for the committed mutation: action, routine ID/revision/state, typed
schedule, next due time, and reservation counts. Reservation counts record admitted
occurrences and attempts; they do not establish execution success. Full instructions,
grants, contract bindings, budgets, and distribution remain available through
`Agent.inspect_routine` and `routine_inspect` and remain the enforcement records.

A foreground model run can stop after a routine was committed. Its failed or
interrupted status does not roll back creation or disable the routine. The terminal
retains this distinction, including after reopening the conversation, and displays
recorded receipt facts alongside the stopped-run notice. Headless `daita run` output
includes bounded tool-result summaries and call IDs without repeating tool arguments
or full authorization records. The public `Transcript.tool_pairs` projection exposes
the original ordered call/result evidence; a missing result remains unknown.
Scheduled execution outcomes continue to appear separately in the Inbox.

## Terminal use

Open the record-backed manager in the Textual app:

```text
/routines
/routines create <self-contained instruction>
/routines promote <basis-run-id> <self-contained instruction>
/routines update <routine-id> <self-contained instruction>
```

The manager lists at most 50 routines and shows the exact schedule, instruction
digest and preview, scope, pinned skills, budget use, next due time, recent
occurrences, failures, expiration, revision, and lifecycle state. Its pause,
resume, run-now, and disable controls call the public `Agent` methods directly.
Create and update use the normal foreground routine tools and existing approval
card. The status bar counts saved assignments separately from active background
reasoning. A saved or queued assignment is not completed work. Receipt-linked
failures remain inspectable through `/effects`; recovery is a separate human
control and never performs an action.

The headless CLI exposes the same record-owned lifecycle:

```bash
daita routines list atlas
daita routines inspect atlas <routine-id>
daita routines pause atlas <routine-id> <revision>
daita routines resume atlas <routine-id> <revision>
daita routines run-now atlas <routine-id> <revision>
daita routines disable atlas <routine-id> <revision>
daita routines create atlas --spec /absolute/path/routine.json
daita routines promote atlas --spec /absolute/path/routine.json \
  --basis-run-id <completed-run-id>
daita routines update atlas <routine-id> <revision> \
  --spec /absolute/path/routine.json
```

The JSON specification contains the foreground origin run and complete frozen
definition. For example:

```json
{
  "origin_run_id": "run-...",
  "title": "Daily paid revenue",
  "authorized_instruction": "Read the exact admitted revenue resource and report paid revenue by region.",
  "schedule": {
    "kind": "calendar",
    "timezone": "America/Chicago",
    "hour": 9,
    "minute": 0,
    "day_selector": "weekdays",
    "weekdays": [1, 2, 3, 4, 5],
    "month_days": [],
    "months": [],
    "nonexistent_time_policy": "next_valid",
    "ambiguous_time_policy": "first"
  },
  "misfire_policy": "latest_only",
  "reporting_mode": "always",
  "precheck": null,
  "allowed_source_ids": ["source:..."],
  "allowed_connector_binding_ids": [],
  "allowed_resource_ids": ["catalog-resource:..."],
  "allowed_capability_ids": ["catalog.inspect", "data.query"],
  "sensitivity_ceiling": "internal",
  "outcome_contract": {
    "require_terminal_conclusion": true,
    "artifact_requirements": [],
    "maximum_total_artifact_bytes": 0,
    "maximum_effective_sensitivity": "internal",
    "require_current_run_provenance": true,
    "require_exact_source_bindings": false,
    "effect_requirements": []
  },
  "distribution_destination_id": "conversation_inbox:conversation-...",
  "eligible_model_routes": ["openai"],
  "per_run_max_tokens": 4000,
  "per_run_max_cost_usd": "0.10",
  "cumulative_max_tokens": 120000,
  "cumulative_max_cost_usd": "3.00",
  "cumulative_max_attempts": 30,
  "cumulative_max_occurrences": 30,
  "maximum_consecutive_failures": 3,
  "expires_at": "2027-08-28T00:00:00+00:00",
  "skill_names": []
}
```

The IDs and model routes must already be admitted to that agent. Create and
update fail closed if any identity, capability contract, sensitivity, route,
pricing, outcome, destination, or skill binding is unavailable. Use
`daita routines --help` for the command surface. CLI creation, promotion and
revision present the exact validated proposal before saving it. Python callers
can pass `confirmation_handler=` to create/update for that review. Without this
explicit confirmation callback, typed Python create/update calls are already
owner-authorized control operations. Model-originated changes always use the
runtime approval handler.

The review summarizes the instruction, schedule, immediate occurrence, budgets,
permissions and completion requirements, followed by the complete validated
authority snapshot. Connector names are presentation; exact identities and
digests remain visible. An approval cannot add a connector permission or promise
stronger evidence than the selected producer supports. Model cost ceilings do not
cap third-party service fees.

## Inbox lifecycle and retention

`/inbox` opens the bounded product projection. The headless equivalents are:

```bash
daita inbox destinations atlas <conversation-id>
daita inbox list atlas
daita inbox inspect atlas <delivery-id>
daita inbox acknowledge atlas <delivery-id>
```

Acknowledgment is idempotent. Unacknowledged available or blocked Deliveries
are never evicted. When the fixed per-agent Delivery bound is full, producer
finalization remains atomic and pending rather than losing the outcome. Once a
Delivery is acknowledged, its oldest retained history entry can be reclaimed
inside a later producer finalization transaction; acknowledgment wakes both
producer drivers so pending work can converge immediately.

A Delivery stores a bounded immutable artifact reference rather than a second
copy of the full artifact manifest. Artifact reads resolve the existing
canonical manifest and require every projected identity, digest, provenance,
authorship, sensitivity, and size fact to match. This lets terminal transcripts
be cleared without losing an artifact that is still rooted by a retained
Delivery.

## Resident host and handoff

Schedules make progress only while an admitted host owns the agent. To keep
one agent open after the TUI exits:

```bash
daita --root /absolute/daita-state \
  --workspace /absolute/path/project \
  host --agent atlas
```

The process reports a JSON readiness record, handles `SIGINT` and `SIGTERM`,
and closes the ordinary supervisors and agent composition before exiting. It
uses the same writer lock as every foreground open. A TUI, CLI invocation, and
resident host cannot own the same agent home concurrently: stop the current
host, open the other process, then restart the resident host. The resident host
does not add IPC, a remote API, a transparent client gateway, a multi-host
queue, or a competing writer.

A headless control command holds its host only for the command lifetime. The TUI
shows its local host as open and identifies background reasoning separately from
foreground activity. Both share one run lock, so one can delay the other. There is
no continuous cloud execution or host availability service.

If a host stops, persisted routines and occurrences remain inspectable but no
new due work runs. On reopen, Daita fences stale claims, finalizes a run that
was already durably terminal without rerunning it, and preserves exactly one
logical inbox delivery.

Each approved revision retains `contract_bindings`: exact capability contracts,
MCP execution origins, structural resource revisions and non-secret model route
configuration digests. The occurrence copies those references into its immutable
scope. Claim and call checks compare current facts with those retained references.
A structural or execution contract change requires a new approved revision; row
values, catalog refresh time and editable discovery hints do not change authority.


## Native update and upsert assignments

The data domain supports exact native update/upsert grants, initially backed by
PostgreSQL. One routine may contain one native write capability, with exactly one
invocation per occurrence and explicit resource, structural revision, key, column,
identity and row ceilings. Update authority never grants insertion. Read the
[relational write contract](RELATIONAL_WRITES.md) before configuring an assignment.

Immediate-first creation produces one immediate occurrence while retaining the
approved interval/calendar schedule. Required effects are checked against current-run
validated results and authenticated receipts. An unchanged upsert consumes the call
and reports zero row changes. Missing effects, transaction failures and uncertainty
cannot become a successful outcome through model text. Unknown commits block further
potentially duplicating work, including future slots and run-now, pending explicit
foreground recovery. No automatic replay or chunking is supported.

This native path is a development implementation with deterministic acceptance;
production release gates and the separate shared MCP external-action work remain.

Authoring and revision validate exact eligible model routes before connector
readiness or contract binding. Invalid choices return a structured error; current
contract and approval rechecks still reject drift. The model sees complete local
route choices and host budget ceilings while an authoring tool is callable.

Native write admission requires the corresponding read-only preview capability in
the same proposal: `data.preview_update_rows` for `data.update_rows`, or
`data.preview_upsert_rows` for `data.upsert_rows`. Admission does not perform or
authorize a future preview; every occurrence must obtain its own authenticated
preview. Structural discovery is separate. A natural assignment that needs table
structure must explicitly request bounded `catalog.schema` access to its approved
source/table. Daita never adds that permission automatically.
