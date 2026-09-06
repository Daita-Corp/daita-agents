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

Admission validates and retains the exact agent, conversation, source,
resource, MCP binding, capability contract, model route, sensitivity, budget,
expiration, typed outcome contract, and immutable inbox distribution plan. Only
capabilities statically declared
`automation_direct` can enter the ceiling. Effectful capabilities additionally
require a domain-normalized `CapabilityGrant`, an exact call ceiling, a receipt
policy and an `EffectRequirement`. Routine-management capabilities remain
interactive-only. Source-free assignments use empty source/resource/binding
ceilings; those empty machine ceilings never mean all currently admitted sources.
Native and MCP unattended actions are enabled only after their concrete adapter
conformance gates; the common routine contract does not confer connector authority.
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
card.

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
`daita routines --help` for the command surface.

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
