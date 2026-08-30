# Scheduled read routines and D2 outcomes

Daita D2 scheduled routines repeat bounded read-only agent work without adding
a second loop, runtime, or state owner. A routine freezes one exact
self-contained instruction and executes each admitted occurrence through the
ordinary `AgentLoop` and `CapabilityRuntime`. Each terminal occurrence converges
atomically with one immutable logical Delivery in the originating
conversation's durable inbox.

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
`scheduled_direct`, with `OperationalEffect.NONE` and read/none data access,
can enter the ceiling. Routine-management capabilities remain interactive-only.
If an artifact is required, admission also proves that at least one allowed
producer can satisfy its media type, authorship, exact-source, sensitivity, and
byte bounds. Impossible contracts fail before the routine is created.

The certified scheduled artifact surface is deliberately small:

- `artifact.create_document` for model-authored text or Markdown;
- `data.sqlite.export_tabular` and `data.postgresql.export_tabular` for exact
  source-data CSV or XLSX;
- `artifact.snapshot_result` for canonical JSON from an exact earlier
  successful result in the current scheduled run.

Artifact inventory, reading, conversion, editing, local publication, and
export-location capabilities remain interactive-only.

An optional `changes_only` resource-revision precheck runs through the ordinary
trusted runtime request. When its canonical observation is unchanged, the
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

D2 cannot write data, start or cancel a durable job, create or manage another
routine, call a remote write tool, deliver externally, run shell commands, or
submit a graph. Email/external delivery, recurring ingestion, and graphs are
not implemented.

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
  "allowed_capability_ids": ["catalog.inspect", "data.sqlite.query"],
  "sensitivity_ceiling": "internal",
  "outcome_contract": {
    "require_terminal_conclusion": true,
    "artifact_requirements": [],
    "maximum_total_artifact_bytes": 0,
    "maximum_effective_sensitivity": "internal",
    "require_current_run_provenance": true,
    "require_exact_source_bindings": false
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
pricing, outcome, destination, or skill binding is unavailable. Use `daita routines
--help` for the command surface.

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
host, open the other process, then restart the resident host. D2 adds no IPC,
remote API, transparent client gateway, multi-host queue, or competing writer.

If a host stops, persisted routines and occurrences remain inspectable but no
new due work runs. On reopen, Daita fences stale claims, finalizes a run that
was already durably terminal without rerunning it, and preserves exactly one
logical inbox delivery.

## Optional live-model certification

The deterministic suite remains authoritative for scheduling, recovery,
fencing, budgets, and exactly-once finalization. Three opt-in acceptance tests
additionally certify the boundaries where real model behavior matters. They
cover the ordinary scheduled model-to-tool path, model correction after one
injected model-visible read failure, and resistance to instruction-like text
returned as untrusted database data. Every test commits exactly one grounded
report to the originating conversation inbox.

The tests are skipped unless explicitly authorized. The complete module makes
at most three live scheduled loop runs. Each run has its own token, wall-time,
step, and estimated-cost ceiling; with the documented default, the complete
module's maximum estimated-cost ceiling is `$0.30`:

```bash
DAITA_RUN_LIVE_STAGE_D1_ROUTINES=1 \
DAITA_STAGE_D1_LIVE_LLM_API_KEY=<provider-api-key> \
DAITA_STAGE_D1_LIVE_MAX_COST_USD=0.10 \
.venv/bin/python -m pytest \
  tests/live/test_stage_d1_routines_live.py -v -s
```

`DAITA_STAGE_D1_LIVE_MODEL_ID` defaults to `openai:gpt-5.6-terra` and may be
set to another release-reviewed, API-backed, tool-capable streaming model. The
tests do not run as part of the default deterministic suite. Running one test
by node ID authorizes only its one scheduled loop; running the complete module
authorizes all three. Their presence does not claim a live certification result.

One separately authorized D2 acceptance test certifies the frozen SQLite CSV
outcome and logical Delivery path. It permits at most one live scheduled loop
with a default `$0.10` estimated-cost ceiling:

```bash
DAITA_RUN_LIVE_STAGE_D2_OUTCOME=1 \
DAITA_STAGE_D2_LIVE_LLM_API_KEY=<provider-api-key> \
DAITA_STAGE_D2_LIVE_MAX_COST_USD=0.10 \
.venv/bin/python -m pytest \
  tests/live/test_stage_d2_outcomes_live.py -v -s
```

`DAITA_STAGE_D2_LIVE_MODEL_ID` defaults to `openai:gpt-5.6-terra`. This test is
also skipped unless explicitly authorized and does not itself claim that a
live certification run has been performed.
