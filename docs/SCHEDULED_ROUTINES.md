# Scheduled routines

A routine is a title, a prompt and a schedule. Optionally it names the sources
to use. Each time it is due, Daita runs the agent on the prompt exactly as if
you had asked it, with the agent's current model, and puts the answer in the
conversation's Inbox. Reads just run; any write or external action asks you at
the time. The background host must be running for routines to progress (see
[The background host](#the-background-host)).

## What a routine holds

| Field | Meaning |
| --- | --- |
| `title` | A short name shown on the work board and approval cards. |
| `prompt` | A self-contained instruction for each run. Name sources and tables by their catalog names, not by a run's `s1`/`t1` handles. |
| `schedule` | When it runs (below). |
| `sources` | Optional source names or ids. Omitted means every current source. |
| `run_immediately` | Creation only: also run once now. Interval and calendar schedules only. |

Everything else is a fixed default, with no option to change it:

- **Delivery:** the routine's conversation Inbox. A routine created outside a
  conversation (from the CLI or Python without `conversation_id`) gets a new
  conversation of its own, which appears with its first result.
- **Missed slots are skipped.** One late slot still runs when the host wakes;
  a backlog of several missed slots does not, and the routine waits for its
  next future slot.
- **Run limits:** each run has the agent's normal step, time, token and cost
  limits and uses the agent's current model routes.
- **Failures:** a recurring routine pauses as **Needs you** after 3
  consecutive failed runs. Resuming it clears the count.
- **Lifetime:** a one-time routine ends (`completed`) after it runs; a
  recurring routine runs until you pause or disable it.

A routine has no budget, expiry date, grant or outcome contract of its own.

## Schedules

A schedule is one of:

- `once`: `exact_at` (ISO 8601 with an offset), or `after_seconds` from when
  it is saved;
- `interval`: `interval_seconds` (60 seconds to five years) from `anchor_at`,
  or from `anchor_after_seconds` after it is saved;
- `once_next_weekday`: the next `weekday` (Monday=1) at `hour`:`minute` in an
  IANA `timezone`;
- `calendar`: `hour`:`minute` in an IANA `timezone`, every day or on
  `weekdays` or `month_days`, optionally limited to `months`. Daylight-saving
  gaps are skipped and repeated hours use the first occurrence by default
  (`nonexistent_time_policy`, `ambiguous_time_policy`).

Relative timing is 1 second to 30 days. Daita turns it into exact UTC times
when the routine is saved; nothing about a stored routine is relative. Cron and
RRULE strings are not accepted. Calendar timezones must be exact IANA names;
abbreviations such as `CST` are refused.

In the TUI, the model is given the code-owned time at run start and your
detected local timezone, so "8 minutes from now" or "every weekday at 9" need
no clock lookup. A timezone you name explicitly wins.

## Creating and changing a routine

Ask for it in a conversation, for example: "Create a one-time routine 8 minutes
from now that writes a two-sentence note on why data quality matters." The
model calls `routine_save` once and Daita shows one confirm card with the
title, the schedule in your local time, the prompt and the sources. Approve it
and the routine is saved; the receipt shows its id, revision, state, schedule
and next run. Relative timing on the card is marked "timed from approval": it
counts from the moment you approve.

To change a routine, pass its `routine_id` with only what changes; omitted
fields keep their stored values. `expected_revision` is optional and defaults
to the current revision; when given and stale, the save is refused with
`routine_revision_changed`. A change shows the same confirm card. `routine_control`
pauses, resumes, runs now, or disables one routine.

A foreground run that stops after a routine was saved does not undo the save;
the terminal shows the saved receipt beside the stopped-run notice.

## Writes ask at the time

A scheduled run can read the routine's sources and use every capability except
local computer files and creating, changing or running routines and jobs. Its
scope is fixed when the run starts: a source added later is seen by the next
run, not the current one.

When a run wants to write (a PostgreSQL update or upsert, an admitted MCP
action), it pauses and the host opens an approval naming the routine. A desktop
notification appears if no window is watching; opening `daita` shows the card.
Approve and the run continues; deny and the model gets the refusal and carries
on. If nobody answers within 24 hours, the run is interrupted (not denied) and
the occurrence shows as Needs you. Every write is receipted before dispatch and
never retried or replayed. An uncertain outcome pauses the routine until you
resolve it; see [Effect receipts and recovery](EFFECT_RECEIPTS.md). Resume and
run-now are refused while such a receipt is unresolved.

A provider quota rejection fails that occurrence without counting toward the 3
failures, and the host leaves due slots unclaimed for a hold that starts at
five minutes, doubles up to one hour, and clears after a completed run.

## Terminal use

In the TUI:

```text
/routines
/routines create <what to do and when>
/routines update <routine-id> <what to change>
```

`/routines` opens the manager: the list, each routine's schedule, prompt,
sources, usage and recent runs, and pause, resume, run-now and disable
controls.

From the CLI:

```bash
daita routines list atlas
daita routines inspect atlas <routine-id>
daita routines create atlas --title "Data quality note" \
  --prompt "Write a two-sentence note on why data quality matters." \
  --schedule '{"kind": "once", "after_seconds": 480}'
daita routines create atlas --spec /absolute/path/routine.json
daita routines update atlas <routine-id> \
  --schedule '{"kind": "calendar", "timezone": "America/Chicago", "hour": 9, "minute": 0, "weekdays": [1, 2, 3, 4, 5]}'
daita routines pause atlas <routine-id> <revision>
daita routines resume atlas <routine-id> <revision>
daita routines run-now atlas <routine-id> <revision>
daita routines disable atlas <routine-id> <revision>
```

`create` and `update` accept `--title`, `--prompt`, `--schedule` (JSON),
repeated `--source`, and `--spec` (a JSON file with the same fields; flags
override it). `create` also takes `--run-immediately`, and `update` takes
`--expected-revision`. Both show the confirm card and ask before saving. A spec
file looks like:

```json
{
  "title": "Daily paid revenue",
  "prompt": "Report paid revenue by region from the revenue table.",
  "schedule": {"kind": "calendar", "timezone": "America/Chicago", "hour": 9, "minute": 0, "weekdays": [1, 2, 3, 4, 5]},
  "sources": ["warehouse"]
}
```

From Python, the same shape is `RoutineIntent`:

```python
from daita import Agent, RoutineIntent

agent = await Agent.open("atlas")
routine = await agent.create_routine(
    RoutineIntent(
        title="Daily paid revenue",
        prompt="Report paid revenue by region from the revenue table.",
        schedule={"kind": "calendar", "timezone": "America/Chicago", "hour": 9, "minute": 0},
        sources=("warehouse",),
    ),
    conversation_id=None,  # or a conversation whose Inbox should receive results
)
await agent.update_routine(routine.routine_id, RoutineIntent(prompt="Also list refunds."))
```

`schedule` takes the JSON shape above or a typed `OnceSchedule`,
`IntervalSchedule` or `CalendarSchedule`. Pass `confirmation_handler=` to review
the confirm card before saving; without it, a typed Python call is the owner's
own decision.

## Results on the work board

The TUI's Work tab (`Ctrl+3` or `/work`) is a board over jobs, routines and their
results. A routine's successful result sits in Done for 24 hours, then
ages off the board. A failed or withheld result stays in Needs you until you
dismiss it. Press `Enter` on a result to read it, and `d` (or the result's
Dismiss button) to dismiss it. Dismissing acknowledges the delivery; it never
reruns the work. Notices about jobs are not listed separately: the job's own card
moves to Needs you while it waits on you. The headless equivalents are:

```bash
daita work show atlas [--done-hours 24]
daita work inspect atlas <delivery-id>
daita work dismiss atlas <delivery-id>
```

`work show` prints the work board as JSON. Finished results stay in Done for
`--done-hours` (1 to 168, default 24). `work dismiss` acknowledges one result.

From Python, `Agent.list_deliveries`, `Agent.inspect_delivery` and
`Agent.acknowledge_delivery` read and dismiss the same records.
`Agent.list_deliveries` also covers what the board leaves out: one
conversation's results and already acknowledged history.
`Agent.work_board()` shows them beside jobs and routines in Scheduled, Running,
Needs you and Done columns; it is not yet available to hosted callers.

A failed run's Inbox item is a code-written notice with its failure code and
any receipt and artifact references, never the model's own claim of success.
Acknowledgment is idempotent. Unacknowledged routine results are never evicted.
When the fixed per-agent Delivery bound is full, a producer's finalization
transaction reclaims the oldest acknowledged Delivery first, then the oldest job
result (the job record keeps the result), then the oldest job notice whose job no
longer waits on the user. If nothing is reclaimable, finalization remains atomic
and pending rather than losing the outcome; acknowledgment wakes both producer
drivers so pending work can converge immediately.

A Delivery stores a bounded immutable artifact reference rather than another
copy of the full artifact manifest. Finalization verifies it against the artifact
store. The independent artifact registry owns current availability and retains
ownership after transcripts are cleared. Explicit artifact deletion preserves the
historical Delivery but makes its attachment unavailable.

## The background host

Routines run in the background Daita host. `daita` and the CLI start it on
demand, and it keeps running after you close the terminal. To keep it running
after a crash and across logins, register it as a login service:

```bash
daita host install
```

`daita host status` shows whether it runs and which agents it holds;
`daita host stop` stops it. It handles `SIGINT` and `SIGTERM`, finishes or
cancels active runs, and closes every agent before exiting. It uses the same
writer lock as a script's `Agent.open()`, which is refused while the host holds
that agent. `daita host --agent <name>` is deprecated and now runs the host.
There is no cloud execution or host availability service.

If a host stops, persisted routines and occurrences remain inspectable but no
new due work runs. On reopen, Daita fences stale claims, finalizes a run that
was already durably terminal without rerunning it, and preserves exactly one
logical inbox delivery.

## Routines from earlier versions

Upgrading a home rewrites each stored routine to this shape. It keeps its
title, prompt, schedule, state, counters and named sources (its former source
ceiling; an empty one now means every current source) and keeps running. Its
frozen access, grants, model routes, sensitivity ceiling, budgets, expiry,
outcome contract, precheck, pinned skills and promotion evidence are dropped,
and any write it makes now asks at the time. An occurrence that was mid-precheck
continues as an ordinary claim; one skipped as unchanged counts as completed.
