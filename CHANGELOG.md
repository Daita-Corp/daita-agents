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
