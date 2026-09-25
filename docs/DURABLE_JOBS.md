# Durable jobs

Daita can carry an investigation beyond one conversation turn. A durable job
records its tasks, dependencies, attempts, results, and blockers in the agent
home. Work can resume after the host reopens, but it progresses only while an
agent host is open.

Ask Daita to profile admitted data without changing it, or to start a bounded
graph job for a larger objective. `start_data_profile` creates a fixed,
read-only profile. `start_graph_job` starts one eligible task and a finalizer;
the job can add bounded tasks when planning is authorized. A model route and
finite cost ceiling are required for model-driven jobs.

The job's original objective, allowed sources and actions, outcome, deadline,
distribution, and total budgets stay fixed. Added tasks can use only a
validated subset of that authority. Source changes, expired contracts, budget
limits, and requests for human input may leave a task needing attention. Daita
does not treat a model's description of permission as a grant.

## Review progress and results

Use `/jobs` in the terminal app to list jobs, inspect tasks and blockers,
review controls, read results, or cancel a job. The Python API provides
`list_jobs`, `inspect_job`, `job_board`, `job_timeline`,
`read_job_result`, and `cancel_job`. The headless CLI can inspect the same
agent-owned records:

```bash
daita jobs list atlas
daita jobs inspect atlas <job-id>
```

A successful job publishes through its approved delivery plan. Its result is
also available by job ID. The originating conversation records where the job
began; agent identity controls who can inspect it. Scheduled routines are a
separate system for work due at specific times.

## External actions in jobs

Most graph work has no external effects. An effectful initial task needs
foreground approval and one exact grant. The supported effects are preview-bound
PostgreSQL updates or upserts and specifically admitted synchronous MCP
actions. A write task must run its frozen preview before applying it once.
An MCP task is bound to the exact admitted server, tool, arguments, and one
call. Neither kind is retried after uncertain dispatch.

If a receipt cannot be attached to an accepted task result, Daita blocks the
task and its descendants with an `effect_uncertain` control. Investigate the
[receipt](EFFECT_RECEIPTS.md) before authorizing future work. See
[relational writes](RELATIONAL_WRITES.md) and
[MCP actions](MCP_CONNECTIVITY.md) for the exact admission limits.
