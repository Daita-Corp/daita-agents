# Durable jobs

Daita can carry an investigation beyond one conversation turn. A durable job
records its tasks, dependencies, attempts, results, and blockers in the agent
home. Work can resume after the host reopens, but it progresses only while an
agent host is open.

Ask Daita to start a bounded graph job for a larger objective.
`start_graph_job` starts one eligible task and a finalizer; the job can add
bounded tasks when planning is authorized. Profiling a table needs no job:
Daita answers with `data_query` aggregates (row counts, distinct counts, null
counts, minimum and maximum) over the whole table. Profile jobs saved before
the revision-2 home upgrade still run through their internal profile task and
finalizer, and their results stay readable with the job tools. A model route is
required for model-driven jobs. Like turn limits in other agent harnesses, each
model task is bounded by its step (model-turn) limit, the host per-run token
ceiling, its wall time and three attempts. Graph jobs carry no dollar ceiling,
so unpriced subscription routes and repeated work can run; a host-wide
estimated-cost limit, when you configure one, still applies to each run.

The job's original objective, allowed sources and actions, outcome, deadline,
distribution, and total budgets stay fixed. Added tasks can use only a
validated subset of that authority. Source changes, expired contracts, budget
limits, and requests for human input may leave a task needing attention. Daita
does not treat a model's description of permission as a grant.

## Host parallelism

`AgentConfig(host_capacity=HostCapacityLimits(graph_parallelism=4))` opts into at
most four concurrent graph attempts in the open host; the default is one. Host
execution and provider capacity still apply, including strict foreground reserves.
The supervisor's model-attempt gate uses the same graph capacity. Every SQLite
claim also enforces the job's immutable root parallelism limit. Higher host
capacity cannot enlarge a job's authority, budgets or parallelism contract.
Capacity changes require drain and reopen and are not persisted as grants.

## Review progress and results

Use `/jobs` in the terminal app to list jobs, inspect tasks and blockers,
review controls, read results, or cancel a job. The Python API provides
`list_jobs`, `inspect_job`, `job_board`, `job_timeline`,
`read_job_result`, and `cancel_job`. `work_board` shows queued, running,
blocked and recently finished jobs beside scheduled routines and their results.
The headless CLI can inspect the same agent-owned records:

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
