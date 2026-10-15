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

For an explicit durable comparison of whole-source row totals, use
`start_count_comparison`. Its model view accepts two through sixteen distinct
`source_ids` with `coverage="complete_admitted_base_tables"`. The data owner resolves
and freezes the required set without catalog pagination or model-visible table
enumeration. Alternatively, supply complete current-run catalog identity
`selection_refs`. These forms are mutually exclusive. Both accept optional
`deadline_seconds` (60–300, default 300). Each source's admitted tables/views must
fit the 50-resource catalog bound. Execution excludes
views and requires at least one base table per source. Ordinary questions remain
foreground requests; Daita does not automatically create jobs.

Typed callers name the required sources directly, without transcript references:

```python
job = await agent.start_count_comparison(
    source_ids=(billing_source.id, application_source.id),
    deadline_seconds=300,
)
inspection = await agent.inspect_job(job.job_id)
result = await agent.read_job_result(job.job_id)  # None until successful finalization
```

An empty source tuple is invalid. Supplied conversations must exist and pass the
ordinary caller checks; omission creates an inbox identity for this job. Current
read permissions, source/resource revisions and sensitivity still apply. This
static path works without a model route or model cost configuration.

Admission freezes required coverage separately from the authorization ceiling.
The complete plan must fit sixteen total batches of at most four resources;
sixteen sources with fifty resources each do not fit. Each internal batch uses
the existing validated count executor, adapters and source permits once. A pure
internal finalizer authenticates every accepted batch, calculates totals and all
tied winners, and publishes one structured `data.count_comparison` result and
one success delivery. These tasks make no model requests or synthetic transcript
exchanges, and create no artifacts. The originating foreground turn still uses
the ordinary model/tool loop for admission and its answer.

Counts are exact sums of the admitted observations, with separate batch and
source snapshots. They do not establish a common instant or physical-database
completeness. There is no automatic source retry: an interrupted read without an
accepted result is fenced and cannot resume after restart, lease expiry or close.
Accepted evidence survives. Pure finalization may recompute within its original
three-attempt work/time allowance. New reading requires fresh job admission;
retry/replacement controls cannot reset the frozen count denominator or budget.

`inspection.count_progress` and `job_inspect.count_progress` expose accepted
counts, missing coverage, attempts and controls. A source with no accepted
observations has an observed subtotal of zero and a null total. Complete per-source
totals may coexist with an incomplete comparison; no winner is established until
successful finalization. Attention notices retain their existing separate delivery
kind. Result readers never present partial progress as a successful comparison.

The development single-source count MODEL-job protocol (`data_count_plan`,
`plan_ref`, and generic `data.count_selection` graph admission) is retired.
Foreground single-source selection counts and explicit/cursor count reads remain
available. Old nonterminal count MODEL jobs need fresh admission before any
provider/source call; terminal history remains readable. There is no single-source
durable compatibility executor.

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
