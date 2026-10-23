# Background jobs

A background job is work that keeps going after your turn ends. Ask Daita for
something long or multi-part ("compare last quarter's sales by region and
write up what changed") and it can start a job with `job_start`. The job shows
on the Work board straight away; there is no confirm card, and you can cancel
it at any time.

A job is an ordinary agent run that keeps going in the Daita host. It:

- keeps its own **checklist** of the work and marks items in progress and done;
- hands independent pieces to **sub-runs**, several at once, and gets back a
  short summary from each;
- **asks you** when it needs you: the job waits as *Needs you* with its
  question, and your answer lets it carry on;
- finishes by writing its result, which arrives in the conversation's Inbox as
  a Done card, with a notification when no window is watching.

Reads just run, using the job's named sources or, without any, all of the
agent's sources. Writes and external actions ask you at the time, exactly as
in a scheduled routine: approve and the job continues; deny and it carries on
with the refusal. An approval nobody answers within 24 hours stops the run and
the job waits for you. Sub-runs cannot start further sub-runs, ask you
questions or start jobs.

## Surviving restarts

A job continues by starting a new run from what it saved: its objective,
checklist, sub-run summaries, your answers, the writes already made and the
last message of its previous run. That happens when the host restarts, when a
run's context fills up, and when you answer a question. Sub-runs that were
running at a restart are reported to the job as interrupted, and it can start
them again. A write that was waiting for your approval at a restart was never
made; the job asks again if it still needs it. A write whose outcome is
uncertain holds the job until you resolve it under [Effects](EFFECT_RECEIPTS.md).

## Limits

Each run has the agent's normal run limits. A job stops as failed after 20
runs or ten times the per-run token limit in total, whichever comes first.
Answer it to raise both limits by the same amount and let it continue. At most
two sub-runs per agent run at once
(`AgentConfig(host_capacity=HostCapacityLimits(job_parallelism=...))`, up to
four); the host's provider capacity still applies.

## Following and answering jobs

Use `/jobs` in the terminal app to list jobs, see one job's checklist,
sub-runs, question and result, answer it or cancel it. The work board shows
running jobs with their progress ("3 of 7 done"), waiting and failed jobs under
*Needs you*, and finished jobs under *Done*. From the shell:

```bash
daita jobs list atlas
daita jobs inspect atlas <job-id>
daita jobs answer atlas <job-id> "Use September."
daita jobs cancel atlas <job-id>
```

The Python API offers `Agent.start_job(objective, sources=None)`, `list_jobs`,
`inspect_job`, `answer_job(job_id, text)` and `cancel_job`. A job belongs to
the agent: any conversation can see it, and the one it started from receives
its result.

Jobs created by an older version of Daita were upgraded: finished ones keep
their result; unfinished ones show as failed with "Created by an older version
of Daita; start it again."
