# Hosting database monitors

`DbMonitorScheduler.run_once()` is the durable, one-shot database-monitor
execution boundary. The library finds persisted monitors, evaluates which are
due, claims their existing durable tick leases, records each decision, and
releases claimed leases. It does not own a recurring loop.

The application host owns cadence, bounded retry, metrics, process signals,
and graceful shutdown. `DbRuntime.setup()` initializes plugins only and never
starts monitor work in the background.

## Responsibilities

| Library | Application host |
| --- | --- |
| Persist monitor definitions, state, runs, operations, and evidence | Construct one scheduler per host process and retain it |
| Evaluate one pass through `run_once()` | Call `run_once()` at least once per desired scheduling resolution |
| Claim and release the existing monitor tick lease | Supply a stable scheduler ID for the lifetime of the process |
| Exclude a second host from a leased monitor tick | Prevent scheduler passes from overlapping within one host |
| Return durable per-monitor decisions | Supply timezone-aware UTC clock input |
| Preserve governance, approvals, actions, and delivery | Retry pass failures with bounded exponential backoff |
| | Stop starting passes on shutdown and await the active pass before teardown |
| | Publish host metrics and alerts |

`DbRuntime.tick_monitors()` remains a one-shot convenience. It constructs a
scheduler, runs exactly one pass, and returns the underlying monitor runs. A
recurring host should retain an explicit `DbMonitorScheduler` so its identity
and lifecycle are visible and stable.

## Cadence and clock

Call `run_once()` at least as frequently as the smallest scheduling resolution
you intend to honor. For example, a host supporting one-minute monitor
schedules should start a pass at least once per minute. Await each pass before
starting the next one; a single host must not overlap its own passes.

Pass a timezone-aware UTC `datetime`, normally
`datetime.now(timezone.utc)`. A stable host-process scheduler ID can combine a
deployment instance identifier and process ID. Create the scheduler once and
reuse it; do not generate a new scheduler ID for each pass.

## Multiple hosts and leases

Multiple hosts may point at the same durable monitor store. A due monitor is
executed only by the scheduler that claims its existing tick lease. A
`lease_lost` decision means another host owns that tick: record the metric and
do not execute or retry that monitor tick from the losing pass. A later normal
scheduler pass may evaluate the monitor again after the current lease and
schedule rules allow it.

The in-memory runtime and monitor stores coordinate only tasks sharing that
same Python object. They are not cross-process coordination mechanisms. Use a
durable store shared by all hosts when cross-process exclusion is required. Do
not add a second distributed lock around monitor execution; the monitor tick
lease is the coordination owner.

## Metrics

Record at least these counters from every scheduler result:

- `due`: increment when `result.claimed` is true or the decision reason is
  `lease_lost`.
- `claimed`: increment when `result.claimed` is true.
- `lease_lost`: increment when `result.run.summary["reason"]` is
  `lease_lost`.
- `succeeded`: increment for run status `succeeded` or `triggered`.
- `blocked`: increment for run status `blocked`.
- `failed`: increment for run status `failed`.

A triggered run is a successful monitor execution, so it contributes to
`succeeded`. A host may also publish a separate `triggered` counter. Skipped
decisions such as `not_due`, `paused`, `cooldown`, and `backoff` do not count as
due or successful executions. Track scheduler-pass exceptions separately from
per-monitor failed runs.

## Host loop example

This is application code, not a library service. Adapt signal registration and
metric emission to the process manager in use.

```python
import asyncio
from datetime import datetime, timezone
import os
import signal

from daita.db import DbMonitorScheduler

RESOLUTION_SECONDS = 30.0
MAX_RETRY_ATTEMPTS = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0


def observe(results, metrics):
    for result in results:
        reason = result.run.summary.get("reason")
        if result.claimed or reason == "lease_lost":
            metrics["due"] += 1
        if result.claimed:
            metrics["claimed"] += 1
        if reason == "lease_lost":
            metrics["lease_lost"] += 1
        if result.run.status in {"succeeded", "triggered"}:
            metrics["succeeded"] += 1
        elif result.run.status == "blocked":
            metrics["blocked"] += 1
        elif result.run.status == "failed":
            metrics["failed"] += 1
        if result.run.triggered:
            metrics["triggered"] += 1


async def serve_monitors(runtime):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for process_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(process_signal, stop.set)

    scheduler = DbMonitorScheduler(
        runtime=runtime,
        scheduler_id=f"orders-api-{os.getpid()}",
    )
    metrics = {
        "due": 0,
        "claimed": 0,
        "lease_lost": 0,
        "succeeded": 0,
        "blocked": 0,
        "failed": 0,
        "triggered": 0,
        "pass_failed": 0,
    }
    failures = 0
    active_pass = None

    try:
        while not stop.is_set():
            pass_started = loop.time()
            try:
                active_pass = asyncio.create_task(
                    scheduler.run_once(now=datetime.now(timezone.utc))
                )
                results = await active_pass
            except Exception:
                metrics["pass_failed"] += 1
                failures += 1
                if failures >= MAX_RETRY_ATTEMPTS:
                    raise
                delay = min(
                    INITIAL_BACKOFF_SECONDS * (2 ** (failures - 1)),
                    MAX_BACKOFF_SECONDS,
                )
            else:
                observe(results, metrics)
                failures = 0  # A successful pass resets exponential backoff.
                delay = max(
                    0.0,
                    RESOLUTION_SECONDS - (loop.time() - pass_started),
                )
            finally:
                if active_pass is not None and active_pass.done():
                    active_pass = None

            if not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
    finally:
        stop.set()  # Do not start another pass.
        if active_pass is not None:
            await active_pass  # Finish the durable pass before runtime teardown.
```

The host should call `runtime.teardown()` only after `serve_monitors()` has
returned or otherwise confirmed that its active pass is complete.
