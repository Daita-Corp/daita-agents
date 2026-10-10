# Local Python analysis

Local foreground sessions share one `analysis_execute` tool across the TUI,
CLI and Python API. It runs ordinary Python with DuckDB, NumPy, pandas, SciPy,
PyArrow, Matplotlib and NetworkX on macOS ARM64. Managed installation includes
the pinned scientific environment. Other platforms and hosted compositions do
not expose this capability. A source checkout needs `pip install -e '.[dev,analysis]'`.
Missing or mismatched packages make the tool unavailable; execution never installs
packages or falls back to an uncontained interpreter.

Ask Daita to analyze your files, combine datasets, calculate statistics or create
a chart. The model discovers and loads Analysis through the ordinary toolbox.
`Agent.analysis_runtime()` reports availability, package versions and the hash
of the interpreter, worker protocol and trusted launch helpers without starting
a worker. Interpreter state lasts for one `Agent.run`, not across messages.

## Cells and read tools

The first call supplies `expected_state={"generation":0,"revision":0}`. Each
accepted cell returns the exact state for the next call. A Python exception
advances the revision and reports `state_may_have_changed`: assignments before
the exception remain. Timeout, resource or protocol failure reports `state_lost`
and closes the interpreter. An explicitly requested later cell can start a fresh
generation using the returned state; the failed code is never replayed and its
consumption remains charged.

The version-1 worker helpers are ordinary Python objects:

```python
import json
result = tools.call("file_read", {"path": "sales.json"})
if result["is_error"]:
    raise ValueError(result["output"])
data = result["output"]["data"]
print({"complete": data["complete"]})
rows = json.loads(data["content"])
```

`tools.call` returns `output`, `is_error`, `evidence_id` and `sensitivity`.
Only already callable read tools from the cell's frozen surface can execute.
It cannot load tools, call analysis recursively, publish artifacts, mutate local
state, make external actions or invoke a model. Every attempt shares the run's
tool-call allowance. The host uses the existing CapabilityRuntime for schema,
permission, classification, permit, output and observer checks. Child results
are durable typed run evidence, separate from assistant/tool messages; the model
receives compact host-issued references rather than every intermediate result.
The existing artifact inventory and preview reads are included through their
owner's exact contracts. Creation, conversion and delivery remain outer calls.

`inputs` maps logical names to exact earlier successful same-run references:
`{"sales":{"kind":"tool_result","call_id":"..."}}` or
`{"saved":{"kind":"artifact","artifact_id":"..."}}`. `inputs.get("sales")`
reads the captured JSON result envelope; `inputs.path("saved")` supplies a private
relative file for an installed reader. Inputs support structured JSON, UTF-8 text,
CSV, Arrow IPC files and Parquet, with bounded isolated validation. JSON integers,
strings and nulls retain their representations. CSV captures retain original bytes,
not an inferred schema. Python must choose explicit decimal/time-zone/table types
and preserve reported coverage. A partial read remains partial; there is no hidden
pagination or shared transaction snapshot across separate reads.

Admitted authorities are checked before cells, child calls and output acceptance,
and periodically during computation. Revocation or artifact deletion closes the
contaminated interpreter. Later row changes alone do not erase an authorized
historical snapshot; request a new read for fresh values.
Read authority stays bound to accepted candidates across interpreter replacement;
state loss cannot make a revoked candidate eligible to save.

## Saving results

Write a file inside scratch and register a candidate:

```python
import json
with open("summary.json", "w", encoding="utf-8") as output:
    json.dump({"total": 42}, output)
outputs.add("summary", "summary.json")
```

The cell's `save_output="summary"` explicitly commits one candidate through the
ordinary artifact store. A later cell may save an existing candidate without
recomputing it. Four candidates per cell and sixteen per run are allowed; each
call commits at most one artifact. Supported outputs are CSV, Parquet, JSON, PNG,
Markdown, UTF-8 text and Python/SQL source text. Source text is retained data.
Host capture requires a bounded regular file inside scratch, rejects symlinks and
traversal, and hashes bytes while the worker is demonstrably stopped. A separate
contained parser validates the format. Changed candidate bytes need a new name.

Artifacts retain runtime identity, executed source or an explicit source omission,
cell digests, authenticated child/input references, known coverage and output hash.
`Agent.artifact_computation_evidence(id)` exposes this retained evidence after
history deletion. Raw input datasets are not automatically retained, so missing
inputs may prevent reproduction. Execution success does not establish analytical
correctness. A failed save reports the actual retained interpreter state and
`save_status="failed"`; it does not rerun computation. Delivery to local files
continues through existing artifact delivery and approval controls.

## Bounds, measurement and closure

`AgentConfig.analysis_limits` accepts `daita.config.AnalysisLimits`. Defaults are
30 seconds per cell, 120 cumulative CPU seconds, 1.5 GiB RSS per worker, 64 MiB
scratch and 256 scratch files. Call time is also bounded by the remaining run
deadline. Captured inputs and immutable output snapshots each have a 64 MiB run
ceiling; individual artifact/parser inputs are at most 16 MiB. Cells allow 64 KiB
source, 16 KiB each stdout/stderr, sixteen child attempts and 1 MiB total RPC.
Broker delivery is bounded to 60 KiB per normalized result. Trace admission
reserves completion capacity within four MiB and 256 records per run.
Byte or record exhaustion terminates the run before another model request.
Cell deadlines include worker admission, input validation and compute waits.
Cleanup retains a separate bounded allowance after a cell times out.

`AnalysisLimits` also configures code/log/result/RPC bytes, child calls, worker
generations, input bytes, output file/count/run bounds, trace bytes/records and
parser seconds/decoded bytes/rows/columns. The table parser defaults to ten
seconds, 128 MiB decoded data, one million rows and 256 columns. Individual
trace records have a fixed 256 KiB ceiling. The protocol permits one outstanding
child read; process creation is denied by the native policy.

The existing admission coordinator holds one worker lifetime slot, a separate
bounded parser slot and one shared computation lane. The computation lane is
released while a suspended worker waits for a broker result. Numerical libraries
receive one-thread settings; OS containment denies new processes, network, host
credentials, state, sibling files, runtime writes and reusable host IPC. CPU and
individual-file bounds also use kernel rlimits. RSS, aggregate scratch and file
count are watchdog controls with 20 ms sampling; reported overshoot is real and
these are not hard quotas. Authority polling defaults to 250 ms; cleanup is bounded
to five seconds. Typed configuration can narrow bounds within their documented
ceilings. `worker_count` selects one or two lifetime slots; the parser retains its
separate single slot. Their finite memory/scratch reservations are retained until
closure, and final evidence distinguishes them from measured usage.
Numerical thread settings start at one and the host monitors a maximum of 32
threads per worker, recording peak thread count and overshoot. Required native
CPU, RSS and thread sampling must be available before admitting generated code.
Scratch entry counts include directories and symlinks. Sampling stops at a quota
breach and retains observed lower bounds; unreadable or incomplete samples expose
unknown current/peak values. Cleanup restores permissions only on authenticated
owned directories after reaping, and never follows scratch symlinks.

`Agent.analysis_evidence(run_id)` returns host facts on success, error, timeout,
cancellation and reopen. These include sampled/final user and system CPU, cell
wall time, broker wait, per-worker peak RSS, current/peak scratch bytes/files,
captured/protocol/output bytes, attempted/dispatched/failed child counts, observed
exit status and deletion postconditions. A run summary reports wall time without
adding overlapping workers, shared call attempts, committed artifact
bytes and released reservations. Unavailable measurements are `null`, never
measured zero. CPU/RSS are for the contained workers; trusted host/guardian CPU
is outside that measurement. Peak RSS is a per-worker peak, not simultaneous
whole-host memory. TUI observer updates are best effort; durable closure facts
remain authoritative.
Protocol input/output counters measure successful OS writes and received bytes,
including failed cells. Captured payloads, format-validation copies and accepted
result bytes are reported separately.
Public summaries include outer attempted/dispatched/denied/failed/unsettled
counts and separate child counts. Saved analytical evidence retains child
contract/schema/argument/result digests and known coverage after history clears.

Before accepting successful completion the host closes broker admission, settles
child I/O/evidence, reaps owned processes, closes descriptors and verifies scratch
absence and zero remaining bytes/files. A trusted native parent performs cleanup
on host death and writes authenticated closure facts outside worker access.
Reopen joins those facts to the abandoned run, removes only authenticated owned
remnants and terminalizes once without replay. Unknown termination, changed
remnant identity or failed deletion prevents successful acceptance and replacement.
Analytics is foreground-only: routines, graph tasks, remote execution and cross-run
interpreter caches remain deferred.

See [testing instructions](TESTING.md#local-python-live-acceptance) for the
real-worker, real-model and managed-install qualification command and reports.
