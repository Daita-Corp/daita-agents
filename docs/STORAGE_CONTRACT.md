# Durable state contract

`daita.storage.protocols.StateStore` describes the durable operations used by
embedded composition, the job supervisor and the routine supervisor. Existing
domain-owned protocols remain the smaller interfaces for their callers. The
contract covers the current SQLite operation surface; an architecture check
rejects omissions, and mypy checks concrete implementations and consumer calls.

The current implementation is `SQLiteStateStore`. Backend construction, home
admission/upgrades, schema revision inspection and filesystem paths remain outside
the protocol. `Agent.create`, `Agent.open` and deletion still use local homes.
There is no Postgres backend or backend selector in this change.

## Semantics to preserve

Implementations must preserve model validation, domain exceptions, bounds,
ordering, timestamps and each operation's transaction boundary. Matching Python
signatures alone is insufficient. In particular:

- A graph claim has one winner across handles. Attempt epochs and claim tokens
  reject stale writes; this is distinct from a future per-agent writer fence.
- Graph mutation and completion retries preserve their existing idempotency and
  conflict rules. Budget settlement cannot charge or release the same reservation
  twice. Only finalizer sealing can publish terminal success and its delivery.
- Transcript append positions are ordered. Completion atomically commits final
  assistant text and terminal state; it rejects writes to a terminal run. It is
  **not** an idempotent retry API. Recover by inspecting persisted state after an
  unknown outcome, rather than replaying writes or external effects.
- Passing `predecessor=None` to `start` asserts that a conversation is empty.
  Passing a predecessor checks the latest terminal turn atomically. Omitting the
  argument retains the legacy unchecked-continuation behavior.
- Identity survives close/reopen and conflicting initialization leaves it intact.
  Recovery terminalizes unfinished runs without replaying their work.
- Closing a handle releases its resources. SQLite performs a WAL checkpoint at
  close, so callers must settle concurrent operations before closing handles.

Graph transaction failures have neutral owners in `storage/errors.py`.
`storage/sqlite_graph.py` retains imports of the same exception classes for
compatibility. Existing record models in `sqlite_records.py` are unchanged.

## Backend conformance tests

```bash
.venv/bin/python -m pytest tests/storage/contracts
```

The parameterized factory in `tests/storage/contracts/conftest.py` opens separate
handles to the same disposable state and closes them explicitly. Today it runs
SQLite only. Add another parameter only with a real backend and a disposable
database fixture; missing prerequisites must fail qualification, not silently
skip it. Backend tests use `StateStore`, never SQL or concrete store internals.

The initial portable suite covers graph claims, attempt fencing, completion and
mutation replay, budgets, seeded topology changes, finalizer delivery, bounded
inspection/event pages, checkpoints/comments/controls, identity and transcript
recovery. SQL fault injection and read-snapshot synchronization stay in the
SQLite-specific storage tests. Existing public acceptance journeys continue to
exercise local `Agent` composition.

Before a hosted backend is usable, expand portable coverage to routines,
deliveries, sources/scopes, catalog, MCP bindings, effects, learning, semantics,
artifact registry and deletion. Then qualify every operation on both backends,
including rollback, cancellation and concurrent-worker recovery. This contract
does not supply remote configuration, artifact/skill bytes, tenant isolation,
distributed writer fencing or a home converter. Those are separate release gates.

## Test reconciliation

The 31 portable cases formerly in `tests/storage/test_graph_transactions.py`
moved to `tests/storage/contracts/test_graph_transactions.py`, retaining names,
all 24 seed values, integration markers and assertions. Node IDs gain a `sqlite`
parameter; contract markers are added. Redundant asyncio markers are removed
because auto mode is configured. Concurrency now uses separate handles.
`test_graph_inspection_uses_one_read_snapshot` keeps its original node ID and
SQLite synchronization. Three new recovery cases live in `test_recovery.py`.
No original case is deleted or skipped.
