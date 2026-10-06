# Durable state contract

`daita.storage.protocols.StateStore` composes the existing domain-owned protocols.
It is the embedded composition boundary, not the interface every consumer should
accept. Job and routine supervisors use `GraphSupervisorStore` and
`RoutineSupervisorStore`; transcript writers, catalog owners, artifact storage,
and other domains retain their smaller contracts. Canonical declarations stay
with their domain. Mypy checks implementation and consumer conformance; the
architecture check verifies composition and dependency boundaries rather than
requiring every public SQLite helper to become a backend operation.

SQLite is the default durable backend. `SQLStateStore` implements the shared
state operations and strict record codecs over an injected `SQLDatabase`.
Applications can supply a database adapter without copying those operations.
`SQLStateStore` is optional: a non-SQL backend can implement the existing
domain-owned protocols composed by `StateStore` and run the same contract suite.

`Agent.create`, `Agent.open`, deletion and home upgrades select local storage.
`Agent.from_storage` composes the same runtime over caller-admitted storage.

## Opening supplied storage

```python
from daita import Agent

# The application has acquired exclusive ownership, fenced prior writers,
# validated/upgraded all storage and initialized the expected agent identity.
agent = await Agent.from_storage(
    agent_id=expected_agent_id,
    state=state_store,
    advisory_storage=documents,
    artifact_storage=artifact_bytes,
    config=agent_config,
    secret_provider=secrets,
    keychain=credentials,
)
try:
    result = await agent.run("Summarize the current catalog.")
finally:
    await agent.close()
# Only after successful close: close supplied resources and release ownership.
```

This entry point verifies the stored identity before recovery, recovers abandoned
effects and unfinished transcripts, opens the existing artifact lifecycle, and
starts the existing supervisors. It borrows all three supplied storage resources;
closing the agent drains its runtime without closing their clients or state
handle. Admission cancellation settles started work and closes a composed runtime
before returning cancellation. Keep resources and ownership alive throughout that
await. Failure to drain must not be followed by releasing ownership beneath work.

No local home or workspace is created. `agent.home`, local-file delivery and
persisted model-configuration controls are unavailable. Supply `AgentConfig` or
an injected provider each time; the application owns configuration persistence.
Creation, deletion, schema validation/upgrades, credentials, ownership renewal
and fencing belong to the application. This API does not implement them.

## Memory and skill documents

`daita.storage.advisory.AdvisoryStorage` supplies synchronous transactions beneath
the existing `MemoryStore` and `SkillStore`. It does not replace their semantics.
The three collections contain canonical bytes:

| Collection | Key | Content |
| --- | --- | --- |
| `memory` | `MEMORY.md` or `USER.md` | Existing UTF-8 document and sensitivity label |
| `skills` | Valid skill name | Existing complete `SKILL.md` bytes |
| `retained-skills` | SHA-256 hex digest, without prefix | Immutable rendered skill bytes |

`get` reads one bounded document; `scan` returns the complete bounded collection
or rejects overflow; `usage` returns count and total bytes without loading content.
`put` and `delete` run only in write transactions. Reads see a consistent snapshot;
writes serialize all affected reads, quota checks and mutations within the agent.
Every changed/recreated document has a fresh opaque revision (1–256 characters),
including restoration of earlier bytes. Preflight uses that revision to reject
changes made while approval was pending. A document's revision is transport
metadata, not a new home-format revision.

The stores retain encoding, sensitivity, bounds, index validation, approval
preflight and retained-skill digest checks. The caller owns physical schema and
transport lifetime. Normalize transport failures to `StorageError`; an unknown
commit or lost ownership propagates to the execution owner and stops the run,
without becoming a model-visible retry opportunity. No adapter may blindly replay
a write. Transactions and transport waits must have finite bounds.

## SQL adapter extension

Supported imports are `SQLStateStore` and `run_sql_transaction` from
`daita.storage.sql`, and `SQLDatabase`, `SQLConnection`, `SQLCursor` and
`StateIntegrityError` from `daita.storage.sql_connection`. A custom adapter may
instantiate `SQLStateStore(database)` or subclass it to own admission and cleanup.
Do not override domain operations to change storage behavior.

The synchronous connection contract runs in worker threads. It uses qmark
parameters, tuple rows, consistent read snapshots and serialized short writes
within a logical home. Trusted dialect expressions cover insertion ordering,
null-safe comparison, conflict columns and caller extraction. Adapters own
parameter translation, physical isolation, bounded waits and driver-error
normalization. An unknown commit must stop further writes; never replay it.
`run_sql_transaction` supplies the existing cancellation owner for adapter writes.

The supplied database is borrowed: `SQLStateStore.close()` clears its own caches,
not the database's pools or credentials. The composing caller must drain active
operations, close the state handle, then release its owned resources. Opening a
connection must not recover effects or acquire execution authority implicitly.

The canonical `CURRENT_DATABASE_SQL`/`CURRENT_SCHEMA` in
`daita.storage.sqlite_schema`, schema inspection in `daita.storage.schema_contract`,
and the registry in `daita.storage.home_migrations` are the shared format sources.
Schema projection must reject unsupported semantics rather than silently discard
them. Released migration definitions remain immutable. Existing migrations and
the whole-home upgrade runner currently use local files; a custom upgrade runner
must reuse the definitions and transformations, and qualify its execution path.
These interfaces do not make existing local upgrade code portable automatically.

Keep concrete drivers, tenant/security policy, object layout, provisioning,
lease services and deployment checks in the consuming application. The framework
owns reusable semantics and local defaults, with no infrastructure-specific
release fingerprint or second feature-migration sequence.

## Ownership and lifecycle

Connection setup and recovery are separate operations. `SQLiteStateStore.open`
initializes or validates a database and opens a handle; it does not change live
receipts into uncertain outcomes. `recover_started_effect_receipts` is explicit,
agent-scoped, atomic, and idempotent. Opening extra reader/test handles cannot
perform recovery.

The lifecycle owner must follow this order:

1. Acquire exclusive ownership of the agent before any recovery or execution.
2. Connect and validate identity, schema and configuration.
3. Recover abandoned effects, then unfinished transcripts, then start supervisors.
4. Drain executions and storage operations, close resources, then release ownership.

Local composition retains `run/host.lock` through this sequence. Direct store
callers are responsible for ownership before invoking recovery. `Agent.from_storage`
requires that ownership already be held; it does not acquire or renew it.

A remote backend must bind each execution handle immutably to one agent and one
writer epoch. Every mutation must validate current ownership in the same short
transaction as its state change. A stale owner must fail even after a process
pause, lease expiration, connection replacement, or network partition. Reads by
run/artifact ID must also remain confined to the admitted agent. Application
filters, graph-attempt fences, and a connection-pool session lock do not provide
these guarantees. Recover only after a successful ownership takeover; connecting
another worker is not evidence that its predecessor stopped.

External I/O must stay outside database transactions. A database fence cannot
undo an external effect that was already dispatched. Persist the effect receipt
before dispatch, retain uncertain outcomes, and reconcile them without replay.

## Transactions, retry and failure outcomes

Implementations preserve validation, domain exceptions, bounds, ordering,
timestamps and atomic boundaries. A successful await means durable commit.
Cancellation before a mutation starts leaves no changes; cancellation after it
starts must settle or expose an unknown outcome before releasing its resources.

`storage/errors.py` owns the backend failure categories:

| Error | Required meaning | Caller behavior |
| --- | --- | --- |
| `StorageUnavailableError` | Transient read failure or mutation known not to have committed; rollback settled | Bounded backoff is safe |
| `StorageCommitUnknownError` | A mutation may have committed; its acknowledgement is missing | Inspect durable state; never blindly resubmit |
| `StorageOwnershipLostError` | The execution handle no longer owns the agent | Stop work; require a fresh ownership acquisition |
| Domain conflict/validation error | A logical precondition failed | Follow that operation's domain contract |

Backends must not translate every transport exception into a retryable error.
SQLite maps busy/locked failures only after a failed read or a completed rollback.
Graph conflict/budget exceptions retain their old SQLite import aliases.

Supervisors retry known non-commits with capped backoff. Unreconciled unknown
commits, lost ownership, and unexpected failures stop their driver and cancel
remaining workers. Existing graph claim/completion paths may reconcile an exact
persisted result before deciding whether an operation failed. Storage errors must
not become ordinary executor failures eligible for automatic replay.
`Agent.background_status()` exposes payload-free running/retrying/failed/stopped
status and code-owned failure codes. Logs and status never include driver error
messages or connection details. Restarting a failed supervisor requires normal
host close/open and ownership admission; `wake()` cannot bypass it.

| Operation family | Atomic boundary and replay rules |
| --- | --- |
| Graph claim | One winner; attempt token/epoch, active slot and budget reservation agree |
| Graph completion/mutation | Exact replay is accepted, conflicting content rejected; budgets settle once |
| Graph finalization | Accepted result, terminal state, budget settlement and delivery commit together |
| Routine claim/finalization | Occurrence, routine revision, reservation/settlement and delivery stay consistent; stale claim tokens fail |
| Transcript start | Explicit `predecessor=None` asserts an empty conversation; a supplied predecessor is checked atomically; omission preserves legacy unchecked continuation |
| Transcript append/completion | Exact ordered positions; final assistant text and terminal result commit together; completion is not an idempotent replay API |
| Source/permission edits | Registration, current snapshot and permission changes retain their composite transaction |
| Effects | Receipt reservation precedes dispatch; terminal observation is immutable; recovery records uncertainty without executing effects |
| Artifacts | Registry lifecycle transitions use current state; immutable bytes are published before a ready reference; byte I/O occurs outside the transaction |

Graph models, validators, reducers, codecs and the existing domain owners remain
the semantic authorities. Completion-authority and finalizer-delivery checks now
live in `jobs/graph/validation.py`. A second backend should reuse these decisions
and extract remaining pure decisions from transaction code as needed. It must
not copy the SQLite scheduler or introduce a second execution engine. Keep
cross-domain atomic operations intact when narrowing consumer interfaces.

## Bounded access

- Graph polling backs off from its responsive interval to one second when no wake
  arrives. Submissions and worker completion wake it immediately. Error backoff
  is separate and capped at five seconds, including worker-originated failures.
- Attempt guards read one consistent job/task/attempt authority snapshot. They
  do not load graph history, results, events, budgets or unrelated attempts.
- `conversation_run_page` returns 1–100 ascending turns after an exclusive
  `after_turn_index` cursor. Continue after the final returned index; an empty
  page ends traversal. Each page uses one read snapshot and batches its messages.
- `conversation_access` checks existence and caller access across all turns
  without materializing transcripts. Restricting the check to a page is unsafe.
- Artifact lists accept an exclusive `(created_at, artifact_id)` cursor with an
  explicit 1–1000 limit. Cursor traversal does not depend on the cursor row still
  existing. Cursor and offset cannot be combined.

Full-history `conversation_runs` and unbounded artifact inventory reads retain
compatibility for callers explicitly requesting complete materialization. They
are not suitable for interactive remote pagination. Full conversation reads now
also batch messages, eliminating the per-turn query. Existing offset-based
artifact consumers remain supported; new large-list consumers should use cursors.

Connection pools and storage concurrency limits must be chosen for the backend
and measured across all active agents. SQLite's pressure permits are local
execution policy, not a PostgreSQL connection budget. Keep deadlines, connection
acquisition and transaction duration bounded; do not hold a database connection
while waiting for a model, source, or artifact operation.

## Conformance and release gates

Maintain the home contract and transformation sequence once. Backend physical
execution remains adapter-owned. Qualify any custom implementation against the
same logical cases, then add its own real transport and failure checks.

```bash
.venv/bin/python -m pytest tests/storage/contracts
.venv/bin/python -m pytest tests/hosting/test_storage_resilience.py tests/storage/test_remote_readiness.py
```

The portable factory opens independent handles to disposable state. The default
is SQLite. Downstream pytest plugins can indirectly parameterize
`state_store_factory` with their own fixture name without editing the shared
cases; see [testing](TESTING.md). The cases use logical operations rather than
concrete store internals. Missing adapter prerequisites must fail qualification.

Coverage includes graph claims/fences/budgets/replay/finalizers, identity and
transcript recovery, passive connection setup, scoped effect recovery, transcript
pages/access, artifact lifecycle/cursors, and routine authority/claims/budgets/
recovery/delivery, catalog/permission detach, semantic digest CAS, learning review
stamps and connector revisions. SQLite-specific cases additionally exercise real
process contention, transaction rollback, read snapshots and query-count budgets.
Supervisor integration cases exercise transient and terminal storage failures,
health reporting, idle polling and prompt wakeup without model calls.

Before integrating a remote agent home, extend qualification to the complete
composition and deployment. Mandatory remaining acceptance areas include:

- Competing processes and connections claiming the same work.
- Ownership takeover while the previous process is paused; rejection of every
  stale mutation family, including transcripts, receipts, configuration and deletion.
- Crash/cancellation before commit and lost acknowledgement after commit;
  readback proves the outcome without duplicate budgets, deliveries or effects.
- Cross-agent reads/writes using reused object IDs and actual database credentials.
- Cold open, configuration and immutable-byte recovery, schema mismatch, and
  a consistent backup/restore across records and referenced bytes.
- Representative latency, query counts, connection peaks, retention growth and
  total database work across many active and idle agents.

The backend contracts do not prove complete remote agent execution, distributed
ownership, transaction-pooler compatibility or production capacity. Remote
configuration, artifact/skill bytes and whole-home conversion remain separate
implementation and qualification work.

## Test reconciliation

The original 31 graph cases moved into the portable suite with all 24 topology
seeds and assertions retained. SQLite read-snapshot fault injection stays in its
original suite. Three identity/transcript cases were added in the initial slice.

Nine existing routine test functions moved into `contracts/test_routines.py`,
retaining their assertions and all changed-authority parameter values. Their
storage setup now uses the backend fixture. Shared record constructors live in
`tests/routines/_storage_support.py`; SQL fault injection and codec tests remain
with routines. Effect restart tests now invoke explicit recovery after opening.
No original behavior case was removed or weakened.
