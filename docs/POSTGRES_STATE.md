# PostgreSQL state backend

`PostgresStateStore` implements structured state persistence: identity, catalog,
permissions, transcripts, graph jobs, routines, deliveries, effect receipts,
learning candidates, semantic annotations, connector bindings and artifact
registry records. SQLite and PostgreSQL use one set of domain operations and
record codecs. PostgreSQL is a state database here; configured PostgreSQL data
sources remain a separate adapter responsibility.

This backend is available in this source tree. Release 1.2.1 contains the portable
contracts only; distribution of the new backend requires a subsequent release.
`Agent.create` and `Agent.open` still compose local SQLite homes. This module does
not yet move configuration, memory/skill files or secrets, import an existing
home, or provide a distributed execution lease. The artifact lifecycle now accepts
remote S3 bytes through an explicitly supplied client, but public agent composition
does not yet select it. See [artifact storage](ARTIFACT_STORAGE.md).

## Schema and authorization

Use a dedicated private schema, by default `daita_state`, containing 30 shared
state tables and three administrative tables. Each logical home has a stable UUID
namespace. Namespace IDs prefix primary keys, foreign keys and lookup indexes;
agents do not create individual schemas. JSON records retain the existing strict
codecs and canonical text representation. Text ordering uses the `C` collation.

Provision schemas through an administrative connection before opening runtime
handles. The schema owner needs database `CREATE`; a superuser is not required.
Role creation and credentials remain the administrator's responsibility. For
example, create a login with no administrative privileges, then provision it:

```python
import os
from uuid import UUID

import psycopg

from daita.storage.postgres_admin import (
    install_postgres_schema,
    provision_postgres_namespace,
)

namespace = UUID(os.environ["DAITA_STATE_NAMESPACE"])
# The administrator supplies a TLS-verified DSN and a pre-existing runtime login.
with psycopg.connect(os.environ["DAITA_STATE_ADMIN_DSN"], autocommit=True) as admin:
    install_postgres_schema(admin)
    provision_postgres_namespace(
        admin, namespace_id=namespace, role="daita_runtime"
    )
```

Installation is transactional and serialized by a transaction-scoped advisory
lock. An existing schema must have the exact supported version/checksum; the
installer refuses an unknown schema rather than modifying it. Namespace grants
are idempotent. Installation removes grants inherited from administrator default
ACLs before provisioning runtime permissions. Admission rejects schema creation,
administrative-table writes and state-table `TRUNCATE` privileges. Opening a
runtime handle creates no tables, roles or state.

The runtime login must not own the schema/tables, belong to an owner role, have
`BYPASSRLS`, or hold administrative privileges. It receives only schema usage,
state CRUD, sequence usage, read access to schema/namespace metadata, and the
ability to advance its namespace epoch. It cannot create tables or edit grants.
Do not grant it broader privileges through other roles.

Row-level security is enabled and forced. Policies require both the selected
transaction-local namespace and an administrator-owned mapping from namespace
to `session_user`. Changing a client setting alone cannot select a foreign home.
A login can access every namespace explicitly granted to it; use separate logins
when database credentials themselves must isolate homes. Poolers must preserve
the authenticated login as `session_user`, rather than force every client onto
one shared backend role. Manage and revoke `namespace_roles` mappings when
retiring or renaming roles.

## Opening and ownership

```python
from daita.storage.postgres import PostgresStateConfig, PostgresStateStore

async def inspect_identity():
    store = await PostgresStateStore.open(
        PostgresStateConfig(
            conninfo=os.environ["DAITA_STATE_DSN"],
            namespace_id=namespace,
            max_connections=2,
            timeout_seconds=15,
        )
    )
    try:
        return await store.load_identity()
    finally:
        await store.close()
```

The DSN must explicitly specify `sslmode=verify-full` and trust the server's CA
through `sslrootcert` or the configured trust store. Encrypted transport without
certificate/hostname verification is rejected. DSNs are omitted from config
representations. Psycopg and its pool load lazily at backend construction.

Opening is passive: it validates admission and observes the current epoch. It
does not recover state or claim execution ownership. Before recovery, the host
must acquire its own exclusive writer lease and then call `advance_fence()` on
the admitted handle. This compare-and-set increments the namespace generation.
Older handles fail every mutation with `StorageOwnershipLostError`, including
after reconnecting. A stale handle cannot advance a newer epoch.

Fencing is not a lease: another newly opened handle observes the same generation,
and PostgreSQL cannot stop an external effect already dispatched by an old host.
Hosts must implement lease renewal, admission stop, drain and handover, and use
the existing receipt recovery rules. Do not run distributed supervisors merely
because a state connection opens successfully. Advance the fence before allowing
concurrent work on the new handle.

## Transactions and failure handling

All mutations use the shared cancellation/transaction boundary. Each PostgreSQL
mutation locks its namespace row, checks its generation and commits atomically.
This preserves existing per-home serial semantics while separate namespaces
proceed independently. Multi-query reads use one repeatable-read snapshot.
Cancellation before mutation starts rolls back; once started, the operation
settles before releasing its connection. Close drains active workers.

The pool opens lazily, defaults to two connections per handle, and permits an
explicit limit from one to sixteen. Acquisition and server statement/lock/idle
timeouts are bounded; TCP keepalives and the supported TCP user timeout limit
transport-loss detection. Size total connections across all active handles. No
provider, source or artifact I/O belongs inside a state transaction.

Prepared statements and pipeline mode are disabled. Namespace, epoch, search
path and timeout settings are transaction-local; no session advisory lock or
temporary table is required. These choices permit transaction pooling, but a
particular pooler's authentication, TLS and failure behavior still need separate
qualification. The local fixture tests direct PostgreSQL, not a managed pooler.

Known rolled-back contention, deadlock, serialization and statement-timeout
failures become `StorageUnavailableError`. A lost mutation COMMIT acknowledgement
becomes `StorageCommitUnknownError`; the handle rejects subsequent and queued
mutations. There is no automatic replay. Stop work, close, reopen under the host's
ownership rules and inspect durable state before deciding what happened. An
idempotent operation can reconcile its existing identity; arbitrary transcript
or effect operations cannot simply be resubmitted. Unexpected driver failures
remain terminal storage errors.

## Compatibility and qualification

Both backends use the same **home revision 3** and migration registry. The
canonical home schema defines tables, columns, constraints and indexes once;
the PostgreSQL adapter derives its physical DDL and supplies namespace keys,
identity allocation, exact text ordering and database security. Adding a home
table or index does not require a second PostgreSQL table definition.

There is one release snapshot, `release/agent-home-contract.json`, and one check:
`python scripts/check_home_release_contract.py check`. The snapshot protects the
shared home contract and fingerprints PostgreSQL's physical/security adaptation.
Changing a previously released backend uses the same home-revision sequence.
The first PostgreSQL fingerprint establishes its baseline without changing the
released SQLite contract or migration checksums. No package version is bumped by
this source change.

Future format changes should be authored once in the existing home migration
registry, with shared record transformations exercised against both backends.
Deriving a target schema does not infer how to rename, split or backfill data.
The remote-home upgrade runner is still part of the remaining integration work;
the current installer only installs a fresh schema or verifies its exact revision.
It does not silently upgrade an existing database. Release readiness requires
actual upgrade tests, in addition to fresh-schema and operation tests.

Run the real disposable database suite using the command in [Testing](TESTING.md).
It qualifies shared operations, namespace isolation with real login credentials,
provisioning, generation fencing, cancellation, termination and unknown commits.
Full remote-home integration still needs configuration/byte ownership, import and
rollback tooling, backup/restore across records and bytes, lease handover,
pooler qualification, and representative multi-agent capacity measurements.
