# Relational updates and upserts

Daita begins with read access only. PostgreSQL updates are an explicit opt-in
for users who need the agent to change selected columns in selected tables.
Every update is structured, previewed, checked against the current catalog and
permissions, approved once, and executed in a transaction. The model never
writes SQL.

Updates can affect one row or many rows within the explicitly approved row ceiling (at most 10,000). Upserts permit one uniform batch of at most 1,000 rows, further narrowed by the permission and routine grant. PostgreSQL is the initial native backend.

## What Daita can update

Daita supports parameterized PostgreSQL `UPDATE` statements built from:

- one current cataloged base table;
- one or more AND-combined filters;
- one or more literal assignments; and
- the exact tables and assignment columns enabled through source permissions.

Filters support equality, inequality, ordered comparisons, set membership, and
null checks where the cataloged PostgreSQL type permits them. Supported values
include ordinary built-in boolean, numeric, text, UUID, date/time, JSON, and
JSONB types. Custom types and arrays are not update targets in the current
contract.

The separate `data_upsert_rows` operation supports an explicitly admitted insertion branch, described below. There is no insert-only tool, delete, arbitrary SQL mutation, DDL, role or grant administration, or mutation of SQLite, CSV, or JSON sources. Updates cannot assign primary-key, identity, or generated columns.

## Safety model

Four independent controls must all permit an update:

1. **PostgreSQL privileges:** the attached database role must have the required
   native privileges.
2. **Daita source permissions:** the exact table and assignment columns must be
   enabled by the user.
3. **Current readiness:** the live table, role, grants, and catalog state must
   pass Daita's non-mutating checks.
4. **Exact approval:** foreground writes require approval of the current-run previewed call. Routines require an approved revision with one native write grant and one invocation per occurrence.

The model cannot enable source permissions, grant database privileges, or
approve its own update.

## Before enabling updates

### Verify recovery first

For any non-disposable database, verify backups and point-in-time recovery
before enabling updates. Know the retention window, recovery objective, restore
owner, and date of the most recent successful restore exercise.

Daita receipts are execution evidence. They are not a backup, PostgreSQL write-
ahead log, replay system, or substitute for a tested restore procedure.

### Use a dedicated least-privileged role

Create and manage the database role outside Daita. Do not attach an
administrator, owner, superuser, `BYPASSRLS`, role-creation, database-creation,
or replication credential.

This example is a starting point for a DBA. Adapt database, schema, table, role,
and column names to the deployment, and provision the login secret through the
organization's normal secret-management process:

```sql
CREATE ROLE daita_writer LOGIN
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    NOREPLICATION
    NOBYPASSRLS;

GRANT CONNECT ON DATABASE application_db TO daita_writer;
GRANT USAGE ON SCHEMA support TO daita_writer;
GRANT SELECT ON TABLE support.tickets TO daita_writer;
GRANT UPDATE (priority) ON TABLE support.tickets TO daita_writer;
```

The role needs `SELECT` because Daita must preview and revalidate the complete
target set before updating it. Grant `UPDATE` only on the columns Daita should
be allowed to assign. PostgreSQL privileges remain effective even if Daita
permissions are later removed, so manage both boundaries.

Never place a database password, password-bearing connection URL, or
administrator credential in a prompt, source name, shell history, log, or test
output. Interactive setup stores the attached password through the configured
OS keychain. Headless attachment accepts the name of a password environment
variable through `--password-env`.

### Confirm table eligibility

The current update contract requires a cataloged base table with a supported
primary key and supported assignment types. Daita rejects:

- views, partitioned tables, and inherited tables;
- tables with row-level security enabled;
- tables with user triggers or custom rewrite rules;
- tables without a supported primary key;
- primary-key, identity, generated, unknown, or unsupported assignment
  columns; and
- powerful roles or roles missing the required `CONNECT`, `USAGE`, `SELECT`,
  or column-level `UPDATE` privileges.

Choose a different table or role, or have a DBA change the external database
configuration. Daita does not accept an administrator credential to remediate
readiness failures.

## Enable update access

Attaching a PostgreSQL source starts with read access and no update access.
During interactive attachment, Daita offers to configure update access after
the source is cataloged. To configure or change it later:

```text
/source permissions
```

Then:

1. Select the PostgreSQL source.
2. Select **PostgreSQL update access**.
3. Choose selected current tables or all current eligible tables.
4. Choose all eligible assignment columns or **Advanced** to select an exact
   subset for each table. Advanced selection gives the narrowest access.
5. Review the before/after permission summary.
6. Confirm only if the exact tables and columns are correct.

Enabling update access can also add the read access required for preview and
revalidation. Future tables are never automatically write-enabled. Changing a
source connection clears all PostgreSQL update scopes; review and enable them
again only after validating the replacement connection.

For automation or diagnostics, readiness can be checked without changing
permissions or data:

```bash
daita relational-update-readiness AGENT SOURCE_ID RESOURCE_ID \
  --assignment-column priority
```

Repeat `--assignment-column` when checking more than one assignment column.
Apply any reported role or grant remediation through the external DBA process.

## Request and review an update

Ask for the intended change in plain language and include an exact target. For
example:

```text
Set priority to high for support tickets where ticket_status is waiting and
category is billing.
```

Daita first runs a read-only preview. It identifies the complete matching
primary-key set, reports the exact matched row count, and displays at most five
bounded before/after samples. The samples help review the change; the matched
count and complete target fingerprint cover every selected row, including rows
not shown in the samples.

Before approving, verify:

- the source and table;
- every AND-combined filter;
- every assigned column and value;
- the exact matched row count; and
- the before/after samples.

Select **Approve once** only when all of those details are correct. Denying the
card performs no update. Approval is bound to that exact plan, preview, and row
count and cannot be reused for different arguments or a later run.

## What happens after approval

After approval, Daita opens one write transaction and locks the target rows. It
rechecks the live catalog, table guardrails, Daita permissions, PostgreSQL
privileges, matched row count, primary-key set, and current assigned values.

If anything changed after the preview, Daita rolls back before the `UPDATE` and
requires a fresh preview and approval. Otherwise it executes the parameterized
statement once and requires PostgreSQL's affected-row count to equal the
approved expectation.

An immutable local receipt records the exact run and call identity, target
fingerprints, expected row count, terminal outcome, and affected count when
known. Daita never automatically retries an update with uncertain commit
status.

## Outcomes and failures

### `committed`

PostgreSQL acknowledged the commit, Daita recorded the terminal receipt, and
the affected count matched the approved count. A separate read can confirm the
current application state when operational policy requires it.

### `not_committed`

Daita established that the approved update did not commit. Common causes
include target drift, constraint violations, permission changes, and statement
or lock timeouts. Correct the underlying issue, then request a new preview. A
previous approval is never reused.

### `outcome_unknown`

The commit was attempted, but Daita could not establish whether PostgreSQL
committed it, or Daita could not durably establish the terminal acknowledgement.
This outcome does not mean that zero rows changed.

When an outcome is unknown:

1. Remove update access for the affected table and columns.
2. Preserve the receipt ID and exact run/call identity shown in the result.
3. Do not retry the update and do not ask the model to repeat it.
4. Read the target rows again and compare their current state with the intended
   result.
5. Review PostgreSQL logs and operational evidence through the authorized DBA
   process.
6. Decide separately whether a corrective update is required, and require a
   new human preview and approval for it.

Seeing the intended state in a later read does not prove that this particular
attempt caused it. The original receipt remains immutable and is never changed
from `outcome_unknown` to `committed` or `not_committed`.

## Disable update access

Use `/source permissions`, select **PostgreSQL update access**, and choose
**No update access**. This removes Daita's update scopes but does not revoke the
database role's PostgreSQL grants.

For incident response, remove the Daita scope first, then revoke or rotate the
database credential through the external DBA or secret-management process.
Detaching the source removes its Daita permission scopes and Daita-owned
credential, but it does not change database grants or restore data.


## Structured upsert

The public tools are `data_preview_update_rows`, `data_update_rows`,
`data_preview_upsert_rows`, and `data_upsert_rows`. The model supplies exact
source/resource IDs and scalar values, never SQL. Both execution tools require
a successful authenticated preview from the same run with matching intent.

Upsert supplies exact `key_columns`, `insert_columns`, `update_columns`, and
`rows`. Every row has exactly the insert-column shape. The key columns must be
included, and update columns must be a nonempty subset of the supplied non-key
columns. Optional `evidence_call_ids` retain ordered, successful current-run
research lineage. These values are explicitly model-derived claims; a URL is
not independent proof and a successful transaction does not verify research.

Conflict keys must be cataloged, immediate, valid, plain, nonpartial unique
B-tree keys with supported built-in operator classes and non-null columns.
Supported key types are boolean, smallint, integer, bigint, UUID, text, and
varchar. Text keys require cataloged `C` or `POSIX` collation. UUID keys are
normalized before duplicate detection. Other key/type/collation semantics are
rejected. All input keys must be distinct under the admitted equality.

An explicit null assigns SQL null when permitted. Omission never clears an
existing column. On insertion, omitted nullable columns without defaults become
null. Omitted required values and other omitted default expressions are rejected.
Generated expressions are unsupported. Integer identity columns outside the
conflict key may be omitted only when explicitly admitted; preview does not
allocate or promise their values. Success returns the observed generated values.

The initial upsert contract retains role, table, RLS, trigger, rule, partition,
and inheritance restrictions. It also rejects check, exclusion, foreign-key and
deferrable constraints (including foreign keys referencing the target), expression/partial indexes, and unsupported index access
methods/operator classes. Its role needs insertion privileges for the selected
columns and table-level UPDATE privilege for the explicit lock. Readiness is a
bounded inspection, not proof that a future call will succeed.

Execution starts one transaction, configures fixed lock/statement deadlines, and
acquires `EXCLUSIVE` on the exact table before its authoritative scan. Ordinary
reads may continue; other writers and row-locking readers wait. The locked scan
rebuilds the preview. Changed existence, relevant values, structure, privileges,
or local permission reject the entire batch before mutation. Missing rows are
inserted, changed rows updated, and unchanged rows skipped. Exact counts must
sum to the input count or the whole transaction rolls back.

A verified unchanged batch succeeds with zero rows changed, consumes its
invocation reservation, and produces a receipt. There is no automatic chunking,
retry, or replay. Unknown commits block future effects until explicit human
recovery. A rollback describes table-row non-application; PostgreSQL identity
sequence allocations can leave gaps and are not restored by rollback.

## Explicit relational write permissions

`RelationalWriteScope` and the SQLite `relational_write_scopes` family bind one
agent/source/resource, structural revision, explicit `update` and/or `upsert`
operations, insertion and update columns, exact keys, admitted identity generation,
and a row ceiling. Update-only permission never implies insertion authority.
Read permission remains separate; detach revokes both permission families.

The existing permission preview/apply flow accepts an exact constraint mapping:

```python
preview = await agent.preview_source_permissions(
    source_id=source_id,
    read_mode="selected",
    read_resource_ids=(companies_resource_id,),
    relational_write_scopes={companies_resource_id: {
        "allowed_operations": ("upsert",),
        "key_columns": ("domain",),
        "allowed_insert_columns": ("domain", "name", "evidence_url"),
        "allowed_update_columns": ("name", "evidence_url"),
        "generated_identity_columns": ("id",),
        "max_rows": 100,
    }},
)
# Inspect the exact before/after state and confirmation before applying it.
await agent.apply_source_permissions(
    source_id=source_id,
    confirmation_fingerprint=preview.confirmation_fingerprint,
)
readiness = await agent.relational_upsert_readiness(source_id, companies_resource_id)
```

The table must already have a supported unique `domain` key and the admitted
integer identity `id`. The API binds current structural facts rather than accepting
a caller's replacement schema. Current TUI column selection configures update-only
permission; explicit upsert configuration uses this typed API. Permission inspection
shows both operations and their exact constraints.

Before write preparation and dispatch, the full request classification must fit
the target's current admitted classification. Higher-classified research/context
cannot be stored in a lower-classified target merely because columns are writable.

## Native routines and release limits

A native routine grant uses `constraints_kind="data.relational_write"` and freezes
`source_id`, `resource_id`, `resource_revision`, `key_columns`,
`allowed_insert_columns`, `allowed_update_columns`, `generated_identity_columns`,
and `max_rows`. The requested call ceiling must be one, and the routine may contain
at most one native write capability. Current permissions and retained contracts
revalidate before dispatch. Research tools must admit the full outbound sensitivity.

An immediate-first weekly assignment creates one immediate occurrence and retains
its calendar schedule. Each occurrence uses the ordinary loop, previews its current
findings, reserves one native call, and reports authenticated transaction counts.
A required write cannot be satisfied by model prose or an unrelated research result.
An approved optional no-findings path can omit the effect; a required effect remains
unsatisfied when research yields no batch.

This native implementation has deterministic acceptance coverage with fake external
I/O. It is not production release approval or live PostgreSQL certification.
Shared MCP external actions and the remaining product/recovery integration are
separate later work. Routines execute only while an eligible host is open.
