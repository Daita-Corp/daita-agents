# PostgreSQL updates

Daita begins with read access only. PostgreSQL updates are an explicit opt-in
for users who need the agent to change selected columns in selected tables.
Every update is structured, previewed, checked against the current catalog and
permissions, approved once, and executed in a transaction. The model never
writes SQL.

Updates can affect one row or many rows. There is no product row-count ceiling,
so treat every preview as a potentially high-impact database operation.

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

Daita does not support inserts, deletes, arbitrary SQL, DDL, role or grant
administration, or writes to SQLite, CSV, or JSON sources. Primary-key,
identity, and generated columns cannot be assigned.

## Safety model

Four independent controls must all permit an update:

1. **PostgreSQL privileges:** the attached database role must have the required
   native privileges.
2. **Daita source permissions:** the exact table and assignment columns must be
   enabled by the user.
3. **Current readiness:** the live table, role, grants, and catalog state must
   pass Daita's non-mutating checks.
4. **Once-only approval:** the user must approve the exact previewed update in
   the terminal.

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
daita postgresql-update-readiness AGENT SOURCE_ID RESOURCE_ID \
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
