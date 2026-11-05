# Relational writes

Daita starts with read access. You can separately allow it to insert, update or
delete rows in named PostgreSQL tables. The agent writes one SQL statement with
`data_write`; Daita previews it, a person approves it, and Daita applies it once
with a receipt. PostgreSQL is the only source type that can be written: SQLite,
files and folders cannot.

For release evaluation with real model generation and a disposable PostgreSQL
database, see the [live model acceptance suite](../tests/live/data/test_model_write_acceptance.py).
Real-server behaviour with a scripted model is in
[the source-first write suite](../tests/live/data/test_source_first_writes.py).

## What a write can be

One statement on one target table:

- `INSERT` with a column list, from `VALUES` or a `SELECT`, including
  `ON CONFLICT (columns) DO UPDATE` and `DO NOTHING`;
- `UPDATE`, including `UPDATE ... FROM`;
- `DELETE`, including `DELETE ... USING`.

A batch upsert is `INSERT ... VALUES ... ON CONFLICT` with bound parameters, at
most 1,000 rows and 64 KiB of SQL and parameters. Parameters are `$1, $2, ...`.

The server receives the statement exactly as written, apart from a trailing `;` and
the `RETURNING` Daita adds to see what it touched. Because Daita adds its own, a
statement cannot have a `RETURNING`; read a generated value with `data_query` after
the write.

## How a write is judged

Four rules decide everything, and they replace lists of risky statement shapes:

1. **Run what was checked.** Daita never rewrites the statement.
2. **Ask the source, don't guess.** Before anything runs, PostgreSQL reports what the
   exact text targets (`EXPLAIN`). The parser only gives early, friendly errors and
   builds the preview.
3. **Verify the effect.** The preview is a claim. Before committing, Daita compares the
   rows the statement touched with the rows on the card, and rolls back if they differ.
4. **Refuse by principle, not by shape.** There are only six refusals:

| Refusal | Examples |
| --- | --- |
| It doesn't parse | SQL that fails to parse never runs |
| Not exactly one write on one table with permission for that operation and those columns | DDL, `TRUNCATE`, `COPY`, `MERGE`, two statements, a data-modifying CTE, `SELECT INTO`, a locking clause, `INSERT` with no column list, a table with no write scope, a view, a table that fails the [eligibility checks](#table-eligibility) |
| A read outside the run's scope, or a function the [function rule](architecture/data-and-files.md) refuses | A table outside the run, `pg_read_file` |
| Too large | Over 64 KiB, 1,000 `VALUES` rows or the scope's `max_rows` |
| The source can't preview it | One error, `write_preview_failed`, carrying PostgreSQL's own message: `LIMIT` on a `DELETE`, `DEFAULT` in `VALUES`, a misspelled column, a `RETURNING` |
| The rows touched don't match the card | `write_state_changed` before the statement runs, `write_affected_rows_mismatch` after it |

Anything else that looks risky is a warning on the card: no `WHERE` or `WHERE true`,
time or randomness, a row that matches twice with different new values, triggers and
cascading foreign keys. A statement that changes no rows has nothing to approve.

## Safety model

All of these must permit a write:

1. **PostgreSQL privileges:** the attached role must hold the native privileges.
2. **Daita write scope:** a person enabled that table, operation and those columns.
   Delete is enabled per table and never implied.
3. **Live checks:** the table and role pass Daita's non-mutating checks.
4. **Approval at the time:** every write is one call a person approves. Daita builds
   the preview itself; routine occurrences, jobs and their sub-runs ask the same way.

The model cannot enable permissions, grant privileges or approve its own write.
The catalog is not involved: a table with no catalog entry can be enabled and
written, and a stale or failing catalog never blocks a write.

## Before enabling writes

### Verify recovery first

For any non-disposable database, verify backups and point-in-time recovery first.
Daita receipts are execution evidence. They are not a backup, a write-ahead log, a
replay system or a substitute for a tested restore.

### Use a dedicated least-privileged role

Create and manage the role outside Daita. Do not attach an administrator, owner,
superuser, `BYPASSRLS`, role-creation, database-creation or replication credential.

```sql
CREATE ROLE daita_writer LOGIN
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;

GRANT CONNECT ON DATABASE application_db TO daita_writer;
GRANT USAGE ON SCHEMA support TO daita_writer;
GRANT SELECT ON TABLE support.tickets TO daita_writer;
GRANT UPDATE (priority) ON TABLE support.tickets TO daita_writer;
```

The role needs `SELECT` because Daita previews and rechecks the rows. PostgreSQL
privileges stay effective if Daita permissions are removed, so manage both.

Never place a password or password-bearing URL in a prompt, source name, shell
history, log or test output.

### Table eligibility

Daita refuses views, partitioned and inherited tables, tables with row-level
security or custom rewrite rules, and powerful roles or roles missing the needed
privileges. Triggers and cascading foreign keys are allowed: the card names them
and says their effects are not previewed. A table with a primary key or a unique
not-null key is checked more closely than one without.

## Enable writes

Attaching a PostgreSQL source grants no write permission. Open `/source permissions`,
choose the source and the maximum rows per call, select **Edit table write access**
and a table, then the operations and columns. Selecting no operation removes that
table's access. The picker lists cataloged tables; the Python API takes any table by
name and reads it live:

```python
preview = await agent.preview_source_permissions(
    source_id=source_id,
    read_mode="all",
    read_resource_ids=(),
    relational_write_scopes={"public.companies": {
        "allowed_operations": ("insert", "update"),
        "allowed_insert_columns": ("domain", "name", "evidence_url"),
        "allowed_update_columns": ("name", "evidence_url"),
        "max_rows": 100,
    }},
)
# Inspect the before/after state and confirmation before applying it.
await agent.apply_source_permissions(
    source_id=source_id,
    confirmation_fingerprint=preview.confirmation_fingerprint,
)
readiness = await agent.relational_write_readiness(source_id, "public.companies")
```

Review the **before/after** document, then **Apply**. Apply executes no write. Enabling
writes can add the read access preview needs. Editing one table keeps the others; a
replacement source connection clears write scopes.

Readiness can be checked without changing anything:

```bash
daita relational-write-readiness AGENT SOURCE_ID public.companies
```

## Preview and approval

`mode=apply` is the default: Daita builds the preview, shows it on the approval card
and writes once. `mode=preview` shows the same facts and changes nothing.

Preview first asks PostgreSQL what the statement targets. Planning a write takes a brief
`ROW EXCLUSIVE` table lock, even in a read-only transaction, so Daita releases it at
once. Behind a schema change or `LOCK TABLE`, preview can hit its lock timeout and
refuse with `write_lock_timeout`; nothing has changed.

It then derives a `SELECT` from the statement and runs it in a read-only transaction,
so no trigger fires and no row lock is taken:

- `UPDATE` and `DELETE`: the target rows, counted once each, with new values for
  `UPDATE`;
- `INSERT`: the incoming rows;
- `ON CONFLICT`: rows that would insert, update or do nothing, by the conflict columns.

Above the scope's `max_rows`, preview refuses. A statement that changes nothing cannot
be applied.

The card shows the exact SQL, the counts, up to five sample rows, the columns changed,
its warnings and how far Daita can verify:

| Verification | Meaning | When |
| --- | --- | --- |
| `rows` | The touched rows are compared before commit | A `DELETE`; or a table with a primary key or unique not-null key the statement does not assign and, for an `INSERT`, lists |
| `count` | Only the number of rows, and how many were inserted or updated, is compared | Everything else: a table with no such key, an update of every key, a generated key |

Verify the source, table and SQL before **Approve once**. Denying performs no write.
Approval is bound to that SQL, parameters, table structure, row identity and count, and
cannot be reused. New row values join that binding only when the statement is free of
time, randomness and functions whose result can change. Otherwise the sample shows new
values as `(computed when written)`, and the card says so.

## What happens after approval

Daita opens one write transaction with a lock timeout, re-reads the table's structure,
asks PostgreSQL again what the statement targets, locks existing target rows with
`FOR UPDATE`, and re-runs the derived `SELECT`. Any change refuses with
`write_state_changed`, including a concurrent change that keeps the same number of rows.

Otherwise it runs the statement once and compares what PostgreSQL returned with the
locked rows; a difference rolls back with `write_affected_rows_mismatch`. So a wrong
preview fails safe, and a table can refuse some writes: a trigger that skips a row or
changes its key fails verification, though one that only stamps another column does not.
A statement that picks rows at random can pass approval and then fail here; that is safe,
and the card warns first.

An immutable local receipt is reserved before dispatch and records the run and call
identity, fingerprints, expected count, outcome and affected count when known. A write
is never retried or replayed. Receipts from before `data_write`, kinds `data.update_rows`
and `data.upsert_rows`, still decode, list and resolve.

## Outcomes and failures

- **`committed`**: PostgreSQL acknowledged the commit and the touched rows matched as
  far as the card said they would be checked.
- **Refused before commit**: drift, constraint violations, permission changes or
  timeouts. Nothing was applied. Correct the issue and ask again; a new call shows a new
  preview.
- **`write_outcome_unknown`**: a commit was attempted and its result is unknown. This
  does not mean zero rows changed.

When an outcome is unknown:

1. Remove write access for the table.
2. Keep the receipt ID and the run and call identity.
3. Do not retry and do not ask the model to repeat it.
4. Read the rows again and compare them with the intended result.
5. Review PostgreSQL logs through the authorized DBA process.
6. Use the [recovery controls](EFFECT_RECEIPTS.md) to record a human decision. Recovery
   performs no action; a corrective write needs its own preview and approval.

## Disable write access

Use `/source permissions`, **Edit table write access**, choose the table and select no
operation, then review and apply. This does not revoke PostgreSQL grants. For an
incident, remove the Daita scope first, then revoke or rotate the credential externally.
Detaching the source removes its scopes and credential, not database grants.

## Routines and jobs

A routine occurrence or job run writes exactly as a foreground run does: one call, one
approval at the time (the card names the routine or job), the same recheck and receipt.
There is no standing permission to write without asking.

The full request's classification must fit the source's: higher-classified context
cannot be stored in a lower-classified source merely because columns are writable.
