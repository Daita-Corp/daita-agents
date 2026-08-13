# PostgreSQL one-row update rollout

## Phase 4 release note

Daita supports one deliberately narrow PostgreSQL mutation: an approved,
typed update of exactly one row selected by one cataloged single-column primary
key. The model supplies structured match and assignment values, never SQL.
Daita first produces a bounded preview, requests approval for the exact frozen
arguments, revalidates inside a fresh transaction, executes once with
`RETURNING`, and records a durable outcome receipt.

This release does not support inserts, deletes, bulk updates, SQLite writes,
arbitrary SQL, DDL, role administration, retries, pending workflows, or
reconciliation APIs. Preview is current evidence, not a guarantee that the
later transaction will commit.

## External least-privilege setup

Database role and grant administration stays outside Daita. A DBA or approved
infrastructure-as-code process must create one dedicated role with:

- `NOSUPERUSER`, `NOCREATEDB`, `NOCREATEROLE`, `NOREPLICATION`, and
  `NOBYPASSRLS`;
- `CONNECT` only to the selected database;
- `USAGE` only on selected schemas;
- the bounded `SELECT` needed for preview and revalidation; and
- `UPDATE` only on the intended tables or assignment columns.

Daita does not accept an administrator credential, create or drop roles, or
execute `GRANT` or `REVOKE`. PostgreSQL privileges remain an independent
authorization boundary even after an exact Daita update scope is enabled.

The first slice rejects superusers, `BYPASSRLS`, row-level security, user
triggers, custom rewrite rules, partitions/inheritance, non-base relations,
missing or composite primary keys, generated/identity assignment columns, and
unsupported types. Resolve role or grant findings externally, then rerun
targeted readiness.

## Credential handling

Attach only the dedicated writer role through a secret reference. Interactive
TUI setup stores the password through the configured OS keychain. Headless
automation should name an environment variable with `--password-env`; never
place a password, connection URL containing a password, or administrator
credential in command history, model text, source names, logs, or test output.

Attachment always begins with read access to all cataloged resources and zero
PostgreSQL update scopes. Permission scopes are stored separately from connection
configuration; public source registrations contain no permission projection.
Rotating the secret changes the value behind the same approved secret reference;
it does not broaden database grants or Daita admission.

## Backup and recovery gate

Before any non-disposable rollout, independently verify database backups and
point-in-time recovery. Record the recovery objective, retention window,
restore owner, and the most recent successful restore exercise. Daita receipts
are execution evidence; they are not a database backup, transaction log,
replay system, or substitute for recovery testing.

Do not use this Phase 4 procedure until the target has a tested recovery path.
For the repository fixture, PostgreSQL data lives in container tmpfs and is
discarded instead of restored.

## Readiness and admission controls

Readiness is non-mutating and scoped to one current resource plus a distinct
tuple of assignment-column names. It requires the exact configured table and
column scope.

Python:

```python
preview = await agent.preview_source_permissions(
    source_id=source_id,
    read_mode="all",
    read_resource_ids=(),
    postgresql_update_scopes={resource_id: ["status"]},
)
await agent.apply_source_permissions(
    source_id=source_id,
    confirmation_fingerprint=preview.confirmation_fingerprint,
)
readiness = await agent.postgresql_update_readiness(
    source_id, resource_id, ("status",)
)
```

The public inspect/preview/apply control plane changes only exact read and update
scopes. It never rewrites connection identity or changes PostgreSQL privileges.
Refresh preserves scopes; detachment deletes them atomically. Always rerun
readiness after permissioning and before using a canary preview.

CLI:

```bash
daita postgresql-update-readiness atlas SOURCE_ID RESOURCE_ID \
  --assignment-column status
```

TUI:

```text
/source permissions
```

Select the source, read mode, cataloged resources, and PostgreSQL tables by
display name. Normal setup uses all eligible assignment columns; Advanced
selects a bounded exact subset. The model cannot invoke this control plane.
One terminal confirmation atomically replaces both scope families.

## Non-production canary procedure

Use only `tests/fixtures/postgresql`, whose `write_canary.accounts` rows and
dedicated `daita_writer` role are disposable. Never substitute production,
shared staging data, a persistent customer database, or
`/Users/jendala/.daita` for the temporary agent root or disposable database.

1. Close other Daita processes using the temporary root.
2. Start the fixture externally and wait for its health check.
3. Use a new root under `/private/tmp` and attach only `write_canary` with the
   `daita_writer` credential. Inspect its permissions and confirm `all` read
   access with zero exact update scopes; the public registration contains no
   permission projection.
4. Run readiness for `write_canary.accounts` and assignment column `status`.
   Confirm it returns `resource_update_not_allowed` before source I/O.
5. Configure the exact table and assignment-column scope through
   `/source permissions`.
6. Rerun readiness and confirm both the exact Daita scope and native PostgreSQL
   privileges are ready.
7. Request a preview for canary key `42`. Confirm the exact before/after value
   without approving an unintended assignment.
8. Approve the exact once-only update card. Confirm one committed row, one
   receipt, and agreement between result and receipt identity.
9. Read the row independently and verify the intended after-state.
10. Remove the exact update scope and confirm preview/update tools are no longer
    projected.
11. Reset or discard the fixture externally.

After deterministic gates pass and the disposable database is explicitly
authorized, run DB-only certification first:

```bash
export DAITA_FIXTURE_POSTGRES_WRITER_PASSWORD='<fixture writer secret>'
export DAITA_FIXTURE_POSTGRES_ADMIN_PASSWORD='<fixture reset-only secret>'
export DAITA_RUN_POSTGRESQL_UPDATE_CERTIFICATION=1

.venv/bin/python -m pytest \
  tests/test_postgresql_update_certification_live.py \
  -m "requires_db and integration" -v -s
```

The fixture administrator credential is used only by external test reset and
verification. It must never be passed to `Agent`, `PostgreSQLSource`,
readiness, or a model tool.

Only after DB-only certification passes, separately authorize the single paid
model/database acceptance test:

```bash
export OPENAI_API_KEY='<approved provider secret>'
export DAITA_RUN_LIVE_POSTGRESQL_UPDATE_ACCEPTANCE=1
export DAITA_POSTGRESQL_UPDATE_ACCEPTANCE_MAX_COST_USD=0.20

.venv/bin/python -m pytest \
  tests/live/test_postgresql_update_acceptance_live.py \
  -m "requires_llm and requires_db and acceptance" -v -s
```

Do not use paid runs to diagnose deterministic failures.

## Disablement and incident response

Remove the exact update scopes through `/source permissions` first when scope,
credentials, schema, privileges, or expected behavior are uncertain.

Then revoke or rotate database access through the external DBA/secret process
when required. Disabling Daita does not revoke PostgreSQL grants, terminate
database sessions, restore data, or delete receipts.

## Unknown-outcome response

An `outcome_unknown` receipt means Daita attempted `COMMIT` but could not
establish whether PostgreSQL acknowledged it, or could not durably record the
terminal acknowledgement after the remote result. It does not mean zero rows
were committed.

When this occurs:

1. remove the affected update scope;
2. preserve the immutable receipt ID and exact run/call identity;
3. do not retry automatically and do not ask the model to repeat the update;
4. perform a fresh read/preview of the target row;
5. report whether the intended after-state is currently observed, without
   claiming that observation proves this exact attempt committed;
6. review PostgreSQL logs and operational evidence through authorized external
   procedures; and
7. require a new human decision before any corrective mutation.

Never rewrite the original receipt from `outcome_unknown` to committed or
rolled back. A later observation or corrective action is separate evidence.
