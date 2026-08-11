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
authorization boundary even after Daita write admission is enabled.

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

Attachment always persists `write_access=false`. It cannot be combined with
enablement. Rotating the secret changes the value behind the same approved
secret reference; it does not broaden database grants or Daita admission.

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
tuple of assignment-column names. It can run while source admission is still
disabled.

Python:

```python
readiness = await agent.postgresql_update_readiness(
    source_id,
    resource_id,
    ("status",),
)
if readiness.ready_for_preview:
    await agent.set_source_write_access(source_id, True)
```

`set_source_write_access` changes only the persisted source-level Daita
admission flag. It never changes PostgreSQL privileges. Always rerun readiness
after enablement and before using a canary preview.

CLI:

```bash
daita postgresql-update-readiness atlas SOURCE_ID RESOURCE_ID \
  --assignment-column status
daita source-write-access inspect atlas SOURCE_ID
daita source-write-access enable atlas SOURCE_ID --yes
daita source-write-access disable atlas SOURCE_ID
```

TUI:

```text
/source config
```

Select the PostgreSQL source, cataloged table, and assignment columns by
display name. The guided screen runs readiness while disabled, explains that
enablement is source-scoped, and requires a separate user confirmation. The
model cannot invoke this control plane. Disabling remains immediately
available, requires no model interaction, and removes write-tool projection
from subsequent runs.

## Non-production canary procedure

Use only `tests/fixtures/postgresql`, whose `write_canary.accounts` rows and
dedicated `daita_writer` role are disposable. Never substitute production,
shared staging data, a persistent customer database, or
`/Users/jendala/.daita` for the temporary agent root or disposable database.

1. Close other Daita processes using the temporary root.
2. Start the fixture externally and wait for its health check.
3. Use a new root under `/private/tmp` and attach only `write_canary` with the
   `daita_writer` credential. Confirm the returned registration is read-only.
4. Run readiness for `write_canary.accounts` and assignment column `status`.
5. Confirm all database guardrails pass and the only expected blocker is Daita
   write admission.
6. Enable the exact source through API, CLI, or `/source config`.
7. Rerun readiness and request a preview for canary key `42`. Confirm the exact
   before/after value without approving an unintended assignment.
8. Approve the exact once-only update card. Confirm one committed row, one
   receipt, and agreement between result and receipt identity.
9. Read the row independently and verify the intended after-state.
10. Disable source admission immediately and confirm preview/update tools are
    no longer projected.
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

Disable Daita admission first when scope, credentials, schema, privileges, or
expected behavior are uncertain:

```bash
daita source-write-access disable atlas SOURCE_ID
```

Then revoke or rotate database access through the external DBA/secret process
when required. Disabling Daita does not revoke PostgreSQL grants, terminate
database sessions, restore data, or delete receipts.

## Unknown-outcome response

An `outcome_unknown` receipt means Daita attempted `COMMIT` but could not
establish whether PostgreSQL acknowledged it, or could not durably record the
terminal acknowledgement after the remote result. It does not mean zero rows
were committed.

When this occurs:

1. disable source write admission;
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
