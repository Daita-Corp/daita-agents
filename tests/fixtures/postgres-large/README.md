# Large multi-schema PostgreSQL fixture

This is Daita's opt-in production-shape PostgreSQL fixture. It complements the
small `tests/fixtures/postgresql` smoke fixture rather than replacing it. The
database exercises real multi-schema discovery, duplicate relation names,
cross-schema and composite foreign keys, bounded catalog projection, qualified
SQL validation, reader-role visibility, and a materially larger fact workload.

The fixture exposes these schemas to `daita_large_reader`:

- `core`
- `catalog`
- `sales`
- `billing`
- `support`
- `analytics`
- `archive`

It also creates an empty `staging` schema and an inaccessible `private` schema.
Both can appear in the bounded schema probe, but neither exposes a base table to
the fixture reader. `sales.orders` and `archive.orders` intentionally have the
same short name. Foreign keys cross the exposed schema boundaries.

The reader-visible catalog contains 34 supported base tables and 47
relationships. It includes approximately 10,000 customers, 5,000 products,
100,000 current orders, 300,000 current order items, 80,000 invoices, 240,000
invoice lines, 20,000 support tickets, and 20,000 archived orders.

The separate `daita_large_writer` role is available only for disposable TUI
update testing. It can catalog and select `support.tickets`, and its
only PostgreSQL mutation privilege is column-scoped `UPDATE (priority)` on
that table. The production-shape `daita_large_reader` remains read-only.

The automated write release tests use `daita_large_write_tester` and the isolated
`write_acceptance.companies` and `write_acceptance.cells` tables. This does not add
tables or permissions to the reader or change the original update canary. The
company table's integer identity and two unique text keys exercise generated IDs,
conflict-key matching, and independent server
constraint failures. The narrow integer `cells` table reaches the 1,000-row ceiling
without exceeding the byte or exact approval bounds. The role has SELECT,
table-level UPDATE (needed for the EXCLUSIVE lock), column-scoped INSERT, and
identity-sequence USAGE. It has no
DELETE, TRUNCATE, DDL, administrative, or role-management privilege. Daita's own
permission scope further restricts each admitted operation and column.

Two deliberate PostgreSQL boundary cases are present:

- `catalog.unsupported_type_probe` uses a custom enum. Daita currently omits
  an entire table containing a non-`pg_catalog` type rather than execute an
  unproven custom output function.
- `analytics.monthly_revenue` is a view. Daita catalogs only PostgreSQL base
  tables, so the view is visible to PostgreSQL but absent from Daita's catalog.

All generated values are deterministic. The database files live in container
tmpfs and are discarded by `docker compose down`.

## Start the fixture

Docker is external and developer-operated. From the repository root:

```bash
docker compose -f tests/fixtures/postgres-large/compose.yaml up -d --wait
```

The default host port is `55433`. Override it without changing the fixture:

```bash
DAITA_LARGE_POSTGRES_PORT=55434 \
docker compose -f tests/fixtures/postgres-large/compose.yaml up -d --wait
```

## Attach Daita

Keep the fixture password in an environment variable:

```bash
export DAITA_LARGE_POSTGRES_PASSWORD=daita_large_fixture_password

daita --root /private/tmp/daita-large attach atlas-large postgresql \
  --host 127.0.0.1 \
  --port 55433 \
  --database daita_large_fixture \
  --username daita_large_reader \
  --password-env DAITA_LARGE_POSTGRES_PASSWORD \
  --schema core \
  --schema catalog \
  --schema sales \
  --schema billing \
  --schema support \
  --schema analytics \
  --schema archive \
  --ssl-mode disable \
  --source-name "Large multi-schema PostgreSQL"
```

Useful manual prompts include:

- “Which region has the most paid invoiced revenue?”
- “Compare current and archived order volume by customer segment.”
- “Which product categories have the highest return rate?”
- “How does support-ticket volume relate to customer revenue?”
- “Explain the relationship path from refunds to products.”

## TUI update tests

Recreate the tmpfs-backed fixture after changing `init.sql`, then use a new
agent root under `/private/tmp`. Attach a second PostgreSQL source with these
values:

```text
Display name: Large PostgreSQL write canary
Host: 127.0.0.1
Port: 55433
Database: daita_large_fixture
Username: daita_large_writer
Password: daita_large_writer_fixture_password
Schema: support
SSL mode: disable
```

Through `/source permissions`, select `support.tickets`, choose update access,
and admit only `priority` as an update column. The
deterministic fresh-fixture canary is `ticket_id = 42`, whose initial priority
is `medium`. Use it to verify that a single-row selection goes through the same
preview and `[Y] Approve once` flow as a bulk selection.

For a deterministic bulk target, select tickets where
`ticket_status = 'waiting'` and `category = 'billing'`. On a fresh fixture,
those rows have priority `low`. Preview changing their priority to `high`,
confirm the exact
matched row count and bounded before/after samples, deny once, and verify no
rows changed. Repeat and approve once, independently read the aggregate back,
then preview and approve restoring the same selection to `low`.

Finally restore ticket 42 to `medium` if it was changed and remove update
access. Never use `daita_large_reader` or the fixture administrator credential
for update testing.

## Run the opt-in fixture test

The test uses PostgreSQL I/O but a fake model boundary, so it incurs no model
cost:

```bash
DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 \
DAITA_LARGE_POSTGRES_PASSWORD=daita_large_fixture_password \
DAITA_LARGE_POSTGRES_WRITER_PASSWORD=daita_large_writer_fixture_password \
.venv/bin/python -m pytest tests/live/data/test_large_fixture.py \
  -o addopts="--tb=short -q --strict-markers" -v
```

## Native write release checks

These opt-in tests use the real public Agent API, catalog discovery, permission
preview/apply, SQL validation, asyncpg, PostgreSQL transactions, and persisted
effect receipts. The model is scripted: no LLM credentials, paid model requests,
MCP calls, or live external services beyond this local fixture are needed.

The suite covers:

- Mixed insert/update/unchanged batches, exact counts and generated identities;
  unchanged repeats; omitted values versus explicit nulls.
- Approval denial, stale row values and structure, revoked database privileges,
  and Daita permission revocation retained after reopen.
- Duplicate/null keys, explicit identity assignments, row and byte ceilings,
  and update permission that cannot authorize upsert.
- A real second-insert unique violation rolling back the entire batch; a
  deliberately corrupted driver row count rolling back a real update.
- Real lock timeout and a concurrent insertion while the upsert waits for its
  EXCLUSIVE lock. The authoritative scan must see the committed competitor.
- Driver cancellation after mutation and after commit; foreground cancellation
  that drains an already-started write and retains its verified commit; lost
  commit confirmation;
  persisted uncertainty, blocked subsequent writes, and both exact human recovery
  decisions performing no action and granting no permission.
- Narrow batch sizes 1, 100 and 1,000, with actual row counts and elapsed-time
  samples in JUnit properties. These are bounded canary measurements, not a throughput or
  p95 latency certification. A wide 1,000-row batch is separately required to fail
  before I/O with an actionable size/shape error; the row ceiling cannot bypass
  byte or approval display bounds.

First start a **fresh disposable fixture built from the current `init.sql`** using
the command above. Existing containers do not rerun initialization SQL. Recreate
only this fixture when its disposable data is no longer needed; the tests never
start, stop or recreate Docker, or delete an existing agent home.

Run serially from the repository root:

```bash
DAITA_RUN_POSTGRES_WRITE_RELEASE=1 \
.venv/bin/python -m pytest tests/live/data/test_write_release.py -v \
  -o addopts="--tb=short -q --strict-markers" \
  -o junit_family=xunit1 --junitxml=/private/tmp/daita-postgresql-write-release.xml
```

The harness is fixed to `127.0.0.1`, database `daita_large_fixture`, and the
dedicated roles. `DAITA_LARGE_POSTGRES_PORT` selects the fixture port (default
55433). Password overrides are `DAITA_LARGE_POSTGRES_ADMIN_PASSWORD` and
`DAITA_LARGE_POSTGRES_WRITE_TESTER_PASSWORD`; defaults match `init.sql` and Compose.
Do not point this harness at customer data.

The fixture administrator verifies the ready marker and holds an advisory lock
before resetting only the two `write_acceptance` canary tables. A concurrent suite
fails instead of sharing canary data. Administrator access is used for seed/readback,
controlled privilege/structure changes, and cleanup; Daita always connects as the
restricted tester. Teardown restores canary privileges, removes the test-added
column, empties the canary tables, and checks that write connections were closed.
The original large workload and `support.tickets` are preserved.

Commit-loss cases inject a disconnect immediately before sending COMMIT, or hide
confirmation after a real successful COMMIT. Cancellation and count corruption
also use explicitly labeled driver-boundary hooks; catalog truth, transactions,
SQL results in ordinary cases, and receipts remain production-owned. These tests
do not claim to exercise actual dropped network packets, a PostgreSQL server
crash, managed-host failover, TLS/pooler behavior, or every PostgreSQL version.
Run those checks separately for the intended deployment where applicable.

Without database authorization, validate the test harness offline:

```bash
.venv/bin/python -m pytest tests/data/postgresql/test_write_release_harness.py \
  tests/data/postgresql/test_write_fixture_contract.py -v
.venv/bin/python -m pytest tests/live/data/test_write_release.py --collect-only \
  -o addopts="--tb=short -q --strict-markers"
```

Release evidence requires a passing authorized database run and review of its
assertions, timings, and remaining deployment-specific limits. Collection and
offline harness checks alone do not establish release readiness.

## Live LLM decisions with this fixture

The separate [live model acceptance suite](../../live/data/test_model_write_acceptance.py)
uses actual API generation and the production router with these same canaries.
It covers model-driven discovery, writes, refusals, failure interpretation,
recovery and immediate/weekly routine authoring. It requires its own paid-run
authorization and environment credential; the default twelve cases admit at most
seventeen bounded agent runs and $2.55 estimated cost per model/repetition.
The explicit `user_flow` profile uses ordinary requests and prose answers,
with thirteen cases, at most nineteen runs and $9.50 estimated per
model/repetition. It uses the production 100,000-token, 24-request, 300-second
outer limits with a $0.50 estimated per-run ceiling; the strict profile remains
unchanged. Exact execution assertions and separate answer review are required.
Independent owner-seeded recovery and scheduled cases are labeled separately
from model-authored end-to-end cases. Database-only results above
do not substitute for a passing live model run.

## Stop and discard it

```bash
docker compose -f tests/fixtures/postgres-large/compose.yaml down
```
