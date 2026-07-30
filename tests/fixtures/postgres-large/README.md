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

Two deliberate PostgreSQL boundary cases are present:

- `catalog.unsupported_type_probe` uses a custom enum. Daita currently omits
  an entire table containing a non-`pg_catalog` type rather than execute an
  unproven custom output function.
- `analytics.monthly_revenue` is a view. Daita's PostgreSQL MVP catalogs only
  base tables, so the view is visible to PostgreSQL but absent from Daita's
  catalog.

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

## Run the opt-in fixture test

The test uses PostgreSQL I/O but a fake model boundary, so it incurs no model
cost:

```bash
DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 \
DAITA_LARGE_POSTGRES_PASSWORD=daita_large_fixture_password \
.venv/bin/python -m pytest tests/test_postgres_large_fixture.py -v
```

## Stop and discard it

```bash
docker compose -f tests/fixtures/postgres-large/compose.yaml down
```
