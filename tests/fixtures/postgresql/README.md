# PostgreSQL read fixture

This developer-operated fixture runs a disposable PostgreSQL database for
catalog and read-query testing. It exposes a commerce-oriented `analytics` schema through the
read-only `daita_reader` role. The database files live in container tmpfs and
are discarded by `docker compose down`.

The schema contains eight related tables and generated test data:

- `regions`
- `customers`
- `products`
- `orders`
- `order_items`
- `payments`
- `shipments`
- `support_tickets`

The dedicated PostgreSQL update fixture is
[`../postgres-large`](../postgres-large). Keeping updates there gives the TUI,
permissions flow, single-row selections, and bulk selections one shared source
instead of maintaining a second write-specific database shape here.

## Start the read fixture

Install the complete application with `pipx install daita-agents` before using
the customer-facing TUI walkthrough.

From the repository root:

```bash
docker compose -f tests/fixtures/postgresql/compose.yaml up -d --wait
```

Attach it in the TUI with:

Use ↑/↓ to move, Space to toggle the `analytics` schema, Enter to confirm, and
Escape to cancel. Starting a live provider call remains explicitly user-driven.

```text
Display name: Fixture PostgreSQL
Host: 127.0.0.1
Port: 55432
Database: daita_fixture
Username: daita_reader
Password: daita_fixture_password
SSL mode: disable
Schema: analytics
```

For headless developer automation:

```bash
export DAITA_FIXTURE_POSTGRES_PASSWORD=daita_fixture_password

daita --root /private/tmp/daita-live attach atlas postgresql \
  --host 127.0.0.1 \
  --port 55432 \
  --database daita_fixture \
  --username daita_reader \
  --password-env DAITA_FIXTURE_POSTGRES_PASSWORD \
  --schema analytics \
  --ssl-mode disable \
  --source-name "Fixture PostgreSQL"
```

Run its opt-in acceptance test only after starting the fixture:

```bash
DAITA_RUN_POSTGRES_FIXTURE=1 \
DAITA_FIXTURE_POSTGRES_PASSWORD=daita_fixture_password \
.venv/bin/python -m pytest tests/test_postgresql_fixture.py -v
```

Stop and discard it with:

```bash
docker compose -f tests/fixtures/postgresql/compose.yaml down
```
