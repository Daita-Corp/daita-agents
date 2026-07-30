# PostgreSQL live fixture

This fixture runs a disposable PostgreSQL database for manual CLI testing and
one opt-in terminal acceptance test. It exposes a larger commerce-oriented
`analytics` schema through the read-only `daita_reader` role. The database
files live in container tmpfs and are discarded by `docker compose down`.
Daita production code never starts, stops, or otherwise manages this fixture.

Every fresh container initialization generates new values with PostgreSQL's
`random()` function. The structure, constraints, value domains, and approximate
distribution remain stable, while customer/product names, relationships,
timestamps, prices, quantities, discounts, channels, payments, shipments, and
support activity change. Use `docker compose down` followed by `up -d --wait`
to obtain a new dataset; restarting the same container does not rerun the init
script.

The schema contains eight related tables:

- `regions`
- `customers`
- `products`
- `orders`
- `order_items`
- `payments`
- `shipments`
- `support_tickets`

It creates 1,000 customers, 250 products, 6,000 orders, one to four randomized
items per order, payments and shipments for completed/refunded orders, and 800
support tickets.

Docker is external and developer-operated. From the repository root, a
developer may explicitly start the fixture and wait for its health check:

```bash
docker compose -f tests/fixtures/postgresql/compose.yaml up -d --wait
```

For the customer-facing terminal walkthrough, install the complete application:

```bash
pipx install daita-agents
daita --root /private/tmp/daita-live
```

Inside `daita`, use ↑/↓ to move and Enter to confirm the provider, its
provider-specific model suggestion (or manual model entry), and PostgreSQL as
the source type. API keys and database passwords are entered only through
hidden prompts; never place them in command-line arguments. Enter these
non-secret PostgreSQL fields when prompted:

```text
Display name: Fixture PostgreSQL
Host: 127.0.0.1
Port: 55432
Database: daita_fixture
Username: daita_reader
SSL mode: disable
```

In the schema picker, move with ↑/↓, press Space to toggle `analytics`, and
press Enter to confirm. The initial selection is empty, at least one schema is
required, and only the selected stable schema names are attached. Press Escape
to cancel the current selector without attaching the source; Ctrl-C or EOF also
uses the terminal application's cleanup and lock-release paths.

This walkthrough reaches `Ready` and terminal chat only if the already-running
fixture and the selected provider are available. Both the PostgreSQL connection
and any live provider call are explicitly opt-in. Production Daita never starts,
stops, or manages Docker.

Good first prompts are “What tables and relationships are available?”,
“Summarize paid revenue and margin by region”, “Which product categories have
the highest average discount?”, and “Does ticket volume correlate with delayed
shipments?”. If `atlas` already exists, reuse it or choose another agent/root.

The advanced/headless attach remains available for developer automation. Keep
the password in an environment variable rather than an argument:

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

To run the opt-in acceptance test against the already-running fixture:

```bash
DAITA_RUN_POSTGRES_FIXTURE=1 \
DAITA_FIXTURE_POSTGRES_PASSWORD=daita_fixture_password \
.venv/bin/python -m pytest tests/test_postgresql_fixture.py -v
```

The acceptance test drives the zero-argument terminal controller from agent
creation through model configuration, PostgreSQL probing, `analytics` schema
selection, catalog summary, and one grounded read-only query. Its model
provider and OS keychain boundaries are fakes: it makes no live provider call
and writes no real keychain entry. Only PostgreSQL I/O goes to the fixture.

## Optional live learning confidence gate

The required learning exit gate remains offline:

```bash
.venv/bin/python -m pytest tests/test_learning_evaluation_phase3.py -v
```

After that gate passes, one explicitly authorized live test measures whether a
real model recognizes an ordinary-language definition without `/learn`, uses
the exact approval path, persists it, recalls it after close/reopen, and applies
the learned paid-order formula against this fixture. Start the fixture first,
then provide an exact release-reviewed model identity, its API key, an absolute
external report directory, and the maximum estimated cost allowed for each
agent run:

```bash
export DAITA_RUN_LIVE_LEARNING_EVAL=1
export DAITA_EVAL_MODEL_ID=openai:gpt-5.6-terra
export DAITA_EVAL_LLM_API_KEY='<provider API key>'
export DAITA_FIXTURE_POSTGRES_PASSWORD=daita_fixture_password
export DAITA_EVAL_OUTPUT_DIR=/private/tmp/daita-learning-evaluation
export DAITA_EVAL_MAX_COST_USD=0.50

.venv/bin/python -m pytest tests/test_learning_evaluation_live.py \
  -m "requires_llm and requires_db" -v -s
```

`DAITA_RUN_LIVE_LEARNING_EVAL=1` is the explicit authorization for both the
external database and paid model calls. The test is never selected by the
default deterministic command. The generated JSON and Markdown contain
judgments, aggregate counters, token usage, duration, cost, and verdicts only;
they exclude prompts, rows, SQL/tool arguments, semantic statements, skill
bodies, credentials, and secrets. The agent home itself remains temporary.

Stop and discard the fixture:

```bash
docker compose -f tests/fixtures/postgresql/compose.yaml down
```
