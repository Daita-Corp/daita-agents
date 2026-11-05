# PostgreSQL schema baseline

A reproducible yardstick for how Daita's PostgreSQL source handles a large,
messy schema: discovery cost and limits, catalog coverage, the effect of
unrelated schema changes, and answer
outcomes on a fixed question set. Rerun it after every change to PostgreSQL
discovery or query revalidation and compare against your last report.

## What it builds

`tests/fixtures/postgres-schema-baseline/seed.sql` drops and recreates six
schemas in a database that must be named `daita_schema_baseline`:

- 1,500 base tables (`sales` 12, `events` 125, `finance` 200, `crm` 300,
  `legacy` 350, `ops` 513) with near-identical filler names and FKs;
- `events.page_views` partitioned by day into 120 partitions;
- a user enum (`sales.order_status`, also used in some filler tables and as an
  array), a domain (`finance.positive_amount`), `money` and `tsvector` columns,
  and a `citext` column when the extension is available (plain text otherwise);
- a view and a materialized view;
- look-alike tables: `sales.orders_archive`, `sales."Customers"`,
  `legacy."Accounts"` beside `legacy.accounts`, `events.page_views_legacy`.

`tests/fixtures/postgres-schema-baseline/questions.json` holds 15 questions,
each with the SQL whose result is the known answer. The harness computes the
answers from the database itself, so they stay correct if the seed changes.
The file is plain JSON so other agents (for example Claude Code in the planned
head-to-head) can run the same set.

## What it measures

`test_postgresql_schema_baseline` (offline, no model cost):

1. Attaches all six schemas (1,500 tables) and records the discovery outcome.
   Attach succeeds and keeps the first 1,000 tables, marked truncated.
2. Attaches the five schemas without `ops` (987 tables), then records attach
   time, metadata round trips by statement, resources and relationships.
3. Compares the catalog with `pg_class` and lists every missing relation with
   its reason (view, enum, domain, extension type, unsupported built-in type),
   missing columns of cataloged tables, and FKs dropped from the catalog.
4. Runs each question through `Agent.run` with a scripted oracle model that
   issues the known-correct SQL. This isolates the catalog gate and source cost
   from model reasoning. Per question: outcome (`correct`, `confident_wrong`,
   `refused`), refusal codes, run seconds, `query_seconds` (pool enabled so the
   backend records them), metadata round trips, loop steps and tokens.
5. Adds a partition, then an unrelated table, and reruns question 1 after each
   change and again after `refresh_source`, recording each outcome and the refresh
   cost.

Round trips are counted by wrapping `asyncpg.Connection` methods and reading
Daita's `/* daita:postgresql.<name> */` statement tags, so the count survives
loader rewrites as long as statements stay tagged. A bounded user read counts
as its `prepare`; cursor fetches are not counted, and a bulk discovery read
counts once as its `Connection.cursor` call.

`test_daita_and_claude_code_answer_the_question_set` (optional, paid) asks
Daita and Claude Code the same 15 questions with the same Anthropic model.
Both connect as the read-only `daita_baseline_reader` role, which sees the 987
tables in the five bounded schemas and nothing in `ops`, so their scope matches.

- Daita attaches the source with a fresh agent per repeat and answers each
  question with `Agent.run`, using its default configuration and loop limits
  (24 steps, 200k tokens, 300 s). `DAITA_SCHEMA_BASELINE_DAITA_MAX_TOKENS`
  raises the token budget for a matched-budget run. Dollar caps are off for
  Anthropic models because Daita cannot price an Anthropic request before it
  is sent.
- Claude Code runs headless (`claude -p`) once per question with its default
  prompt and tools, in an empty working directory, with an empty
  `CLAUDE_CONFIG_DIR` (no user `CLAUDE.md`, plugins, hooks, memory or login),
  no MCP servers, no inherited `CLAUDE*` variables or stdin, and Bash allowed
  only for `psql`. It sees the database but not this repository, which holds
  the answer SQL. Connection settings arrive as `PG*` environment variables;
  `DAITA_SCHEMA_BASELINE_MAX_COST_USD` caps each question.

Both answers are scored by the requested last line, with one rule:
`correct` when it holds the known value; `no_answer` when the run declines or
asks for clarification; `incomplete` when a limit or an error stopped the run;
otherwise `confident_wrong`. That rule is a keyword heuristic, not a human
label; `answer_elsewhere` in the JSON flags runs whose answer appears outside
the last line. The report also records seconds, turns, tool calls, input and
output tokens, input tokens per turn and cost, plus each Daita tool call (SQL,
resources, error code) and each Claude Code `psql` command.

## Run it

Start the disposable server (tmpfs, discarded on `down`):

```bash
docker compose -f tests/fixtures/postgres-schema-baseline/compose.yaml up -d --wait
```

Run the offline baseline from the repository root:

```bash
DAITA_SCHEMA_BASELINE_POSTGRES_URL=postgresql://postgres:fixture_admin_password@127.0.0.1:55434/daita_schema_baseline \
.venv/bin/python -m pytest tests/live/schema_baseline -o addopts="--tb=short -q --strict-markers" -s
```

Any PostgreSQL 13+ works if the URL's database is named
`daita_schema_baseline`; the harness refuses other names because it drops
schemas. Add `?sslmode=require` for a non-local server.

Results go to `live-evidence/schema-baseline/` (git-ignored) as
`postgresql-schema-baseline-<commit>.json` and `.md`. Set
`DAITA_SCHEMA_BASELINE_REPORT_DIR` to write elsewhere. Reports stay out of git.

For the comparison, export an Anthropic API key as `ANTHROPIC_API_KEY` (Claude
Code runs with an empty config, so it cannot use a claude.ai login) and set:

```bash
DAITA_RUN_LIVE_SCHEMA_BASELINE_COMPARISON=1
DAITA_SCHEMA_BASELINE_LLM_API_KEY="$ANTHROPIC_API_KEY"   # Daita's key
DAITA_SCHEMA_BASELINE_MODEL_ID=anthropic:claude-sonnet-5 # the default; a reviewed anthropic: model
DAITA_SCHEMA_BASELINE_REPEATS=3                          # 1 to 5, default 1
DAITA_SCHEMA_BASELINE_MAX_COST_USD=0.50                  # Claude Code, per question
DAITA_SCHEMA_BASELINE_DAITA_MAX_TOKENS=250000            # optional matched budget
DAITA_SCHEMA_BASELINE_SYSTEMS=daita,claude_code          # optional subset
```

Then run the comparison test. A run outlasts pytest's 300-second hang guard,
so turn the guard off:

```bash
.venv/bin/python -m pytest tests/live/schema_baseline -k claude_code -o addopts="--tb=short -q --strict-markers" -o faulthandler_timeout=0 -s
```

The comparison writes `postgresql-schema-comparison-<commit>-<model>.json` and
`.md` into the report directory, also when a run fails partway. These hold
model answers and commands: keep them out of git.

Stop the server with:

```bash
docker compose -f tests/fixtures/postgres-schema-baseline/compose.yaml down
```
