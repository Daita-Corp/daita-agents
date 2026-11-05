# Cross-source benchmark

A reproducible yardstick for questions that need PostgreSQL, SQLite and a
folder of files together, with known answers and planted key mismatches. Rerun
it after every change to combined queries and compare against your last report.

## What it builds

`tests/fixtures/cross-source-benchmark/`:

- `compose.yaml`: a disposable PostgreSQL 16 (tmpfs, database
  `daita_cross_source`, port `DAITA_CROSS_SOURCE_POSTGRES_PORT`, default 55435).
- `seed.sql`: drops and recreates schema `sales` (`orders` 3,000 rows,
  `order_items` 120,000 rows, `invoices` 100 rows; arithmetic only, no
  `random()`) and the read-only role `daita_cross_source_reader`.
- `seed_sqlite.sql`: the CRM (`customers` 51 rows, `accounts` 50 rows), built
  into a fresh SQLite file on every run.
- `files/`: `products.jsonl` (23 products), `country_targets.csv`,
  `account_managers.csv`.
- `questions.json`: 12 questions. Each has `expected`, the `sources` it needs,
  an `oracle` (the exact `data_query` arguments that yield the answer, naming
  sources by display name: "Sales database", "CRM database", "Business files"),
  and for `planted_mismatch` questions a `mismatch` block.

Planted mismatches (counts are distinct keys; `row_level` is the count when an
input carries raw rows):

| Question | Join | Plant | Unmatched |
| --- | --- | --- | --- |
| q07, q08 | `orders.customer_id` = `customers.id` | orders for customers 51 and 52 have no CRM row (115 orders); CRM customer 60 never ordered | 2 / 1 (115 / 1) |
| q09 | `invoices.account_code` = `accounts.account_code` | CRM spells three codes `acc-013`, `ACC-022 `, `ACC-034 ` | 3 / 3 (6 / 3) |
| q10 | `order_items.product_id` = `products.product_id` | integers in PostgreSQL, zero-padded text (`"007"`) in the JSONL; three products never sold | 0 / 3, types differ |
| q11 | `customers.country` = `country_targets.country` | seven customers in BR, which has no target; JP has no customers | 1 / 1 (7 / 1) |
| q12 | `accounts.account_code` = `account_managers.account_code` | the CRM and the manager file disagree on five codes by case or trailing space | 5 / 5 |

The product ids are JSON strings because a CSV column of `007`..`023` is
inferred as integers by the file reader and would hide the type mismatch.

`q10` also requires aggregating inside the input: `sales.order_items` has more
rows (120,000) than the combined query lets all inputs share (50,000), so
reading it whole refuses the call. If that limit grows, grow the seed.

Every `expected` was computed by hand from the seed arithmetic and confirmed
through the `oracle` (see "Verification" below). A mismatch question's answer is
the plain-meaning one: it counts the accounts and orders a human would count
despite the spelling or orphan problem.

## What it measures

1. `test_sqlite_and_folder_oracle_answers_are_correct`: offline, no model, no
   PostgreSQL. A scripted model (`MockModelProvider`) issues each question's
   `oracle` call through the real `Agent.run` path for the questions that need
   only SQLite and the folder (q01, q03, q05, q11, q12). It asserts the value
   equals `expected`, that mismatch questions show the planted counts in
   `join_checks`, and that `warnings` is not empty. It proves the harness and
   the combined path, not model behaviour.
2. `test_full_oracle`: needs `DAITA_CROSS_SOURCE_POSTGRES_URL`. Seeds, runs all
   12 oracles, writes `cross-source-oracle.{json,md}`, asserts 100%.
3. `test_daita_and_claude_code_answer_the_question_set`: paid, separately gated.
   Per repeat, a fresh Daita agent (read-only PostgreSQL role, a copy of the
   SQLite file, a copy of the folder, default configuration and loop limits)
   answers all 12 questions through `Agent.run`; Claude Code runs headless
   (`claude -p`) once per question in an empty working directory with an empty
   `CLAUDE_CONFIG_DIR`, no MCP servers, no inherited `CLAUDE*`/`DAITA_*`
   variables, Bash allowed only for `psql`, `sqlite3`, `duckdb` and the read-only
   helpers `ls`, `cat`, `head`, `echo` (each simple command is matched on its own,
   so compound commands, `VAR=` prefixes and `find`/`env` are refused), and
   read-only copies of the SQLite file and folder. Its prompt says where the
   three sources are (PG environment variables, absolute SQLite path, absolute
   folder path) and nothing about this repository. Same model on both sides.

Scoring per run uses the shared rule in `tests/support/live_comparison.py`
(`correct`, `confident_wrong`, `no_answer`, `incomplete`, keyword heuristic on
the requested last line). Mismatch questions add `reported_in_answer` (the final
text matches a `mentions` regex; for q08 and q11, whose answer is itself the
unmatched count, a correct answer counts as reported) and, for Daita, `reported_in_result` (a
`join_checks` entry shows the planted counts at key or row level, either side
first, or shows the planted type difference, or (q09, q12) a positive
`left_near_matches`/`right_near_matches`, or has counts it could not measure).

The report states an acceptance block, pass or fail:

- Daita correct is at least 90% of questions and repeats;
- every planted mismatch is reported in every Daita run, meaning in the result's
  `join_checks` or in the answer (both are counted separately in the table);
- the Claude Code baseline is recorded for every question and repeat and no
  Claude Code run ended on a denied tool call (a baseline that could not run is
  not a baseline).

The report also holds per system: correct, confident wrong, no answer,
incomplete, tool calls, turns, tokens, seconds, cost; per question detail; each
Daita tool call's SQL and error code; each Claude Code shell command. Reports
never contain the PostgreSQL passwords (they are scrubbed before writing), but
they hold model answers and commands: keep them out of git.

## Run it

Offline proof (no server, no model):

```bash
.venv/bin/python -m pytest tests/live/cross_source -k sqlite_and_folder -o addopts="--tb=short -q --strict-markers"
```

Start the disposable server (discarded on `down`):

```bash
docker compose -f tests/fixtures/cross-source-benchmark/compose.yaml up -d --wait
export DAITA_CROSS_SOURCE_POSTGRES_URL=postgresql://postgres:fixture_admin_password@127.0.0.1:55435/daita_cross_source
```

Any PostgreSQL 13+ works if the URL's database is named `daita_cross_source`;
the harness refuses other names because the seed drops schema `sales`. Add
`?sslmode=require` for a non-local server.

Full oracle (free):

```bash
.venv/bin/python -m pytest tests/live/cross_source -k full_oracle -o addopts="--tb=short -q --strict-markers" -s
```

Paid comparison. Claude Code runs with an empty config, so export an Anthropic
API key. The `psql`, `sqlite3`, `duckdb` and `claude` executables must be on
`PATH`. A run outlasts pytest's hang guard, so turn the guard off:

```bash
export ANTHROPIC_API_KEY=...
DAITA_RUN_LIVE_CROSS_SOURCE_COMPARISON=1 \
.venv/bin/python -m pytest tests/live/cross_source -k claude_code -o addopts="--tb=short -q --strict-markers" -o faulthandler_timeout=0 -s
```

Optional settings:

```bash
DAITA_CROSS_SOURCE_MODEL_ID=anthropic:claude-sonnet-5   # default; an anthropic: model
DAITA_CROSS_SOURCE_REPEATS=3                            # 1 to 5, default 3
DAITA_CROSS_SOURCE_MAX_COST_USD=0.50                    # Claude Code, per question
DAITA_CROSS_SOURCE_SYSTEMS=daita,claude_code            # optional subset
DAITA_CROSS_SOURCE_REPORT_DIR=...                       # default live-evidence/cross-source/
DAITA_CROSS_SOURCE_CLAUDE_EXECUTABLE=...                # default: claude on PATH
```

Reports are `cross-source-oracle.{json,md}` and
`cross-source-comparison-<commit>-<model>.{json,md}`, written also when a run
fails partway; the paths are printed. Stop the server with:

```bash
docker compose -f tests/fixtures/cross-source-benchmark/compose.yaml down
```

## Cost

Claude Code is capped per question by `--max-budget-usd`, which Claude Code
checks between turns, so a question can overshoot slightly:

    Claude Code worst case = 12 questions x REPEATS x MAX_COST_USD
                           = 12 x 3 x 0.50 = $18.00 with the defaults

A Daita run on an Anthropic model has **no dollar cap**: Daita cannot price an
Anthropic request before sending it, and under a dollar cap it refuses every
unpriced request (the schema baseline explains the same limit). Each run is
bounded only by the default loop limits, 24 steps, 200,000 tokens and 300
seconds, and may exceed the token limit by at most one response:

    Daita worst case = 12 x REPEATS x 200,000 tokens
                     = 7.2 million tokens with the defaults,
                       priced at the model's current list rates

Total worst case is the sum. Set `DAITA_CROSS_SOURCE_REPEATS=1` for a first run.

## Verification of the fixtures

The five SQLite and folder questions are verified by test 1. The seven that need
PostgreSQL (q02, q04, q06, q07, q08, q09, q10) were checked offline by running
their oracles with a SQLite file standing in for PostgreSQL, built with the same
arithmetic, but a real PostgreSQL run (`test_full_oracle`) is the confirmation.

## Reading a report

Per system the table gives runs with a denied tool call, with an errored tool
call, and ended on a denial (Claude Code's last tool result was a permission
refusal), plus the sum of known costs and how many runs were unpriced. A final
text that only says it lacked permission scores `no_answer`. Daita runs record
`provider_failure` and the last six observed events, so a `timeout` exit shows
whether a model call or a tool was in flight.
