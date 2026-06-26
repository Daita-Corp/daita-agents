# Data Team Agent

A copyable Daita project centered on `Agent.from_db()` and
`DbRuntime`. It runs locally by default against a seeded SQLite database, with a
persistent runtime store and local memory under `.daita/local/`.

## What It Demonstrates

- `Agent.from_db(..., mode="data_team")`
- Catalog backed planning from the runtime catalog plugin
- Persistent `DbRuntime` storage with `DbRuntimeOptions(store="sqlite", ...)`
- Data quality, lineage, and memory through extension first runtime plugins
- Runtime native monitor creation and inspection
- Deterministic local operation output unless live LLM mode is explicitly enabled

The template seeds fixture data only. It does not hand write schema inspection,
SQL execution, relationship search, evidence, verification, synthesis,
approval/resume, or monitor polling logic.

## Setup

```bash
cd examples/deployments/data-team-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When running from this repository, an editable install is also useful:

```bash
pip install -e "../../..[sqlite]"
```

## Run

Local deterministic mode requires no credentials:

```bash
python run.py
```

Setup only mode seeds SQLite, initializes `DbRuntime`, and creates/inspects the
monitor without running demo operations:

```bash
python run.py --setup-only
```

Monitor execution is opt in and still local:

```bash
python run.py --tick-monitor
```

Expected output includes local file paths, registered runtime plugins, monitor
details, operation ids, task capability sequences, evidence kinds, and
deterministic answers for memory, quality, lineage, and a catalog backed query.

## Test

```bash
pytest tests/
```

The smoke tests run without credentials or external databases. They validate
imports, local SQLite seeding, `Agent.from_db()` setup, registered runtime
plugins, persistent runtime store creation, and monitor inspection.

## Live LLM Opt-In

The default path does not call external LLM services. To use OpenAI synthesis,
set credentials and pass `--live-llm`:

```bash
export OPENAI_API_KEY=sk-...
python run.py --live-llm
```

If `OPENAI_API_KEY` is missing, `--live-llm` falls back to deterministic runtime
output.

## Local Files

Runtime files are created under `.daita/local/`:

- `data/team_sales.sqlite`
- `runtime/runtime_store.sqlite`
- `memory/`

These files are intentionally git ignored because runtime stores and memory can
contain local or sensitive data.
