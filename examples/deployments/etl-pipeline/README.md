# etl-pipeline

A three-agent linear workflow that ingests raw event logs, normalises them,
and loads them into a reporting database.

**Pipeline:**
```
Extractor → (raw_data) → Transformer → (transformed_data) → Loader
```

**Use case:** Nightly job that reads unprocessed event records from a source
database, cleans and deduplicates them with pandas, and writes the results to
a fact table in the destination database.

## Highlights

- `Workflow` with two relay channels (`raw_data` → `transformed_data`)
- Each agent has a single, focused responsibility
- Idempotent loader — safe to re-run with the same data
- `validate_and_clean` tool is pure Python (tested without any services)

## Required environment variables

```
OPENAI_API_KEY        sk-...
SOURCE_DATABASE_URL   postgresql://user:pass@host:5432/source_db
DEST_DATABASE_URL     postgresql://user:pass@host:5432/dest_db
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export SOURCE_DATABASE_URL=postgresql://user:pass@localhost:5432/source_db
export DEST_DATABASE_URL=postgresql://user:pass@localhost:5432/dest_db
```

## Expected schemas

**Source database (`SOURCE_DATABASE_URL`):**
```sql
CREATE TABLE raw_events (
    id          TEXT PRIMARY KEY,
    event_type  TEXT,
    user_id     TEXT,
    session_id  TEXT,
    properties  JSONB,
    created_at  TIMESTAMPTZ,
    processed   BOOLEAN DEFAULT false
);
```

**Destination database (`DEST_DATABASE_URL`):**
```sql
CREATE TABLE fact_events (
    id           TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    session_id   TEXT,
    properties   JSONB,
    event_date   DATE,
    processed_at TIMESTAMPTZ
);
```

## Run

```bash
# Run one ETL pass (default 1000 records)
python run.py

# Process a larger batch
python run.py 5000
```

## Test

```bash
# Fast tests — no database or API key required (transformer tool uses pandas)
pytest tests/ -v

# Full integration tests — requires all env vars
pytest tests/ -v
```

## Project structure

```
etl-pipeline/
├── agents/
│   ├── extractor.py       # Fetches unprocessed events from source DB
│   ├── transformer.py     # Cleans and normalises with pandas
│   └── loader.py          # Inserts into destination DB
├── workflows/
│   └── etl_workflow.py    # Wires up the three-agent pipeline
├── tests/
│   └── test_basic.py
├── daita-project.yaml     # Agents, workflow, nightly schedule
├── requirements.txt
└── run.py
```

## Deploy

```bash
daita push
```

After deploying, the nightly ETL run is scheduled automatically at 02:00 UTC
via the cron entry in `daita-project.yaml`.
