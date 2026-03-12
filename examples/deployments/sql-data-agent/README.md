# sql-data-agent

A single agent that translates natural language questions into SQL, runs them
against a PostgreSQL database, and explains the results in plain English.

**Use case:** "Show me customers who haven't ordered in 90 days" → the agent
inspects the schema, generates the right SQL, runs it, and summarises the output.

## Highlights

- PostgreSQL plugin with lazy import (`daita-agents[postgresql]`)
- `inspect_schema` tool — queries `information_schema` so the agent knows exact column names
- `run_query` tool — executes SELECT-only queries with row limits and error recovery
- Single-agent pattern — the simplest plugin-backed starting point

## Required environment variables

```
OPENAI_API_KEY   sk-...
DATABASE_URL     postgresql://user:pass@host:5432/dbname
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
```

## Run

```bash
# Demo mode — runs 5 sample questions
python run.py

# One-shot question
python run.py "Which products generated the most revenue last quarter?"
```

## Test

```bash
# Fast tests — no database or API key required
pytest tests/ -v

# Full suite — requires DATABASE_URL and OPENAI_API_KEY
pytest tests/ -v
```

## Project structure

```
sql-data-agent/
├── agents/
│   └── sql_agent.py        # Agent definition + tools
├── workflows/              # No workflows — single-agent example
├── data/                   # Not used in this example
├── tests/
│   └── test_basic.py       # Unit + integration tests
├── daita-project.yaml      # Project config
├── requirements.txt
└── run.py                  # Entry point
```

## Deploy

```bash
daita push
```
