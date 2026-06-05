# Database Health Monitor

PostgreSQL health investigation agent with diagnostic tools for slow queries,
connection pressure, and table bloat. Runtime-native monitoring should be
declared through `daita.runtime.monitors` and `daita.runtime.scheduler`; this
example focuses on the agent and tools operators can invoke from those actions.

## Diagnostic Tools

| Tool | Checks |
|------|--------|
| `get_slow_queries` | Long-running active queries in `pg_stat_activity` |
| `get_connection_stats` | Active, idle, total, and maximum connections |
| `get_table_bloat` | Dead tuples and autovacuum status in user tables |

## Quick Start

```bash
# 1. Start a test PostgreSQL instance (max_connections=20, autovacuum=off)
docker compose up -d

# 2. Seed the database with sample data
cp .env.example .env
python scripts/seed.py

# 3. Ask the agent for an investigation
python run.py
```

## Simulating Anomalies

In a separate terminal, simulate data conditions and then ask the agent to investigate:

```bash
# Slow queries — 4 connections sleeping 60s
python scripts/simulate.py slow_queries

# Connection pressure — hold 13 of 20 connections for 90s
python scripts/simulate.py connections

# Table bloat — 150k dead tuples on orders
python scripts/simulate.py bloat

# All three in sequence
python scripts/simulate.py all
```

## Running Tests

```bash
# Unit tests (no database needed)
pytest tests/test_basic.py -v

# Integration tests (requires running docker compose)
DATABASE_URL=postgresql://daita:daita@localhost:5499/healthmon pytest tests/test_basic.py -v
```

## Cleanup

```bash
docker compose down -v
```
