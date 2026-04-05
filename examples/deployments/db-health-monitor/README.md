# Database Health Monitor

Continuously monitors a PostgreSQL database for operational anomalies using the
`@agent.watch()` system. When a threshold is crossed, the agent investigates
with diagnostic tools and recommends corrective action.

## Watches

| Watch | Polls | Fires when | Interval |
|-------|-------|------------|----------|
| `on_slow_queries` | `pg_stat_activity` | 3+ queries > 30s | 1m |
| `on_connection_pressure` | connection utilization % | > 80% of `max_connections` | 30s |
| `on_table_bloat` | `pg_stat_user_tables` | any table > 100k dead tuples | 5m |

All three watches use `on_resolve=True` — they fire again when the condition clears.

## Quick Start

```bash
# 1. Start a test PostgreSQL instance (max_connections=20, autovacuum=off)
docker compose up -d

# 2. Seed the database with sample data
cp .env.example .env
python scripts/seed.py

# 3. Start the monitor (fast mode = 10-15s intervals)
python run.py --fast
```

## Simulating Anomalies

In a separate terminal, trigger each watch:

```bash
# Slow queries — 4 connections sleeping 60s (threshold is 3)
python scripts/simulate.py slow_queries

# Connection pressure — hold 13 of 20 connections for 90s
python scripts/simulate.py connections

# Table bloat — 150k dead tuples on orders (autovacuum is off)
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
