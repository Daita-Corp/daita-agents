# Deployment Examples

This directory contains one production-shaped project template:

```text
data-team-agent/
```

The template mirrors the data first learning path in a copyable project layout.
It uses `Agent.from_db()` and `DbRuntime` with local SQLite fixtures, persistent
runtime storage, catalog backed planning, data quality, lineage, memory, and
runtime native monitors.

## Template

### [`data-team-agent`](./data-team-agent/)

**Use case:** start from a local SQLite fixture, inspect runtime capabilities,
profile quality, trace lineage, persist memory and operation state, and create a
runtime monitor without writing custom SQL tools.

**Required env vars:** none by default. `OPENAI_API_KEY` is optional with
`--live-llm`.

Run it locally:

```bash
cd examples/deployments/data-team-agent
python run.py --setup-only
python run.py
pytest tests/
```

Local runtime files are created under `.daita/` and are ignored by git because
they can contain operation history, evidence, and memory.

## Related Learning Path

The numbered examples in [`../README.md`](../README.md) teach the individual
runtime concepts used by this project template, especially:

- `07_monitor_orders.py` for runtime native monitoring.
- `08_infrastructure_catalog.py` for catalog owned infrastructure discovery.
- `10_csv_to_sqlite_data_app.py` for CSV/file data through the DB runtime.
