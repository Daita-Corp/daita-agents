# Daita Data-First Examples

These examples teach Daita as a data agent runtime. The main entry
point is `Agent.from_db()`: the DB runtime handles schema discovery, planning,
SQL validation, execution, evidence, verification, catalog relationships, and
answer synthesis.

The numbered scripts form a learning path from a local SQLite quickstart to
catalog assisted planning, governance, persistence, memory, quality, lineage,
monitors, infrastructure discovery, custom extensions, and CSV ingestion. The
`deployments/data-team-agent/` template shows the same runtime model in a
copyable project layout.

## Setup

```bash
pip install -e ".[dev,sqlite]"
```

The first examples use a temporary local SQLite database seeded at runtime.
They do not require an external database or an LLM. Pass `--live-llm` to opt
into OpenAI backed planning or synthesis when `OPENAI_API_KEY` is configured;
otherwise the scripts use the deterministic DB runtime path.

## Learning Path

| Step | Example                                                                    | Scope                                     | What it teaches                                                                |
| ---- | -------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------ |
| 1    | `00_quickstart_sqlite_from_db.py`                                          | Local only, LLM optional                  | Ask useful questions through `Agent.from_db()` with no custom SQL tools.       |
| 2    | `01_inspectable_operation.py`                                              | Local only, LLM optional                  | Inspect operation id, intent, capabilities, tasks, evidence, and verification. |
| 3    | `02_catalog_assisted_joins.py`                                             | Local only, LLM optional                  | Let catalog-owned relationships guide join-heavy questions.                    |
| 4    | `03_governed_reads_and_writes.py`                                          | Local only, LLM optional                  | Add governance, approval, and resume.                                          |
| 5    | `04_persistent_runtime_store.py`                                           | Local only, LLM optional                  | Persist and reopen operation state.                                            |
| 6    | `05_data_quality_and_lineage.py` and `06_memory_for_business_semantics.py` | Local only, LLM optional                  | Add quality, lineage, and memory.                                              |
| 7    | `07_monitor_orders.py`                                                     | Local only, LLM optional                  | Create durable data monitors.                                                  |
| 8    | `08_infrastructure_catalog.py`                                             | Local dry-run, external services optional | Discover infrastructure through catalog capabilities.                          |
| 9    | `09_custom_data_plugin_extension.py`                                       | Local only, advanced                      | Extend the runtime with declared capabilities and evidence.                    |
| 10   | `10_csv_to_sqlite_data_app.py`                                             | Local only, LLM optional                  | Promote file data into the same DB runtime path.                               |

## Run The Local Examples

```bash
python examples/00_quickstart_sqlite_from_db.py
python examples/01_inspectable_operation.py
python examples/02_catalog_assisted_joins.py
python examples/03_governed_reads_and_writes.py
python examples/04_persistent_runtime_store.py
python examples/05_data_quality_and_lineage.py
python examples/06_memory_for_business_semantics.py
python examples/07_monitor_orders.py
python examples/08_infrastructure_catalog.py
python examples/09_custom_data_plugin_extension.py
python examples/10_csv_to_sqlite_data_app.py
```

To validate setup without asking questions, pass `--setup-only`:

```bash
python examples/00_quickstart_sqlite_from_db.py --setup-only
```

To try live LLM backed synthesis explicitly:

```bash
OPENAI_API_KEY=sk-... python examples/00_quickstart_sqlite_from_db.py --live-llm
```

## Example Categories

- Local only: examples `00` through `07`, `09`, and `10` are intended to run
  against generated local SQLite data by default.
- LLM required: none of the local examples require an LLM; future examples may
  mark live model paths explicitly.
- External service required: example `08` has a credential free local dry run
  path by default. AWS and GitHub discovery are opt in with `--aws` or
  `--github` and are skipped unless the relevant environment variables exist.
- Advanced/full project templates: `deployments/data-team-agent/` collects the
  pieces into one copyable project.

## Runtime Ownership

The examples use runtime owned schema discovery, SQL execution, relationship
search, evidence, verification, synthesis, approval/resume, and monitor
control plane APIs. Those behaviors are owned by the catalog, `DbRuntime`,
runtime capabilities, and persisted evidence.
