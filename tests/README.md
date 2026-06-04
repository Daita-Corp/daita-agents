# Test Suite Layout

Tests are grouped first by execution type, then by product area.

- `unit/`: fast, isolated tests that do not require real credentials or services.
- `integration/`: tests that exercise component boundaries, live providers, databases,
  or larger end-to-end flows. Use `requires_llm` and `requires_db` markers when a test
  needs external credentials or infrastructure.
- `performance/`: benchmark-style tests for latency, scale, or resource-sensitive
  behavior.
- `fixtures/`: reusable fixture data and local test servers.
- `mocks/`: reusable mock providers and external-service doubles.

Within each execution type, prefer the existing domain folder:

- `agents/`: `Agent`, runtime, streaming, retry, and conversation behavior.
- `catalog/`: catalog discovery, normalization, graph/search, and catalog-owned query
  planning helpers.
- `core/`: core utilities, config, tracing, tools, security, and exception behavior.
- `data/`: data quality, transformation, and lineage behavior.
- `evals/`: eval engine tests and eval factory helpers.
- `focus/`: Focus DSL and SQL pushdown behavior.
- `from_db/`: reserved for future `DbAgent`/`DbRuntime` integration tests. The
  legacy generic-agent `from_db` integration suite has been removed.
- `llm/`: LLM providers, embedding providers, pricing, and provider contracts.
- `memory/`: memory stores, graph memory, reinforcement, and working memory.
- `plugins/`: plugin-specific unit tests.
- `watch/`: watch sources, triggers, and watch lifecycle behavior.

Safe default:

```bash
pytest tests/ -m "not requires_llm and not requires_db"
```

Single area examples:

```bash
pytest tests/unit/agents -v
pytest tests/unit/plugins -v
pytest tests/unit/db -v
```
