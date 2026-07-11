# Daita Agents

**Build AI agents that can reason over real data.**

Daita Agents is a Python framework for data agents.

Inspect structured sources, plan governed work, execute through declared capabilities, collect typed evidence, verify results, and leave an audit trail. The primary
entry point is `Agent.from_db()`.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-daita--agents-orange)](https://pypi.org/project/daita-agents/)
[![Version](https://img.shields.io/pypi/v/daita-agents.svg)](https://pypi.org/project/daita-agents/)

## Why Daita?

- **Data native agents**: connect to SQLite or PostgreSQL and ask questions in
  natural language.
- **Production runtime**: operation records, persisted tasks, governance,
  approvals, evidence, verification, resume, and audit summaries.
- **Extension first plugins**: integrations declare capabilities, executors,
  evidence schemas, policies, context providers, workers, and tool views.
- **Local Python ergonomics**: install with pip, write normal async Python, and
  add local tools with a decorator.
- **Broad integration surface**: databases, vector stores, cloud services,
  web search, Slack, Google Drive, memory, catalog, lineage, data quality, and
  evals live behind optional extras.

## Quickstart

Install the SQLite extra and set an LLM key:

```bash
pip install "daita-agents[sqlite]"
export OPENAI_API_KEY=sk-...
```

Create `quickstart.py`:

```python
import asyncio
import sqlite3
from pathlib import Path

from daita import Agent


DB_PATH = Path("sales.db")


def seed_database() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS orders;
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                product TEXT NOT NULL,
                revenue REAL NOT NULL
            );
            INSERT INTO orders (product, revenue) VALUES
                ('Notebook', 1250.00),
                ('Keyboard', 875.50),
                ('Monitor', 2410.00);
        """)


async def main() -> None:
    seed_database()

    agent = await Agent.from_db(
        str(DB_PATH),
        mode="analyst",
        read_only=True,
        memory=False,
    )

    try:
        answer = await agent.run("What product had the most revenue?")
        print(answer)
    finally:
        await agent.stop()


asyncio.run(main())
```

Run it:

```bash
python quickstart.py
```

Need PostgreSQL instead?

```bash
pip install "daita-agents[postgresql]"
export DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

```python
agent = await Agent.from_db(
    os.environ["DATABASE_URL"],
    mode="governed",
    read_only=True,
    allowed_tables=["orders", "customers", "products"],
)
```

## What You Get Back

`agent.run(...)` returns the synthesized answer string. Use
`run_detailed(...)` when you need the operation contract, evidence, diagnostics,
or audit identifiers.

```python
result = await agent.run_detailed("Which customers had the largest refunds?")

print(result.operation_id)
print(result.contract.required_capabilities)
print(result.evidence)
print(result.answer)
```

Runtime inspection is built in:

```python
inspection = await agent.describe()
print(inspection.plugin_ids)
print(inspection.capability_ids)
```

## `Agent.from_db()`

Use `Agent.from_db()` for agents that answer questions from structured data with
a durable operation trail.

```python
agent = await Agent.from_db(
    "postgresql://user:pass@localhost/warehouse",
    mode="governed",
    read_only=True,
    query_default_limit=50,
    query_max_rows=200,
    query_timeout=30,
    lineage=True,
    memory={"enabled": True, "retrieval_mode": "structured"},
)
```

Current direct source support:

| Source                                                 | Status                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------- |
| SQLite file path, `:memory:`, or `sqlite://...`        | Supported by `Agent.from_db()`                                            |
| PostgreSQL URL, `postgresql://...` or `postgres://...` | Supported by `Agent.from_db()`                                            |
| Converted `BaseDatabasePlugin` instance                | Supported by `Agent.from_db()`                                            |
| Other database URL schemes                             | Available through plugin APIs while direct `from_db` routing is converted |

Built in modes:

| Mode        | Default posture                                            |
| ----------- | ---------------------------------------------------------- |
| `simple`    | Read-only, conservative row and character limits           |
| `analyst`   | Default read-only analysis profile                         |
| `governed`  | Stricter limits with lineage enabled by default            |
| `data_team` | Broader data-team profile with quality and lineage enabled |

`DbAgent` exposes:

- `run(prompt)`: return the answer string.
- `run_detailed(prompt)`: return a typed `DbOperationResult`.
- `describe()`: inspect registered plugins, capabilities, and runtime state.
- `operations` and `audit_log`: retained operation summaries.
- `monitor(...)`: create durable database observations.
- `stop()` / `teardown()`: release runtime resources.

Database monitor scheduling is deliberately host-driven. The library provides
durable one-shot passes through `DbMonitorScheduler.run_once()` (and the
one-shot `DbRuntime.tick_monitors()` convenience); the application owns
cadence, retry, metrics, signals, and shutdown. See
[`docs/MONITOR_HOSTING.md`](docs/MONITOR_HOSTING.md) for the complete hosting
contract and multi-host lease guidance.

## Local Tool Agents

The generic `Agent` is useful for non DB assistants, local tool calling,
streaming, conversation history, and experiments. Data agents should usually
start with `Agent.from_db()`.

```python
import asyncio
from daita import Agent, tool


@tool
def calculate_discount(price: float, pct: float) -> float:
    """Calculate a discounted price."""
    return round(price * (1 - pct / 100), 2)


async def main() -> None:
    agent = Agent(
        name="shopping_assistant",
        llm_provider="openai",
        model="gpt-4o-mini",
        tools=[calculate_discount],
    )

    result = await agent.run(
        "A jacket is $120 with a 15% discount. What is the final price?"
    )
    print(result)


asyncio.run(main())
```

For richer diagnostics:

```python
result = await agent.run("Use the discount tool.", detailed=True)
print(result["operation_id"])
print(result["tool_calls"])
```

## Architecture

Daita's runtime is operation centric. Runtime owned work flows through declared
capabilities, persisted tasks, registered executors, and the shared governance
boundary before any executor runs.

```mermaid
flowchart LR
    User["Data question"] --> FromDB["Agent.from_db()"]
    FromDB --> Agent["DbAgent"]
    Agent --> Runtime["DbRuntime"]
    Runtime --> Kernel["RuntimeKernel"]
    Kernel --> Store["RuntimeStore"]
    Kernel --> Registry["ExtensionRegistry"]
    Registry --> Plugins["Plugins and skills"]
    Plugins --> Contracts["Capabilities, executors, evidence, policies, workers"]
    Runtime --> Catalog["Catalog plugin"]
    Runtime --> Verify["Verification and synthesis"]
```

Ownership rules that matter:

- `DbRuntime` owns DB planning, task execution, governance, approvals, resume,
  evidence, verification, monitors, and synthesis.
- `execute_task()` is the executor choke point.
- The catalog plugin owns cataloging infrastructure, normalized schemas,
  relationships, and graph traversal/search over data assets.
- Generic `Agent` projects tools and local chat behavior; it should not grow a
  parallel DB planner.

## Integrations

Install the smallest extra you need:

```bash
pip install "daita-agents[postgresql]"
pip install "daita-agents[sqlite]"
pip install "daita-agents[data]"
pip install "daita-agents[websearch]"
pip install "daita-agents[cloud]"
```

Common extras:

| Area               | Extras                                                                           |
| ------------------ | -------------------------------------------------------------------------------- |
| LLM providers      | `anthropic`, `google`, `llm-all`                                                 |
| Databases          | `sqlite`, `postgresql`, `mysql`, `mongodb`, `snowflake`, `bigquery`, `databases` |
| Search and vectors | `websearch`, `exa`, `chromadb`, `pinecone`, `qdrant`, `vectordb`                 |
| Cloud and apps     | `aws`, `azure`, `gcp`, `google-drive`, `slack`, `mcp`, `cloud`                   |
| Data and runtime   | `data`, `memory`, `data-quality`, `lineage`, `redis`, `neo4j`, `otlp`            |
| Bundles            | `recommended`, `complete`, `all`                                                 |

Direct plugin APIs are available under `daita.plugins`:

```python
import asyncio
from daita.plugins import sqlite


async def main() -> None:
    async with sqlite(path="./sales.db") as db:
        rows = await db.query("SELECT product, revenue FROM orders LIMIT 5")
        print(rows)


asyncio.run(main())
```

## Skills, Memory, And Evals

Skills package reusable behavior: instructions, activation rules, runtime
effects, context providers, and optional tool views.

```python
from daita import Agent, Skill

reporting = Skill(
    name="executive_reporting",
    description="Write concise executive summaries.",
    instructions="Use: summary, key metrics, risks, and next actions.",
)

agent = Agent(
    name="ops_analyst",
    llm_provider="openai",
    skills=[reporting],
)
```

Tool-backed skills use `Skill.with_tools(...)`:

```python
from daita import Skill, tool


@tool
def normalize_region(value: str) -> str:
    """Normalize a sales region name."""
    return value.strip().lower().replace(" ", "_")


region_skill = Skill.with_tools(
    name="region_cleanup",
    tools=[normalize_region],
    instructions="Normalize region names before comparing reports.",
)
```

Eval suites run agents locally or in CI and write structured artifacts such as
`report.json`, `summary.md`, JUnit XML, per case artifacts, judge artifacts, and
baseline comparisons. The eval API is developer-preview.

## Project Layout

| Path                     | Responsibility                                                                 |
| ------------------------ | ------------------------------------------------------------------------------ |
| `daita/agents/`          | `Agent`, `BaseAgent`, chat runtime, tools, streaming, conversation history     |
| `daita/db/`              | `Agent.from_db()`, `DbAgent`, planning, SQL analysis, verification, synthesis  |
| `daita/db/runtime/`      | DB operation lifecycle, tasks, governance, resume, monitors, cache, results    |
| `daita/runtime/`         | Operations, tasks, capabilities, evidence, policies, workers, stores, kernel   |
| `daita/plugins/`         | Extension-first connectors and domain services                                 |
| `daita/plugins/catalog/` | Discovery, normalization, profiling, persistence, relationships, graph views   |
| `daita/plugins/memory/`  | Semantic, keyword, graph, working-memory, contradiction, and storage helpers   |
| `daita/skills/`          | Skill declarations, activation, discovery, runtime effects, tool adapters      |
| `daita/llm/`             | OpenAI, Anthropic, Gemini, Grok, Ollama, OpenAI-compatible, and mock providers |
| `daita/embeddings/`      | OpenAI, Gemini, Voyage, sentence-transformers, and mock embeddings             |
| `daita/evals/`           | Eval suites, assertions, judges, reporters, artifacts, datasets, baselines     |
| `examples/`              | Data-first learning path scripts and one deployment-style project template     |
| `tests/`                 | Unit, integration, performance, fixture, mock, and live-gated tests            |

## Examples

- [`examples/`](examples/): numbered data first learning path from a local
  SQLite quickstart through inspection, catalog joins, governance, persistence,
  memory, quality, lineage, monitors, infrastructure discovery, extensions, and
  CSV-to-SQLite ingestion.
- [`examples/deployments/data-team-agent/`](examples/deployments/data-team-agent/):
  production shaped local data-team template using `Agent.from_db()`,
  `DbRuntime`, catalog, quality, lineage, memory, monitors, and a persistent
  runtime store.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -m "not requires_llm and not requires_db"
```

Useful focused targets:

```bash
pytest tests/unit/ -v
pytest tests/unit/db/test_agent_from_db.py -v
pytest tests/unit/core/test_skills.py -v
```

Development rules that matter most:

- Optional dependencies must be imported lazily inside `connect()` or a client
  property body.
- Packages needed by one integration belong in an optional extra, not core
  dependencies.
- `asyncio_mode = "auto"` is configured globally; do not add per-test
  `@pytest.mark.asyncio`.
- Production DB behavior belongs in `DbRuntime`; catalog behavior belongs in
  catalog plugins.

## Resources

- [Documentation](https://docs.daita-tech.io)
- [examples/](examples/)
- [tests/README.md](tests/README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0 - see [LICENSE](LICENSE).

Built by [Daita](https://daita-tech.io).
