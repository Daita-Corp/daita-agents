# Daita Agents

**Open-source Python SDK for building production data agents.**

Daita Agents is built for AI systems that operate on real data: databases, schemas, catalogs, pipelines, metrics, files, APIs, and memory. It gives agents the context and guardrails they need to inspect data sources, generate safe queries, validate results, monitor changes, trace execution, and remember business context across runs.

```mermaid
flowchart LR
    Source["Database or data source"] --> DB["Database plugin<br/>connect, policy, schema discovery"]
    DB --> Catalog["Catalog<br/>normalized schema snapshots<br/>schema lookup capabilities"]
    Catalog --> Runtime["DbRuntime<br/>typed contracts + evidence"]
    Runtime --> Agent["DbAgent<br/>run, stream, inspect"]

    Catalog -.->|"persisted schema nodes"| Graph["Core graph<br/>schema, lineage, impact"]
    Lineage["Lineage plugin<br/>(optional)"] -.-> Graph
    Memory["Memory plugin<br/>(optional)"] -.->|"business context evidence"| Runtime

    Runtime --> Query["Governed DB operations<br/>plan, validate, query, sample"]
    Runtime --> Analyst["Analyst operations<br/>pivot, correlate, anomalies, forecast"]
    Runtime -.->|"quality enabled"| Quality["Data quality evidence<br/>assertions + violations"]
    Runtime -.->|"lineage / graph evidence"| Graph
    Agent -.->|"agent watch"| Watch["Watches<br/>poll sources and trigger handlers"]
    Watch -.->|"trigger handler"| Agent

    Runtime --> Tracing["Tracing + DB audit log<br/>runs, LLM calls, capability execution, retries, cost"]
    Query --> Tracing
    Analyst --> Tracing
    Quality --> Tracing
```

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-daita--agents-orange)](https://pypi.org/project/daita-agents/)
[![Version](https://img.shields.io/pypi/v/daita-agents.svg)](https://pypi.org/project/daita-agents/)

---

## Quickstart

```bash
pip install daita-agents
```

Point an agent at a database and start asking questions:

```python
import asyncio
from daita import Agent

async def main():
    agent = await Agent.from_db(
        "sqlite:///sales.db",
        model="gpt-4o",
    )

    result = await agent.run("What were the top 5 products by revenue last quarter?")
    print(result)

asyncio.run(main())
```

`Agent.from_db()` builds a `DbAgent` backed by `DbRuntime`, catalog-owned schema evidence, governed operation contracts, and runtime inspection/audit state — no manual configuration needed.

---

## Features

- **Database-native agents** — `Agent.from_db()` connects to PostgreSQL, MySQL, MongoDB, SQLite, BigQuery, Snowflake, and more; discovers schema through catalog-owned snapshots; and executes governed operation contracts.
- **Catalog and profiling** — `CatalogPlugin` discovers infrastructure, profiles stores, persists normalized schemas, compares live schemas against baselines, and declares table/schema lookup capabilities with optional model-visible tool views.
- **Governed query execution** — read-only defaults, table/column allowlists and blocklists, query limits, timeouts, result compaction, SQL validation, and audit logs are built into DB agents.
- **Analyst operations** — DB agents can execute higher-level data operations for pivots, correlations, anomaly detection, entity comparison, similarity search, and trend forecasting.
- **Data quality enforcement** — `ItemAssertion` + `query_checked()` validate every returned row and fail fast with structured `DataQualityError` violations.
- **Graph-backed lineage and impact analysis** — the core graph layer models tables, columns, pipelines, APIs, files, metrics, queries, and transformations; expose traversal tools with `register_graph_tools()`.
- **Runtime monitors** — monitor specifications and scheduler paths create runtime operations and tasks for threshold-driven work.
- **Data-aware memory** — persistent semantic memory, working memory, fact extraction, contradiction checks, and memory graphs let agents keep business context without stuffing every run into the prompt.
- **Tracing and auditability** — OpenTelemetry-backed spans cover agent runs, LLM calls, capability and tool-view executions, retries, plugin operations, latency, tokens, and cost, with optional OTLP export.
- **Plugin ecosystem** — databases, vector stores, catalogs, memory, data quality, lineage, cloud storage, APIs, messaging, MCP, and search integrations.
- **Agent evals** — developer-preview suites check answers, tools, SQL shape, data operations, budgets, latency, plugin behavior, stability, baselines, and optional LLM judges.
- **General agent foundations** — multi-provider LLM support, autonomous tool calling, `@tool`, skills, streaming, conversation history, workflows, retries, and the Focus DSL.

---

## Data Operations Architecture

At runtime, a Daita data agent is built around a cataloged view of the data estate. The LLM is not handed a raw connection and left to improvise; `DbRuntime` plans against declared capabilities, verifies typed evidence, and only projects model-visible tool views when a chat loop needs them.

| Layer                         | Where it lives                                              | What it does                                                                                                                                                                                                           |
| ----------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Database runtime              | `daita/db`                                                  | Builds `DbAgent` / `DbRuntime`, profiles schema through catalog-owned snapshots, plans typed operations, executes governed connector capabilities, and exposes runtime inspection, audit summaries, and typed evidence |
| Catalog                       | `daita/plugins/catalog`                                     | Discovers infrastructure, profiles stores, normalizes schemas, persists catalog snapshots, compares schema drift, and declares schema lookup capabilities plus typed evidence                                          |
| Query and analysis operations | `daita/db/query_*`, `daita/db/analyst_tools.py`             | Resolves intent, validates SQL/IR, executes governed queries, and adds analyst operations such as pivots, anomaly detection, correlation, comparison, similarity, and forecasting                                      |
| Graph and lineage             | `daita/core/graph`                                          | Stores tables, columns, services, files, queries, transformations, and lineage edges for impact analysis and graph traversal                                                                                           |
| Assertions and quality        | `daita/core/assertions.py`, `daita/plugins/data_quality.py` | Validates returned rows and structured datasets with fail-fast, inspectable violations                                                                                                                                 |
| Watches                       | `daita/core/watch.py`                                       | Polls or streams data sources, handles retries, and triggers agent handlers when operational conditions activate                                                                                                       |
| Memory                        | `daita/plugins/memory`                                      | Stores and recalls business facts, working context, extracted facts, contradiction checks, and memory graph relationships                                                                                              |
| Tracing                       | `daita/core/tracing.py`                                     | Captures agent runs, LLM calls, capability/tool-view executions, plugin operations, retries, latency, tokens, and cost with OpenTelemetry-backed spans                                                                 |

`Agent.from_db()` wires the main pieces together in one call:

```mermaid
sequenceDiagram
    participant App
    participant Factory as daita.db.from_db()
    participant Runtime as DbRuntime
    participant DB as Database plugin
    participant Catalog as CatalogPlugin

    App->>Factory: connection string or database plugin
    Factory->>Runtime: create DB runtime with extension registry
    Runtime->>DB: connect and apply query policy
    Runtime->>Catalog: load fresh persisted schema snapshot
    Runtime->>DB: refresh schema when cache is stale or missing
    Runtime->>Catalog: persist normalized schema evidence
    App->>Runtime: ask data questions or run data operations
    Runtime->>DB: execute typed operation contract
    Runtime->>App: return synthesized answer with inspection/audit evidence
```

---

## Examples

### Database agent with `Agent.from_db()`

The fastest way to build a data agent. Pass a connection string or plugin instance and get a configured runtime with schema-aware operation planning, catalog-backed evidence, result compaction, audit logging, and optional lineage/memory:

```python
import asyncio
from daita import Agent

async def main():
    agent = await Agent.from_db(
        "postgresql://user:pass@localhost/sales_db",
        model="gpt-4o",
        mode="analyst",
        read_only=True,
        lineage=True,   # track data lineage automatically
        memory=True,    # remember business context across sessions
    )

    result = await agent.run("What were our top 5 products by revenue last quarter?")
    print(result)

asyncio.run(main())
```

For generic chat agents, you can still add a database plugin manually and let the agent consume its model-visible tool views:

```python
from daita import Agent
from daita.plugins import postgresql

agent = Agent(name="Sales Analyst", llm_provider="openai", model="gpt-4o")
agent.add_plugin(postgresql(host="localhost", database="sales_db", user="analyst", password="secret"))

result = await agent.run("What were the top 5 products by revenue last quarter?")
```

DB agents support policy controls for production data access:

- `read_only`, `allowed_tables`, `blocked_tables`, and `blocked_columns`
- `query_default_limit`, `query_max_rows`, `query_max_chars`, and `query_timeout`
- `budget`, `schema_prompt_policy`, and `tool_result_policy`
- `mode="simple"`, `"analyst"`, `"governed"`, or `"data_team"`

---

### Custom tools with `@tool`

```python
import asyncio
from daita import Agent, tool

@tool
def search_products(query: str, max_results: int = 5) -> list:
    """Search the product catalog.

    Args:
        query: Search terms
        max_results: Maximum number of results to return
    """
    return [{"name": "Widget A", "price": 9.99}]

@tool
def calculate_discount(price: float, pct: float) -> float:
    """Calculate a discounted price.

    Args:
        price: Original price
        pct: Discount percentage (0-100)
    """
    return round(price * (1 - pct / 100), 2)

async def main():
    agent = Agent(
        name="Shopping Assistant",
        llm_provider="openai",
        model="gpt-4o",
        tools=[search_products, calculate_discount],
    )

    result = await agent.run("Find me a widget and apply a 15% discount.")
    print(result)

asyncio.run(main())
```

Both sync and async functions work with `@tool`. Parameter types and descriptions are auto-extracted from type hints and docstrings.

---

### Skills — reusable units of capability

Skills bundle domain instructions with a set of tools. Use the `Skill` helper for simple cases, or subclass `BaseSkill` when you need dynamic instructions or plugin dependencies.

```python
import asyncio
from daita import Agent, Skill, tool

@tool
def format_report(data: list, title: str) -> str:
    """Render a markdown report."""
    rows = "\n".join(f"- {r}" for r in data)
    return f"# {title}\n\n{rows}"

@tool
def generate_chart(series: list, kind: str = "bar") -> str:
    """Generate a chart description."""
    return f"{kind} chart with {len(series)} series"

report_skill = Skill(
    name="report_gen",
    description="Produces polished analytical reports",
    instructions="Always render results as markdown with a title and bulleted rows.",
    tools=[format_report, generate_chart],
)

async def main():
    agent = Agent(name="Analyst", llm_provider="openai", model="gpt-4o")
    agent.add_skill(report_skill)

    result = await agent.run("Summarize Q3 revenue with a chart.")
    print(result)

asyncio.run(main())
```

For skills that need plugin access, subclass `BaseSkill` and declare `requires()`:

```python
from daita import BaseSkill
from daita.plugins.base_db import BaseDatabasePlugin

class MigrationsSkill(BaseSkill):
    name = "migrations"
    instructions = "Follow forward-only migration policy."

    def requires(self):
        return {"db": BaseDatabasePlugin}
```

---

### Data quality enforcement with `ItemAssertion`

Validate every row returned by a database query; violations raise `DataQualityError` (permanent, non-retried) with the full list attached.

```python
import asyncio
from daita import ItemAssertion, DataQualityError
from daita.plugins import postgresql

async def main():
    async with postgresql(host="localhost", database="sales_db") as db:
        try:
            rows = await db.query_checked(
                "SELECT id, amount, customer_id FROM transactions WHERE day = CURRENT_DATE",
                assertions=[
                    ItemAssertion(lambda r: r["amount"] > 0, "All amounts must be positive"),
                    ItemAssertion(lambda r: r["customer_id"] is not None, "Every row needs a customer_id"),
                ],
            )
            print(f"{len(rows)} clean rows")
        except DataQualityError as exc:
            print(f"Data quality failure: {exc}")

asyncio.run(main())
```

---

### Agent evals

Agent evals are a developer-preview system for testing runnable Daita agents locally or in CI. An eval suite loads an agent through a Python factory, runs one or more prompts, and writes structured artifacts that show what passed, failed, and changed.

Eval suites can check:

- final-answer text and numeric values
- required or forbidden tools
- SQL safety and query shape
- non-SQL data operations across files, APIs, storage, and vector search
- skill and plugin usage, latency, and errors
- cost, latency, token, and iteration budgets
- repeat-run stability
- baselines and optional structured LLM judges

```yaml
name: sales-agent-evals
version: 1

agent:
  factory: "myapp.agents:create_sales_agent"
  kwargs:
    model: gpt-4o-mini

defaults:
  runs: 2
  max_iterations: 8

cases:
  - id: top-products
    prompt: What were the top 5 products by revenue?
    expectations:
      answer:
        contains: ["Widget A"]
        numeric:
          - label: revenue
            expected: 12840.50
            tolerance: 0.01
      tools:
        required: ["sqlite_query"]
        max_calls: 4
      sql:
        read_only: true
        require_limit: true
        must_include: ["SUM", "GROUP BY"]
        must_not_include: ["DELETE", "DROP"]
      skills:
        required: ["schema_discovery"]
        max_errors: 0
      plugins:
        required: ["sqlite"]
        max_latency_ms: 3000
      budgets:
        max_tokens: 8000
        max_latency_ms: 15000
      stability:
        require_same_tools: true
        max_answer_variants: 1
```

Run a suite from Python:

```python
import asyncio
from daita.evals import EvalSuite
from daita.evals.reporters import render_pretty

async def main():
    report = await EvalSuite.from_file("evals/sales-agent.yaml").run()
    print(render_pretty(report))

asyncio.run(main())
```

Eval runs write `report.json`, `summary.md`, JUnit XML, per-case artifacts, per-run artifacts, repeat-run diffs, judge artifacts, and baseline comparisons. The CLI command (`daita eval`) is planned; use the Python API while evals are in developer preview.

---

### Streaming with `agent.stream()`

Use `agent.stream()` to receive real-time events as an async generator:

```python
import asyncio
from daita import Agent
from daita.core.streaming import EventType

async def main():
    agent = Agent(name="assistant", llm_provider="openai", model="gpt-4o")

    async for event in agent.stream("Explain transformer attention mechanisms"):
        if event.type == EventType.THINKING:
            print(event.content, end="", flush=True)
        elif event.type == EventType.TOOL_CALL:
            print(f"\n[calling {event.tool_name}]")
        elif event.type == EventType.COMPLETE:
            print(f"\n\nDone. Tokens used: {event.token_usage}")

asyncio.run(main())
```

Alternatively, pass an `on_event` callback to `run()`:

```python
await agent.run("...", on_event=lambda e: print(e))
```

---

### Multi-turn conversations with `ConversationHistory`

```python
import asyncio
from daita import Agent, ConversationHistory

async def main():
    agent = Agent(name="Support Bot", llm_provider="anthropic", model="claude-sonnet-4-6")
    history = ConversationHistory(session_id="alice-session")

    await agent.run("My name is Alice and I prefer concise answers.", history=history)
    result = await agent.run("What's my name and preference?", history=history)
    print(result)  # "Your name is Alice and you prefer concise answers."

asyncio.run(main())
```

Sessions persist to `.daita/sessions/` between process restarts.

---

### Monitor data sources with runtime monitors

Declare monitor specs and execute actions through the runtime kernel when a threshold is crossed:

```python
import asyncio
from daita.runtime import MonitorRuntime, MonitorSpec, RuntimeKernel, InMemoryRuntimeStore
from daita.plugins import ExtensionRegistry

registry = ExtensionRegistry()
store = InMemoryRuntimeStore()
kernel = RuntimeKernel(
    runtime_id="ops-monitor",
    runtime_kind="monitor",
    extension_registry=registry,
    runtime_store=store,
)
monitor = MonitorRuntime(kernel=kernel)

spec = MonitorSpec(
    id="failed-jobs",
    name="Failed jobs",
    trigger={"gt": 10},
    action_capability_id="ops.alert",
    action_input={"severity": "warning"},
)

asyncio.run(monitor.tick(spec, value=12, execute_actions=True))
```

Scheduled polling belongs in `daita.runtime.scheduler`; monitor actions execute through `RuntimeKernel`.

---

### Multi-agent workflow

```python
import asyncio
from daita import Agent, Workflow

async def main():
    fetcher  = Agent(name="Data Fetcher",  llm_provider="openai", model="gpt-4o")
    analyzer = Agent(name="Analyzer",      llm_provider="openai", model="gpt-4o")

    workflow = Workflow("Sales Pipeline")
    workflow.add_agent("fetcher",  fetcher)
    workflow.add_agent("analyzer", analyzer)
    workflow.connect("fetcher", "raw_data", "analyzer")

    await workflow.start()
    await workflow.inject_data("fetcher", {"query": "Q3 sales"}, task="fetch")
    await workflow.stop()

asyncio.run(main())
```

---

### Memory-enabled agent

```python
import asyncio
from daita import Agent
from daita.plugins import memory

async def main():
    agent = Agent(name="Assistant", llm_provider="anthropic", model="claude-sonnet-4-6")
    agent.add_plugin(memory())

    await agent.run("My name is Alex and I prefer concise answers.")
    result = await agent.run("What's my preference?")
    print(result)

asyncio.run(main())
```

Memory auto-detects local or cloud backend and includes working memory, fact extraction, contradiction handling, and a memory graph for association.

---

### Custom embedding providers

```python
from daita import BaseEmbeddingProvider
from daita.embeddings import create_embedding_provider

# Built-in: "openai", "gemini", "voyage", "sentence_transformers", "mock"
embedder = create_embedding_provider("voyage", model="voyage-3")
vectors = await embedder.embed(["hello world", "another doc"])
```

Subclass `BaseEmbeddingProvider` to plug in any embedding model you want.

---

### Vector database search

```python
import asyncio
from daita import Agent
from daita.plugins import chroma

async def main():
    agent = Agent(name="Knowledge Assistant", llm_provider="openai", model="gpt-4o")
    agent.add_plugin(chroma(path="./vectors", collection="docs"))

    result = await agent.run("What do our docs say about authentication?")
    print(result)

asyncio.run(main())
```

---

### Expose graph traversal to agents

Lineage and catalog plugins populate a shared agent graph automatically. Call `register_graph_tools()` to let the agent traverse it directly:

```python
from daita import Agent
from daita.plugins import lineage
from daita.core.graph import register_graph_tools

agent = Agent(name="Impact Analyst", llm_provider="openai", model="gpt-4o")
agent.add_plugin(lineage())
register_graph_tools(agent)   # adds graph_subgraph, graph_shortest_path, impact_analysis

await agent.run("What downstream tables break if we drop customers.email?")
```

---

### OTLP tracing export

```python
from daita import configure_tracing

configure_tracing(
    exporter="otlp",
    endpoint="https://otel.example.com",
    service_name="my-daita-agent",
)
```

Install with `pip install "daita-agents[otlp]"` to enable the OTLP exporter. Spans cover LLM calls, tool invocations, retries, and plugin operations.

---

### MCP (Model Context Protocol) integration

```python
import asyncio
from daita import Agent
from daita.plugins import mcp

async def main():
    agent = Agent(
        name="File Analyst",
        llm_provider="openai",
        model="gpt-4o",
        mcp=mcp.server(command="uvx", args=["mcp-server-filesystem", "/data"]),
    )

    result = await agent.run("Read report.csv and summarize the totals.")
    print(result)

asyncio.run(main())
```

---

## Plugins

### Databases

| Plugin          | Description                           | Extra             |
| --------------- | ------------------------------------- | ----------------- |
| `postgresql`    | Query and write PostgreSQL (pgvector) | `[postgresql]`    |
| `mysql`         | Query and write MySQL                 | `[mysql]`         |
| `mongodb`       | Query MongoDB collections             | `[mongodb]`       |
| `sqlite`        | Query and write SQLite                | `[sqlite]`        |
| `snowflake`     | Query Snowflake data warehouse        | `[snowflake]`     |
| `bigquery`      | Query Google BigQuery                 | `[bigquery]`      |
| `elasticsearch` | Search Elasticsearch indices          | `[elasticsearch]` |

### Vector Databases

| Plugin     | Description                  | Extra        |
| ---------- | ---------------------------- | ------------ |
| `chroma`   | Local/embedded vector search | `[chromadb]` |
| `pinecone` | Managed cloud vector search  | `[pinecone]` |
| `qdrant`   | Self-hosted vector search    | `[qdrant]`   |

### Integrations & Cloud

| Plugin            | Description                      | Extra            |
| ----------------- | -------------------------------- | ---------------- |
| `rest`            | Call REST APIs                   | _(included)_     |
| `s3`              | Read/write S3 objects            | `[aws]`          |
| `slack`           | Send Slack messages              | `[slack]`        |
| `email`           | Send/receive email (SMTP/IMAP)   | _(included)_     |
| `google_drive`    | Read files from Google Drive     | `[google-drive]` |
| `websearch`       | AI-optimized web search (Tavily) | `[websearch]`    |
| `exa_search`      | AI-powered semantic search (Exa) | `[exa]`          |
| `mcp`             | Model Context Protocol servers   | `[mcp]`          |
| `redis_messaging` | Redis pub/sub messaging          | `[redis]`        |
| `redis`           | Redis data store operations      | `[redis]`        |
| `neo4j`           | Graph database (Cypher queries)  | `[neo4j]`        |

### Knowledge & Orchestration

| Plugin         | Description                                 |
| -------------- | ------------------------------------------- |
| `memory`       | Persistent semantic agent memory            |
| `catalog`      | Schema discovery and metadata management    |
| `lineage`      | Data lineage tracking and impact analysis   |
| `orchestrator` | Multi-agent coordination and task routing   |
| `data_quality` | Data profiling and quality checks           |
| `transformer`  | SQL transformation management and execution |

---

## Installation

### Core (OpenAI included)

```bash
pip install daita-agents
```

### LLM providers

```bash
pip install "daita-agents[anthropic]"   # Claude
pip install "daita-agents[google]"      # Gemini
pip install "daita-agents[llm-all]"     # All LLM providers
```

### Database plugins

```bash
pip install "daita-agents[postgresql]"
pip install "daita-agents[mysql]"
pip install "daita-agents[mongodb]"
pip install "daita-agents[sqlite]"
pip install "daita-agents[bigquery]"
pip install "daita-agents[snowflake]"
pip install "daita-agents[databases]"   # All traditional databases
```

### Vector database plugins

```bash
pip install "daita-agents[chromadb]"
pip install "daita-agents[pinecone]"
pip install "daita-agents[qdrant]"
pip install "daita-agents[vectordb]"    # All vector databases
```

### Embedding providers

```bash
pip install "daita-agents[voyage]"                # Voyage AI
pip install "daita-agents[sentence-transformers]" # Local sentence-transformers
```

### Cloud

```bash
pip install "daita-agents[aws]"          # boto3
pip install "daita-agents[gcp]"          # Google Cloud services
pip install "daita-agents[google-drive]" # Drive + document parsers
pip install "daita-agents[cloud]"        # All cloud integrations
```

### Observability & production

```bash
pip install "daita-agents[otlp]"         # Export traces to OTLP collectors
pip install "daita-agents[api-server]"   # FastAPI + Uvicorn
pip install "daita-agents[production]"   # AWS + API server
```

### Data & content

```bash
pip install "daita-agents[data]"         # pandas, numpy, openpyxl, parsing libs
pip install "daita-agents[web]"          # beautifulsoup4, lxml
pip install "daita-agents[data-quality]" # Advanced quality checks (scipy)
pip install "daita-agents[lineage]"      # networkx graph support
```

### Bundles

```bash
pip install "daita-agents[recommended]"  # Anthropic + pandas + beautifulsoup4
pip install "daita-agents[complete]"     # Most features, no heavy packages
pip install "daita-agents[all]"          # Everything (large install)
```

---

## Exception hierarchy

All exceptions are importable from `daita`:

`DaitaError` → `AgentError`, `LLMError`, `ConfigError`, `PluginError`, `SkillError`, `WorkflowError`, `TransientError`, `RetryableError`, `PermanentError`, `RateLimitError`, `AuthenticationError`, `ValidationError`, `FocusDSLError`, `DataQualityError`

---

## Documentation

See the [`examples/`](examples/) directory for full working examples, or the [documentation](https://docs.daita-tech.io).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

_Built by [Daita](https://daita-tech.io)_
