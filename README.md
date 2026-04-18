# Daita Agents

**Open-source Python SDK for building production AI agents.**

Daita Agents gives you a clean, minimal API for autonomous tool-calling agents that work with any LLM provider — OpenAI, Anthropic, Gemini, Grok, and more. Zero-configuration tracing, pluggable data sources, composable skills, and a workflow system for multi-agent pipelines.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-daita--agents-orange)](https://pypi.org/project/daita-agents/)
[![Version](https://img.shields.io/badge/version-0.16.0-green)](https://pypi.org/project/daita-agents/)

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

`Agent.from_db()` inspects the schema, generates tool wrappers, and composes a system prompt — no manual configuration needed.

---

## Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Gemini, Grok (or bring your own)
- **Autonomous tool calling** — agents plan and execute multi-step tool chains without manual orchestration
- **`@tool` decorator** — turn any sync or async Python function into an LLM-callable tool in one line
- **`Agent.from_db()`** — point at a database connection string and get a fully-configured data agent in one call
- **Skills** — reusable, composable units of agent capability that bundle instructions + tools (subclass `BaseSkill` or use the `Skill` helper)
- **Streaming** — real-time event-based output via `agent.stream()` or `on_event` callback
- **Conversation history** — stateful multi-turn sessions with local persistence
- **Plugin ecosystem** — PostgreSQL, MySQL, MongoDB, SQLite, BigQuery, Snowflake, S3, Slack, Elasticsearch, Pinecone, ChromaDB, Qdrant, Neo4j, Redis, MCP, and more
- **Embeddings** — pluggable providers (OpenAI, Gemini, Voyage, sentence-transformers) via `BaseEmbeddingProvider`
- **Memory** — persistent semantic memory with working memory, memory graph, and automatic local/cloud detection
- **Watch system** — monitor databases and APIs continuously; trigger agent actions when thresholds are crossed
- **Workflows** — connect multiple agents into pipelines via relay channels
- **Data quality enforcement** — `ItemAssertion` + `query_checked()` validate every row and fail fast with structured violations
- **Agent graph** — built-in graph backend powering lineage & catalog; expose traversal tools to agents with `register_graph_tools()`
- **Zero-config tracing** — every LLM call and tool execution is automatically traced (tokens, latency, cost); optional OTLP export to Datadog, Jaeger, Honeycomb, etc.
- **Retry & reliability** — configurable exponential backoff with permanent-error detection
- **Focus DSL** — pre-filter tool results before the LLM sees them, reducing token usage

---

## Examples

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

### Database agent with `Agent.from_db()`

The fastest way to build a data agent. Pass a connection string (or plugin instance) and get a fully-configured agent with schema-aware tools, an auto-generated system prompt, and optional lineage/memory:

```python
import asyncio
from daita import Agent

async def main():
    agent = await Agent.from_db(
        "postgresql://user:pass@localhost/sales_db",
        model="gpt-4o",
        lineage=True,   # track data lineage automatically
        memory=True,    # remember business context across sessions
    )

    result = await agent.run("What were our top 5 products by revenue last quarter?")
    print(result)

asyncio.run(main())
```

You can also add a database plugin manually for more control:

```python
from daita import Agent
from daita.plugins import postgresql

agent = Agent(name="Sales Analyst", llm_provider="openai", model="gpt-4o")
agent.add_plugin(postgresql(host="localhost", database="sales_db", user="analyst", password="secret"))

result = await agent.run("What were the top 5 products by revenue last quarter?")
```

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

### Monitor data sources with `@agent.watch()`

Continuously poll a data source and trigger the agent when a threshold is crossed:

```python
import asyncio
from daita import Agent, WatchEvent
from daita.plugins import postgresql

db = postgresql(host="localhost", database="ops_db")
agent = Agent(name="Ops Monitor", llm_provider="openai", model="gpt-4o")
agent.add_plugin(db)

@agent.watch(
    source=db,
    condition="SELECT COUNT(*) FROM failed_jobs WHERE created_at > NOW() - INTERVAL '5m'",
    threshold=lambda v: v > 10,
    interval="1m",
)
async def on_job_failures(event: WatchEvent):
    await agent.run(f"There are {event.value} failed jobs in the last 5 minutes. Diagnose and suggest fixes.")

asyncio.run(agent.start())
```

Watches start lazily on the first `run()` call, or explicitly with `await agent.start()`.

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
