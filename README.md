# Daita Agents

**Open-source Python SDK for building production AI agents.**

Daita Agents gives you a clean, minimal API for autonomous tool-calling agents that work with any LLM provider — OpenAI, Anthropic, Gemini, Grok, and more. Zero-configuration tracing, pluggable data sources, and a composable workflow system for multi-agent pipelines.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-daita--agents-orange)](https://pypi.org/project/daita-agents/)

---

## Quickstart

```bash
pip install daita-agents
```

```python
import asyncio
from daita import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

async def main():
    agent = Agent(
        name="assistant",
        llm_provider="openai",
        model="gpt-4o",
        tools=[get_weather],
    )

    result = await agent.run("What's the weather in Tokyo?")
    print(result)

asyncio.run(main())
```

---

## Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Gemini, Grok (or bring your own)
- **Autonomous tool calling** — agents plan and execute multi-step tool chains without manual orchestration
- **`@tool` decorator** — turn any sync or async Python function into an LLM-callable tool in one line
- **`Agent.from_db()`** — point at a database connection string and get a fully-configured data agent in one call
- **Streaming** — real-time event-based output via `agent.stream()` or `on_event` callback
- **Conversation history** — stateful multi-turn sessions with local persistence
- **Plugin ecosystem** — PostgreSQL, MySQL, MongoDB, SQLite, S3, Slack, Elasticsearch, Pinecone, ChromaDB, Neo4j, MCP, and more
- **Memory** — persistent agent memory with local or custom backends
- **Watch system** — monitor databases and APIs continuously; trigger agent actions when thresholds are crossed
- **Workflows** — connect multiple agents into pipelines via relay channels
- **Zero-config tracing** — every LLM call and tool execution is automatically traced (tokens, latency, cost)
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

    # History is carried across run() calls
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
from daita.plugins.memory import MemoryPlugin

async def main():
    agent = Agent(name="Assistant", llm_provider="anthropic", model="claude-sonnet-4-6")
    agent.add_plugin(MemoryPlugin())

    await agent.run("My name is Alex and I prefer concise answers.")
    result = await agent.run("What's my preference?")
    print(result)

asyncio.run(main())
```

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

| Plugin          | Description                    | Extra              |
| --------------- | ------------------------------ | ------------------ |
| `postgresql`    | Query and write PostgreSQL     | `[postgresql]`     |
| `mysql`         | Query and write MySQL          | `[mysql]`          |
| `mongodb`       | Query MongoDB collections      | `[mongodb]`        |
| `sqlite`        | Query and write SQLite         | `[sqlite]`         |
| `snowflake`     | Query Snowflake data warehouse | `[snowflake]`      |
| `elasticsearch` | Search Elasticsearch indices   | `[elasticsearch]`  |

### Vector Databases

| Plugin     | Description                      | Extra        |
| ---------- | -------------------------------- | ------------ |
| `chroma`   | Local/embedded vector search     | `[chromadb]` |
| `pinecone` | Managed cloud vector search      | `[pinecone]` |
| `qdrant`   | Self-hosted vector search        | `[qdrant]`   |

### Integrations & Cloud

| Plugin            | Description                      | Extra            |
| ----------------- | -------------------------------- | ---------------- |
| `rest`            | Call REST APIs                   | *(included)*     |
| `s3`              | Read/write S3 objects            | `[aws]`          |
| `slack`           | Send Slack messages              | `[slack]`        |
| `email`           | Send/receive email (SMTP/IMAP)   | *(included)*     |
| `google_drive`    | Read files from Google Drive     | `[google-drive]` |
| `websearch`       | AI-optimized web search (Tavily) | `[websearch]`    |
| `mcp`             | Model Context Protocol servers   | `[mcp]`          |
| `redis_messaging` | Redis pub/sub messaging          | `[redis]`        |
| `neo4j`           | Graph database (Cypher queries)  | `[neo4j]`        |

### Knowledge & Orchestration

| Plugin         | Description                               |
| -------------- | ----------------------------------------- |
| `memory`       | Persistent semantic agent memory          |
| `catalog`      | Schema discovery and metadata management  |
| `lineage`      | Data lineage tracking and impact analysis |
| `orchestrator` | Multi-agent coordination and task routing |

---

## Installation

### Core (OpenAI included)

```bash
pip install daita-agents
```

### Add LLM providers

```bash
pip install "daita-agents[anthropic]"   # Claude
pip install "daita-agents[google]"      # Gemini
pip install "daita-agents[llm-all]"     # All LLM providers
```

### Add database plugins

```bash
pip install "daita-agents[postgresql]"
pip install "daita-agents[mysql]"
pip install "daita-agents[mongodb]"
pip install "daita-agents[sqlite]"
pip install "daita-agents[databases]"   # All traditional databases
```

### Add vector database plugins

```bash
pip install "daita-agents[chromadb]"
pip install "daita-agents[pinecone]"
pip install "daita-agents[qdrant]"
pip install "daita-agents[vectordb]"    # All vector databases
```

### Bundles

```bash
pip install "daita-agents[recommended]"  # Anthropic + pandas + beautifulsoup4
pip install "daita-agents[complete]"     # Most features, no heavy packages
pip install "daita-agents[all]"          # Everything (large install)
```

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
