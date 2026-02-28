# Daita Agents

**Open-source Python SDK for building production AI agents.**

Daita Agents gives you a clean, minimal API for autonomous tool-calling agents that work with any LLM provider — OpenAI, Anthropic, Gemini, Grok, and more. Zero-configuration tracing, pluggable data sources, and a composable workflow system for multi-agent pipelines.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![CI](https://github.com/daita-tech/daita-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/daita-tech/daita-agents/actions)

---

## Quickstart

```bash
pip install daita-agents
```

```python
import asyncio
from daita import Agent

async def main():
    agent = Agent(
        name="Analyst",
        llm_provider="openai",
        model="gpt-4o",
    )

    result = await agent.run("Summarize the key trends in Q3 sales data.")
    print(result)

asyncio.run(main())
```

---

## Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Gemini, Grok (or bring your own)
- **Autonomous tool calling** — agents plan and execute multi-step tool chains without manual orchestration
- **Streaming** — real time token-by-token output with `stream=True`
- **Plugin ecosystem** — PostgreSQL, MySQL, MongoDB, REST APIs, S3, Slack, Elasticsearch, Neo4j, and more
- **Memory** — persistent agent memory with local or custom backends
- **Workflows** — connect multiple agents into pipelines via relay channels
- **Zero-config tracing** — every LLM call and tool execution is automatically traced (tokens, latency, cost)
- **Retry & reliability** — configurable exponential backoff with permanent-error detection

---

## Examples

### Tool-calling agent with a database

```python
from daita import Agent
from daita.plugins import postgresql

agent = Agent(
    name="Sales Analyst",
    llm_provider="openai",
    model="gpt-4o",
)

agent.add_plugin(postgresql(
    host="localhost",
    database="sales_db",
    user="analyst",
    password="secret",
))

result = await agent.run("What were the top 5 products by revenue last quarter?")
```

### Streaming output

```python
async for chunk in agent.generate("Explain transformer attention mechanisms", stream=True):
    print(chunk.content, end="", flush=True)
```

### Multi-agent workflow

```python
from daita import Agent, Workflow

fetcher  = Agent(name="Data Fetcher",  llm_provider="openai", model="gpt-4o")
analyzer = Agent(name="Analyzer",      llm_provider="openai", model="gpt-4o")

workflow = Workflow("Sales Pipeline")
workflow.add_agent("fetcher",  fetcher)
workflow.add_agent("analyzer", analyzer)
workflow.connect("fetcher", "raw_data", "analyzer")

await workflow.start()
await workflow.inject_data("fetcher", {"query": "Q3 sales"}, task="fetch")
await workflow.stop()
```

### Memory-enabled agent

```python
from daita import Agent
from daita.plugins.memory import MemoryPlugin

agent = Agent(name="Assistant", llm_provider="anthropic", model="claude-sonnet-4-6")
agent.add_plugin(MemoryPlugin())

# Memory persists across calls
await agent.run("My name is Alex and I prefer concise answers.")
result = await agent.run("What's my preference?")
```

## Plugins

| Plugin         | Description                              |
|----------------|------------------------------------------|
| `postgresql`   | Query and write PostgreSQL               |
| `mysql`        | Query and write MySQL                    |
| `mongodb`      | Query MongoDB collections                |
| `rest`         | Call REST APIs                           |
| `s3`           | Read/write S3 objects                    |
| `slack`        | Send Slack messages                      |
| `elasticsearch`| Search Elasticsearch indices             |
| `neo4j`        | Graph queries via Neo4j                  |
| `memory`       | Persistent agent memory                  |
| `websearch`    | Web search via Tavily                    |
| `email`        | Send email via SMTP/Gmail                |
| `snowflake`    | Query Snowflake data warehouse           |

---

## Installation

### Core (OpenAI only)
```bash
pip install daita-agents
```

### Recommended (common providers + tools)
```bash
pip install "daita-agents[recommended]"
```

### All providers
```bash
pip install "daita-agents[all]"
```

---

## Documentation

See the [`examples/`](examples/) directory for full working examples.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built by [Daita](https://daita-tech.io)*
