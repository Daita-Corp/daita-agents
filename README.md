# Daita Agents

**Open-source Python framework for building production data agents.**

Daita Agents is built around `Agent.from_db()`: the production interface for
agents that answer questions from real structured data with planning,
governance, typed evidence, verification, and auditability.

- `Agent.from_db()` is the primary entry point. It returns a
  `DbAgent` backed by `DbRuntime`, with operation contracts, governed task
  execution, typed evidence, verification, monitors, resume, and audit state.
- Generic `Agent` exists as a secondary support surface for local tools, skills,
  streaming, conversation history, and non-DB experiments.
- Plugins and skills declare capabilities into a shared runtime registry.
  Tools are only the model-visible projection of those capabilities.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-daita--agents-orange)](https://pypi.org/project/daita-agents/)
[![Version](https://img.shields.io/pypi/v/daita-agents.svg)](https://pypi.org/project/daita-agents/)

## Quickstart

```bash
pip install "daita-agents[sqlite]"
```

Ask a question over a SQLite or PostgreSQL source:

```python
import asyncio
from daita import Agent


async def main():
    agent = await Agent.from_db(
        "sales.db",
        mode="analyst",
        read_only=True,
    )

    answer = await agent.run("What were the top 5 products by revenue last quarter?")
    print(answer)

    await agent.stop()


asyncio.run(main())
```

Inspect the operation instead of only reading the answer:

```python
result = await agent.run_detailed("Which customers had the largest refunds?")
print(result.operation_id)
print(result.contract.required_capabilities)
print(result.evidence)

inspection = await agent.describe()
print(inspection.plugin_ids)
print(inspection.capability_ids)
```

## Architecture

```mermaid
flowchart LR
    User["Data question"] --> Primary["Agent.from_db()<br/>primary public interface"]
    Primary --> DBAgent["DbAgent"]

    DBAgent --> DbRuntime["DbRuntime<br/>planning + governance + evidence"]
    DbRuntime --> Kernel["RuntimeKernel<br/>operations + tasks + approvals"]
    Kernel --> Store["RuntimeStore<br/>operation, task, evidence, events"]
    Kernel --> Registry["ExtensionRegistry"]

    Registry --> Plugins["Plugins<br/>connectors + domain services"]
    Registry --> Skills["Skills<br/>instructions + declarations"]
    Plugins --> Capabilities["Capabilities + executors<br/>evidence + policies + workers"]
    Skills --> Capabilities

    DbRuntime --> Catalog["Catalog plugin<br/>schemas + relationships + graph search"]
    DbRuntime --> Synthesis["Verification + synthesis"]

    Support["Generic Agent<br/>secondary support surface"] -.-> ChatRuntime["ChatRuntime<br/>model turns + tool views"]
    ChatRuntime -.-> Kernel
```

The important boundary is that runtime-owned work goes through declared
capabilities, persisted tasks, registered executors, and the shared governance
boundary. `DbRuntime` owns database planning, SQL validation, approval state,
resume, evidence, verification, and synthesis. The generic `Agent` should not
grow a parallel DB planner or catalog graph owner.

## Package Structure

| Path                     | Responsibility                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| `daita/db/`              | Public `from_db`, `DbAgent`, contracts, query planning, SQL analysis, synthesis, verification           |
| `daita/db/runtime/`      | Operation-centric DB runtime mixins for tasks, governance, resume, monitors, cache, results             |
| `daita/agents/`          | Secondary generic `Agent`, `BaseAgent`, conversation history, and chat facade                           |
| `daita/agents/chat/`     | Runtime-native generic chat loop, tool execution, guardrails, retries, evidence                         |
| `daita/runtime/`         | Domain-neutral primitives: operations, tasks, capabilities, evidence, policies, workers, stores, kernel |
| `daita/plugins/`         | Extension-first connectors and domain services plus `ExtensionRegistry`                                 |
| `daita/plugins/catalog/` | Catalog-owned discovery, normalization, profiling, persistence, relationship search, graph views        |
| `daita/plugins/memory/`  | Semantic, keyword, graph, working-memory, contradiction, and storage helpers                            |
| `daita/skills/`          | Skill declarations, activation, discovery, runtime effects, and skill-owned tool adapters               |
| `daita/llm/`             | OpenAI, Anthropic, Gemini, Grok, Ollama, OpenAI-compatible, and mock providers                          |
| `daita/embeddings/`      | OpenAI, Gemini, Voyage, sentence-transformers, and mock embedding providers                             |
| `daita/evals/`           | Eval suites, assertions, judges, reporters, artifacts, datasets, baselines                              |
| `daita/core/`            | Tools, exceptions, tracing, streaming, Focus DSL, assertions, graph helpers                             |
| `daita/config/`          | Agent config, retry policies, retry strategies, settings                                                |
| `tests/`                 | Unit, integration, performance, fixtures, mocks, and live-gated suites                                  |
| `examples/`              | Basic examples and deployment-style project templates                                                   |

## `Agent.from_db()`

Use `Agent.from_db()` when an agent needs to answer questions from structured
data with a durable operation trail.

```python
agent = await Agent.from_db(
    "postgresql://user:pass@localhost/warehouse",
    mode="governed",
    read_only=True,
    allowed_tables=["orders", "customers", "products"],
    query_default_limit=50,
    query_max_rows=200,
    query_timeout=30,
    lineage=True,
    memory=True,
)
```

Current source support on the new `DbRuntime` path:

| Source                                                 | Status                                                                            |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- |
| SQLite file path, `:memory:`, or `sqlite://...`        | Supported                                                                         |
| PostgreSQL URL, `postgresql://...` or `postgres://...` | Supported                                                                         |
| Converted `BaseDatabasePlugin` instance                | Supported                                                                         |
| Other database URL schemes                             | Direct plugin APIs exist, but URL routing into `from_db` is still being converted |

Built-in modes:

| Mode        | Default posture                                                      |
| ----------- | -------------------------------------------------------------------- |
| `simple`    | Conservative read-only questions with small row and character limits |
| `analyst`   | Default read-only analytical profile                                 |
| `governed`  | Stricter limits with lineage enabled by default                      |
| `data_team` | Broader data-team profile with quality and lineage enabled           |

`DbAgent` exposes:

- `run(prompt)`: returns the synthesized answer string.
- `run_detailed(prompt)`: returns a typed `DbOperationResult`.
- `describe()`: returns a `DbRuntimeInspection` registry and runtime snapshot.
- `operations` and `audit_log`: retained operation summaries.
- `monitor(...)` and monitor management methods for durable DB observations.
- `stop()` / `teardown()`: releases runtime resources.

### DB Operation Flow

1. `Agent.from_db()` resolves a source plugin, catalog plugin, optional memory,
   lineage, and data-quality plugins.
2. `DbRuntime.setup()` registers plugin declarations in `ExtensionRegistry`.
3. A prompt becomes a `DbRequest`, `DbIntent`, and `DbOperationContract`.
4. The runtime persists an operation and planned tasks through `RuntimeKernel`.
5. Governance evaluates operation and task facts; approvals can block execution.
6. Executors produce typed evidence such as schema, SQL validation, query
   results, quality profiles, lineage, or synthesis payloads.
7. Verification and synthesis produce a final `DbOperationResult` with audit
   diagnostics.

## Secondary Generic `Agent`

Most users building data agents should start with `Agent.from_db()`. The generic
agent is a secondary non-DB runtime surface for lightweight assistants, local
tool calling, skill experiments, streaming events, and conversation history. It
uses the same registry and kernel primitives, but its owner is `ChatRuntime`
rather than `DbRuntime`.

```python
from daita import Agent, tool


@tool
def calculate_discount(price: float, pct: float) -> float:
    """Calculate a discounted price."""
    return round(price * (1 - pct / 100), 2)


agent = Agent(
    name="shopping_assistant",
    llm_provider="openai",
    tools=[calculate_discount],
)
```

```python
result = await agent.run(
    "Use the calculator and show the final price.",
    detailed=True,
)
print(result["operation_id"])
print(result["tool_calls"])
```

Streaming:

```python
from daita.core.streaming import EventType

async for event in agent.stream("Explain transformer attention in one paragraph"):
    if event.type == EventType.THINKING:
        print(event.content, end="", flush=True)
    elif event.type == EventType.COMPLETE:
        print("\nDone")
```

Conversation state:

```python
from daita import Agent, ConversationHistory

history = ConversationHistory(session_id="alice-session", workspace="support")
agent = Agent(name="support_bot", llm_provider="openai")

await agent.run("My name is Alice and I prefer concise answers.", history=history)
answer = await agent.run("What is my preference?", history=history)
```

## Runtime Primitives

Most framework behavior is expressed as declarations:

| Primitive         | Purpose                                                                       |
| ----------------- | ----------------------------------------------------------------------------- |
| `Capability`      | Runtime-plannable behavior with access, risk, evidence, and executor metadata |
| `Executor`        | Performs a capability for one persisted task                                  |
| `EvidenceSchema`  | Declares the typed evidence shape produced by executors                       |
| `Evidence`        | Runtime output accepted from executors, workers, or policies                  |
| `Policy`          | Governance decision logic over runtime facts                                  |
| `ContextProvider` | Renders context blocks for models, synthesizers, reviewers, or inspectors     |
| `ToolView`        | Model-visible projection over a capability                                    |
| `Worker`          | Specialist or background worker declaration                                   |
| `Operation`       | Durable top-level unit of runtime work                                        |
| `Task`            | Durable executable unit within an operation                                   |
| `RuntimeKernel`   | Shared operation/task/governance/executor choke point                         |
| `RuntimeStore`    | Operation, task, evidence, event, approval, and audit persistence             |

## Plugins

Plugins can expose direct Python APIs and declare runtime contracts. Extension
plugins should declare stable manifests and contribute capabilities, executors,
evidence schemas, policies, context providers, tool views, and workers through
the registry.

### Database And Search

| Plugin        | Factory              | Extra             |
| ------------- | -------------------- | ----------------- |
| PostgreSQL    | `postgresql(...)`    | `[postgresql]`    |
| SQLite        | `sqlite(...)`        | `[sqlite]`        |
| MySQL         | `mysql(...)`         | `[mysql]`         |
| MongoDB       | `mongodb(...)`       | `[mongodb]`       |
| Snowflake     | `snowflake(...)`     | `[snowflake]`     |
| BigQuery      | `bigquery(...)`      | `[bigquery]`      |
| Elasticsearch | `elasticsearch(...)` | `[elasticsearch]` |
| Chroma        | `chroma(...)`        | `[chromadb]`      |
| Pinecone      | `pinecone(...)`      | `[pinecone]`      |
| Qdrant        | `qdrant(...)`        | `[qdrant]`        |

### Integrations

| Plugin           | Factory                | Extra            |
| ---------------- | ---------------------- | ---------------- |
| REST APIs        | `rest(...)`            | core             |
| S3               | `s3(...)`              | `[aws]`          |
| Google Drive     | `google_drive(...)`    | `[google-drive]` |
| Slack            | `slack(...)`           | `[slack]`        |
| Email            | `email(...)`           | core             |
| MCP              | `mcp` module           | `[mcp]`          |
| Web search       | `websearch(...)`       | `[websearch]`    |
| Exa search       | `exa_search(...)`      | `[exa]`          |
| Redis data store | `redis(...)`           | `[redis]`        |
| Redis messaging  | `redis_messaging(...)` | `[redis]`        |
| Neo4j            | `neo4j(...)`           | `[neo4j]`        |

### Domain Services

| Plugin       | Factory             | Purpose                                                   |
| ------------ | ------------------- | --------------------------------------------------------- |
| Catalog      | `catalog(...)`      | Schema, infrastructure, relationship, and graph discovery |
| Memory       | `memory(...)`       | Persistent semantic and working memory                    |
| Lineage      | `lineage(...)`      | Data lineage and impact evidence                          |
| Data quality | `data_quality(...)` | Profiling, freshness, anomaly, and report evidence        |
| Transformer  | `transformer(...)`  | SQL transformation management and execution               |

Direct plugin example:

```python
import asyncio
from daita.plugins import sqlite


async def main():
    async with sqlite(path="./sales.db") as db:
        rows = await db.query("SELECT product, revenue FROM sales LIMIT 5")
        print(rows)


asyncio.run(main())
```

## Skills

Skills are reusable units of agent behavior. They can provide instructions,
runtime discovery metadata, activation rules, runtime effects, and optional
runtime declarations.

Simple instruction skill:

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

Tool-backed skill:

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

Advanced skills subclass `BaseSkill` when they need dynamic instructions,
capability requirements, policies, workers, or custom runtime declarations.

`Skill(...)` no longer accepts `tools=` directly. Use `Skill.with_tools(...)` or
declare capabilities, executors, and tool views explicitly.

## Data Quality

Use `ItemAssertion` with database plugin queries when row-level guarantees
matter:

```python
import asyncio
from daita import DataQualityError, ItemAssertion
from daita.plugins import postgresql


async def main():
    async with postgresql(host="localhost", database="sales_db") as db:
        try:
            rows = await db.query_checked(
                "SELECT id, amount, customer_id FROM transactions",
                assertions=[
                    ItemAssertion(lambda row: row["amount"] > 0, "amount must be positive"),
                    ItemAssertion(lambda row: row["customer_id"] is not None, "customer_id required"),
                ],
            )
            print(f"{len(rows)} clean rows")
        except DataQualityError as exc:
            print(f"Data quality failure: {exc}")


asyncio.run(main())
```

## Evals

Eval suites run Daita agents locally or in CI and write structured artifacts:
`report.json`, `summary.md`, JUnit XML, per-case run artifacts, repeat-run
diffs, judge artifacts, and baseline comparisons.

```yaml
name: sales-agent-evals
version: 1

agent:
  factory: "myapp.agents:create_sales_agent"

cases:
  - id: top-products
    prompt: What were the top 5 products by revenue?
    expectations:
      answer:
        contains: ["Widget"]
      sql:
        read_only: true
        require_limit: true
        must_not_include: ["DELETE", "DROP"]
      budgets:
        max_latency_ms: 15000
```

```python
import asyncio
from daita.evals import EvalSuite
from daita.evals.reporters import render_pretty


async def main():
    report = await EvalSuite.from_file("evals/sales-agent.yaml").run()
    print(render_pretty(report))


asyncio.run(main())
```

The eval API is developer-preview.

## Installation

Core install:

```bash
pip install daita-agents
```

Common extras:

```bash
pip install "daita-agents[recommended]"
pip install "daita-agents[postgresql]"
pip install "daita-agents[sqlite]"
pip install "daita-agents[databases]"
pip install "daita-agents[data]"
pip install "daita-agents[websearch]"
pip install "daita-agents[otlp]"
```

LLM providers:

```bash
pip install "daita-agents[anthropic]"
pip install "daita-agents[google]"
pip install "daita-agents[llm-all]"
```

Other useful extras include `[memory]`, `[voyage]`, `[sentence-transformers]`,
`[data-quality]`, `[aws]`, `[azure]`, `[gcp]`, `[google-drive]`, `[slack]`,
`[mcp]`, `[redis]`, `[neo4j]`, `[lineage]`, `[cloud]`, `[complete]`, and `[all]`.

## LLM And Embeddings

LLM providers are created lazily when the agent first needs a model. Built-in
provider names are `openai`, `anthropic`, `grok`, `gemini`, `ollama`, and
`mock`.

```python
from daita import create_llm_provider

llm = create_llm_provider("openai", "gpt-5.4-mini", api_key="sk-...")
```

Embedding providers live under `daita.embeddings` and include OpenAI, Gemini,
Voyage, sentence-transformers, and mock implementations.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -m "not requires_llm and not requires_db"
```

Useful test targets:

```bash
pytest tests/unit/ -v
pytest tests/unit/db/test_agent_from_db.py -v
pytest tests/unit/core/test_skills.py -v
pytest tests/ -m "not requires_llm and not requires_db"
```

Development rules that matter most in this codebase:

- Optional dependencies must be imported lazily inside `connect()` or a client
  property body.
- Packages needed by one integration belong in an optional extra, not core
  dependencies.
- `asyncio_mode = "auto"` is configured globally; do not add per-test
  `@pytest.mark.asyncio`.
- Production DB behavior belongs in `DbRuntime`; catalog behavior belongs in
  catalog plugins; generic `Agent` should consume shared primitives rather than
  reimplementing DB paths.

## Exceptions

Public exceptions are importable from `daita`:

`DaitaError`, `AgentError`, `LLMError`, `ConfigError`, `PluginError`,
`SkillError`, `TransientError`, `RetryableError`, `PermanentError`,
`RateLimitError`, `AuthenticationError`, `ValidationError`, `FocusDSLError`,
and `DataQualityError`.

## More

- [examples/](examples/) contains basic examples and deployment templates.
- [tests/README.md](tests/README.md) documents test organization.
- [CONTRIBUTING.md](CONTRIBUTING.md) covers contribution workflow.

## License

Apache 2.0 - see [LICENSE](LICENSE).

Built by [Daita](https://daita-tech.io).
