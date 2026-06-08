# Daita Agents

**Open-source Python framework for building production data agents.**

Daita Agents is built for AI systems that operate on real data: databases,
schemas, catalogs, pipelines, metrics, APIs, files, and memory. Its primary
interface is `Agent.from_db()`, which returns a `DbAgent` backed by an
operation-centric `DbRuntime`: catalog-aware planning, governed task execution,
typed evidence, verification, audit state, and deterministic synthesis.

The package also includes a lightweight generic `Agent` / `ChatRuntime` for
non-DB assistants, local tool calling, plugin experiments, skill development,
streaming, and shared runtime primitive testing. Production data-agent behavior
belongs in `DbRuntime`; generic `Agent` is the supporting general capability
runtime.

```mermaid
flowchart LR
    Source["Database or data source"] --> DB["Database plugin<br/>connect + schema capabilities"]
    DB --> Catalog["Catalog<br/>normalized schema snapshots<br/>relationship lookup"]
    Catalog --> Runtime["DbRuntime<br/>operation contracts + evidence"]
    Runtime --> Agent["DbAgent<br/>run + inspect + audit"]

    Runtime --> Tasks["RuntimeKernel<br/>operations + tasks + leases"]
    Tasks --> Exec["Executors<br/>declared capabilities"]
    Exec --> Evidence["Typed evidence<br/>query results + schema + validation"]
    Evidence --> Verify["Verification + synthesis"]
    Verify --> Agent

    Runtime -.-> Governance["Governance<br/>policies + approvals"]
    Runtime -.-> Memory["Memory plugin<br/>business context"]
    Runtime -.-> Lineage["Lineage / graph evidence"]
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

Point the runtime at a database and ask a data question:

```python
import asyncio
from daita import Agent

async def main():
    agent = await Agent.from_db(
        "sqlite:///sales.db",
        mode="analyst",
        read_only=True,
    )

    answer = await agent.run("What were the top 5 products by revenue last quarter?")
    print(answer)

asyncio.run(main())
```

For inspection and audit detail:

```python
result = await agent.run_detailed("Which customers had the largest refunds?")
print(result.operation_id)
print(result.contract.required_capabilities)
print(result.evidence)

inspection = await agent.describe()
print(inspection.capability_ids)
```

`Agent.from_db()` currently supports SQLite paths/URLs, PostgreSQL URLs, and
converted extension-first database plugin instances. Additional database plugins
are available as integrations and are being moved onto the same runtime path.

---

## Runtime Model

### Primary: `Agent.from_db()`

Use `Agent.from_db()` for production data agents. It owns:

- schema discovery through database and catalog capabilities
- catalog-owned structural truth for tables, columns, and relationships
- governed operation contracts
- persisted `Operation` and `Task` records
- executor invocation through `RuntimeKernel`
- typed evidence and audit events
- DB policy, approval, resume, verification, and synthesis

```python
agent = await Agent.from_db(
    "postgresql://user:pass@localhost/warehouse",
    mode="governed",
    read_only=True,
    allowed_tables=["orders", "customers", "products"],
    query_max_rows=200,
    lineage=True,
    memory=True,
)
```

Supported modes:

- `simple`: conservative read-only data questions
- `analyst`: default read-only analytical work
- `governed`: stricter limits and lineage defaults
- `data_team`: broader data-team profile with quality and lineage support

### Supporting: `Agent`

Use generic `Agent` for lightweight non-DB assistants, local tools, model-visible
tool views, streaming chat flows, and plugin/skill development.

```python
import asyncio
from daita import Agent, tool

@tool
def calculate_discount(price: float, pct: float) -> float:
    """Calculate a discounted price."""
    return round(price * (1 - pct / 100), 2)

async def main():
    agent = Agent(
        name="Shopping Assistant",
        llm_provider="openai",
        tools=[calculate_discount],
    )

    answer = await agent.run("What is 15% off a $24 item?")
    print(answer)

asyncio.run(main())
```

Generic `Agent` can project registered `ToolView`s and execute capabilities, but
it should not grow a parallel DB planner, SQL validator, catalog graph owner, or
DB governance path.

---

## Features

- **DB-native agents**: `Agent.from_db()` builds a `DbAgent` with operation
  contracts, typed evidence, runtime inspection, and audit state.
- **Extension-first plugins**: plugins declare `Capability`, `Executor`,
  `EvidenceSchema`, `Policy`, `ContextProvider`, `ToolView`, and `Worker`
  contracts through `ExtensionRegistry`.
- **Catalog ownership**: catalog plugins own normalized schemas, infrastructure
  discovery, relationship search, and graph traversal over data assets.
- **Governance and approvals**: DB work flows through shared governance and
  persisted task execution before executors run.
- **Data quality**: `ItemAssertion` and `query_checked()` validate returned rows
  with structured `DataQualityError` violations.
- **Memory and lineage**: optional plugins add persistent business context,
  contradiction checks, graph relationships, and lineage/impact evidence.
- **Generic tool calling**: `@tool`, local tools, model-visible tool views,
  streaming, conversation history, retries, and Focus DSL support the lightweight
  chat runtime.
- **Evals**: developer-preview eval suites inspect answers, tools, SQL shape,
  data operations, budgets, stability, plugin behavior, and optional judges.

---

## Skills

Skills are reusable operating patterns. A tool is a concrete executable action;
a skill can include context, capability requirements, skill-owned capabilities,
tool views, policies, evidence schemas, workers, planning hints, verification
requirements, and synthesis preferences.

Current stable API:

```python
from daita import Agent, Skill

reporting = Skill(
    name="executive_reporting",
    description="Write concise executive summaries.",
    instructions=(
        "Use a crisp format: summary, key metrics, risks, and next actions."
    ),
)

agent = Agent(
    name="Ops Analyst",
    llm_provider="openai",
    skills=[reporting],
)
```

For advanced skills, subclass `BaseSkill` and declare runtime contracts:

```python
from daita import BaseSkill

class DataQualityInvestigationSkill(BaseSkill):
    name = "data_quality_investigation"
    description = "Investigate nulls, duplicates, drift, and schema anomalies."

    def requires_capabilities(self):
        return (
            "catalog.schema.search",
            "quality.profile",
            "db.sql.execute_read",
        )

    def get_instructions(self, user_prompt: str = ""):
        return (
            "Separate confirmed evidence from hypotheses. Prefer small "
            "diagnostic queries before broad scans."
        )
```

`Skill` no longer accepts `tools=`. Tool-backed skill ergonomics and first-class
`Agent.from_db(..., skills=[...])` integration are planned in
[docs/skills_runtime_architecture_plan.md](docs/skills_runtime_architecture_plan.md).

---

## Data Quality

Use `ItemAssertion` with database plugin queries when row-level guarantees
matter:

```python
import asyncio
from daita import DataQualityError, ItemAssertion, postgresql

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

---

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

The CLI command is planned; use the Python API while evals are in developer
preview.

---

## Streaming And Conversation History

Generic `Agent` supports streaming events:

```python
import asyncio
from daita import Agent
from daita.core.streaming import EventType

async def main():
    agent = Agent(name="Assistant", llm_provider="openai")

    async for event in agent.stream("Explain transformer attention in one paragraph"):
        if event.type == EventType.THINKING:
            print(event.content, end="", flush=True)
        elif event.type == EventType.COMPLETE:
            print("\nDone")

asyncio.run(main())
```

Conversation state is available through `ConversationHistory`:

```python
from daita import Agent, ConversationHistory

history = ConversationHistory(session_id="alice-session", workspace="support")
agent = Agent(name="Support Bot", llm_provider="openai")

await agent.run("My name is Alice and I prefer concise answers.", history=history)
answer = await agent.run("What is my preference?", history=history)
```

---

## Runtime Primitives

Most runtime behavior is expressed through extension declarations:

| Primitive | Purpose |
| --- | --- |
| `Capability` | Runtime-plannable behavior with access, risk, evidence, and executor metadata |
| `Executor` | Performs a capability for a persisted task |
| `ToolView` | Model-visible projection over a capability |
| `EvidenceSchema` | Typed evidence shape produced by executors |
| `Policy` | Governance decision logic over operation/runtime facts |
| `ContextProvider` | Audience-specific context blocks |
| `Worker` | Specialist/background worker declaration |
| `RuntimeKernel` | Operation/task/governance/executor choke point |
| `RuntimeStore` | Operation, task, evidence, event, approval, and audit persistence |

Runtime-owned DB work must pass through declared capabilities, persisted tasks,
registered executors, and the shared governance boundary.

---

## Plugins

Plugins expose direct Python APIs and may also declare runtime capabilities,
executors, evidence schemas, policies, context providers, workers, and
model-visible tool views.

### Database And Search

| Plugin | Extra |
| --- | --- |
| PostgreSQL | `[postgresql]` |
| SQLite | `[sqlite]` |
| MySQL | `[mysql]` |
| MongoDB | `[mongodb]` |
| Snowflake | `[snowflake]` |
| BigQuery | `[bigquery]` |
| Elasticsearch | `[elasticsearch]` |
| Chroma | `[chromadb]` |
| Pinecone | `[pinecone]` |
| Qdrant | `[qdrant]` |

### Integrations

| Plugin | Extra |
| --- | --- |
| S3 | `[aws]` |
| Slack | `[slack]` |
| Google Drive | `[google-drive]` |
| MCP | `[mcp]` |
| Web search | `[websearch]` |
| Exa search | `[exa]` |
| Neo4j | `[neo4j]` |
| Redis | `[redis]` |

### Domain Services

| Plugin | Purpose |
| --- | --- |
| Catalog | Schema, infrastructure, relationship, and graph discovery |
| Memory | Persistent semantic and working memory |
| Lineage | Data lineage and impact analysis |
| Data quality | Profiling and quality evidence |
| Transformer | SQL transformation management and execution |

---

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

Full bundles:

```bash
pip install "daita-agents[complete]"
pip install "daita-agents[all]"
```

Development:

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -m "not requires_llm and not requires_db"
```

---

## Exception Hierarchy

All public exceptions are importable from `daita`:

`DaitaError` -> `AgentError`, `LLMError`, `ConfigError`, `PluginError`,
`SkillError`, `TransientError`, `RetryableError`, `PermanentError`,
`RateLimitError`, `AuthenticationError`, `ValidationError`, `FocusDSLError`,
`DataQualityError`

---

## Documentation

See [examples/](examples/) for deployment examples and
[docs/skills_runtime_architecture_plan.md](docs/skills_runtime_architecture_plan.md)
for the proposed skills runtime architecture.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New production data-agent behavior should
land in `DbRuntime`; generic `Agent` should receive shared runtime primitives or
lightweight projections of those primitives.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

_Built by [Daita](https://daita-tech.io)_
