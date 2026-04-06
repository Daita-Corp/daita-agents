# Deployment Examples

Full project examples that mirror what `daita init` creates. Each is a self-contained
project you can run locally and deploy to production with `daita push`.

## Learning path

Start simple and work up to multi-agent pipelines:

```
csv-data-analyst         → single agent, no external services
sql-data-agent           → single agent + database plugin
infrastructure-catalog   → single agent + catalog plugin (AWS + GitHub discovery)
slack-reporter           → single agent + two plugins + scheduling
etl-pipeline             → multi-agent linear workflow
customer-support-bot     → multi-agent routing workflow
deep-research            → multi-agent pipeline with memory + web search
```

---

## Examples

### [`csv-data-analyst`](./csv-data-analyst/)

A single agent that answers natural language questions about a CSV file.

**Use case:** Drop a CSV in `data/`, ask "what are the top 5 products by revenue?" and
get a plain-English answer backed by real computation.

**Highlights:**
- `@tool` decorator for wrapping pandas operations
- `data` extra (`pip install 'daita-agents[data]'`)
- Single-agent pattern — the simplest useful starting point

**Required env vars:**
```
OPENAI_API_KEY
```

---

### [`sql-data-agent`](./sql-data-agent/)

A single agent that translates natural language questions into SQL, runs them against
a PostgreSQL database, and explains the results.

**Use case:** "Show me customers who haven't ordered in 90 days" → agent inspects the
schema, generates the right SQL, runs it, and summarises the output.

**Highlights:**
- PostgreSQL plugin with lazy import
- Schema-inspection tool so the agent understands the database structure
- Result formatting and error recovery

**Required env vars:**
```
OPENAI_API_KEY
DATABASE_URL   # postgresql://user:pass@host:5432/dbname
```

---

### [`infrastructure-catalog`](./infrastructure-catalog/)

A single agent that discovers data stores across AWS accounts and GitHub
repositories, builds a unified catalog, and answers questions about the
organization's data landscape.

**Use case:** "What production databases do we have?" -> the agent scans AWS
RDS/DynamoDB/S3/ElastiCache/Redshift, scans GitHub repos for connection strings,
deduplicates results, and returns a categorized summary.

**Highlights:**
- CatalogPlugin with pluggable `AWSDiscoverer` and `GitHubScanner`
- Store deduplication and environment inference
- Schema profiling via `PostgresProfiler` / `MySQLProfiler`
- Paginated tool responses for large catalogs

**Required env vars:**
```
OPENAI_API_KEY
AWS credentials (AWS_ACCESS_KEY_ID / AWS_PROFILE / AWS_ROLE_ARN)
```

**Optional env vars:**
```
GITHUB_TOKEN       # enables GitHub repo scanning
GITHUB_ORG         # scan all repos in an org
AWS_REGIONS        # comma-separated (default: us-east-1)
```

---

### [`slack-reporter`](./slack-reporter/)

A single agent that runs on a schedule, queries a database, builds a summary, and
posts it to a Slack channel.

**Use case:** Daily sales digest posted to `#analytics` every morning at 9 AM.

**Highlights:**
- Slack plugin
- PostgreSQL plugin
- Cron scheduling via `daita-project.yaml`
- Composing multiple plugins in one agent

**Required env vars:**
```
OPENAI_API_KEY
DATABASE_URL
SLACK_BOT_TOKEN
SLACK_CHANNEL_ID
```

---

### [`etl-pipeline`](./etl-pipeline/)

A three-agent linear workflow: **Extractor** pulls raw data from a source →
**Transformer** cleans and reshapes it with pandas → **Loader** writes the result
to a destination table.

**Use case:** Nightly job that ingests raw event logs, normalises them, and loads
them into a reporting database.

**Highlights:**
- `Workflow` with relay channels (`raw_data` → `transformed_data`)
- Each agent has a single, focused responsibility
- Demonstrates the linear multi-agent pipeline pattern

**Required env vars:**
```
OPENAI_API_KEY
SOURCE_DATABASE_URL
DEST_DATABASE_URL
```

---

### [`customer-support-bot`](./customer-support-bot/)

A two-agent workflow: a **Classifier** reads incoming tickets and routes them to the
appropriate **Specialist** agent (billing, technical, or general).

**Use case:** Support queue automation that tags and triages tickets before a human
ever sees them.

**Highlights:**
- Conditional relay routing (unlike the linear pattern in `etl-pipeline`)
- `MemoryPlugin` for per-conversation history
- Agent specialization — each agent has a narrow, well-defined prompt

**Required env vars:**
```
OPENAI_API_KEY
```

---

### [`deep-research`](./deep-research/)

A four-agent pipeline: **Orchestrator** plans the research → **Web Researcher**
searches the web → **Analyst** synthesises findings → **Report Writer** produces a
cited report.

**Use case:** "What are the latest breakthroughs in solid-state batteries?" → receive
a structured research report with sources 5 minutes later.

**Highlights:**
- Full relay pipeline across four specialised agents
- `WebSearchPlugin` (Tavily) + `MemoryPlugin` + `LineagePlugin`
- Shared project-scoped memory so prior research is recalled on repeat queries

**Required env vars:**
```
OPENAI_API_KEY
TAVILY_API_KEY
```

---

## Project structure

Every example follows the standard `daita init` layout:

```
example-name/
├── agents/                 # One file per agent
│   └── my_agent.py
├── workflows/              # Workflow definitions (multi-agent examples only)
│   └── my_workflow.py
├── data/                   # Input/output data files
├── tests/                  # pytest tests
│   └── test_basic.py
├── daita-project.yaml      # Project config (name, agents, schedules)
├── requirements.txt        # Pinned dependencies
└── .gitignore
```

## Running an example

```bash
cd examples/deployments/csv-data-analyst

pip install -r requirements.txt

# Set required env vars (see example README for specifics)
export OPENAI_API_KEY=sk-...

# Run locally
python agents/my_agent.py

# Test
pytest tests/

# Deploy
daita push
```
