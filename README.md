![Daita data operations agent](assets/banner.png)

# Daita

**An autonomous data operations agent that learns your data systems.**

Data systems are more than tables. Daita maps their structure, learns the
business meaning you approve, and turns individual investigations into durable
jobs and recurring checks. It answers from current evidence and carries out
admitted actions. The goal is fully autonomous data operations within the
access, budgets, and actions you authorize.

## What Daita can do

- **Understand your data.** Explore SQLite and PostgreSQL catalogs, follow
  relationships between resources, and run durable data profiles without
  changing source data.
- **Answer across systems.** Query admitted databases, local CSV, TSV, JSON,
  NDJSON, and Parquet files, and remote MCP tools using current source evidence.
- **Carry work forward.** Run durable task graphs or set assignments to run
  once or regularly. Review their results in the inbox while a Daita host is open.
- **Improve with use.** Keep conversations, approved business knowledge,
  meanings tied to specific sources, and reusable Markdown procedures. An
  optional, explicit review can propose lessons from completed runs for you to accept.
- **Deliver useful output.** Create reports backed by evidence and CSV or XLSX
  exports, including exact relational snapshots.
- **Take controlled action.** Preview and approve scoped PostgreSQL updates or
  upserts, and admit specific remote MCP actions. Durable receipts support
  inspection and recovery when an outcome is uncertain.

Daita works with OpenAI, Anthropic, Gemini, Grok, Ollama, endpoints compatible
with OpenAI's API, and [supported model subscriptions](docs/SUBSCRIPTION_MODEL_SOURCES.md).

## Quick start

On supported macOS and Linux systems, install Daita with:

```bash
curl -fsSL https://daita-tech.io/install.sh | bash
```

The installer supplies its own Python runtime and starts onboarding when run
in a terminal. Run `daita` in a new terminal to return later.

The first launch guides you through creating an agent, configuring a model, and
optionally attaching a data source for reading. The launch directory becomes
the default local working directory; foreground Files tools also accept
explicit local paths.

Try asking:

```text
How do our customers, orders, and payments tables relate?
Profile the orders and payments tables without changing them.
Compare paid revenue by region across our admitted sources.
Every Monday, summarize last week's revenue and put the result in my inbox.
```

Use `/learn <material>` to teach Daita a business definition or procedure with
approval. `/jobs` shows durable work, `/routines` manages saved assignments, and
`/inbox` shows their results. `/sources` explores connections and relationships;
`@` narrows a question to one source. Type `/help` for the full command list.

Scheduled work progresses only while the terminal or `daita host --agent <name>`
keeps that agent open. One host can open an agent home at a time.

## Authority stays explicit

Database connections start with read access only. Daita validates SQL against
the current catalog, checks tool calls against current scope and permissions,
and treats source content, remote output, memory, and skills as untrusted input.
Saved knowledge helps interpret data; it cannot grant access or override
current source facts.

External effects require exact permissions and approval or a bounded standing
grant. Native writes use a preview from the current run and transactional
checks. Daita records effect receipts and never silently retries an uncertain
action.

See [relational writes](docs/RELATIONAL_WRITES.md),
[remote MCP connectivity](docs/MCP_CONNECTIVITY.md), and
[action receipts and recovery](docs/EFFECT_RECEIPTS.md) for the exact limits.

## Guides

| Topic | Guide |
| --- | --- |
| Local file access and edits | [Local computer files](docs/LOCAL_WORKSPACES.md) |
| Reports, exports, and provenance | [Artifacts](docs/ARTIFACTS.md) |
| Source scope and retained context | [Context and scope](docs/CONTEXT_AND_SCOPE.md) |
| Scheduled work, outcomes, and hosting | [Scheduled routines](docs/SCHEDULED_ROUTINES.md) |
| Model setup and subscriptions | [Model sources](docs/SUBSCRIPTION_MODEL_SOURCES.md) |
| Python API walkthroughs | [Offline examples](examples/README.md) |
| Architecture and development | [Repository guide](AGENTS.md) · [Contributing](CONTRIBUTING.md) |
| Private vulnerability reporting | [Security policy](SECURITY.md) |

## Upgrade or uninstall

Close any running Daita terminal or host first. Run the Quick start command
again to upgrade a managed installation. To remove it, run:

```bash
curl -fsSL https://daita-tech.io/install.sh | bash -s -- --uninstall
```

Agent state under `~/.daita` is separate from the installation. Existing pipx
installations still use pipx for upgrades and removal. Daita 0.19.0 and earlier
belong to a different framework family and cannot be migrated into 1.x; preserve
that state before upgrading. See [local state compatibility](docs/LOCAL_STATE_UPGRADES.md).

## License

[MIT](LICENSE)
