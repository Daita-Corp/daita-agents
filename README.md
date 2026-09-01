![Daita: persistent data agents](assets/banner.png)

# Daita

The data agent that learns how your business works.

Daita connects to SQLite, PostgreSQL, and an admitted local workspace, then
returns grounded answers to questions asked in plain language. Conversations,
approved memory, and reusable skills persist across sessions so useful business
context does not have to be explained again.

[Quick start](#quick-start) ·
[Local workspaces](docs/LOCAL_WORKSPACES.md) ·
[Model sources](docs/SUBSCRIPTION_MODEL_SOURCES.md) ·
[Remote MCP](docs/MCP_CONNECTIVITY.md) ·
[Scheduled routines](docs/SCHEDULED_ROUTINES.md) ·
[Examples](examples/README.md)

```text
You:   Which region led paid revenue last quarter?
Daita: EMEA led with $4.2M, followed by North America with $3.7M.
```

## Why Daita?

| | |
| --- | --- |
| **Talk to real data** | Query SQLite and PostgreSQL, or analyze admitted CSV, TSV, JSON, NDJSON, and Parquet files without writing SQL. |
| **Get grounded answers** | Daita validates queries against the current catalog before reading a source. |
| **Choose your model** | Use OpenAI, Anthropic, Gemini, Grok, Ollama, an OpenAI-compatible endpoint, or supported model subscriptions. |
| **Keep useful context** | Persist conversations, user-approved memory, and reusable Markdown skills. |
| **Stay in control** | Sources begin read-only, access is explicitly scoped, and operational effects require exact approval. |

## Quick start

You need Python 3.11 or 3.12 and
[pipx](https://pipx.pypa.io/stable/installation/).

```bash
pipx install daita-agents
daita
```

If pipx would otherwise select a newer unsupported interpreter, choose an
installed Python 3.11 or 3.12 explicitly:

```bash
pipx install --python python3.12 daita-agents
```

The first launch guides you through choosing a workspace, creating an agent,
selecting a model, and optionally attaching a read-only data source. API-backed
model credentials are stored in the OS keychain. Ollama needs no API key, and
supported Codex, Claude Code, and Grok Build subscriptions can use their
documented sign-in flows. See
[Subscription model sources](docs/SUBSCRIPTION_MODEL_SOURCES.md) for setup and
security boundaries.

Once setup is complete, try asking:

```text
Which products grew fastest month over month?
How many customers have not ordered in 90 days?
Compare paid revenue by region and plan.
Summarize the CSV files in this workspace.
```

Run `daita` again for a returning launch. Daita reopens the only agent or shows
a picker when several exist. Use `daita --agent atlas` to select one directly.
Inside the terminal, `/help` lists commands and controls, `/` opens the command
palette, and `@` selects a source for one question.

## Read-first by design

Daita treats source metadata, query results, file content, remote tool output,
memory, and skills as untrusted input. None of them can grant authority or
change the execution policy.

- SQLite and PostgreSQL sources begin read-only.
- SQL is validated against the current catalog before source I/O.
- Workspace reads reject traversal, symlinks, secret-like paths, and special
  files.
- Remote MCP tools require an explicitly admitted read-only binding and are
  revalidated at call time.
- The only supported source-data mutation is an explicitly enabled structured
  PostgreSQL update with an exact target preview, once-only approval,
  transactional drift detection, and an immutable receipt.

Learn more in [Local workspaces](docs/LOCAL_WORKSPACES.md),
[Remote MCP read connectivity](docs/MCP_CONNECTIVITY.md), and
[PostgreSQL updates](docs/POSTGRESQL_UPDATES.md).

## How it works

Daita uses one direct model/tool loop:

```text
user message -> model -> tool calls -> ordered tool results -> model -> answer
```

The current transcript is the loop state. Tool failures are returned to the
model like ordinary results so it can correct a call on the next step. Steps,
wall time, tokens, and estimated cost are bounded.

Agent identity, source registrations, catalog snapshots, transcripts, jobs,
routines, and results are stored in one SQLite database inside the agent home.
Memory and skills are bounded advisory Markdown—not source truth, evidence, or
authorization. Durable jobs and scheduled routines use the same catalog,
capability runtime, and execution loop as foreground questions.

For the full implementation boundaries, see the
[repository architecture guide](AGENTS.md).

## Documentation

| Topic | Guide |
| --- | --- |
| Workspace selection, file reads, queries, and edits | [Local workspaces](docs/LOCAL_WORKSPACES.md) |
| Codex, Claude Code, and Grok Build subscriptions | [Subscription model sources](docs/SUBSCRIPTION_MODEL_SOURCES.md) |
| Read-only remote tools | [Remote MCP connectivity](docs/MCP_CONNECTIVITY.md) |
| Schedules, outcomes, inboxes, and resident hosting | [Scheduled routines](docs/SCHEDULED_ROUTINES.md) |
| Scoped PostgreSQL updates and receipts | [PostgreSQL updates](docs/POSTGRESQL_UPDATES.md) |
| State compatibility and automatic upgrades | [Local state compatibility](docs/LOCAL_STATE_UPGRADES.md) |
| Managed installer release status | [Managed installer](docs/MANAGED_INSTALLER_RELEASE.md) |
| Public Python API walkthroughs | [Offline examples](examples/README.md) |
| Development and architecture contracts | [Repository guide](AGENTS.md) |
| Contribution workflow | [Contributing](CONTRIBUTING.md) |
| Private vulnerability reporting | [Security policy](SECURITY.md) |

For command discovery, use:

```bash
daita --help
daita routines --help
```

Python users can start with the deterministic SQLite quickstart:

```bash
PYTHONPATH=src .venv/bin/python examples/00_quickstart_sqlite_from_db.py
```

## Upgrade or uninstall

Close every running Daita terminal before managing the installation:

```bash
pipx upgrade daita-agents
pipx reinstall daita-agents
pipx uninstall daita-agents
```

Application state under `~/.daita` is separate from the installation and is
not removed by pipx. Daita 0.19.0 and earlier belong to a different legacy
framework family; a 0.x-to-1.0 migration is unsupported. Preserve legacy state
before installing Daita 1.x. See [Local state compatibility](docs/LOCAL_STATE_UPGRADES.md)
and the [managed installer status](docs/MANAGED_INSTALLER_RELEASE.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull-request
guidance. Report security issues privately through [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
