![Daita: persistent, read-only data agents](assets/banner.png)

# Daita

**The data agent that learns how your business works.**

Daita connects to SQLite, PostgreSQL, CSV, and JSON, catalogs your data, and
returns grounded answers to questions asked in plain language.

With your approval, Daita learns recurring query patterns, business semantics,
and important operational context. It carries that knowledge into future
conversations through inspectable memory and reusable skills. The more you use
it, the better it understands your data and the way your business works.

```text
You:   Which region led paid revenue last quarter?
Daita: EMEA led with $4.2M, followed by North America with $3.7M.
```

Daita keeps source access read only and learned context transparent, so it can
become more useful over time without giving up human control.

## Why Daita?

| | |
| --- | --- |
| **Talk to real data** | Query SQLite, PostgreSQL, CSV, and JSON without writing SQL. |
| **Use your preferred model** | OpenAI, Anthropic, Gemini, Grok, Ollama, or a custom OpenAI compatible endpoint. |
| **Keep useful context** | Persist conversations, user approved memory, and reusable Markdown skills. |
| **Stay in control** | Validate SQL against the current catalog and approve agent-proposed local changes. |

## Quick start

You need Python 3.11 or 3.12 and
[pipx](https://pipx.pypa.io/stable/installation/).

```bash
pipx install daita-agents
daita
```

The first launch guides you through creating an agent, choosing a model,
storing its API key in the OS keychain, and attaching a read only source. No
credentials need to appear in CLI arguments or configuration files.
Local Ollama models do not require an API key.

Once setup is complete, ask a question:

```text
Which products grew fastest month over month?
How many customers have not ordered in 90 days?
Compare paid revenue by region and plan.
```

Run `daita` again for a returning launch. Daita reopens the only agent
automatically or shows a picker when several exist. Use `daita --agent atlas`
to select one directly.

Inside the terminal, use `/help` to see available commands. Press Enter to
submit, Ctrl-J for a newline, Ctrl-C to cancel an active run, and Ctrl-D to
exit from an empty prompt.

## How it works

Daita uses one direct model/tool loop:

```text
user message -> model -> tool calls -> ordered tool results -> model -> answer
```

The current transcript is the loop state. A tool error is returned to the
model like any other result, so it can correct the call on the next step. The
loop has bounded steps, wall time, tokens, and estimated cost; it does not add
a verifier pass or a session runtime.

Data access remains read-only. SQL and local paths are checked against the
current catalog before source I/O, and every requested tool call receives one
ordered result even if another call fails.

Agent identity, source registrations, catalog snapshots, transcripts, and
terminal results live in a small SQLite database inside the agent home.
Conversation continuity projects a bounded tail of completed runs: at most 8
runs, 40 messages, and 24,000 UTF-8 bytes.

Memory and skills are bounded, advisory Markdown. They are not source truth,
authorization, or evidence. Agent-proposed changes occur only in the
foreground through an in-process approve-once callback. The optional observer
is best-effort and does not persist events, collect telemetry, or direct
execution.

Candidate review is disabled by default. When explicitly requested, review
uses one tool-free model request outside `AgentLoop` and places proposals in an
inactive inbox. `/memory accept <id>` handles exactly one candidate through a
fresh foreground run and the normal approval path. There is no bulk
acceptance.

For the complete implementation boundaries, see [AGENTS.md](AGENTS.md).

## Advanced/headless CLI

The zero-argument `daita` command is the normal path. Automation-friendly
commands use the same public API:

```bash
daita --root /private/tmp/daita create atlas
daita --root /private/tmp/daita attach atlas sqlite /absolute/path/sales.db
daita --root /private/tmp/daita run atlas "Summarize sales" \
  --model openai:gpt-4.1-mini
```

`run` writes one JSON record. Provider credentials and PostgreSQL passwords
must come from their documented environment or keychain references, never
secret command-line values.

Discover all commands and options with:

```bash
daita --help
daita memory --help
daita skills --help
```

## Python and examples

The public async API supports creating and opening agents, attaching sources,
running questions, continuing conversations, and inspecting transcripts.
Start with the deterministic offline
[SQLite quickstart](examples/00_quickstart_sqlite_from_db.py), then explore
the [examples guide](examples/README.md).

```bash
PYTHONPATH=src .venv/bin/python examples/00_quickstart_sqlite_from_db.py
```

## Upgrade or uninstall

```bash
pipx upgrade daita-agents
pipx reinstall daita-agents
pipx uninstall daita-agents
```

Use `pipx reinstall daita-agents` to repair missing or damaged application
dependencies. Uninstalling the application does not delete existing agent
homes or credentials stored in the OS keychain.

## Development

```bash
git clone https://github.com/Daita-Corp/daita-agents.git
cd daita-agents
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Run the deterministic suite before submitting a change:

```bash
.venv/bin/python -m pytest tests/ -m "not requires_llm and not requires_db"
.venv/bin/python -m black --check src tests
.venv/bin/python -m mypy src/daita tests
```

`tests/pipx_lifecycle_smoke.py` additionally builds and exercises an isolated
pipx installation; it may download declared dependencies.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.

## License

[Apache 2.0](LICENSE)
