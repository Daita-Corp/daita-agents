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
| **Use your preferred model** | OpenAI, Anthropic, Gemini, Grok, Ollama, a custom OpenAI compatible endpoint, or an existing Codex/Claude Code subscription. |
| **Keep useful context** | Persist conversations, user approved memory, and reusable Markdown skills. |
| **Stay in control** | Validate SQL against the current catalog and approve agent proposed local changes. |

## Quick start

The first managed installer release targets macOS on Apple Silicon and Intel,
plus glibc Linux on x86_64 and arm64.

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

The first launch guides you through creating an agent, choosing a model, and
attaching a read only source. API-backed models store their key in the OS
keychain; local Ollama models need no key. Choosing **Codex subscription** starts
a ChatGPT device-code sign-in inside Daita—installing the Codex CLI is not
required. **Claude Code subscription** uses the installed, signed-in Claude Code
client, so run `claude auth login` before selecting it.

Subscription calls consume the allowance and follow the model availability of
the connected plan. Daita keeps all data tools inside its validated direct loop
for both transports. See
[Subscription model sources](docs/SUBSCRIPTION_MODEL_SOURCES.md) for the exact
boundary and setup details.

Once setup is complete, ask a question:

```text
Which products grew fastest month over month?
How many customers have not ordered in 90 days?
Compare paid revenue by region and plan.
```

Run `daita` again for a returning launch. Daita reopens the only agent
automatically or shows a picker when several exist. Use `daita --agent atlas`
to select one directly.

Inside the terminal, use `/help` to see available commands and shipped
controls. Press Enter to submit, Ctrl-J for a newline, and Ctrl-D to exit from
an empty prompt. Press Escape twice to clear the current input. Ctrl-C copies
an application-owned selection; without a selection it cancels an active run.
The animated status shows the active tool without filling the transcript with
tool cards. After a run, press Ctrl-O to show or hide that run's recorded tool
calls and results. Clipboard requests that a terminal cannot acknowledge are
reported as requests, not successful copies. If pointer or clipboard support
is unavailable, use the terminal's own selection bypass modifier (often
Shift) and copy command.

## How it works

Daita uses one direct model/tool loop:

```text
user message -> model -> tool calls -> ordered tool results -> model -> answer
```

The current transcript is the loop state. A tool error is returned to the
model like any other result, so it can correct the call on the next step. The
loop has bounded steps, wall time, tokens, and estimated cost; it does not add
a verifier pass or a session runtime.

Data access remains read only. SQL and local paths are checked against the
current catalog before source I/O, and every requested tool call receives one
ordered result even if another call fails.

Agent identity, source registrations, catalog snapshots, transcripts, and
terminal results live in a small SQLite database inside the agent home.
Conversation continuity projects a bounded tail of completed runs: at most 8
runs, 40 messages, and 24,000 UTF-8 bytes.

Memory and skills are bounded, advisory Markdown. They are not source truth,
authorization, or evidence. Agent proposed changes occur only in the
foreground through an in process approve once callback. The optional observer
is best effort and does not persist events, collect telemetry, or direct
execution.

Candidate review is disabled by default. When explicitly requested, review
uses one tool free model request outside `AgentLoop` and places proposals in an
inactive inbox. `/memory accept <id>` handles exactly one candidate through a
fresh foreground run and the normal approval path. There is no bulk
acceptance.

File requests use the same direct loop. Exact SQL results can become CSV or
XLSX artifacts, while attached cataloged CSV and JSON resources can be copied
byte-for-byte without passing source bytes through the model. A later turn can
use a bounded model-only `artifact_list` for the current conversation,
`artifact_read` for a bounded preview, and `artifact_convert` for the supported
Daita XLSX `Data` snapshot to CSV conversion. There is no public artifact
inventory, CLI list command, hidden current-file pointer, or prompt keyword
router.

Local files are normally delivered automatically to the authorized default
destination and reported with the verified saved path. Public recovery remains
known-ID only through `Agent.read_artifact`, `Agent.save_artifact`, and
`daita artifacts save`. Clearing conversations invalidates the corresponding
internal artifact IDs but never deletes copies already delivered to user-owned
directories.

For the complete implementation boundaries, see [AGENTS.md](AGENTS.md).

## Advanced/headless CLI

The zero argument `daita` command is the normal path. Automation-friendly
commands use the same public API:

```bash
daita --root /private/tmp/daita create atlas
daita --root /private/tmp/daita attach atlas sqlite /absolute/path/sales.db
daita --root /private/tmp/daita run atlas "Summarize sales" \
  --model openai:gpt-4.1-mini
```

`run` writes one JSON record. Provider credentials and PostgreSQL passwords
must come from their documented environment or keychain references, never
secret command line values.

Discover all commands and options with:

```bash
daita --help
daita memory --help
daita skills --help
```

## Manage local data

The terminal provides confirmed lifecycle commands:

```text
/source detach <source>
/conversation clear
/agent delete
```

Source detachment disables access and deletes a Daita-owned PostgreSQL
credential. Its non-secret registration remains as inactive lifecycle history
until the agent is deleted. Clearing conversations deletes all transcripts,
learning candidate records, and review stamps while preserving separately
approved memory, user profile, semantics, and skills. Agent deletion removes
the complete agent home and its Daita-owned keychain credentials; it never
changes the attached source data itself.

Automation must pass an explicit confirmation flag:

```bash
daita detach atlas <source-id> --yes
daita conversations clear atlas --yes
daita delete atlas --yes
```

If owned credential cleanup fails, agent deletion preserves the agent home so
the operation can be retried.

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

Close every running Daita terminal before upgrading:

```bash
pipx upgrade daita-agents
pipx reinstall daita-agents
pipx uninstall daita-agents
```

Version 1.0.0 establishes the first supported agent home format. Beginning with
the next release, every release candidate must open and preserve agent homes
created by the immediately preceding release. An upgrade preserves agent
identity, model configuration, source registrations and catalogs,
conversations, approved memory and user profile, semantics, learning
candidates, and skills. It never silently resets incompatible state.

Before changing versions, an optional complete local backup can be made while
Daita is closed:

```bash
cp -a ~/.daita ~/.daita-backup-before-upgrade
```

The backup contains the agent homes but not secret values held by the OS
keychain. Those keychain entries remain installed and are referenced by the
backed-up configuration.

If an installed release cannot admit an existing state database, it exits
before writing to that database and reports that the agent home was preserved.
Restore the complete backup before installing an older version; opening a home
with an older release after a newer release has changed its format is not a
supported downgrade path.

Use `pipx reinstall daita-agents` to repair missing or damaged application
dependencies. Uninstalling the application does not delete existing agent
homes or credentials stored in the OS keychain.

The managed installer owns only `~/.local/bin/daita` and
`~/.local/share/daita`; application data remains separately owned under
`~/.daita`. Its install, verify, repair, rollback, and uninstall actions never
roll back or delete application data or OS-keychain entries. Once published,
the stable command will be:

```bash
curl -fsSL --proto '=https' --tlsv1.2 https://daita-tech.io/install.sh | bash
```

The managed installer has not been promoted to the public endpoint, which is
not live yet. See
[the managed installer release status](docs/MANAGED_INSTALLER_RELEASE.md).

Daita 0.19.0 and earlier belong to a different legacy framework family. A
0.x-to-1.0 migration is unsupported. Preserve or archive legacy `~/.daita`
data and explicitly remove the old application before a clean 1.0 install;
the managed installer does not adopt, migrate, delete, or overwrite it.

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

Release CI builds the candidate wheel once, then passes those exact bytes to
both `tests/managed_installer_lifecycle_smoke.py` and
`tests/pipx_lifecycle_smoke.py`. The managed smoke uses deterministic local
downloads and verifies install, repeat, repair, rollback, verification,
uninstall, and data preservation. The pipx smoke exercises dependency checks,
the real XLSX runtime, reinstall, state preservation, and uninstall on Python
3.11 and 3.12 and may download declared dependencies.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.

## License

[MIT](LICENSE)
