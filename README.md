![Daita: persistent data agents](assets/banner.png)

# Daita

**The data agent that learns how your business works.**

Daita connects to a data source, catalogs your data, and
returns grounded answers to questions asked in plain language.

Daita learns recurring query patterns, business semantics,
and important operational context. It carries that knowledge into future
conversations through inspectable memory and reusable skills. The more you use
it, the better it understands your data and the way your project works.

```text
You:   Which region led paid revenue last quarter?
Daita: EMEA led with $4.2M, followed by North America with $3.7M.
```

Daita begins every source read only and keeps learned context transparent. An
explicitly opt-in PostgreSQL source can additionally expose the narrow,
previewed, once approved structured update described below; other source access
remains read only.

## Why Daita?

| | |
| --- | --- |
| **Talk to real data** | Query SQLite, PostgreSQL, CSV, and JSON without writing SQL. |
| **Use your preferred model** | OpenAI, Anthropic, Gemini, Grok, Ollama, a custom OpenAI compatible endpoint, or supported Codex, Claude Code, and Grok Build subscriptions. |
| **Keep useful context** | Persist conversations, user approved memory, and reusable Markdown skills. |
| **Stay in control** | Validate reads against the current catalog and require resource scoped readiness, preview, and exact approval for the limited PostgreSQL update. |

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
attaching a read only source inside our Textual application; setup,
chat, pickers, secret entry, confirmations, and approvals never fall back to a
second line oriented interface. API backed models store their key in the OS
keychain; local Ollama models need no key. Choosing **Codex subscription** starts
a ChatGPT device code sign in inside Daita. Installing the Codex CLI is not
required. **Claude Code subscription** uses the installed, signed in Claude Code
client, so run `claude auth login` before selecting it. **Grok Build
subscription** uses a signed in `grok` client (`grok login`). The CLI route does
not store a provider credential in Daita configuration. Gemini remains available
through its explicit API key billed `gemini:<model>` route.

Subscription calls consume the allowance and follow the model availability of
the connected plan. Daita keeps all data tools inside its validated direct loop
for every subscription transport. See
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
to select one directly. The retained `daita chat` spelling is a strict alias of
the same Textual application, not a separate chat implementation.

Inside the terminal, use `/help` to see available commands and shipped
controls. Type `/` to open the command palette, continue typing to filter it,
and use Up/Down plus Tab or Enter to insert the highlighted command. Type `@`
to choose a registered source for one question without changing the persistent
active source. Escape closes an open palette before participating in the normal
double-Escape input clear. Press Enter to submit, Ctrl-J for a newline, and
Ctrl-D to exit from an empty prompt. Ctrl-C copies an application-owned
selection; without a selection it cancels an active run. The animated status
shows the active tool without filling the transcript with tool cards. After a
run, press Ctrl-O to show or hide that run's recorded tool calls and results.
Clipboard requests that a terminal cannot acknowledge are reported as
requests, not successful copies. If pointer or clipboard support is
unavailable, use the terminal's own selection bypass modifier (often Shift)
and copy command.

`/memory edit`, `/user edit`, `/memory edit <candidate-id>`, and `/skills
create` or `/skills edit` use the configured `$EDITOR`. Textual temporarily
restores the ordinary terminal while that external editor runs, then reacquires
the full-screen UI. Source configuration, passwords, selection, and approval
remain inside Textual.

## How it works

Daita uses one direct model/tool loop:

```text
user message -> model -> tool calls -> ordered tool results -> model -> answer
```

The current transcript is the loop state. A tool error is returned to the
model like any other result, so it can correct the call on the next step. The
loop has bounded steps, wall time, tokens, and estimated cost; it does not add
a verifier pass or a session runtime.

Data access is read first. All current data tools are non-side-effecting reads
except the explicitly scoped PostgreSQL update. SQL and local paths are
checked against the current catalog before source I/O, and every requested tool
call receives one ordered result even if another call fails.

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

The zero argument `daita` command is the normal path. Automation friendly
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
/source edit
/source permissions
/source detach <source>
/conversation clear
/agent delete
```

`/source edit` changes the active source connection without dropping the
working connection first. Daita validates and catalogs the edited connection,
shows a redacted Textual confirmation before one atomic handoff. Read
intent is preserved only for exact matching resources, PostgreSQL update scopes
are cleared, and a new conversation starts. If validation, discovery, review,
or commit fails, the existing connection remains active.

`/source permissions` is the single guided terminal flow for read access
(`all`, exact selected current resources, or `none`) and exact PostgreSQL
update access. Users can select one, many, or all current eligible
tables, use all eligible assignment columns by default, or choose an exact
subset through Advanced. One before/after summary and confirmation applies both
scope families atomically. Future tables are never automatically write-enabled,
and the flow never changes PostgreSQL roles or grants.

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

Local state has its own immutable, checksummed migration journal, independent
of the package version. On the first normal launch after an upgrade, Daita
validates a supported journal prefix under the existing per-agent writer lock
and automatically applies its known missing suffix to a verified staged copy.
It validates the complete result and atomically activates it only after every
check succeeds. No separate state command, import, restore, or backup step is
part of a routine upgrade, and existing agent homes should never be reset for a
normal application update.

The upgrade preserves identity, configuration and secret references, settings,
sources and catalogs, active-source selection, exact permissions, complete
conversations and results, artifacts, approved memory and user profile, skills,
semantics, learning state, and database-write receipts. Before activation Daita
retains the verified prior database as one bounded `state.db.rollback-*`
recovery point. Failure or cancellation before activation removes the staging
files and will leave the prior database unchanged.

Unknown, reordered, gapped, or checksum-mismatched journals, newer state from
an attempted package downgrade, recognizable pre-1.0 homes, and damaged schemas
fail closed without moving, replacing, emptying, or recreating the home. Install
the same or a newer Daita release for newer state; opening an upgraded home with
an older package is not a supported downgrade path. See the
[local state upgrade contract](docs/LOCAL_STATE_UPGRADES.md).

No manual backup is required for compatibility. A complete copy made while
Daita is closed is optional disaster recovery, not a compatibility mechanism:

```bash
cp -a ~/.daita ~/.daita-backup-before-upgrade
```

Secret values stored in the OS keychain are referenced by the copied
configuration but are not duplicated into the agent home.

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

The PostgreSQL update plan, least-privilege role requirements, exact-impact
preview, approval, transactional drift handling, receipts, and unknown-outcome
response are documented in
[PostgreSQL updates](docs/POSTGRESQL_UPDATES.md).

## License

[MIT](LICENSE)
