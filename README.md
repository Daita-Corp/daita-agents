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
| **Talk to real data** | Query SQLite and PostgreSQL without writing SQL, and search, read, or safely approve targeted text edits in one admitted workspace. |
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

The first launch admits one local workspace, then guides you through creating
an agent, choosing a model, and optionally attaching a read-only source inside
our Textual application; setup,
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

The Files toolbox is independent of attached Sources. Use `/files <question>`
for a files-only turn, `/workspace` to inspect the admitted root, and ordinary
questions when Daita may use both workspace files and the selected source.
Workspace file names and content are always treated as untrusted data.

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

Data access is read first. Capability metadata records data access separately
from operational effect: ordinary queries read data without an operational
effect, the durable-profile start reads current catalog scope and starts one
job, and the explicitly scoped PostgreSQL update is the only current data
mutation. SQL is checked against the current catalog before source I/O.
Workspace paths are separately validated through descriptor-relative
containment, with no symlink following, traversal, secret-file reads, or
special-file reads. Every requested tool call receives one ordered result even
if another call fails.

Agent identity, source registrations, catalog snapshots, transcripts, and
terminal results live in a small SQLite database inside the agent home.
Conversation continuity projects a bounded tail of completed runs: at most 8
runs, 40 messages, and 24,000 UTF-8 bytes.

The `start_data_profile` capability can freeze one exact read-only profile job
over current admitted resources. Its `JobRun` is persisted before background
work, continues under the currently open agent host after the originating
model run finishes, and produces a bounded result plus one verified JSON
artifact. `Agent.list_jobs`, `inspect_job`, `read_job_result`, and `cancel_job`
provide bounded lifecycle access. The model-facing lifecycle tools use the same
agent ownership boundary, so a new conversation can list the agent's jobs,
inspect one, read its result, or cancel it. The originating conversation remains
visible as provenance rather than acting as an access gate. Work pauses while no
agent host is open and safe stale attempts are fenced on reopen. Daita does not
self-install or silently fork a daemon; the explicit resident host described
below is the only way to keep an agent open after the foreground process exits.

D1 scheduled routines freeze one self-contained foreground-authorized read
instruction, canonical one-shot/interval/calendar schedule, exact source,
resource, connector and capability ceilings, budgets, expiration, inbox
destination, and optional skill content digests. Every due slot enters the same
`AgentLoop` and `CapabilityRuntime` used by foreground work. `/routines` shows
record-owned lifecycle truth and supports pause, resume, run-now, and disable.
An optional exact resource-revision check can advance an unchanged occurrence
without a model call. Scheduled execution is read-only: it cannot update data,
start or cancel a job, manage another routine, deliver externally, or submit a
graph. Repeated pre-run or execution failures move the routine to
`needs_attention` at its configured threshold and create one conversation-inbox
escalation; a pre-run escalation truthfully records that no model run started.
See [Scheduled read routines](docs/SCHEDULED_ROUTINES.md).

The external-executor contract currently has deterministic offline conformance
coverage only. No real external job profile ships, and no connected service is
selected automatically or used as a fallback.

Explicitly admitted remote MCP servers can contribute read-only tools through
the same registry, runtime, ordered result, transcript, and context path.
Bindings use remote Streamable HTTP, exact allowlists, secret references,
call-time identity/schema rechecks, and per-binding revocation; remote metadata
never authorizes or instructs the agent. Agent open is network-free: accepted
metadata composes immutable declarations locally, while clients initialize and
recheck exact remote identity only at an admitted call. Large surfaces use a
bounded per-run catalog with pinned definitions and transcript-bound
`toolbox_search` → atomic `toolbox_load` → ordinary tool invocation. In the
TUI, `/mcp` opens a grouped
server manager and `/mcp add` guides endpoint inspection, multi-tool selection,
read-only attestation, and controlled activation. See
[Remote MCP read connectivity](docs/MCP_CONNECTIVITY.md).

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

File requests use the same direct loop. The pinned `file_search` and
`file_read` tools expose only bounded workspace-relative results and never add
a Files-domain writer. The on-demand `file_query` tool filters or aggregates
one homogeneous CSV, TSV, JSON-records/NDJSON, or Parquet dataset through a
private one-call DuckDB worker. Daita expands and revision-binds the relative
pattern itself; validated SQL can see only the relation `data`, and every
result retains the complete exact input manifest. For bounded UTF-8 text, a
current-run `file_read`
binding can feed `artifact_edit_text`, which commits a complete replacement
artifact without changing the workspace. `artifact_save_local` then requests
one approval and atomically replaces only that exact unchanged bound file;
drift requires a fresh read and never triggers a silent merge or retry. Exact
SQL results can become CSV or XLSX artifacts. A later turn can use a bounded
model-only `artifact_list` for the current conversation, `artifact_read` for a
bounded preview of an exact known ID owned by the agent, and `artifact_convert`
for the supported current-conversation Daita XLSX `Data` snapshot to CSV
conversion. This lets a new conversation read an exact artifact reference
returned by a durable job without creating an agent-wide inventory. There is
no `file_write`, public artifact inventory, CLI list command, hidden
current-file pointer, or prompt keyword router.

Local files are normally delivered automatically to the authorized default
destination and reported with the verified saved path. Public recovery remains
known-ID only through `Agent.read_artifact`, `Agent.save_artifact`, and
`daita artifacts save`. Clearing conversations invalidates the corresponding
internal artifact IDs but never deletes copies already delivered to user-owned
directories.

For the complete implementation boundaries, see [AGENTS.md](AGENTS.md).
For workspace selection and read guarantees, see
[Local workspaces](docs/LOCAL_WORKSPACES.md).

## Advanced/headless CLI

The zero argument `daita` command is the normal path. Automation friendly
commands use the same public API:

```bash
daita --root /private/tmp/daita --workspace /absolute/path/project create atlas
daita --root /private/tmp/daita --workspace /absolute/path/project attach atlas sqlite /absolute/path/sales.db
daita --root /private/tmp/daita --workspace /absolute/path/project run atlas "Summarize sales" \
  --model openai:gpt-4.1-mini
daita --root /private/tmp/daita --workspace /absolute/path/project run atlas "Summarize the notes" --files-only
daita --root /private/tmp/daita --workspace /absolute/path/project host --agent atlas
```

`run` writes one JSON record. Provider credentials and PostgreSQL passwords
must come from their documented environment or keychain references, never
secret command line values.

Discover all commands and options with:

```bash
daita --help
daita memory --help
daita skills --help
daita routines --help
```

## Manage jobs and local data

The terminal provides confirmed lifecycle commands:

```text
/jobs
/jobs inspect <id>
/jobs results <id>
/jobs cancel <id>
/routines
/routines create <self-contained instruction>
/routines promote <basis-run-id> <self-contained instruction>
/source edit
/source permissions
/source detach <source>
/conversation clear
/agent delete
```

`/jobs` opens the bounded durable-job manager. It lists up to 50 current jobs
owned by the agent and provides direct refresh, lifecycle details, validated
results, exact artifact references, and confirmed cancellation without using a
model turn. The inspect, results, and cancel forms are equivalent power-user
commands for a known job ID. Jobs still run only while that agent host is open;
the manager does not add another daemon, scheduler, retry path, or generic job
starter.

`/routines` opens the bounded routine manager. Direct headless lifecycle
operations are available under `daita routines` for create/promote from an
exact JSON specification, list, inspect, update, pause, resume, run-now, and
disable. To continue due work after closing the TUI, run `daita host --agent
atlas` with the same `--root` and `--workspace`. The resident process owns the
ordinary agent-home writer lock, so it must be stopped before another TUI or
CLI process opens that agent. No schedule progresses while no admitted host is
open.

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

The public async API requires an explicit `LocalWorkspace` when creating or
opening a local agent. It also supports attaching sources,
administering explicit remote MCP read bindings, running questions, continuing
conversations, and inspecting transcripts.
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

The first production state format has not been frozen. During active North Star
development, local agent homes have no backward-compatibility guarantee and may
need to be recreated after a state-shape change. Daita keeps one current schema,
one current shape per persisted record family, and one mutable checksummed
development baseline; it does not migrate between unreleased formats.

At the explicitly approved first production release, that complete state shape
will become the first immutable baseline. Subsequent package upgrades will use
the existing staged-copy migration engine automatically on first open. See
[local state during development](docs/LOCAL_STATE_UPGRADES.md).

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
