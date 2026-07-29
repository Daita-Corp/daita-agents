# Daita Agent

Daita is a persistent, read-only data agent for SQLite, PostgreSQL, CSV, and
JSON. Install the complete application with pipx, then launch the interactive
terminal:

```bash
pipx install daita-agents
daita
```

## Start with `daita`

Pipx keeps Daita in an isolated environment and exposes the `daita` command on
your `PATH`. The default package includes every currently supported model
provider, keychain, database driver, and SQL validation dependency; no extras
expression or separate virtual environment is required.

The first launch stays inside the terminal. It creates or selects an agent,
uses ↑/↓ and Enter to choose a provider and one of its curated model
suggestions, reads the API key through a hidden prompt, saves it in the OS
keychain, and explains before making one small exact-model validation request.
Every provider menu also offers manual model-ID entry through the same
validation boundary. Source choices use ↑/↓ and Enter. For PostgreSQL, Daita
tests the connection, lists non-system schemas, uses Space to toggle stable
schema names and Enter to confirm the nonempty selection, and catalogs only
that selection before chat begins. Escape cancels a selector without persisting
its terminal position or incomplete onboarding state.

No API key, database password, model flag, source flag, environment variable,
or conversation ID is required on the normal first-run path. Daita persists
only keychain references and non-secret configuration. It never accepts an API
key or database password as a command-line argument.

A ready first launch moves into a full-screen, transcript-first shell:

```text
 DAITA  atlas                                      Local PostgreSQL
───────────────────────────────────────────────────────────────────

 You
 Which region leads paid revenue?

 Daita
 EMEA leads paid revenue with $4.2M.

╭─────────────────────────────────────────────────────────────────╮
│ › Ask a follow-up…                                              │
╰─────────────────────────────────────────────────────────────────╯
 atlas · gpt-5.6-sol · ● ready       1 step · 824 tokens · $0.01
```

The shell uses the terminal's alternate screen, so responsive redraws do not
accumulate in native scrollback and the prior shell screen returns on exit.
The transcript has no embedded scrollbar and moves vertically with Page
Up/Page Down. The composer starts at one visible row and grows to as many as
six rows when keyboard-typed input wraps, shrinking again when text is deleted
or the terminal widens. A paste that would cross the current row boundary, or
that contains multiple lines, appears as `[Pasted Text #1]`, `[Pasted Text #2]`,
and so on rather than expanding the composer; Daita restores the exact bounded
contents when the message is submitted. Deleting or editing a placeholder
immediately purges its hidden contents from the current draft and recalled
composer history. The row boundary follows terminal resizes. Model answers
render as sanitized Markdown. Enter submits nonempty input, Ctrl-J inserts a
newline, Ctrl-C cancels an active run without closing Daita, and Ctrl-D exits
when the composer is empty. Existing slash commands continue to use their
current local prompts and selectors. Redirected and other non-interactive
streams retain the deterministic plain-text chat path.

Run `daita` again to reopen the sole agent automatically. When several agents
exist, Daita shows a deterministic picker; `daita --agent atlas` selects one
explicitly. Use `daita --root /absolute/path` to keep agent homes under a
non-default root. A returning launch retains the configured model and current
cataloged sources, skips their completed onboarding fields, derives `Ready`
from that current state, and starts a new terminal conversation unless the
customer explicitly uses `/resume <conversation-id>`.

### Upgrade, repair, or uninstall

Use pipx for the installed application lifecycle:

```bash
pipx upgrade daita-agents
pipx reinstall daita-agents
pipx uninstall daita-agents
```

Reinstall repairs missing or damaged application dependencies. Uninstall
removes the pipx-managed application environment; it does not delete Daita
agent homes or credentials already stored in the OS keychain.

### Troubleshooting onboarding

- “The API key was rejected” means the provider rejected the saved credential;
  replace it and retry. “This account cannot access `<model>`” means the exact
  model is unavailable to that account.
- Rate-limit, provider-unavailable, timeout, and invalid-model messages are
  normalized at the provider boundary. They never include raw SDK responses.
- A missing runtime dependency message means the isolated application
  environment needs repair; run `pipx reinstall daita-agents`. Other keychain
  messages indicate that the OS credential store could not save or read the
  secret; unlock or repair keychain access, then retry.
- For PostgreSQL connection failures, confirm that the database is already
  running and verify host, port, database, reader username, password, and SSL
  mode. Schema-inspection failures usually mean the reader role lacks catalog
  permissions. Daita does not start or manage PostgreSQL or Docker.
- Local SQLite and CSV/JSON paths must exist, be readable, resolve to absolute
  paths, and pass Daita's containment and symlink checks.

Setup failures leave the last committed model configuration and catalog intact.
Provider and database availability are checked only when an operation runs;
Daita does not persist a health or readiness claim.

## Architecture

Daita uses one direct model/tool loop:

```text
user message -> model -> zero or more tool calls -> tool results -> model -> answer
```

The exact current-run transcript is the loop state. Tool errors are ordinary
tool results, so the model can correct a call on its next step. The loop has
only outer limits for steps, time, tokens, and estimated cost. It does not
restore the deleted operation, session, monitor, extension, event-stream, or
telemetry lifecycles. Explicit conversation IDs add bounded multi-turn
continuity, including continuation after close and reopen, without changing
that direct progression or creating a session runtime. Bounded advisory memory
and bounded Markdown skills provide inspectable context; full skill instructions
are loaded only when the model calls `skill_view`.

SQLite and local-file sources are admitted through bounded adapters. SQL is
validated against the current catalog before execution. Every model-requested
tool call receives one ordered success or error result; one failing call does
not suppress the others. Data access remains read-only. The only built-in side
effects are local memory, semantic-annotation, and skill mutations proposed
during a foreground run and guarded by an in-process approve-once callback.

## Python API

```python
from daita import Agent, SQLiteSource, create_llm_provider
from daita.llm import ModelProfile

provider = create_llm_provider("openai:gpt-4.1-mini")
profile = ModelProfile(
    id=provider.provider_id,
    context_window_tokens=128_000,
    max_output_tokens=4_096,
    supports_tools=True,
    supports_parallel_tools=True,
)

agent = await Agent.create(
    "atlas",
    model=provider,
    model_profile=profile,
)
await agent.attach(SQLiteSource("/absolute/path/sales.db"))
result = await agent.run("What were the top five products by revenue?")
print(result.final_text)

follow_up = await agent.run(
    "Now only EMEA",
    conversation_id=result.conversation_id,
)
turns = await agent.conversation_runs(result.conversation_id)
print(follow_up.final_text, len(turns))
await agent.close()
```

Omitting `conversation_id` creates a new opaque conversation. Supplying the
returned ID appends an agent-scoped turn and projects a deterministic bounded
tail of completed prior runs: at most 8 runs, 40 messages, and 24,000 UTF-8
bytes, subject to the complete model-input budget and an 8,000-token current-run
growth reserve. Failed, interrupted, unfinished, or structurally incomplete
runs remain inspectable but are not replayed. Prior messages are never copied
into the new run's canonical transcript. There is no current-conversation
pointer, search, summary, compression checkpoint, or interrupted-run resume.

### Advisory memory

Each agent home may contain two fixed UTF-8 documents:

- `MEMORY.md` holds user-confirmed durable business semantics, terminology,
  conventions, and non-secret environment notes (at most 2,200 characters and
  8,800 UTF-8 bytes).
- `USER.md` holds durable user preferences and communication expectations (at
  most 1,375 characters and 5,500 UTF-8 bytes).

Use `await agent.read_memory()`, `await agent.set_memory(text)`,
`await agent.read_user_profile()`, and `await agent.set_user_profile(text)` for
explicit caller-controlled access. These methods do not invoke a model,
request approval, create a tool call, or emit an observation event.

The foreground model may propose a complete replacement through
`memory_set(target, content)`, where `target` is `memory` or `user`. The same
bounds and atomic file owner apply, but the exact frozen invocation executes
only after the configured in-process approval callback returns `approve`.
The callback sees the proposed content for semantic review; observation events
never include that content.

The documents are advisory context, not policy, evidence, approval,
authorization, capability configuration, catalog truth, or current source
truth. Current catalog structure and validated tool results remain
authoritative within their domains. Do not store secrets or credentials, raw
rows or copied query results, current source availability or freshness,
catalog revisions or schema snapshots, whole messages or tool results, or
approval or policy claims in either document.

Agent identity, attached-source registrations, current catalog snapshots, and
exact run transcripts are stored in one small SQLite database. There is no
schema migration or backward-compatibility layer in the unreleased MVP. Memory
content stays only in the two bounded files and is not copied into SQLite or
persisted transcripts merely because it was included in a model request.

### Procedural skills

Each skill is stored only at
`<agent-home>/skills/<skill-name>/SKILL.md`. Names match
`[a-z][a-z0-9-]{0,63}`. An agent may have at most 32 skills; descriptions are
single-line, already-trimmed text of at most 240 characters; instructions are
already-trimmed LF-only text of at most 12,000 characters. A rendered skill is
limited to 50,000 UTF-8 bytes. The complete deterministic name/description
index is limited to 4,000 characters and 16,000 UTF-8 bytes.

Use `await agent.list_skills()`, `await agent.read_skill(name)`,
`await agent.save_skill(name, description, instructions)`, and
`await agent.delete_skill(name)` for explicit caller-controlled access. Lists
contain only immutable name/description summaries; reads return a full
name/description/instructions record or `None`. Save returns whether rendered
bytes changed, and delete returns whether the named skill was present.

The default model context contains the complete shallow index, never every
instruction body. The fixed read-only `skill_view(name)` tool progressively
loads one full skill when requested. Skills are user-authorized procedural
guidance, not catalog evidence, capability declarations, policy, approval,
authorization, or permission to bypass validation. Current catalog structure,
validated tool results, runtime checks, and the existing governance and approval
boundaries remain authoritative. Foreground `skill_save` and `skill_delete`
calls cross the same approval branch as `memory_set`; direct API mutations are
already explicit caller actions and do not invoke the model or approval flow.

In the terminal, `/skills create` starts a guided flow for the name, one-line
description, and multiline instructions. Finish the instructions with `.` on
its own line, or enter `/cancel` at any step. `/skills create <name>` remains an
editor shortcut: Daita opens `$EDITOR` on the canonical `SKILL.md`, validates
the complete document, and saves it only when the name is new. An unchanged
template cancels editor creation; invalid content can be reopened without
losing the draft. Both forms are local caller actions, so neither enters the
conversation nor asks the model or approval handler to perform the write.

Each installed skill is exposed as `/<skill-name> [request]`. Use
`/skills use <name> [request]` when a skill name collides with a built-in
command; built-ins always win the short form. Invocation is an ordinary Daita
run, and the exact slash message is persisted as the user message. The model
loads the selected procedure through `skill_view` as its only first-step tool
call before continuing. A bare invocation loads the skill and asks what the
user wants to do. The terminal never injects a skill body into the transcript
or bypasses the existing trust, validation, or approval boundaries.

### Optional learning-candidate review

Candidate review is disabled by default. It runs only when a caller explicitly
invokes `await agent.review_learning_candidates()` or `/review` and
provides one-call or process-level cost authorization. Ordinary `Agent.run()`
calls never start, wait for, or depend on candidate review, and Daita has no
review scheduler, inactivity daemon, or retry worker.

One review makes at most one tool-free model request outside `AgentLoop`. It
uses one direct provider with no retry or fallback, receives bounded projections
of at most eight completed runs and 40 messages, and can return at most four
proposals. Projected transcript material is limited to 24,000 UTF-8 bytes; the
whole request is limited to 60 seconds and 24,000 model tokens. The reviewer has
no source clients, SQL tools, mutation tools, approval authority, or external
side effects. Paid routes require an explicit finite non-negative estimated-cost
ceiling. Invoking review can incur one provider charge; the ceiling is checked
against the returned usage estimate and is not a prepaid billing guarantee.

Each agent retains at most 64 candidate records in the state database.
Candidates are bounded, inactive review records with stable record IDs,
immutable review provenance, editable proposed content, exact supporting-run
references, and an effective status of `awaiting_review`, `accepted`,
`rejected`, or `obsolete`. They are untrusted and inactive: they do not enter
ordinary context, alter memory or skills, establish semantic or catalog truth,
select a source, expand a tool, or authorize a write.

Review history is a best-effort advisory projection. A bounded historical run
that cannot be decoded is preserved in storage, excluded from review input, and
reported in `skipped_run_count`; it cannot suppress compatible newer runs or be
misreported as a provider failure. If the bounded tail contains no readable
runs, review returns `history_unavailable` without making a model call. Daita
does not rewrite historical records or add a migration runtime for this
unreleased state format.

`/memory` displays the bounded inbox and status counts. The actions
`/memory list`, `/memory show <id>`, `/memory edit <id>`,
`/memory reject <id> [reason]`, and `/memory clear-rejected` operate on inactive
records. `/memory accept <id>` handles exactly one candidate by starting a fresh
ordinary foreground run. Only a matching mutation that passes current
validation and the existing exact approval prompt marks that candidate
accepted. Denial, model refusal, validation failure, changed catalog/artifact
state, or interruption before execution leaves active knowledge unchanged. If
cancellation arrives only after the exact mutation has definitely completed,
Daita finalizes the candidate as accepted before propagating cancellation so
active state and candidate status cannot split. There is no bulk acceptance.

In the zero-argument terminal, `/review` asks for one-time cost
authorization when review is disabled. Press Enter to use the displayed
default, enter a different finite non-negative USD amount, or enter `/cancel`.
The equivalent expert form is `/review 0.02`. Authorization applies
only to that review and is not persisted. Review can make at most one model
call and only adds inactive suggestions to the inbox; memory and skills still
change only after explicit acceptance. The ceiling is checked against the
model's reported estimate after the response, so provider charges can still
apply when a result is rejected for exceeding it.

To preauthorize manually triggered reviews for one terminal process and skip
that prompt, set a cost ceiling before launch:

```bash
DAITA_CANDIDATE_REVIEW_MAX_COST_USD=0.05 daita
```

The terminal derives a bounded, direct, no-retry reviewer from the primary
persisted model route. Candidate review is disabled by default, and Daita never
runs it periodically or in the background. The public
`Agent.review_learning_candidates(max_estimated_cost_usd=...)` method provides
the same one-call authorization. `Agent.create()` and `Agent.open()` also offer
host-controlled `reviewer_model`, `reviewer_profile`, and
`reviewer_max_estimated_cost_usd` arguments; the reviewer profile must be
explicitly bounded below the 24,000-token total review budget.

### Approval and observation

```python
from daita import ApprovalDecision, ApprovalRequest

async def approve(request: ApprovalRequest) -> ApprovalDecision:
    # Inspect request.tool_name, request.capability_id, and the exact frozen
    # request.arguments. The caller remains responsible for semantic review.
    return ApprovalDecision.APPROVE

agent = await Agent.create(
    "atlas",
    model=provider,
    model_profile=profile,
    approval_handler=approve,
    observer=events.append,
)
```

Approval is approve-once and in-process. Missing handlers, denial, callback
failure, invalid decisions, and state changes become ordinary model-visible
tool errors. No approval or pending invocation is persisted, and there is no
approve-later or resume API. Read-only tool groups may execute concurrently;
the local memory side effect is a sequential barrier. After approval, the
runtime reacquires the same mutation lock used by direct memory and skill
writes, revalidates current state, and rejects stale approval with
`state_changed`. Cancellation before mutation never writes; after an atomic
replacement starts, Daita waits for a definite outcome before propagating
cancellation.

Active learning is only this visible foreground interaction: the model
proposes `memory_set`, `semantic_save`, `semantic_delete`, `skill_save`, or
`skill_delete`; the caller reviews the exact frozen arguments; the existing
tool runtime returns the approved mutation or error as a normal tool result;
and the model continues. Daita never performs implicit post-run review or
background learning. The optional candidate reviewer described above is an
explicit, synchronous, one-request operation that can create only inactive
review material.

Semantic maintenance is derived without mutation whenever current knowledge is
read or recalled. Missing resources or fields, revision mismatches, conflicts,
supersession, and exact normalized duplicates remain inspectable. Ordinary
recall excludes stale and conflicting statements, collapses each exact
duplicate group to its lowest-ID representative, and requires the complete
scope of multi-resource meaning to be selected. Related requests receive only
bounded review notices with annotation IDs, resource IDs, and reasons. The
ordinary foreground model can inspect the affected record and propose an exact
digest-protected correction through the same approval boundary.

One optional synchronous observer receives bounded `run.started`,
`model.completed`, `tool.started`, `approval.requested`, `approval.decided`,
`tool.completed`, and `run.completed` events. Events contain identifiers,
outcomes, durations, counts, and usage—not prompts, memory content, skill
instructions, tool arguments, results, or secrets. Observer exceptions are
ignored; events are not persisted and are not a durable audit log. Callers
should enqueue and return promptly if they need external telemetry work.
This callback data is the entire event and telemetry surface: Daita does not
persist events, collect telemetry, upload analytics, or provide an event bus,
trace store, exporter, or delivery guarantee.

`daita.evaluation` provides a caller-owned, in-memory benchmark and reporting
harness. Callers supply human labels plus bounded observer-derived counters;
the deterministic report makes baseline-versus-learned correctness, safety,
and efficiency differences explicit. Reports contain no prompts, rows, tool
arguments, semantic statements, skill bodies, credentials, or secrets, and
stored artifact counts are not treated as evidence of improvement.

Learning evaluation has two deliberately separate gates. The required offline
gate uses scripted model responses but executes the real agent loop, SQLite
adapter, approval path, semantic persistence, close/reopen lifecycle, automatic
recall, denial, observer measurement, and report rendering:

```bash
.venv/bin/python -m pytest tests/test_learning_evaluation_phase3.py -v
```

The optional live confidence gate uses the disposable PostgreSQL fixture and
one explicitly selected release-reviewed model. It is excluded from default
tests, requires both `requires_llm` and `requires_db`, enforces an explicit
per-run cost ceiling, and writes only content-free JSON and Markdown reports to
the caller-selected output directory. See
[`tests/fixtures/postgresql/README.md`](tests/fixtures/postgresql/README.md) for
the complete authorized command.

An explicitly injected custom tool runtime remains caller-owned code and owns
authorization, tool-level observation, and side-effect safety for its tools.

## Advanced/headless CLI

The commands below remain available for automation, scripting, and debugging.
They are separate from the zero-argument onboarding flow. `run` emits one JSON
record; other non-interactive commands retain their existing JSON contracts.
Provider credentials and PostgreSQL passwords must be supplied through their
documented environment/keychain references, never as secret command-line
values.

```bash
daita --root /private/tmp/daita create atlas
daita --root /private/tmp/daita attach atlas sqlite /absolute/path/sales.db
daita --root /private/tmp/daita run atlas "Summarize sales" \
  --model openai:gpt-4.1-mini
DAITA_CANDIDATE_REVIEW_MAX_COST_USD=0.05 \
  daita --root /private/tmp/daita chat atlas \
  --model openai:gpt-4.1-mini
```

For a developer-owned disposable PostgreSQL source with a stable commerce
schema and freshly randomized sample values on each initialization, start the
repository fixture separately and attach its read-only role:

```bash
docker compose -f tests/fixtures/postgresql/compose.yaml up -d --wait
export DAITA_FIXTURE_POSTGRES_PASSWORD=daita_fixture_password

daita --root /private/tmp/daita attach atlas postgresql \
  --host 127.0.0.1 --port 55432 \
  --database daita_fixture --username daita_reader \
  --password-env DAITA_FIXTURE_POSTGRES_PASSWORD \
  --schema analytics --ssl-mode disable
```

The fixture is ephemeral and can be discarded with `docker compose -f
tests/fixtures/postgresql/compose.yaml down`. See
[`tests/fixtures/postgresql/README.md`](tests/fixtures/postgresql/README.md) for
the complete live CLI and opt-in acceptance-test workflow.

`run` remains the one-record JSON automation surface. `chat` is the readable
interactive surface and keeps only the explicit current conversation ID and
current-process usage totals. Its local `/help`, `/status`, `/conversation`,
`/new`, `/resume <conversation-id>`, `/sources`, and `/exit` commands never
enter the model transcript.

In the zero-argument terminal, `/source` opens the active-source picker and
`/source use <name>` selects directly. The selection is persisted and pinned in
the status line. Prefix one question with a display-name alias such as
`@postgres-large` to override that source without changing the conversation
selection.

Memory and user-profile documents can be read, replaced from a complete UTF-8
file or stdin, or edited with `$EDITOR`:

```bash
daita --root /private/tmp/daita memory read atlas
daita --root /private/tmp/daita memory inspect atlas
daita --root /private/tmp/daita memory read atlas --target user
daita --root /private/tmp/daita memory set atlas --target memory --file notes.md
printf '%s' 'Prefer concise answers.' | \
  daita --root /private/tmp/daita memory set atlas --target user --file -
EDITOR='code --wait' daita --root /private/tmp/daita memory edit atlas
```

Omitting `--target` from `memory read` or `memory edit` selects `memory`.
`memory set` requires an explicit `memory` or `user` target. A `-` input reads
stdin through EOF. File and stdin content is passed in full to the public
`Agent` method: the CLI does not truncate, normalize, or bypass the existing
bounds and validation.

Candidate review and the inactive inbox use the same `memory` namespace:

```bash
daita --root /private/tmp/daita memory review atlas \
  --model openai:gpt-4.1-mini \
  --cost-limit 0.05
daita --root /private/tmp/daita memory list-candidates atlas \
  --status awaiting_review
daita --root /private/tmp/daita memory show-candidate atlas <candidate-id>
EDITOR='code --wait' \
  daita --root /private/tmp/daita memory edit-candidate atlas <candidate-id>
daita --root /private/tmp/daita memory accept-candidate atlas <candidate-id> \
  --model openai:gpt-4.1-mini
daita --root /private/tmp/daita memory reject-candidate atlas <candidate-id> \
  --reason user_declined
daita --root /private/tmp/daita memory clear-rejected atlas
```

`memory review` is the only command above that invokes the auxiliary reviewer;
it always requires both an explicit direct model and an estimated-cost ceiling.
Listing, showing, editing, rejecting, and clearing candidates are bounded local
operations. `accept-candidate` handles one candidate through a fresh foreground
model run and the normal exact approval prompt; it is not approval of a stored
tool invocation.

Skills use the same public API:

```bash
daita --root /private/tmp/daita skills list atlas
daita --root /private/tmp/daita skills show atlas monthly-revenue
daita --root /private/tmp/daita skills save atlas monthly-revenue \
  --description 'Monthly revenue procedure.' \
  --instructions-file monthly-revenue.md
daita --root /private/tmp/daita skills edit atlas monthly-revenue
daita --root /private/tmp/daita skills delete atlas monthly-revenue
```

`$EDITOR` is parsed as a command plus arguments without a shell. Edit commands
use a private temporary file outside the agent home and save only after the
editor exits successfully and public validation passes. Skill editing uses this
exact deterministic document representation, with the command-supplied name
authoritative:

```markdown
# monthly-revenue

Monthly revenue procedure.

## Instructions

Group paid invoices by UTC month.
```

`memory inspect` returns bounded global-memory and semantic-maintenance state,
including stale, conflicting, duplicate, superseded, and revalidation fields.

Interactive chat additionally provides `/review`, `/memory`, `/memory edit`,
`/memory list`, `/memory show <id>`,
`/memory edit <candidate-id>`, `/memory accept <candidate-id>`,
`/memory reject <candidate-id> [reason]`, `/memory clear-rejected`, `/user`,
`/user edit`, `/skills`, `/skills show <name>`, `/skills create [name]`,
`/skills edit <name>`, `/skills delete <name>`,
`/skills use <name> [request]`, and `/<skill-name> [request]`. Skill deletion
asks for explicit confirmation and defaults to no. These direct caller actions
invoke neither the model nor an approval callback except for explicit
`/review` and individual candidate acceptance. Skill invocations do enter the
model transcript as ordinary user requests. Model-requested memory and skill
side effects still use the exact, once-only, in-process approval prompt in
`chat`; `run` installs no approval handler.

## Development

```bash
cd /path/to/daita-agents
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python tests/pipx_lifecycle_smoke.py
```

Production integration dependencies are installed by default but imported
lazily at their execution boundaries. The release smoke builds a local wheel
and sdist, uses temporary explicit pipx home/bin/man/cache directories,
installs and reinstalls the local wheel, runs installed metadata and
`daita --help` checks outside the checkout, uninstalls it, and proves separately
located agent-home data remains. It may download declared dependencies from the
configured package index, so run it only where that read access is authorized;
it does not upload artifacts or change an index.
