# Daita MVP

Daita is a persistent, read-only data agent built around one direct loop:

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

The MVP supports catalog-backed reads from SQLite, PostgreSQL, CSV, and JSON.
SQLite and local-file sources are admitted through bounded adapters. SQL is
validated against the current catalog before execution. Every model-requested
tool call receives one ordered success or error result; one failing call does
not suppress the others. Data access remains read-only. The only built-in side
effects are local, file-backed memory and skill replacements proposed during a
foreground run and guarded by an in-process approve-once callback.

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

Learning is only this visible foreground interaction: the model proposes
`memory_set`, `skill_save`, or `skill_delete`; the caller reviews the exact
frozen arguments; the existing tool runtime returns the approved mutation or
error as a normal tool result; and the model continues. Daita performs no
post-run review, auxiliary model call, curator, or background learning.

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

An explicitly injected custom tool runtime remains caller-owned code and owns
authorization, tool-level observation, and side-effect safety for its tools.

## CLI

```bash
daita --root /private/tmp/daita create atlas
daita --root /private/tmp/daita attach atlas sqlite /absolute/path/sales.db
daita --root /private/tmp/daita run atlas "Summarize sales" \
  --model openai:gpt-4.1-mini
daita --root /private/tmp/daita chat atlas \
  --model openai:gpt-4.1-mini
```

`run` remains the one-record JSON automation surface. `chat` is the readable
interactive surface and keeps only the explicit current conversation ID and
current-process usage totals. Its local `/help`, `/status`, `/conversation`,
`/new`, `/resume <conversation-id>`, `/sources`, and `/exit` commands never
enter the model transcript.

Memory and user-profile documents can be read, replaced from a complete UTF-8
file or stdin, or edited with `$EDITOR`:

```bash
daita --root /private/tmp/daita memory read atlas
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

Interactive chat additionally provides `/memory`, `/memory edit`, `/user`,
`/user edit`, `/skills`, `/skills show <name>`, `/skills edit <name>`, and
`/skills delete <name>`. Skill deletion asks for explicit confirmation and
defaults to no. These direct caller actions invoke neither the model nor an
approval callback. Model-requested memory and skill side effects still use the
exact, once-only, in-process approval prompt in `chat`; `run` installs no
approval handler.

## Development

```bash
cd /path/to/daita-agents
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,sqlite]"
.venv/bin/python -m pytest
```

Optional provider and database SDKs are imported lazily at their execution
boundaries.
