# Local operations

## Install and initialize

Install only the extras needed by the selected providers and sources. For
example, a SQLite/OpenAI environment uses the `sqlite` and `openai` extras.

```bash
python -m pip install "daita-agents[sqlite,openai]"
daita --root /srv/daita-v2 agent create atlas --idempotency-key create-atlas-v1
daita --root /srv/daita-v2 model set atlas --provider openai --model gpt-4.1-mini \
  --secret env:OPENAI_API_KEY --idempotency-key model-atlas-v1
daita --root /srv/daita-v2 model show atlas
```

The root must be private to the operating-system user. Keep it separate from
v1 state and from attached source files.

P9.5-08 passed this retained configuration and source lifecycle from a clean
wheel on CPython 3.11 and 3.12. Each environment started a real local host,
chatted and followed committed events, stopped cleanly, cold-reopened without
model reinjection, ran again, uninstalled the package, and retained Agent Home.

## Run a foreground host

```bash
daita --root /srv/daita-v2 serve atlas --cadence-seconds 1
```

One host owns the agent writer lock, startup recovery, monitor scheduling,
durable inbox, and event wakeups. Run it under a process supervisor; v2 does
not hide a daemon or background thread inside `Agent.create/open`.

The first create/open binds the agent's future-operation budgets and default
governance profile in `state.db`. Later opens may omit them and load the exact
binding; supplying different values fails closed. A programmatically configured
model route is append-only/versioned, contains `SecretReference` values rather
than resolved secrets, binds each new operation, and reconstructs lazily for a
model-free cold `Agent.open()` or host start. `model set` changes only a stopped
host's future-operation route; `model show/status` expose non-secret retained
configuration and readiness.

Programmatic configured extensions must be supplied explicitly and identically
on every reopen. Agent Home binds their ordered IDs, versions, kinds, and
declaration/manifest fingerprints; omission, order/version/declaration drift,
or collision with a built-in fails closed. Only capability-provider manifests
execute in the MVP. Package scanning, auto-install, hot reload, resource-
adapter/backend-provider extension kinds, and alternate execution paths are
not supported.

Use `daita host health atlas` for liveness and `daita host status atlas` for
bounded runtime status. Mutating CLI requests require an idempotency key.

## Learning review

Successful user operations accept ordinary direct statements such as
`Remember that fiscal weeks start on Monday.` through the governed memory
lifecycle. Alias-shaped statements such as `Remember that completed status is
stored as complete.` require one accepted current read resource binding in that
operation and become stale when the bound catalog revision changes. The exact
canonical resource-alias message remains supported for programmatic callers.

`Propose a fact from the evidence: ...` creates a visible proposal only when
the operation contains accepted applicable current read evidence. It does not
silently commit memory. `Propose a skill named reconcile-status: ...` likewise
creates an inert version proposal. Python callers inspect these with
`Agent.list_learning_proposals()` and must call `Agent.accept_skill_change()`
before the skill can become active. Rejected sensitive or executable candidates
retain no candidate payload. The CLI supports bounded memory and skill
list/inspection. Skill acceptance remains the explicit audited Python API
operation; inspection never activates a proposal.

## Backup and recovery

- Stop the host or use an application-consistent snapshot mechanism.
- Back up the whole agent directory, including `state.db`, WAL-related files
  when present, `agent.toml`, and blobs.
- Restore into a private path with the same authoritative agent identity.
- Reopen normally; startup recovery resumes persisted nonterminal operations,
  skips terminal tasks, and fails unknown side-effect outcomes closed.

Do not edit SQLite rows, manifests, blobs, leases, approval fingerprints, or
event cursors manually. Inspect the operation and resolve manual-recovery work
through an explicit operator decision.

## Stop and uninstall

Stop the foreground host cleanly before upgrading or uninstalling. Package
uninstallation leaves the selected state root intact. Retain or remove that
root according to the operator's data-retention policy, as a separate action.
