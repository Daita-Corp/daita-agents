# Local operations

## Install and initialize

Install only the extras needed by the selected providers and sources. For
example, a SQLite/OpenAI environment uses the `sqlite` and `openai` extras.

```bash
python -m pip install "daita-agents[sqlite,openai]"
daita --root /srv/daita-v2 agent init atlas --idempotency-key init-atlas-v1
```

The root must be private to the operating-system user. Keep it separate from
v1 state and from attached source files.

## Run a foreground host

```bash
daita --root /srv/daita-v2 serve atlas \
  --openai-model gpt-4.1-mini \
  --cadence-seconds 1
```

One host owns the agent writer lock, startup recovery, monitor scheduling,
durable inbox, and event wakeups. Run it under a process supervisor; v2 does
not hide a daemon or background thread inside `Agent.create/open`.

The first create/open binds the agent's future-operation budgets and default
governance profile in `state.db`. Later opens may omit them and load the exact
binding; supplying different values fails closed. The MVP has no in-place
configuration mutation command.

Use `daita host health atlas` for liveness and `daita host status atlas` for
bounded runtime status. Mutating CLI requests require an idempotency key.

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
