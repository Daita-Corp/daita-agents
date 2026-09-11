# Offline examples

These examples use the package in `src` and the current public `Agent` API.
Unless `--root` is provided, each walkthrough creates a fresh
temporary root and removes it on exit. A supplied root should be dedicated to
that example. The shared helper also admits a separate sibling workspace,
matching the required local `Agent.create(..., workspace=LocalWorkspace(...))`
contract.

The data walkthroughs exercise the read-only catalog and data tools with a tiny
scripted provider, so they are deterministic, offline, and free of
credentials. Production providers are configured through
`daita.create_llm_provider()` and secret providers or provider SDK environment
configuration; no example contains an API key.

The public `Agent` also supports explicit conversation IDs with bounded
cold continuation, bounded `MEMORY.md`/`USER.md` context, bounded Markdown
skills loaded through `skill_view`, foreground approval-gated memory/skill
learning, and best-effort non-persisted events. These walkthroughs stay offline
and do not perform external data writes or live model calls.

Run a walkthrough from the repository root:

```bash
PYTHONPATH=src .venv/bin/python examples/00_quickstart_sqlite_from_db.py
PYTHONPATH=src .venv/bin/python examples/02_catalog_assisted_joins.py
PYTHONPATH=src .venv/bin/python examples/03_offline_assignments_and_recovery.py
PYTHONPATH=src .venv/bin/python examples/10_csv_to_sqlite_data_app.py
```

The same offline state can be inspected and managed through the public-API CLI:

```bash
PYTHONPATH=src .venv/bin/python -m daita.cli --root /private/tmp/daita \
  memory set atlas --target memory --file confirmed-semantics.md
PYTHONPATH=src .venv/bin/python -m daita.cli --root /private/tmp/daita \
  skills save atlas monthly-revenue \
  --description 'Monthly revenue procedure.' \
  --instructions-file monthly-revenue.md
PYTHONPATH=src .venv/bin/python -m daita.cli --root /private/tmp/daita \
  skills show atlas monthly-revenue
```

Use `--file -` or `--instructions-file -` to read complete UTF-8 content from
stdin. `memory read` and `memory edit` default to the `memory` target; pass
`--target user` for the user profile. The edit commands require an available
`$EDITOR`, accept editor arguments safely, and write only after a successful
exit and public validation. In the interactive Textual app, the bounded
`/memory`, `/user`, and `/skills` commands are local and make no model call.

These CLI writes are explicit caller mutations, so they require no model
approval. A memory or skill change requested by the model during an interactive
run remains on the exact once-only in-process approval path. The non-interactive
`run` command never installs an approval handler.

The assignment walkthrough uses `httpx.MockTransport`, a scripted model with
explicit fictional usage, and a fresh temporary agent home. It approves one
immediate and weekly research/briefing/notification assignment, closes and reopens
the host, simulates a lost action response, and records human recovery without
another dispatch. Its automatic approval callback is for simulated I/O only;
use the TUI or an exact human approval handler for real actions. No live service,
credentials, model call, or external database is used.

Native company research/upsert is exercised offline by
`tests/data/writes/test_public_writes.py`, with the real data/runtime/scheduler owners
and deterministic PostgreSQL I/O. The guided permission acceptance is in
`tests/acceptance/test_product_workflows.py`. See [relational write authoring](../docs/RELATIONAL_WRITES.md)
for the runnable public API pattern against a separately admitted target.

A reusable research procedure can follow [the procedure example](research_and_store/SKILL.md).
Importing that Markdown never supplies a connector or write permission. Scheduled
assignments retain exact skill content; adopting an edit requires an approved
routine revision.
