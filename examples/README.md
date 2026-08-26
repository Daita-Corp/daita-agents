# Daita MVP examples

These examples use only the package in `src` and the completed public
`Agent` API. Unless `--root` is provided, each walkthrough creates a fresh
temporary root and removes it on exit. A supplied root should be dedicated to
that example. The shared helper also admits a separate sibling workspace,
matching the required local `Agent.create(..., workspace=LocalWorkspace(...))`
contract.

The data walkthroughs exercise the MVP's read-only catalog and data tools with
a tiny scripted provider, so they are deterministic, offline, and free of
credentials. Production providers are configured through
`daita.create_llm_provider()` and secret providers or provider SDK environment
configuration; no example contains an API key.

The same public `Agent` also supports explicit conversation IDs with bounded
cold continuation, bounded `MEMORY.md`/`USER.md` context, bounded Markdown
skills loaded through `skill_view`, foreground approval-gated memory/skill
learning, and best-effort non-persisted events. These examples do not enable a
session runtime, external data writes, background learning, telemetry
collection, or live services.

Run a walkthrough from the repository root:

```bash
PYTHONPATH=src .venv/bin/python examples/00_quickstart_sqlite_from_db.py
PYTHONPATH=src .venv/bin/python examples/02_catalog_assisted_joins.py
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
