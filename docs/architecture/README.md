# Architecture notes

Detailed notes on how Daita is built, for contributors and coding agents. Start
with the root [`AGENTS.md`](../../AGENTS.md): it has the system map, the
invariants and the recipes. These files hold the detail behind them.

| File | Covers |
| --- | --- |
| [`runtime.md`](runtime.md) | Composition, the model loop, context building, budgets, source scope, capability execution, approval and observation |
| [`data-and-files.md`](data-and-files.md) | Source types, catalog and declared links, SQL validation, combined queries, relational writes, folders and artifacts |
| [`mcp.md`](mcp.md) | Remote MCP bindings, calls and actions |
| [`jobs-and-routines.md`](jobs-and-routines.md) | Job graph boundaries, job supervision, routines, scheduled-run limits, outcomes and deliveries |
| [`persistence.md`](persistence.md) | State boundary, home revisions, migrations, upgrades and the release contract |
| [`models.md`](models.md) | Canonical model records, routing and retries, call policy, provider lifecycle |
| [`product-surfaces.md`](product-surfaces.md) | CLI and TUI rules |

When code and a note disagree, the code wins: fix the note in the same change.
