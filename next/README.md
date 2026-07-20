# Daita autonomous agent v2 replacement

This is the isolated replacement project for Daita 2.0. Phase 9 produced a
functional, hardened persistent local-agent substrate: the generic loop,
governed execution, SQLite recovery, catalog-backed SQLite and sandboxed-file
data paths, provenance-bearing context, session compression, scoped
memory/learning, versioned procedural skills, durable monitors, a foreground
local host, controlled approved SQLite writes, PostgreSQL reads, retained-model
routing, packaging/lifecycle, live integrations, restart reliability, and
security hardening are implemented. `daita-agents` owns the Python API and the
one installed `daita` command; the separate `daita-cli` and `daita-client`
packages remain legacy Daita 1.x products and are excluded from 2.0.

A post-gate plan-to-source audit opened mandatory Phase 9.5 work to join those
components into the supported default product: exact persisted read authority,
reconstructable model routing, operational default monitor outcomes, ordinary
learning ingress, additive extensions, and the complete in-package CLI/host
journey. That joined gate now passes. The candidate is replacement-ready and
eligible for human Phase 10 review, but cutover remains a separately authorized
Phase 10 decision.

The governing plan is the local
`../docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md` fingerprinted in
`STATUS.md`. Root `../daita/` is a frozen behavioral oracle, not a
dependency. Phase 10 cutover is explicitly outside the authorized work.

## Architectural promise

The replacement is built around:

- one persistent agent identity;
- one provider-neutral generic agent loop;
- one governed operation runtime as the only executor invocation boundary;
- one canonical resource catalog;
- provenance-backed memory and versioned procedural skills;
- monitors that create ordinary triggers and operations;
- authoritative current state plus append-only committed runtime events; and
- narrow adapters/providers that cannot introduce alternate agent frameworks.

The model chooses semantics. Deterministic runtime code validates authority,
scope, policy, persistence, execution, evidence, readiness facts, and recovery.

Candidate operations, migration, support, security, CLI/client, and breaking-
change guidance is indexed in the source-tree `docs/README.md`. Those detailed
project records are intentionally excluded from the runtime distribution.

## Isolation rules

- V2 source lives at `src/daita/` and already uses the final `daita` import
  name.
- V2 production code never imports or executes root v1 production code.
- Do not install the v1 and v2 editable distributions into one interpreter.
- Compare v1 and v2 only through subprocess results or neutral serialized
  fixtures.
- Run v2 tests from this directory in an isolated environment.
- V2 defaults to `~/.daita-next/` until an explicitly approved cutover.

The import-firewall tests enforce package origin, source-reference, symlink,
and root-distribution boundaries.

## Development

Create an environment that does not contain the root editable distribution:

```bash
cd next
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -m "not requires_llm and not requires_db"
```

Python 3.11 and 3.12 are the tested candidate versions. Optional source and
provider SDKs will be added only to matching extras and imported lazily.

Do not co-install legacy `daita-cli` with Daita 2.0: both distributions claim
the `daita` console script. Upgrade guidance is included in the candidate's
CLI/client and breaking-change documentation.

## Local development host

The CLI is intentionally thin. Except for initial bootstrap, mutations travel
over the private per-agent Unix socket to the foreground `AgentHost`, which
remains the only local writer. Configure the retained route once, then start a
model-free host with an isolated state root:

```bash
daita --root /tmp/daita-next agent create atlas --idempotency-key create-atlas
daita --root /tmp/daita-next model set atlas --provider openai \
  --model gpt-4.1-mini --secret env:OPENAI_API_KEY \
  --idempotency-key model-atlas-v1
daita --root /tmp/daita-next serve atlas
```

`serve` resolves `OPENAI_API_KEY` through the provider SDK at request time; the
secret is not stored in the agent home. In another terminal, attach a source,
inspect the model/host, submit work, and inspect its durable operation:

```bash
daita --root /tmp/daita-next source add atlas sqlite /absolute/path/sales.db \
  --idempotency-key attach-sales-v1
daita --root /tmp/daita-next model show atlas
daita --root /tmp/daita-next model status atlas
daita --root /tmp/daita-next host health atlas
daita --root /tmp/daita-next chat atlas --session-id sales-chat
daita --root /tmp/daita-next operation inspect atlas <operation-id>
daita --root /tmp/daita-next events read atlas --limit 100
```

SQLite sources remain read-only unless write admission is explicit on their
first attachment. `--write-access` enables only the catalog-validated impact
preview and conditional single-row update capabilities; it does not expose an
arbitrary mutation-SQL surface:

```bash
daita --root /tmp/daita-next source add atlas sqlite /absolute/path/orders.db \
    --write-access --idempotency-key attach-orders-write-v1
```

Phase 8 PostgreSQL support uses the same catalog, capability, runtime, and
evidence path as SQLite, but its admission boundary is intentionally narrower.
Only attached base tables whose columns use supported `pg_catalog` native types
are advertised for queries; views, foreign tables, and tables containing
custom/extension types are omitted. Generated SQL must use the exact
schema-qualified `native_identity` returned by `catalog_inspect`; execution
adds `ONLY` and a transaction-local `pg_catalog` search path. Credentials are
resolved from secret references and are never stored as raw passwords.

The PostgreSQL database and login role are trusted deployment inputs: use a
least-privilege read-only role and do not attach a database whose row-security
policies or server-owned type/operator code is untrusted. PostgreSQL reads are
classified non-idempotent and non-replay-safe, so ambiguous interrupted calls
fail into manual recovery instead of being repeated. Fake-driver coverage is
deterministic Phase 8 evidence. Phase 9 also passed the real asyncpg path
against a throwaway PostgreSQL 16 service using an ordinary SELECT-only role.

If inspection reports a waiting approval, the decision updates only the
persisted approval and wakes the same operation through the host:

```bash
daita --root /tmp/daita-next approval approve atlas <approval-id> \
  --actor local-user --reason "Reviewed impact" \
  --idempotency-key approve-<approval-id>
```

Monitor creation is proposal-first. The definition is inert until its exact
candidate hash is confirmed:

```bash
daita --root /tmp/daita-next monitor propose atlas daily-sales \
  --definition '{"name":"Daily sales","objective":"Summarize sales changes.","scope":{"source_ids":[],"resource_ids":[]},"schedule":{"kind":"cron","expression":"0 8 * * 1-5","timezone":"UTC"}}' \
  --idempotency-key propose-daily-sales-v1
daita --root /tmp/daita-next monitor confirm atlas <proposal-id> \
  --candidate-hash <candidate-hash> --actor local-user \
  --reason "Reviewed schedule and scope" \
  --idempotency-key confirm-daily-sales-v1
daita --root /tmp/daita-next monitor inspect atlas daily-sales
daita --root /tmp/daita-next monitor run-now atlas daily-sales \
  --idempotency-key run-daily-sales-1
```

Every non-streaming CLI response is one strict JSON object. Interactive chat
and `events follow` emit strict JSON Lines. Follow reconnects from its last
durable event sequence and observes events committed after it starts; every
underlying socket page remains bounded. Python callers can use
`AgentHost.subscribe_events()` while the host is running. OS-service
installation and remote transports remain deferred.

## Phase gates

The architecture MVP proves the generic loop, persistence/recovery, fake
approval-controlled side effect, SQLite and sandboxed local-file data paths,
catalog, memory/skills, local host/monitors, and mock plus OpenAI operation.

The Phase 9 hardening gate additionally proved a controlled real write,
PostgreSQL, every retained provider's conformance suite, public-feature
dispositions, the in-package CLI/legacy-package decision, packaging, live
checks, recovery and security hardening, and tested fresh-state behavior.

P9.5-Q02 through P9.5-Q08 prove exact default-product read authority, cold
configured model reconstruction, scoped evidence-linked monitor findings, safe
natural memory/skill proposals, additive configured capability providers with
exact reopen binding, the installed-console real-socket product journey, and a
clean-wheel joined lifecycle on both supported Python versions. The affected
real OpenAI/default-domain/SQLite path also passes across cold reopen.

Even passing Phase 9.5 does not authorize Phase 10. Root `daita/` may be
removed or replaced only after explicit human approval.

## Project records

- `STATUS.md` — active task ledger and exact resume action
- `PARITY_MATRIX.md` — v1 feature and behavior dispositions
  (created in P0-04)
- `QUALITY_GATES.md` — commands, environments, and results
  (created in P0-06 and maintained thereafter)
- `decisions/` — accepted numbered architecture decisions

Later phases add a module only when a working vertical slice requires its
owner. The target tree in the plan is not a scaffolding checklist.
