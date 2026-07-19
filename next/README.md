# Daita autonomous agent v2 replacement

This is the isolated replacement project for Daita 2.0. Through Phase 5 it is
a functional persistent embedded agent: the generic loop, governed execution,
SQLite recovery, catalog-backed SQLite and sandboxed-file data paths,
provenance-bearing context, session compression, scoped memory/learning, and
versioned procedural skills are implemented. Local hosting, monitors, broader
candidate integrations, and cutover remain later phases.

The governing plan is the local
`../docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md` fingerprinted in
[`STATUS.md`](STATUS.md). Root `../daita/` is a frozen behavioral oracle, not a
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

## Phase gates

The architecture MVP proves the generic loop, persistence/recovery, fake
approval-controlled side effect, SQLite and sandboxed local-file data paths,
catalog, memory/skills, local host/monitors, and mock plus OpenAI operation.

The replacement-candidate gate additionally requires a controlled real write,
PostgreSQL, every retained provider's conformance suite, public-feature
dispositions, CLI/client integration, packaging, live checks, recovery and
security hardening, and tested fresh-state behavior.

Passing the replacement-candidate gate does not authorize Phase 10. Root
`daita/` may be removed or replaced only after explicit human approval.

## Project records

- [`STATUS.md`](STATUS.md) — active task ledger and exact resume action
- [`PARITY_MATRIX.md`](PARITY_MATRIX.md) — v1 feature and behavior dispositions
  (created in P0-04)
- [`QUALITY_GATES.md`](QUALITY_GATES.md) — commands, environments, and results
  (created in P0-06 and maintained thereafter)
- [`decisions/`](decisions/) — accepted numbered architecture decisions

Later phases add a module only when a working vertical slice requires its
owner. The target tree in the plan is not a scaffolding checklist.
