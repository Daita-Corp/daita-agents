# CLI and TUI control surfaces

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

Rules for the human-facing controls built on the public `Agent` API.

## Product control surfaces

CLI and TUI routine inspection reuse the routine domain's current projections.
Approval summaries derive only from the exact validated request and retain its
complete bounded details. An explicit `confirmation_handler` reviews direct routine
create/update controls before mutation; typed Python callers without that callback
remain the authorizing owner. CLI create/update supplies this callback. Approval cannot grant missing connector permission.

Native approval documents contain the exact execution `arguments`, catalog-backed
`target`, and bounded `preview` review facts. The data domain reuses its existing
preflight preview; rendering performs no I/O. Execution arguments and intent digests
remain separate from review labels and samples. The runtime still approves and
rechecks one exact plan before reserving an effect. A positive count alone does not
prove the selected business entity; there is no second target authorization record.

Model-authored routine creation requires an explicit `run_immediately` boolean.
Model-authored updates accept omission or false and reject true; typed owner defaults
remain false. Immediate creation and later run-now use the existing owner/supervisor.

The source permission editor authors one exact table scope through existing
preview/apply APIs. Catalog-backed column/key choices are presentation, not a
second validator or proof of live database readiness. Apply confirms the complete
before/after state, including required read additions; it performs no source write.

Receipt list, inspect and human recovery controls call the existing Agent APIs.
Recovery never calls the model or an external executor. The original observation
and separate immutable resolution stay owned by SQLite.

## Host status

The TUI and the CLI are clients of the Daita host. The TUI status bar shows the
real host state (`● host`, `◌ reconnecting`, `○ host stopped`) from the host
status route and `agent_state` events, never a static indicator; `/status`
prints the connection as a `Host` line with the same guidance. `daita host
status` reports whether the host runs, its pid, version, socket and log path,
the agents it holds and how many clients watch them. Agent commands go
through the host; `daita --no-host <command>` opens the agent in-process for
one command and fails with `HostActiveError` when the host holds it. `daita`
and `daita chat` always use the host.
