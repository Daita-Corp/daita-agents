# ADR 0003: Canonical runtime vocabulary

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- **Trigger:** typed request that wakes the agent. MVP kinds are `user`,
  `schedule`, `monitor`, and `internal`; `event` is reserved for post-MVP and
  must be rejected unless explicitly enabled later.
- **Operation:** one durable objective created by a trigger. It owns the loop
  checkpoint and may be sessionless.
- **Loop phase:** current progression checkpoint, distinct from operation
  success/failure/waiting status.
- **Turn:** one context-build/model-response/action-observation iteration.
- **Model tool call:** untrusted provider-neutral proposal from the model.
- **Action proposal:** domain-validated capability request that is still not
  executable until the operation runtime validates and persists it.
- **Task:** persisted executable capability invocation or deterministic recipe
  step owned by the operation runtime.
- **Evidence:** typed, validated, durably accepted task output with provenance.
- **Observation:** bounded model-facing projection of evidence or an explicit
  validation, governance, waiting, or failure result. It is not evidence.
- **Readiness:** domain evaluation of a candidate answer against authoritative
  current-operation facts.
- **Exit result:** typed waiting, interrupted, succeeded, failed, or cancelled
  result returned by the loop/host boundary.

## Consequences

There is no separate one-to-one `AgentRun` record in the MVP. Provider-native
call IDs and payloads are metadata, never replacements for these records.
