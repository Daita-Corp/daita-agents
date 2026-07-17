# ADR 0007: Memory, learning, and skill safety

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- Memory is scoped, versioned, provenance-backed, sensitivity-aware,
  freshness/revision constrained, inspectable, supersedable, and reversible.
- The loop may propose learning but cannot commit durable memory or skills.
- A learning service consumes explicit corrections and completed-operation
  events, validates proposals, and commits according to policy.
- Explicit user requests to remember may commit after safety validation.
  Narrow evidence-backed facts may auto-commit when configured. Inferred
  business meanings and skill changes remain proposals requiring acceptance by
  default.
- Raw sensitive rows and PII are rejected or specially governed. Failed or
  policy-blocked actions cannot be learned as successful rules.
- Memory can never change policy, secrets, credentials, or executable code.
- Skills are versioned `SKILL.md` procedures that reference registered
  capabilities. They cannot declare, invoke, or bypass executors or policy.

## Consequences

Learning follows `observe -> propose -> validate -> commit -> version ->
recall`; rollback and audit history are required behavior, not later extras.
