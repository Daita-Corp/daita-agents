# ADR 0002: Persistent agent and single generic loop

- **Status:** Accepted
- **Date:** 2026-07-16

## Context

V1 splits generic chat and database-specific agent behavior. Reproducing that
split for every source type would create parallel planners and runtimes.

## Decision

- The durable root identity is one persistent `Agent`; a session is only an
  optional conversational episode.
- Exactly one provider-neutral loop handles user, schedule, monitor, and
  bounded internal triggers.
- The model owns semantic choice. The runtime owns authority and facts.
- The loop owns progression, budgets, cancellation/interruption checks,
  action sequencing, readiness correction, and the decision to complete.
- The operation runtime atomically commits terminal operation state after the
  loop's completion decision or a terminal runtime fault.
- Domain controllers validate and project domain facts but cannot execute I/O.
- Hosting composes the loop and services; it cannot introduce another loop.

## Consequences

The generic loop may import only canonical model, context, domain-controller,
and operation-runtime contracts. Database, file, provider, cloud, monitor,
governance, and durable-learning branches are forbidden in the loop.
