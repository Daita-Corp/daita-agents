# ADR 0012: Monitor scheduling defaults

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- Monitors persist cron or interval definitions and emit ordinary typed
  triggers that create normal operations.
- The host owns cadence; the scheduler exposes one-shot `run_due(now)` and does
  not start background work on library import or embedded-agent construction.
- The default missed-run policy is `catch_up = once`, followed by normal next
  schedule calculation.
- Tick leases, stable occurrence keys, checkpoints, cooldown, and backoff make
  a due occurrence create at most one operation.
- Monitor activation requires confirmation of schedule, scope, condition,
  policy, and budget.
- Local MVP notification is a durable finding/event linked to operation and
  evidence. Outbound email/chat/webhook delivery is a separate extension.

## Consequences

Monitor-triggered work uses the same loop, policy, task, approval, evidence,
readiness, and recovery paths as user-triggered work.
