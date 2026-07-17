# ADR 0014: Licensing boundary

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- The local v2 core, embedded runtime, local host contracts, generic loop,
  local persistence, built-in MVP adapters, and extension protocols remain
  under the repository's Apache-2.0 license for this replacement candidate.
- Premium value is hosted operational infrastructure and enterprise product
  control: managed availability, scheduling, runners, multi-tenancy, RBAC/SSO,
  approvals, secrets, audit retention, backups, notifications, observability,
  residency, support, and SLAs.
- Local behavior is not intentionally crippled to create the premium boundary.
- Final commercial packaging, trademarks, and hosted-service terms require
  product/legal review and are outside implementation Phases 0 through 9.

## Consequences

Local and premium hosting must preserve the same semantic runtime contracts;
premium evolution replaces infrastructure rather than forking the agent brain.
