# ADR 0015: Model routing and provider security

- **Status:** Accepted
- **Date:** 2026-07-19

## Decision

Phase 8 adds one provider-neutral `ModelRouter` that implements the same model
boundary consumed by the generic loop. Provider adapters remain translation
leaves and may neither retry another provider nor inspect runtime, capability,
executor, approval, or evidence state.

One route contains an ordered primary/fallback candidate list. Each candidate
binds an exact canonical model profile and an explicit set of allowed request
sensitivity classes. The request sensitivity classes are `public`, `internal`,
`confidential`, and `restricted`; absence defaults to `internal`. Permission is
checked before every attempt, including the primary. A denied destination
receives no request. Free-form model-profile routing labels are descriptive and
are never interpreted as security policy.

Only `rate_limit_error`, `provider_unavailable`, and `timeout` are eligible for
bounded retry or fallback. Cancellation propagates unchanged. Authentication,
model-selection, context, request, content-policy, and malformed-response
failures are terminal so routing cannot hide configuration, policy, or adapter
defects. Retry counts are explicit and bounded; there is no unbounded backoff
or retry loop inside an adapter.

The router identity includes a SHA-256 fingerprint of candidate order, exact
profiles, sensitivity grants, retry bound, and fallback error set. Agent model
profile binding therefore rejects a changed route after restart. Successful
responses carry typed, bounded attempt and selected-provider facts; provider
error text, credentials, and request content do not enter that trace. Canonical
messages and Daita tool-call IDs remain authoritative. Opaque continuation
metadata is replayed only by its originating provider and is stripped for a
different provider.

Model cost estimates use the exact `Decimal` rates in the selected candidate's
persisted profile and normalized token usage. There is no mutable global price
table. Provider-reported usage remains canonical input to that calculation.

Generic OpenAI-compatible endpoints require explicit construction. HTTPS is
required for remote endpoints; plaintext HTTP is accepted only for loopback
hosts. User information, query strings, fragments, and credentials in endpoint
URLs are rejected. Ollama is a loopback specialization and Grok is a fixed xAI
endpoint specialization; neither silently accepts an arbitrary remote URL.

## Consequences

The generic loop gains no vendor or fallback branch. Restart detects route
drift through its bound identity, fallback cannot exfiltrate sensitive context,
and every retained adapter can share one conformance suite without sharing a
provider base framework or mutable global registry.
