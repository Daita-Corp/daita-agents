# ADR 0013: Retention and redaction defaults

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- Always retain normalized fields required for resume and audit: ordered tool
  calls, validated arguments, bounded observations, status/error reasons,
  final-answer candidate, usage metadata, fingerprints, and record linkages.
- Do not retain raw secrets. Raw provider requests/responses, full executor
  payloads, rows, file content, and connector errors are disabled from
  telemetry and public projections by default.
- Local audit evidence may retain a bounded structured payload or a
  content-addressed artifact according to its sensitivity and explicit
  retention class. Model and public projections are separately bounded and
  redacted.
- Store content hashes and references instead of duplicate verbatim content.
  Large/binary artifacts use the blob store with provenance, retention, and
  tombstone state.
- Memory rejects raw sensitive rows by default and applies scope, freshness,
  revision, sensitivity, and provenance filters before recall.
- Redaction covers connection strings, credentials, headers, query parameters,
  file content, and configured PII fields before logs/events/telemetry.

## Consequences

Operational correlation cannot depend on raw sensitive payload retention.
Future retention expansion requires an explicit policy change and tests.
