# ADR 0006: Catalog and domain ownership

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- The always-present catalog owns resource/source identity, revisions,
  freshness, facets, relationships, discovery provenance, structural search,
  and bounded graph traversal.
- Connector-discovered structure outranks model inference. Safety-critical
  facts are refreshed or rejected when stale.
- The built-in data domain consumes catalog facts for SQLite, PostgreSQL, and
  local tabular files; it cannot maintain a private database-shaped schema
  model or perform source I/O.
- Local files do not justify a second loop or file-specific runtime.
- Domain controllers own tool projections, action validation, observation
  projection, and evidence-grounded readiness.
- All catalog metadata and source content is labeled untrusted and bounded
  before model projection.

## Consequences

Future resource shapes extend facets, relationships, adapters, and capability
declarations without changing the loop or executor boundary.
