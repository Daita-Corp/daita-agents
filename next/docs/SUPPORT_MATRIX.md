# Daita 2.0 candidate support matrix

This is the tested Daita 2.0 candidate surface. Phase 9 component, packaging,
live, fault, and security evidence and the Phase 9.5 joined default-product
gate pass. That makes the candidate eligible for human Phase 10 review; it
does not authorize cutover.

| Area | Candidate support | Acceptance status |
| --- | --- | --- |
| Python | CPython 3.11 and 3.12 | P9.5-08 passed 1,631 deterministic cases per interpreter and the clean-wheel joined lifecycle on 3.11.15/3.12.7 |
| Distribution and CLI | `daita-agents` with its internal `daita.cli` module and sole `daita` console entry point | P9.5-07/P9.5-08 pass the installed-console and clean-wheel real-socket journeys for durable model setup, model-free serve, source lifecycle, interactive chat, bounded inspection, natural monitor proposal/confirmation, reconnecting event follow, and cold reopen |
| Local state | SQLite WAL state, migrations, backup-before-migrate, blobs, fresh v2 root, immutable agent runtime defaults | Deterministic |
| Hosting | Embedded mode and one foreground local host over a private Unix socket | Host/restart/socket substrate, cold configured model reconstruction, and default monitor outcome semantics passed through P9.5-04 |
| Data sources | SQLite, sandboxed local CSV/JSON, conservative PostgreSQL base-table reads | Connector validation, real SELECT-only PostgreSQL, exact persisted validator-owned read authority, and canonical evidence metadata passed through P9.5-02 |
| Controlled writes | Parameterized single-row SQLite update with impact evidence and approval | Deterministic |
| Models | Mock, OpenAI, Anthropic, Gemini, Grok, Ollama, explicit OpenAI-compatible endpoints | Shared conformance and every retained-provider live row passed; persisted reconstructable primary/fallback routes and injection-free cold reopen passed in P9.5-03 |
| Autonomy | Persistent sessions, recovery, events, memory, skills, monitors, approvals | Durable lifecycle mechanics plus ordinary natural learning, inert skill proposals, and default-host scoped condition/finding semantics pass individually and in the joined P9.5-08 gate |
| Extensions | Narrow explicit capability-provider manifests configured by the caller; no scanning or auto-install | Additive composition alongside built-ins, exact durable manifest-set binding, sole-runtime execution, and fail-closed missing/drift/collision behavior passed in P9.5-06; resource adapters, backend providers, and event observers are post-MVP |
| Telemetry | Optional projection from committed canonical events | Phase 9 hardened; never required for commits |

Explicitly deferred from the Daita 2.0 replacement scope: MySQL, MongoDB,
Snowflake, BigQuery, Elasticsearch/OpenSearch, object stores, cloud inventory,
Google Drive, Slack, GitHub, web search, vector/graph acceleration, embeddings,
rich documents, data-quality/lineage providers, arbitrary event triggers,
managed-cloud deployment, `daita cloud ...` commands, a remote v2 SDK, and a
concrete OTLP transport adapter. The v1 eval framework and the multi-agent
delegation model are also explicitly deferred.

The separately distributed `daita-cli` and `daita-client` are legacy Daita 1.x
products, not deferred Daita 2.0 components. They are unsupported and excluded
from the candidate. Uninstall legacy `daita-cli` before installing Daita 2.0 so
the two distributions do not compete for the `daita` console script.

Deferred packages are absent; selecting one must not import or execute v1.
The complete name-by-name disposition remains in
[`../PARITY_MATRIX.md`](../PARITY_MATRIX.md).

Every Phase 9.5 contract above has executed evidence in `QUALITY_GATES.md`.
The support surface is replacement-ready and eligible for separately
authorized human Phase 10 review. Passing Phase 9.5 does not authorize Phase
10 or cutover by itself.
