# Daita 2.0 candidate support matrix

This is the proposed replacement-candidate surface. “Deterministic” means the
local contract suite passes; “live passed” means Phase 9 exercised the real
credential- or service-backed integration without substituting a mock.

| Area | Candidate support | Acceptance status |
| --- | --- | --- |
| Python | CPython 3.11 and 3.12 | Deterministic and clean-install lifecycle passed in Phase 9 |
| Distribution and CLI | `daita-agents` with its internal `daita.cli` module and sole `daita` console entry point | Deterministic packaging and lifecycle passed; legacy packages excluded |
| Local state | SQLite WAL state, migrations, backup-before-migrate, blobs, fresh v2 root, immutable agent runtime defaults | Deterministic |
| Hosting | Embedded mode and one foreground local host over a private Unix socket | Deterministic; thin local CLI uses the same host/client contract |
| Data sources | SQLite, sandboxed local CSV/JSON, conservative PostgreSQL base-table reads | Deterministic; real SELECT-only PostgreSQL live passed |
| Controlled writes | Parameterized single-row SQLite update with impact evidence and approval | Deterministic |
| Models | Mock, OpenAI, Anthropic, Gemini, Grok, Ollama, explicit OpenAI-compatible endpoints | Shared conformance deterministic; every retained-provider live row passed |
| Autonomy | Persistent sessions, recovery, events, memory, skills, monitors, approvals | Deterministic |
| Extensions | Narrow manifests, capability providers, resource adapters, and event observers | Phase 9 hardened |
| Telemetry | Optional projection from committed canonical events | Phase 9 hardened; never required for commits |

Explicitly deferred after the replacement candidate: MySQL, MongoDB,
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
