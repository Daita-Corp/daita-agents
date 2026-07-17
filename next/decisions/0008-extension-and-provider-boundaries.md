# ADR 0008: Extension and provider boundaries

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

V2 recognizes four narrow extension categories:

1. resource adapters for discovery, inspection, health, declarations, and
   source-specific executor implementations;
2. capability providers for optional domain contracts and evidence schemas;
3. backend providers for models, embeddings, storage, secrets, and telemetry;
4. validated manifests that declare identity, version, kind, dependency hints,
   and contributed contracts.

Built-ins and an explicit configured extension list are allowed. Arbitrary
cross-lifecycle hooks, implicit package scanning, and automatic installation
are excluded.

Model providers translate canonical inference I/O only. They never receive the
operation runtime or invoke tools. Provider-specific formats stop at adapters;
only the router handles normalized retry/fallback, subject to data-routing
policy.

All optional SDKs import lazily inside connection/client creation and raise
`ImportError` with `pip install 'daita-agents[extra]'` guidance.

## Consequences

Extensions can add environments without adding planners, loops, storage
owners, governance paths, or hidden model tools.
