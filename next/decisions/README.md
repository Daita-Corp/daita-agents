# Daita v2 Architecture Decisions

These accepted Phase 0 decisions govern the isolated replacement project.
They are derived from
`docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md` as fingerprinted in
`next/STATUS.md`. If an ADR conflicts with that plan, the plan wins until a
new ADR explicitly documents the approved change.

| ADR | Decision |
| --- | --- |
| [0001](0001-replacement-isolation-and-baseline.md) | Replacement isolation, namespace, baseline, and v1 freeze |
| [0002](0002-persistent-agent-and-single-loop.md) | Persistent agent identity and the single generic loop |
| [0003](0003-canonical-runtime-vocabulary.md) | Canonical runtime vocabulary and lifecycles |
| [0004](0004-sole-executor-boundary.md) | Sole executor boundary and recovery semantics |
| [0005](0005-state-events-and-crash-consistency.md) | Current state, canonical events, and crash consistency |
| [0006](0006-catalog-and-domain-ownership.md) | Catalog and data-domain ownership |
| [0007](0007-memory-learning-and-skills.md) | Memory, learning, and skill safety |
| [0008](0008-extension-and-provider-boundaries.md) | Extension categories and provider boundaries |
| [0009](0009-local-state-hosting-and-migration.md) | Local state, writer ownership, and migration posture |
| [0010](0010-public-api-cli-and-secrets.md) | Public API, CLI ownership, and secrets |
| [0011](0011-support-matrix-and-phase-boundaries.md) | Supported platforms/providers and phase boundaries |
| [0012](0012-monitor-scheduling-defaults.md) | Monitor scheduling and missed-run defaults |
| [0013](0013-retention-and-redaction-defaults.md) | Data retention and redaction defaults |
| [0014](0014-licensing-boundary.md) | Local-core and premium-hosting licensing boundary |
| [0015](0015-model-routing-and-provider-security.md) | Model routing, fallback, sensitivity, continuation, and endpoint security |

## Status vocabulary

- **Accepted:** binding for the current replacement candidate.
- **Proposed:** not binding; implementation must not depend on it.
- **Superseded:** replaced by a later ADR, which must link back here.
