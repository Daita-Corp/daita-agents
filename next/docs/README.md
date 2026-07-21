# Daita 2.0 candidate documentation

These documents describe the isolated v2 replacement project. Phase 9
hardening and the mandatory Phase 9.5 joined replacement-readiness closure
pass. The candidate is eligible for human Phase 10 review, but these documents
do not authorize Phase 10 or substitute for the executed project ledgers. The
subsequently proposed live LLM production-readiness gate is not complete: its
Wave 1 deterministic harness passes, but the first authorized real-provider
run failed all 12 prompt variants. The complete gate must pass before
destructive cutover.

- [Fresh state and migration](FRESH_STATE_AND_MIGRATION.md)
- [V1 export guidance](V1_EXPORT_GUIDE.md)
- [Breaking changes](BREAKING_CHANGES.md)
- [Support matrix](SUPPORT_MATRIX.md)
- [Live LLM production-readiness test plan](LIVE_LLM_PRODUCTION_READINESS.md)
- [Wave 1 live failure root-cause analysis](LIVE_MVP_WAVE1_FAILURE_ANALYSIS_2026-07-20.md)
- [Wave 1 explicit repair specification](LIVE_MVP_WAVE1_REPAIR_SPEC_2026-07-20.md)
- [Local operations](OPERATIONS.md)
- [Security model](SECURITY.md)
- [CLI and client boundary](CLI_AND_CLIENT.md)

The implementation status and executed evidence remain authoritative in
[`../STATUS.md`](../STATUS.md) and [`../QUALITY_GATES.md`](../QUALITY_GATES.md).
