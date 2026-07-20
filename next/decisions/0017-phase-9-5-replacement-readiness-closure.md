# ADR 0017: Phase 9.5 replacement-readiness closure

- **Status:** Accepted
- **Date:** 2026-07-19
- **Supersedes:** ADR 0011 only where it says Phases 0 through 9 alone
  produce the replacement candidate

## Context

Phase 9 passed its declared packaging, provider/database, fault, performance,
security, frozen-v1, and clean-install gates. A subsequent plan-to-source audit
confirmed that the replacement has strong individual primitives, but found six
places where the supported default composition does not yet satisfy the full
product contract:

1. read adapters revalidate scope safely, but exact validator-owned authority
   and sensitivity facts are not carried through every task and evidence
   record;
2. model-route fingerprints persist, but a new process cannot reconstruct a
   configured retained provider/router from Agent Home alone;
3. monitor scheduling and deduplication work, but the default host has no
   production data outcome projector and does not fully bind scope, effective
   policy/budget, condition evaluation, and findings;
4. one precise learning trajectory works, but ordinary correction/remember and
   skill-proposal ingress is incomplete;
5. extension declarations are validated, but explicitly configured extensions
   do not compose additively with built-ins through the normal Agent/host path;
   and
6. the bundled CLI/local protocol is safe but does not yet provide the complete
   first-run, interactive, inspection, natural-monitor, and event-follow
   experience required by the plan.

Treating those component seams as a complete product would make the cutover
claim stronger than the executed evidence. Reopening the architecture itself
would be unnecessary and would put the single-loop/runtime design at risk.

## Decision

- Add a mandatory Phase 9.5 between Phase 9 hardening and Phase 10 cutover.
  Phase 9 remains historical PASS evidence, but replacement-readiness sign-off
  now requires the Phase 9.5 joined gate.
- Close the six gaps through their existing owners: data validators and
  evidence repositories; model configuration/router/factory and secret
  provider; monitor service/scheduler plus data-domain outcome projection;
  learning/memory/skill services; extension/capability registries; and the
  Agent/host/local-protocol/in-package CLI facades.
- Persist exact read authority and canonical model-route configuration with
  schema versions and fail-closed migration/reopen behavior. Persist secret
  references only, never secret values.
- Treat monitor scope as an enforced maximum. MVP monitor budget and policy
  overrides may only restrict agent defaults, and admitted conditions are
  deterministic typed evaluations over accepted current-operation evidence.
  Free-form expression execution is not supported.
- Compose extensions only from an explicit configured list or registry.
  Built-ins remain present, declaration identity/version drift fails before
  partial visibility, and every executor still runs only through the operation
  runtime. There is no implicit scanning, installation, or hot reload.
- Let ordinary interactions propose learning through the existing safety and
  provenance lifecycle. The loop cannot directly commit hidden memory or
  activate a skill change.
- Keep `daita-agents` as the sole Daita 2.0 distribution and `daita` entry-point
  owner. Phase 9.5 completes the in-package product journey; it does not revive
  `daita-cli` or `daita-client`.
- Use focused tests within coherent slices and one consolidated broad/live gate
  at P9.5-08. Reuse unaffected Phase 9 evidence rather than repeating it for
  ceremony.

## Consequences

The candidate is no longer described as replacement-ready until Phase 9.5
passes. Phase 10 still requires separate explicit human authorization after
that gate. Root v1 remains frozen, and no push, publication, release, package
replacement, or legacy-repository mutation is authorized here.

P9.5 must prove joined default behavior, not merely more unit seams: a
configured agent reopens and runs without model reinjection; trusted read facts
govern real I/O and provider routing; the default host produces a scoped,
evidence-linked monitor finding; natural learning changes later behavior; one
extension coexists with built-ins; and the installed in-package CLI completes
the real local-host journey.

No new loop, operation runtime, catalog, policy engine, state store, learning
store, extension framework, source family, model-provider family, remote SDK,
managed deployment, outbound notification system, or autonomous skill
activation belongs in this phase.

## Outcome

P9.5-Q08 passed on 2026-07-20. The candidate is now replacement-ready and
eligible for separately authorized human Phase 10 review. The gate did not
start or authorize cutover, modify root v1, revive either legacy package, push,
publish, open a pull request, or create a release.

## Alternatives rejected

- Calling the six gaps post-MVP would contradict the existing acceptance
  journeys and replacement definition of done.
- Marking Phase 9 failed retroactively would discard valid executed hardening
  evidence and obscure what is actually missing.
- Solving each gap through test injection or caller assembly would preserve the
  same product-level incompleteness.
- Reintroducing separate CLI/client packages would recreate version skew and a
  second local product boundary already rejected by ADR 0016.
