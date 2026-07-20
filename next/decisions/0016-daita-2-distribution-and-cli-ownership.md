# ADR 0016: Daita 2.0 distribution and CLI ownership

- **Status:** Accepted
- **Date:** 2026-07-19
- **Supersedes:** The external CLI/client integration decision in ADR 0010

## Context

The replacement already contains the public `daita` Python package, a thin
`daita.cli` parser, and the private Unix-socket host/client protocol. The
separate `daita-cli` and `daita-client` repositories implement Daita 1.x cloud
and local-HTTP contracts. Adapting either package would create competing
console-entry-point ownership, release skew, or a second local runtime path.

## Decision

- `daita-agents` is the sole supported Daita 2.0 product distribution. It owns
  the `daita` import package, the internal `daita.cli` module, and the installed
  `daita` console entry point.
- Keeping command parsing in an internal module is an implementation boundary,
  not a reason to require a separately installed CLI distribution. The parser
  remains a thin client of `AgentHost` and the public local protocol; it owns no
  planning, execution, persistence, policy, retry, or recovery behavior.
- The `daita-cli` and `daita-client` repositories are preserved as legacy Daita
  1.x repositories. They are unsupported and excluded from Daita 2.0: the
  replacement does not depend on them, co-install them, adapt their HTTP
  behavior, or use them as compatibility fallbacks.
- A user moving from Daita 1.x must uninstall legacy `daita-cli` before
  installing Daita 2.0. Both distributions otherwise claim a console script
  named `daita`, and Python packaging does not safely arbitrate that collision.
  Legacy `daita-client` should also be removed unless the user intentionally
  remains on the Daita 1.x hosted API.
- Managed-cloud commands and a remote SDK are deferred until a stable Daita 2.0
  cloud API exists. The preferred future command surface is `daita cloud ...`,
  loaded behind a narrow optional boundary while `daita-agents` retains entry-
  point ownership. A separately versioned remote SDK may be considered then
  for consumers that do not install the local runtime.

## Consequences

Phase 9 proves one install, one local protocol, and one command owner instead
of making legacy packages appear compatible. The core installation stays free
of the legacy CLI's HTTP, MCP, ASGI, and YAML dependency stack, and the port
8123/secondary-SQLite runtime is not carried into Daita 2.0.

The Phase 10 cutover checklist must call out legacy-package removal before the
new console script is installed. Publishing, yanking, or otherwise changing
the legacy distributions remains release work and is not authorized by this
ADR.

## Alternatives rejected

- A separate `daita-cli` owning `daita` would make the host protocol and CLI
  release independently while recreating the existing console-script
  collision.
- Renaming the legacy command to `daita-cloud` would expose repository
  boundaries to users and preserve a v1 cloud contract in the 2.0 product.
- Adapting `daita-client` for local sockets would duplicate the public
  `Agent`/local-client surface and risk a second retry or resume policy.
- A meta-package would add another version and dependency coordinator without
  adding a user capability.
