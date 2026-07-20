# Distribution, CLI, and client boundary

The local host owns runtime semantics. A CLI or client may validate arguments,
frame a request, submit an idempotency key, and render a bounded response; it
may not plan operations, invoke executors, mutate state files, or implement a
second retry/resume policy.

`daita-agents` is the sole supported Daita 2.0 product distribution. It owns
the `daita` import package, the internal `daita.cli` parser, and the installed
`daita` console entry point. `daita.cli` uses the same local request methods as
the public Python surface; keeping the parser in its own module does not make
it a separately installed product.

The Phase 6 protocol covers agent bootstrap, health/status, chat submission,
operation inspection/cancellation, approval decisions, source lifecycle,
model status, committed events, and monitor lifecycle.

## Phase 9.5 closure status

P9.5-07 proves the supported local first-run path through the installed
`daita` entry point and a real private Unix socket. The bundled surface now
includes `agent create`; `model set/show/status`; model-free `serve`; source
add/list/detach/health; line-oriented `chat`; operation and approval
list/inspection; catalog search/show; memory and skill list/inspection;
monitor list/inspection and natural proposal/confirmation; and reconnecting
cursor-based `events follow`.

Non-streaming responses are one strict JSON object. Interactive chat emits one
JSON object per committed event followed by one result object for each input
line. Event follow emits one committed event per line, advances only after a
durable sequence, reconnects after transport loss, and bounds each page. The
CLI never writes state directly or creates runtime background work; all
mutations remain idempotent `AgentHost` operations through the private local
protocol. P9.5-08 also passes the clean-wheel joined lifecycle on CPython 3.11
and 3.12, including cold model-free host reopen and a second grounded run.

The separate `daita-cli` and `daita-client` repositories are preserved legacy
Daita 1.x repositories. They are unsupported and excluded from Daita 2.0 and
must not be installed as candidate dependencies or used as HTTP compatibility
fallbacks. In particular, the legacy CLI's port-8123 server and private SQLite
ledger are not alternate ways to run a Daita 2.0 agent.
They must not be used as a fallback.

Before installing Daita 2.0 into an environment that contained Daita 1.x,
uninstall `daita-cli`. It also installs a `daita` console script, and two Python
distributions cannot safely own the same executable. Remove `daita-client` as
well unless the environment intentionally continues to target the legacy
hosted API.

In a Daita 2.0 environment, make the ownership change explicit and then verify
the executable that the shell resolves:

```bash
python -m pip uninstall daita-cli daita-client
python -m pip install "daita-agents[recommended]"
hash -r
command -v daita
daita --help
```

Use a fresh virtual environment when legacy hosted-API access must remain
available elsewhere; do not try to share one `daita` script between versions.

Managed-cloud commands and a remote SDK remain deferred until a stable Daita
2.0 cloud API exists. The intended future CLI shape is `daita cloud ...`
behind a narrow optional boundary, with `daita-agents` still owning the one
console entry point. A remote-only SDK can be evaluated separately when that
protocol has a concrete consumer and compatibility contract.

Protocol requests are strict, bounded JSON frames over a private Unix socket.
Every mutation is tied to one method/parameter hash and durable idempotency key;
reusing a key for different input is an error. Committed-event pagination
replays from a durable cursor. The Phase 9.5 follow loop advances only from
those durable pages and treats in-process notifications only as wake hints.

This decision is recorded in ADR 0016.
The mandatory product-contract closure before replacement readiness is
recorded in ADR 0017.
