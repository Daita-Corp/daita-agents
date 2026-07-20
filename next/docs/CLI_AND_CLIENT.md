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
reusing a key for different input is an error. Event following replays from a
durable cursor and treats in-process notifications only as wake hints.

This decision is recorded in ADR 0016.
