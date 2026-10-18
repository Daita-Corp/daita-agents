# Model provider implementation guide

Daita keeps provider discovery declarative while retaining explicit native
protocol adapters. Built-in provider identity, display name, authentication
mode, endpoint behavior, construction binding, request-policy support, and
unreviewed-profile defaults have one owner:
`src/daita/llm/provider_definitions.py`.

The consumers are deliberately separate:

- `llm/factory.py` resolves configured credentials and builds routes;
- `hosting/configuration.py` admits and persists model selections;
- `tui/` derives provider choices and authentication prompts; and
- `llm/providers/` owns native requests, responses, streaming grammar, and SDK
  or official-client shutdown.

Adding a built-in provider should require one `ProviderDefinition`, its explicit
lazy construction callable, the protocol implementation, and focused tests. Do
not add provider lists or vendor branches to the factory, embedded host, TUI, or
model loop. Unknown provider names already use the custom OpenAI-compatible path
and must supply an HTTP(S) base URL; that path does not confer built-in identity
or privileges.

## Host request capacity

`AgentConfig.host_capacity` is an immutable `HostCapacityLimits` for one open
composition. Defaults remain five executions, two simultaneous provider requests,
one strictly reserved foreground slot in each, one graph task, two PostgreSQL
reads per source and four SQLite state-pressure permits. SQLite source reads and
MCP binding calls remain capacity one. Capacity never grants a capability or
widens a frozen execution scope.

```python
from daita import AgentConfig, HostCapacityLimits

config = AgentConfig(
    host_capacity=HostCapacityLimits(
        execution_capacity=8,
        foreground_execution_reserve=1,
        provider_capacity=4,
        foreground_provider_reserve=1,
        graph_parallelism=4,
        postgresql_read_capacity=4,
    ),
)
```

Supply the configuration with the existing model route or injected provider to
`Agent.create`, `Agent.open` or `Agent.from_storage`. These capacity settings are
runtime-only and must be supplied on each open. Drain and reopen to change them.
The initial opt-in ceilings are 16 executions, eight provider requests, four graph
tasks and eight PostgreSQL reads per source. They are qualification bounds, not
published safe operating levels. Reserves must be integers below capacity; zero
explicitly permits full background utilization. Reserved slots are never borrowed.
Independent conversations can overlap; turns within a conversation remain FIFO.
Weighted background admission and the effect lane retain their existing owners.
Provider quotas shared across agent homes remain with the consuming application.
Use `Agent.admission_diagnostics()` to inspect applied capacities and bounded
queue, execution and cleanup counters. Current defaults remain the rollback
configuration until combined offline and live qualification supports a change.

## Adapter boundaries

Use `llm._lifecycle` for the common canonical attempt boundary and once-only
close coordination. Keep SDK/process shutdown with the adapter that owns the
native resource. `providers._fields` owns identical native field extraction and
validation; protocol-specific error codes, headers, billing dimensions, message
formats, and continuation rules remain local.

For a substantial native protocol, separate these responsibilities:

- `<provider>/adapter.py`: client ownership, request admission, native calls,
  and release;
- `<provider>/messages.py`: canonical-to-native serialization and continuation
  replay; and
- `<provider>/stream.py`: native event grammar, incremental state, and terminal
  decoding.

The provider package `__init__.py` exports its supported provider classes and
preserves imports such as `daita.llm.providers.openai.OpenAIProvider`. Keep a
small specialization such as Codex, Grok, or Ollama as one module when it reuses
an existing protocol adapter and has no independent translation grammar.

Official-client subscriptions share bounded process execution in
`providers.subscription_cli.process` and their canonical request/response
envelope in `providers.subscription_cli.envelope`. Claude Code and Grok Build
retain their own flags, environment restrictions, inspection, and output
decoding. They do not create another model/tool loop.

## Qualification

Test definition uniqueness, derived TUI choices, lazy SDK imports, endpoint and
authentication rules, and an unknown compatible provider without dispatching to
its synthetic endpoint. Then exercise success, malformed output, timeout,
cancellation, early stream exit, cleanup failure, borrowed clients, usage/cost
accounting, routing, native continuation, and architecture isolation for every
affected protocol. Live or paid integrations are opt-in and are not part of the
deterministic provider qualification suite.

PostgreSQL source reads can opt into bounded connection reuse for one open home:

```python
from daita import AgentConfig, PostgreSQLReadLimits

config = AgentConfig(postgresql_reads=PostgreSQLReadLimits(
    pool_enabled=True,
    pool_size_per_source=2,
    max_connections=8,
    max_cached_sources=8,
    acquire_timeout_seconds=5,
    idle_timeout_seconds=60,
    statement_timeout_seconds=5,
    cleanup_timeout_seconds=1,
))
```

Pooling is disabled by default. Pool size is clamped to the source permit and total
connection ceilings. Reserved slots, including idle pools and pools being retired,
never exceed `max_connections`; idle pools are retired when a new source needs
capacity. Integer ceilings are eight; timeout bounds are finite and positive
(acquire/statement at most 60 seconds, idle 3600, cleanup 10). These settings are
runtime-only and apply through create, open and from-storage. Drain and reopen to
change them. They grant no source, resource, effect or scheduled authority.

The default backend owns lazy asyncpg pools and closes them after host work drains,
before credentials and state close. Every operation has a fresh read-only
repeatable-read transaction, transaction-local search path/timeout and full live
structural comparison. Checkout and initialization fit the read deadline. Failed
or cancelled operations discard their connection; successful operations reset it
before return. A reset failure discards the connection without replaying the read.

Pools bind exact source registration, readable scope and resolved credential
admission. Credentials are resolved again at checkout and after waiting, comparing
only private in-memory values. Configuration changes, detach, permission changes,
rotation or idle expiry retire old pools; secret values/fingerprints are never
persisted or reported. Stale idle connections may be replaced before dispatch;
a dispatched user query is never automatically retried. Native effects retain
their separate connection and receipt owner.

`Agent.postgresql_read_diagnostics()` reports bounded aggregate counts, applied
limits, checkout wait, successful structural/query durations, cleanup time and pool reservations,
including after close. Pool counters are zero with reuse disabled; cold-connection
benchmarks measure transport creation separately. Opt-in admission observations use the existing observer;
failed callbacks cannot fail reads. These source-local limits cannot enforce
shared database quotas across homes or consuming applications.
