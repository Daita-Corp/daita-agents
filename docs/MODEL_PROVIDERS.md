# Model provider implementation guide

Daita keeps provider discovery declarative while retaining explicit native
protocol adapters. Built-in provider identity, display name, authentication
mode, endpoint behavior, construction binding, request-policy support, and
unreviewed-profile defaults have one owner:
`src/daita/llm/provider_definitions.py`.

The consumers are deliberately separate:

- `llm/factory.py` resolves configured credentials and builds routes;
- `hosting/embedded.py` admits and persists model selections;
- `tui/` derives provider choices and authentication prompts; and
- `llm/providers/` owns native requests, responses, streaming grammar, and SDK
  or official-client shutdown.

Adding a built-in provider should require one `ProviderDefinition`, its explicit
lazy construction callable, the protocol implementation, and focused tests. Do
not add provider lists or vendor branches to the factory, embedded host, TUI, or
model loop. Unknown provider names already use the custom OpenAI-compatible path
and must supply an HTTP(S) base URL; that path does not confer built-in identity
or privileges.

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
