# Models and providers

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

Canonical model records, routing, retries, call policy and provider lifecycle. See also `docs/MODEL_PROVIDERS.md`.

## Models and providers

Canonical messages, tool calls and results, usage, requests, responses, and
errors live in `daita.llm.models` and `daita.llm.errors`. Provider-native
payloads end inside provider adapters. `daita.llm.routing` handles retry and
fallback decisions from normalized failures; `AgentLoop` does not retry a
whole run or inspect provider-specific failures.

`daita.llm.provider_definitions` is the sole static source for built-in provider
identity, display metadata, authentication and endpoint modes, lazy construction,
request-policy facts, and unreviewed-profile capability defaults. The factory,
embedded host, and TUI derive their views from those definitions; do not add a
parallel provider list or vendor dispatch branch. Reviewed model limits and
prices remain in `profiles.py` and `pricing.py`.

One immutable `ModelCallPolicy` in `AgentConfig` and `ModelRequest` governs
configured and conforming injected providers, both delivery modes, foreground,
routines, job runs and validation. Defaults are 180 seconds
per logical request, 120 per attempt, 60 to first substantive progress, 30 idle,
15 counting, 5 connect, 120 read, 30 write, 5 pool and 5 cleanup. Every logical
request intersects its caller/run deadline before setup; retries retain that
logical deadline and receive a fresh, narrower attempt deadline. Counting has
its own phase cap (`call_policy.input_count_timeout_seconds`, at most 60).
Monotonic deadlines are runtime-only; policy durations and both retry ceilings
are serialized and included in the machine model-contract digests that a run's
scope fixes when it starts. Routines no longer freeze routes: each occurrence
uses the agent's current model routes and normal run limits, bound when that
run starts, and so does each job run and sub-run.
Counting transport failures retain their canonical cause and proven zero generation
usage; invalid count data and unsupported counting remain permanent admission
failures. External cancellation remains cancellation.

Bounded OpenAI, Anthropic, and Gemini API requests use provider-owned input
counting over the prepared generation input, including tools and retained
provider content. Count and generation calls share the run deadline and SDK
ownership. Token counts are admission estimates; returned usage is incurred
consumption. Byte bounds must not substitute for billed-token counts. Routes
without complete counting retain usage-based progression and supported output
caps, and cannot admit estimated-cost ceilings. A failed count never submits
generation. Preserve known zero usage when counting is cancelled while retaining
normal cancellation/deadline behavior; dispatched generation can remain unmeasured.

The router owns model retries. API SDK retries are disabled on owned clients and
on borrowed-client request views without changing or closing the caller's client.
Every retried attempt receives the remaining logical-request allowance. Unknown
failed-attempt consumption stops budgeted routing; it cannot fund a retry or
fallback. The one exception is a transient failure before any output (a rate
limit, an unavailable provider or dropped connection, a silent stall or an
unsent transport timeout): without a dollar cap it may replay, so a token limit
can be exceeded by at most one unmeasured attempt per replay. An HTTP rejection
or unsent request before any native event proves zero usage, so dollar-capped
routes replay it too; unknown usage under a dollar cap still fails closed.
A provider-reported charge alone does not establish advance price coverage.
Subscription output bounds remain advisory where the external surface does not
support a wire limit; returned usage still controls loop progression.

Retry accounting covers active attempts, backoff, cancellation, and stream cleanup.
Visible stream progress, including terminal completion, closes retry/fallback
eligibility. Completed usage remains authoritative during shutdown. Valid HTTP
retry delays are honored without shortening; waits exceeding 60 seconds or the
remaining deadline stop recovery. Local exponential backoff uses bounded jitter.
`RetryPolicy` defaults to `max_attempts_per_candidate=2` and
`max_total_attempts=3`. Both limits are one for one-shot validation.
Every host route uses `ModelRouter`, including a one-candidate route. The host
places an injected non-router provider in a one-candidate router with the
default `RetryPolicy` (one attempt for validation) under its
`injected:` provider permit key, so every route shares retry, admission and
attempt diagnostics. Injected providers retain caller ownership, must honor the
canonical policy and are never closed by that router; their frozen model
contracts still describe the declared provider and profile.

Provider lifecycle follows explicit ownership. Providers constructed from an
agent's persisted model route are closed by `EmbeddedAgent` after runs and
supervisors drain. Providers injected by a caller remain caller-owned and may
be reused across agent instances. Each provider closes only SDK clients it
created; injected SDK clients remain borrowed. Temporary validation providers
are closed by the component that creates them.

Owners stop new work and drain active calls before closing a provider. Close
callers share one absolute cleanup deadline and a retained once-only outcome;
repeated calls cannot retry native cleanup or renew grace. Native SDK scopes
enter, iterate and exit in one owned task. A bounded supervisor retains native
work that fails to retire, poisons its owner, discards late output and rejects
replacement work. Only native I/O, never tools or persistence, can remain there.
Transport bytes, decoded activity, substantive progress, canonical emission and
terminal completion are separate facts. Empty introductions, snapshots without
growth and keepalives do not reset progress. Opaque nonstreaming/CLI calls report
progress as unobservable and obey fixed deadlines. Terminal usage survives
cleanup failure; successful completion is published only after native release. Adapters scope
request-stream cleanup to completion, failure, cancellation, or early exit.
Canonical stream wrappers finalize in the iteration context and propagate
closure to their delegate; releasing a request stream never closes a borrowed
provider or SDK client. Verify underlying transport release with actual-SDK
tests rather than assuming that closing a public SDK generator releases it.

To add a provider:

1. implement `ManagedModelProvider` under `src/daita/llm/providers/`, delegating
   common attempt and once-only close behavior to `llm._lifecycle`; use a
   provider-named package with `adapter.py`, `messages.py`, and `stream.py` when
   the protocol has substantial translation or stream grammar, while a small
   specialization of an existing adapter remains one module;
2. keep client ownership and native calls in the package's orchestration
   adapter, message translation in its message module, and substantial stream
   grammar in its decoder; the package `__init__.py` exports only the supported
   provider classes so the public provider import stays stable;
3. reuse `providers._fields` for identical native-field validation and import a
   new SDK only at client construction, with normalized pipx repair guidance;
4. add one immutable `ProviderDefinition`, including its explicit lazy
   construction callable, in `llm/provider_definitions.py`; factory, host, and
   TUI provider choices require no provider-specific edit;
5. add a bounded SDK dependency only when the protocol needs one; and
6. add focused definition, lazy-import, translation, lifecycle, accounting,
   routing, and architecture tests.

Unknown configured names continue through the explicit OpenAI-compatible path
and require a base URL; they never inherit built-in authentication, endpoint, or
profile privileges. Subscription subprocess mechanics live in
`providers.subscription_cli.process`, the canonical envelope in
`providers.subscription_cli.envelope`, and vendor flags, inspection, and
decoding in the package's Claude or Grok adapters. Preserve dispatch-time
revalidation even when it uses the same definition fact as lazy route preflight:
those checks protect different boundaries.

Do not add provider branches to `AgentLoop`.
