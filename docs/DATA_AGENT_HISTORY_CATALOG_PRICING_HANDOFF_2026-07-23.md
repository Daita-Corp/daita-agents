# Data Agent History, Catalog, and Pricing Improvement Handoff

- Date: 2026-07-23
- Status: Ready for implementation
- Audience: Coding agent extending the stabilized Daita MVP
- Scope: Conversation projection, catalog discovery efficiency, and model-cost
  estimation

## 1. Objective

Implement three related improvements to the current transcript-driven data
agent:

1. preserve useful conversation continuity when a prior analytical turn is
   larger than the fixed history projection bound;
2. reduce repeated, token-heavy catalog inspection while preserving catalog
   authority and query safety; and
3. replace fabricated `$0` model costs with auditable, provider-aware pricing
   estimates.

These are post-remediation product improvements. R1 through R12 in
`DATA_AGENT_MVP_REVIEW_REMEDIATION_HANDOFF_2026-07-23.md` are already
implemented; do not reopen or rework them unless a focused regression proves
that one of their accepted contracts was broken by this work.

The three improvements reinforce one another:

```text
compact catalog evidence
        ↓
smaller current-run transcripts
        ↓
more reliable follow-up projection
        ↓
fewer repeated model and catalog calls
        ↓
lower, measurable cost
```

The goal is not to maximize context usage. It is to preserve the smallest
deterministic evidence needed for accurate follow-ups, acquire structural facts
at the right granularity, and report the resulting cost honestly.

## 2. Architecture constraints

Follow `AGENTS.md` and preserve the current ownership model.

- `DataContextBuilder` owns bounded prior-run projection and request budgeting.
- `CatalogService` remains the sole owner of normalized current catalog truth.
- Catalog tool views remain presentations over registered catalog
  capabilities.
- Provider adapters own provider-native response and usage translation.
- Canonical usage and cost records belong under `daita.llm`.
- `AgentLoop` remains provider-neutral and owns no pricing tables or catalog
  behavior.
- Durable transcripts remain exact. Compact history is constructed at request
  time and is not written back to the transcript store.
- Do not introduce a conversation runtime, LLM-authored history summary,
  compression checkpoint, vector store, second catalog, pricing web service,
  middleware framework, or background refresh worker.
- Preserve all existing trust, source containment, SQL validation, output
  bounds, lazy imports, approval, and single-writer protections.
- Do not make paid model calls while implementing or validating this work.

## 3. Current evidence and failure mechanisms

### 3.1 Oversized analytical turns can erase all continuity

The reproduced conversation was:

1. inspect eight PostgreSQL tables and relationships;
2. calculate paid revenue, order count, AOV, gross profit, and gross margin by
   region; and
3. ask, “Now restrict that analysis to enterprise customers and compare it
   with the overall results.”

The prior analytical turn projected to approximately 46,554 UTF-8 bytes.
`src/daita/domains/data/context.py` currently applies:

```text
maximum completed runs = 8
maximum prior messages = 40
maximum prior UTF-8 bytes = 24,000
```

`_project_completed_history()` projects complete turns, iterates newest first,
and stops when the next whole turn exceeds any bound. Therefore an oversized
newest turn causes zero prior turns to be retained. The follow-up model receives
only an omission marker and asks the user to restate the metric, grouping, and
filters.

Raising 24,000 above 46,554 would repair this one transcript but would retain
the wrong material: repeated catalog snapshots, duplicated relationships, and
raw query output. The same failure would recur when the next analytical turn
crossed the new constant.

### 3.2 Catalog inspection is audit-rich but planning-inefficient

The schema question used eight parallel `catalog_inspect` calls. Parallelism
reduced elapsed time but did not reduce input tokens.

Each inspection includes:

- a complete resource payload;
- all typed facets;
- incident relationships;
- neighboring resource payloads;
- sync IDs, revisions, observation timestamps, provenance, and confidence; and
- selection facts.

Foreign-key relationships are repeated when both endpoints are inspected.
Most audit metadata is important inside the catalog but unnecessary for common
SQL planning.

Catalog search currently matches resource kind, name, native identity, and
external URI. It does not search column/key facts stored in facets, and its
hits do not provide enough compact structure to replace repeated inspection.

### 3.3 `$0` currently means “not calculated”

`ModelUsage.estimated_cost_usd` defaults to zero, and the OpenAI, Anthropic,
Gemini, and OpenAI-compatible adapters explicitly return zero. The terminal
renders that value as `$0`.

The two optional price fields on `ModelProfile` cannot express current provider
billing:

- uncached input, cache reads, and cache writes have different rates;
- providers disagree on whether reasoning/thinking is included in another
  token total;
- long-context thresholds may change the price of the entire request;
- batch, flex, priority, region, and data-residency modes can apply
  multipliers;
- provider aliases may resolve to a differently priced concrete model; and
- some providers charge separately for server-side tools, storage, images,
  audio, or execution time.

Unknown cost must not be represented as zero.

## 4. Accepted design decisions

| Area | Decision |
| --- | --- |
| History | Fix the projection unit before considering a larger limit |
| History | Construct deterministic compact history from exact transcripts; do not ask an LLM to summarize it |
| History | Guarantee the newest useful continuity projection before adding older or fuller history |
| History | Historical values remain stale context; follow-ups requery current sources |
| Catalog | Add one compact schema-slice projection at the existing catalog owner |
| Catalog | Keep full `catalog_inspect` for diagnostics and progressive disclosure |
| Catalog | Extend deterministic lexical search to structural facet facts; do not add embeddings or a search service |
| Pricing | Replace the scalar zero default with explicit priced, partial, and unavailable states |
| Pricing | Keep stable capability facts separate from time-varying price schedules |
| Pricing | Provider adapters normalize exclusive billable quantities; a shared pure calculator applies reviewed schedules |
| Pricing | Persist the schedule and component breakdown used for each estimate |
| Pricing | Public pricing produces a list-price estimate, never a claim about the provider invoice |

## 5. Improvement A: fix history projection before raising the limit

### 5.1 Required outcome

A referential follow-up must retain the prior analytical contract even when the
full prior transcript exceeds 24,000 bytes.

For the reproduced enterprise follow-up, the model must be able to recover:

- paid/captured-payment scope;
- regional grouping;
- all-date scope;
- paid revenue and order-count definitions;
- AOV definition;
- tax-exclusive merchandise revenue;
- COGS and gross-margin definitions; and
- the need to compare enterprise with overall results.

It must then rerun current source queries. It must not treat historical result
values as current evidence.

### 5.2 Use a deterministic continuity projection

For each eligible completed run, construct a bounded projection with two
possible representations:

1. **Continuity representation**

   - the exact user message;
   - selected compact, complete historical tool exchanges described below; and
   - the terminal assistant answer.

2. **Full representation**

   - the current projected transcript, including all retained tool exchanges,
     only when it is already small enough and the remaining budget permits it.

The continuity representation is the default for analytical turns. Full
catalog snapshots should not be replayed merely because there is available
context.

Do not persist either representation. Continue loading exact completed runs and
derive the projection inside the context owner.

### 5.3 Preserve only useful historical tool evidence

Every retained tool call must still have exactly one retained result, and calls
and results must retain their original order. It is valid to omit an entire
historical call/result pair. It is not valid to keep a call without its result
or a result without its call.

Use capability/evidence identity, not tool-name string heuristics, to select a
projection rule where the current records expose stable capability metadata.

| Historical evidence | Continuity projection |
| --- | --- |
| `catalog.search_result` | Omit; current catalog context is authoritative |
| `catalog.resource_snapshot` | Omit; current catalog context or a fresh schema slice is authoritative |
| `catalog.traversal_result` | Omit; rerun when the current relationship path matters |
| SQLite/PostgreSQL query result | Keep the exact normalized call arguments, plus result kind, success/error state, columns, row count, truncation state, and source/resource revisions when available; omit raw rows |
| File read result | Keep identity, revision/freshness, format, and bounded shape metadata; omit raw file content |
| Memory or skill reads/writes | Preserve the existing argument and body redaction rules; never turn historical authorization into current authorization |
| Side-effect or approval-related evidence | Omit from continuity; approvals are once-only and never carry forward |
| Unknown future evidence kind | Fail closed to user message plus terminal assistant answer; do not copy an unbounded unknown payload |

The compact result must be explicitly marked as a historical projection so the
model cannot confuse it with a fresh tool result.

The exact SQL call is useful procedural context, not evidence that its old
result remains true. The system prompt must continue stating that current
catalog and fresh tool results outrank historical messages.

### 5.4 Selection algorithm

Replace newest-turn starvation with this order:

1. Determine eligible completed runs exactly as today.
2. Build a continuity representation for the newest run.
3. Reserve and include that representation if any prior history can fit.
4. Add continuity representations for older runs newest first, within the run,
   message, byte, and whole-request bounds.
5. Optionally upgrade already selected small turns to their fuller
   representation, newest first, only when doing so adds useful evidence and
   stays inside every bound.
6. Add one omission marker when any completed history was not retained.

An oversized full newest turn must never cause the selector to break before
trying its continuity representation.

If even the terminal assistant answer is too large, use a deterministic,
explicitly marked bounded projection rather than silent truncation:

- retain the complete user message where possible;
- retain compact query receipts;
- retain bounded beginning and ending portions of the assistant answer; and
- insert an exact omission marker with original byte count.

Do not invent a semantic summary. Do not use a provider tokenizer or remote
counting call merely to create history.

### 5.5 Budget policy

Implement and evaluate the continuity representation while retaining the
existing 24,000-byte history ceiling. The reproduced follow-up must pass under
that ceiling before any increase is considered.

After the projection is correct, separately evaluate whether a larger or
model-relative ceiling improves quality enough to justify its cost.

The long-term request budget should remain:

```text
model maximum input
  - mandatory system/catalog/tool context
  - current-run messages
  - current-run growth reserve
  = space eligible for prior-run continuity
```

The fixed byte bound remains a provider-neutral safety limit. It must not be
used as a reason to replay raw catalog and row payloads.

### 5.6 Owners and likely files

- `src/daita/domains/data/context.py`
- `src/daita/llm/models.py` only if a small canonical historical receipt record
  is required
- `tests/test_conversation_mvp.py`
- `tests/test_architecture.py`

Do not change durable transcript persistence or add a conversation table.

### 5.7 Required tests

Add deterministic tests for:

- a newest completed turn larger than 24,000 bytes that still contributes its
  user request, compact query receipt, and final answer;
- the exact reproduced follow-up wording, proving that metric/group/filter
  context is present;
- removal of catalog snapshots and raw query rows from compact history;
- preservation of complete selected tool exchanges and rewritten historical
  call IDs;
- preservation of existing memory and skill redactions;
- no historical approval or authorization reuse;
- one omission marker when material is dropped;
- current-run messages remaining complete when history is reduced;
- old historical values not being inserted into current catalog context; and
- a small prior turn still being eligible for full projection when useful.

Add one live-model-marked acceptance contract only if it is opt-in and not
required for deterministic validation. Do not make paid model calls during
implementation.

### 5.8 Acceptance

- The oversized regional-analysis turn no longer erases all continuity.
- “Now restrict that analysis to enterprise customers” does not trigger a
  request to restate the prior analysis.
- The model receives enough exact procedural context to rerun and compare the
  metrics.
- Historical tool payload size is materially smaller than the exact durable
  transcript.
- The solution introduces no LLM summary, persisted projection, session
  runtime, or second loop.

## 6. Improvement B: make catalog discovery progressively disclosed

### 6.1 Required outcome

Common schema and query-planning questions should not require one verbose
inspection per table.

Target behavior:

| Prompt class | Target catalog behavior |
| --- | --- |
| “What tables and relationships are available?” | One compact schema projection |
| “Summarize paid revenue and margin by region” | One query-relevant schema slice, then the SQL query |
| Follow-up with unchanged catalog revisions | No repeated deep inspection |
| Freshness, containment, or connector diagnostics | Explicit full `catalog_inspect` |

Parallel inspection may remain supported, but it is not the optimization goal.
Reduce result volume and duplicated structural evidence.

### 6.2 Add a catalog-owned schema slice

Add a bounded schema projection to `CatalogService`, exposed through one
registered read capability and model-facing tool view. A suitable tool name is
`catalog_schema`; follow repository naming conventions if a better exact name
already exists when implementation begins.

This is a projection of the existing catalog, not a new schema graph.

The capability should accept:

- optional `query`;
- optional bounded `resource_ids`;
- optional `source_id`;
- bounded `limit`; and
- an option to include current relationships, defaulting to true.

Require at least a query or explicit resource IDs. Resolve all identities
against current active catalog truth before projection.

The output should contain:

- current source/snapshot or sync identities needed to detect staleness;
- resources with:
  - resource ID;
  - source ID;
  - kind;
  - schema-qualified/native name;
  - current revision;
  - columns with name, type, and nullability;
  - primary-key fields;
  - unique-key fields; and
  - bounded query-relevant structural facts;
- relationships emitted once, with:
  - relationship ID and kind;
  - endpoint resource IDs;
  - source and target field pairs;
  - connector provenance where relevant; and
  - endpoint revisions;
- total matches and truncation state; and
- `trust_classification: untrusted_external_data`.

Do not include every facet revision, observation timestamp, external URI,
neighbor copy, sync field, or audit property in the model projection. Keep that
truth in the catalog and continue exposing it through full inspection when
needed.

### 6.3 Deduplicate relationships

A relationship between two selected resources must appear once in the schema
slice, keyed by its stable relationship identity. Do not repeat it under both
resources.

When a relationship connects a selected resource to an unselected neighbor,
include a compact endpoint identity only when required to make the edge
understandable. Do not recursively expand the entire graph.

### 6.4 Improve structural search without adding a search system

Extend the current deterministic lexical search to include bounded structural
facet facts:

- column names;
- key/index field names;
- schema-qualified resource names; and
- relationship endpoint field names.

Maintain deterministic ranking:

1. exact resource-name match;
2. resource-name prefix;
3. resource-name containment;
4. exact structural-field match;
5. structural-field containment; and
6. bounded one-hop relationship neighbors.

One-hop expansion must carry a match reason such as
`relationship_neighbor`. It should fill remaining result capacity rather than
displacing stronger direct matches.

Do not add embeddings, synonyms presented as structural truth, a vector store,
or a separately persisted search index. The model may choose useful schema
terms; the catalog layer should not invent a business ontology for words such
as “margin.”

### 6.5 Integrate progressive disclosure into model context

Update catalog tool descriptions and the data system prompt so the intended
order is clear:

1. use current catalog context for identity candidates;
2. use `catalog_schema` for compact query planning;
3. use `catalog_traverse` for a bounded path question not answered by the
   slice; and
4. use `catalog_inspect` only for complete facets, freshness, containment, or
   diagnostics.

Do not rely on prompt wording alone. The compact capability and output contract
are the primary fix.

The initial `catalog_context()` projection may include a bounded count and
compact relationship hints among returned resources, but it must remain small.
Do not place full columns for an arbitrarily large catalog in every system
message.

### 6.6 Revision-aware reuse

The schema slice must identify the current source sync/revisions from which it
was projected.

A later turn may reuse the historical procedural shape only when current
catalog context shows the same identities. If the relevant source or resource
revision changed, acquire a fresh schema slice.

Historical schema still does not become catalog authority. The current catalog
comparison is what permits reuse.

### 6.7 Owners and likely files

- `src/daita/catalog/models.py`
- `src/daita/catalog/protocols.py`
- `src/daita/catalog/service.py`
- `src/daita/catalog/capabilities.py`
- `src/daita/storage/sqlite.py` for structural lexical matching at the current
  store boundary
- `src/daita/domains/data/context.py`
- `src/daita/hosting/embedded.py` for ordinary capability composition only
- `tests/test_catalog_summary.py`
- a focused catalog-capability test file if no current file cleanly owns the
  new contract
- `tests/test_postgresql_fixture.py` for the public vertical slice
- `tests/test_architecture.py`

### 6.8 Required tests

Add deterministic coverage for:

- a schema slice across the eight-table PostgreSQL fixture catalog;
- exact columns, PKs, unique keys, and FK field pairs;
- each relationship appearing exactly once;
- active/current source filtering after detach and refresh;
- structural search matching a column absent from the resource name;
- direct matches ranking above one-hop neighbors;
- strict resource/source scope;
- deterministic ordering independent of insertion order;
- explicit truncation at every bound;
- output-schema validation through `CapabilityRegistry`;
- no source connector I/O during schema projection;
- full inspection remaining available and unchanged for diagnostics; and
- catalog/tool content retaining its untrusted-data classification.

Add an acceptance assertion that the table-and-relationship question uses no
more than one catalog tool call with the deterministic fixture provider.

### 6.9 Efficiency gates

Record and assert bounded behavior, not exact provider token counts:

- catalog tool-call count;
- serialized catalog result bytes;
- number of duplicated relationship IDs;
- number of resources projected; and
- truncation state.

For the eight-table fixture:

- schema inventory should require one schema projection rather than eight deep
  inspections;
- relationship IDs should have no duplicates; and
- the compact projection should be materially smaller than concatenated full
  inspections.

### 6.10 Acceptance

- The model can plan the regional revenue/margin join from one current schema
  slice.
- Full `catalog_inspect` remains the progressive-disclosure path for audit-rich
  facts.
- Structural search finds relevant fields without a second catalog or vector
  index.
- Catalog authority, revision checking, validation, and source containment are
  unchanged.

## 7. Improvement C: implement honest, auditable pricing

### 7.1 Required outcome

The terminal must never display `$0` merely because no price was calculated.

Every run must report one of:

- a complete list-price estimate;
- a partial/lower-bound estimate;
- unavailable pricing; or
- a true zero only when an admitted price schedule explicitly produces zero.

Provider invoices remain authoritative for actual charges, discounts, credits,
tax, and negotiated agreements.

### 7.2 Separate usage, billable quantities, and price schedules

Keep three concepts distinct:

1. **Observed provider usage**

   Canonical informational counters such as input, output, reasoning, cache
   reads, and cache writes.

2. **Exclusive billable quantities**

   Provider-adapter-normalized quantities that can be multiplied without
   double counting, for example:

   - uncached input tokens;
   - cached-read input tokens;
   - cache-write input tokens;
   - billed output tokens;
   - separately billed reasoning tokens, only when the provider contract makes
     them exclusive;
   - tool invocations;
   - storage token-hours; or
   - execution seconds.

3. **Price schedule**

   Time-varying rates and threshold/multiplier rules keyed by exact provider,
   model, endpoint/service tier, region, and effective period.

`ModelProfile` should continue owning capability and hard-token facts. Remove
pricing from that stable profile or stop treating its existing two scalar
fields as authoritative.

### 7.3 Canonical cost result

Replace the unconditional decimal with a canonical cost record along these
lines:

```text
CostEstimate
  amount_usd: Decimal | None
  status: complete | partial | unavailable
  basis: public_list | configured_contract | provider_reported
  rate_schedule_id: str | None
  effective_at: datetime | None
  components: bounded tuple of CostComponent
  note/code: bounded machine-readable reason when incomplete
```

Each component should retain:

- billable quantity kind and amount;
- unit;
- applied unit rate;
- threshold or multiplier rule, if any; and
- component subtotal.

The representation must be bounded, immutable, JSON-serializable, and safe to
persist with the terminal run result.

Aggregation rules:

- all complete components/runs -> complete;
- known amount plus any unknown attempt/component -> partial, with the amount
  treated as a lower bound;
- no priced component -> unavailable;
- zero -> complete only when supported by an explicit schedule.

### 7.4 Provider-adapter responsibilities

Each provider adapter must:

- extract the actual response model identity when the provider returns it;
- extract the actual service tier, region, cache mode, or other billing
  dimensions returned by the provider;
- preserve provider usage counters;
- translate them into mutually exclusive billable quantities; and
- invoke the shared pure price calculator with those quantities and the
  applicable schedule.

Do not calculate provider cost in `AgentLoop`. Do not add provider branches to
routing or context construction.

Important normalization examples:

- OpenAI `output_tokens` already includes reasoning tokens for billing; do not
  add the reasoning subset again.
- OpenAI uncached input is total input minus cache-read and cache-write
  quantities when those fields are present.
- Anthropic exposes base input, cache-read input, and cache-creation input as
  separate provider fields.
- Gemini thinking-token treatment must follow the exact API/price contract for
  the selected endpoint; do not assume its counters share OpenAI semantics.
- OpenAI-compatible and custom providers remain unpriced unless exact pricing
  and usage semantics were explicitly configured.

Malformed or internally inconsistent usage must produce unavailable/partial
pricing or a normalized provider response error. Never clamp inconsistent
values into a plausible-looking price.

### 7.5 Price schedules

Add a small release-reviewed pricing owner under `src/daita/llm/`. A dedicated
`pricing.py` module is justified because multiple current provider adapters
need the same bounded records, effective-date selection, and arithmetic.

Each built-in schedule entry must include:

- exact provider/model identity;
- applicable endpoint or service tier;
- effective start and optional end;
- component rates;
- long-context threshold rules;
- region/data-residency modifiers where supported;
- source URL;
- review date; and
- stable schedule ID.

Seed schedules only when exact public pricing and usage semantics are
documented. Unknown is safer than a guessed match.

Official sources reviewed for this handoff:

- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
- [OpenAI GPT-5.6 Sol model and long-context notes](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [OpenAI prompt-caching usage fields](https://developers.openai.com/api/docs/guides/prompt-caching#requirements)
- [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [xAI pricing](https://docs.x.ai/developers/pricing)

Do not scrape these pages at runtime. Public sites are not stable machine
interfaces, runtime fetching harms offline reliability, and public prices
cannot know a customer's negotiated contract.

Support explicit user-configured schedules for custom endpoints or negotiated
rates. Persist only rates and identifiers, never credentials or provider
billing-account data.

### 7.6 Effective dates and historical stability

Select the price schedule effective when each provider request was made.
Persist the applied schedule ID and component rates with the run result.

Do not recalculate an old run using today's prices when displaying history.
Price changes must affect new requests only.

The schedule table must support future-dated changes and closed effective
periods. This is required for documented temporary or introductory provider
pricing.

### 7.7 Aliases, routing, retries, and failed attempts

- Prefer the concrete response model for pricing.
- If the request used an alias and the provider does not return a concrete
  billable model, mark pricing unavailable unless the alias has an exact
  schedule contract.
- Price each routed provider response independently before aggregation.
- Preserve usage from every completed attempt.
- If an attempted request may have been billed but returned no usable usage,
  mark the aggregate partial rather than silently omitting the uncertainty.
- A failed request with authoritative zero-billing documentation may remain
  complete, but do not assume all failures are free.

Routing still owns retry/fallback decisions. It may aggregate attempt cost
records, but it must not contain provider rate tables.

### 7.8 Cost limits

`max_estimated_cost_usd` cannot be enforceable when any eligible route candidate
is unpriced.

Required behavior:

- reject a configured cost limit before execution when the route lacks complete
  pricing;
- compare the complete aggregate after every provider response;
- preserve token, step, and wall-time limits independently; and
- describe the cost limit as an estimate-based post-response bound unless a
  conservative preflight bound is later designed.

Do not treat unavailable cost as zero for limit enforcement.

### 7.9 Terminal and observation presentation

Examples:

```text
$0.1769 estimated at public list rates
≥$0.1200 estimated; some attempts were unpriced
cost unavailable
provider API charge $0; local compute not estimated
```

Use `$0` only for a complete explicit zero.

Add cost status, basis, and schedule ID to the existing observation payload in
a bounded form. Observation remains best effort and non-directive.

### 7.10 Reproduced-cost sanity check

Using the persisted usage from the three reproduced turns, if the provider was
GPT-5.6 Sol on the standard direct-OpenAI tier, the current public rates applied,
and cache-write tokens were zero, the combined list-price estimate is
approximately `$0.34`, not `$0`.

This is a test fixture/sanity check with explicit assumptions, not an invoice
claim. Add deterministic arithmetic tests from synthetic usage rather than
making live calls.

### 7.11 Owners and likely files

- `src/daita/llm/models.py`
- `src/daita/llm/pricing.py`
- `src/daita/llm/profiles.py` for reviewed identity linkage, not arithmetic
- `src/daita/llm/providers/openai.py`
- `src/daita/llm/providers/anthropic.py`
- `src/daita/llm/providers/gemini.py`
- `src/daita/llm/providers/openai_compatible.py`
- `src/daita/llm/providers/grok.py` where it has distinct response semantics
- `src/daita/llm/providers/ollama.py`
- `src/daita/llm/routing.py` for aggregate completeness only
- `src/daita/loop/driver.py` for provider-neutral cost aggregation/limits only
- `src/daita/storage/sqlite.py` for canonical run-result encoding
- `src/daita/cli.py`
- `src/daita/terminal.py`
- `tests/test_loop.py`
- `tests/test_routing.py`
- `tests/test_observation.py`
- focused provider translation tests
- a focused deterministic pricing test file
- `tests/test_architecture.py`

### 7.12 Required pricing tests

Add deterministic coverage for:

- complete uncached input/output pricing;
- cache-read and cache-write rates;
- reasoning/thinking without double counting;
- long-context threshold applying to the complete request where documented;
- standard versus batch/flex/priority schedules;
- regional/data-residency multipliers;
- exact effective-date boundary selection;
- actual response model taking precedence over an ambiguous alias;
- custom/unreviewed model with no schedule -> unavailable;
- explicit configured custom schedule -> complete;
- failed retry without usage -> partial aggregate;
- multiple routed complete attempts -> exact aggregate;
- persisted historical estimate remaining stable after the current schedule
  changes;
- cost limit rejecting an unpriced route;
- `$0` displayed only for a complete explicit zero;
- no secret or billing-account data in persisted records or diagnostics; and
- malformed usage failing closed.

Use `Decimal` throughout rate and subtotal arithmetic. Do not use binary float.

### 7.13 Acceptance

- No live provider adapter fabricates zero cost.
- Terminal output clearly distinguishes complete, partial, and unavailable
  pricing.
- Every complete estimate identifies the exact applied schedule and component
  arithmetic.
- Provider-specific token semantics end inside provider adapters.
- Historical cost does not change when a new price schedule is released.
- Cost limits never treat unknown as free.

## 8. Implementation order

Use focused vertical slices:

1. **History correctness**

   - add compact continuity projection;
   - pass the oversized-turn deterministic regression under the existing
     24,000-byte bound; and
   - confirm trust/redaction invariants.

2. **Catalog schema slice**

   - add the catalog-owned projection and capability;
   - add structural lexical search and deduplicated relationships; and
   - update tool guidance and fixture-provider acceptance.

3. **Pricing semantics**

   - introduce complete/partial/unavailable cost records;
   - make all existing adapters return unavailable instead of zero before
     adding schedules; and
   - update terminal, persistence, aggregation, and observations.

4. **Provider pricing schedules**

   - add reviewed schedules one provider at a time;
   - normalize exclusive billable quantities; and
   - add threshold, cache, tier, and effective-date tests.

5. **Efficiency and completion audit**

   - compare tool calls and serialized evidence against the reproduced
     baseline;
   - run full deterministic/static gates; and
   - inspect the final diff for architecture expansion.

Do not combine all three areas into one refactor. Each slice must be independently
testable and reviewable.

## 9. Validation

At minimum, run focused deterministic tests for the touched owners, then:

```bash
.venv/bin/python -m pytest tests/ -m "not requires_llm and not requires_db"
.venv/bin/python -m black --check src tests
.venv/bin/python -m mypy src/daita tests
npx --yes pyright@1.1.411 --pythonpath .venv/bin/python
git diff --check
```

Run the existing PostgreSQL fixture acceptance only against the explicitly
started loopback fixture:

```bash
docker compose -f tests/fixtures/postgresql/compose.yaml up -d --wait
DAITA_RUN_POSTGRES_FIXTURE=1 \
DAITA_FIXTURE_POSTGRES_PASSWORD=daita_fixture_password \
.venv/bin/python -m pytest tests/test_postgresql_fixture.py -v
docker compose -f tests/fixtures/postgresql/compose.yaml down
```

The fixture provider must remain deterministic. Do not substitute a paid model
call for a missing deterministic assertion.

## 10. Non-goals

Do not implement:

- a larger history constant as the first or only history fix;
- LLM-authored conversation summaries;
- persisted history compaction/checkpoints;
- provider-native conversation state as the canonical Daita history;
- a second schema graph or catalog cache;
- embeddings or vector retrieval for catalog facts;
- source connector calls from catalog projection;
- runtime scraping of pricing websites;
- claims of invoice-exact pricing;
- a provider-specific pricing branch in `AgentLoop`;
- a generic billing/telemetry platform;
- automatic cost-based model switching; or
- background price-refresh infrastructure.

## 11. Completion checklist

Before handoff, prove:

- [ ] The reproduced oversized prior turn contributes useful continuity under
      the current 24,000-byte history bound.
- [ ] Historical catalog snapshots and raw rows are absent from compact
      history.
- [ ] Retained historical tool exchanges are complete and ordered.
- [ ] Current source queries are rerun for referential analytical follow-ups.
- [ ] One compact schema slice replaces eight deep inspections for the fixture
      inventory question.
- [ ] Schema-slice relationships are deduplicated and revision-scoped.
- [ ] Structural search covers columns and relationship fields without a new
      search subsystem.
- [ ] Full catalog inspection remains available for diagnostics.
- [ ] No provider adapter reports fabricated zero cost.
- [ ] Complete, partial, unavailable, and explicit-zero states render
      distinctly.
- [ ] Complete estimates retain schedule IDs, effective dates, and component
      arithmetic.
- [ ] Cache, reasoning, tier, region, long-context, retry, and alias semantics
      have focused deterministic coverage where supported.
- [ ] Cost limits reject unpriced routes.
- [ ] Existing R1-R12 remediation tests remain green.
- [ ] Full deterministic, Black, mypy, Pyright, architecture, and diff checks
      pass.
- [ ] No paid model call, new runtime, second catalog, vector store, pricing
      service, or background worker was introduced.
