# Live MVP Wave 1 failure analysis — 2026-07-20

Status: **root-cause analysis complete; no repair or live rerun performed.**

This record analyzes the first real-provider execution of LIVE-MVP-01 through
LIVE-MVP-04. It separates the observed terminal cause from latent failures that
the test would encounter after recovery. It does not promote LLM-G02, authorize
Phase 10, change a prompt, relax an oracle, increase a budget, or claim launch
readiness.

## 1. Evidence analyzed

- Provider/model: OpenAI / `gpt-4.1-mini`
- Interpreter: CPython 3.11.15
- Prompt corpus: `wave1-prompts-v1`
- Fixture: `wave1-commerce-v1`, digest
  `sha256:a313d61d1e4dc0411b02e3d3314ba49b18dc238ea62621f1d77b9f5b4f01439c`
- Result: **12 failed, 0 passed, 0 skipped in 340.079 seconds**
- Provider calls: 74 completed; zero provider errors, retries, or fallbacks
- Usage: 273,205 input tokens, 11,727 output tokens, 284,932 total
- Persisted work: 47 tasks and 47 accepted evidence records
- JUnit: `/private/tmp/daita-live-mvp-wave1.xml`
- Retained synthetic Agent Homes:
  `/private/var/folders/gd/x18f67x50kg8mj6k5hplr0k80000gn/T/pytest-of-jendala/pytest-3959/`
- Credential and session-sentinel scans: passed

The fixture/oracle values were independently checked. LIVE-MVP-01 expects four
active customers and 16,200 net cents ($162.00). LIVE-MVP-02 expects Europe and
12,500 net cents ($125.00). There is no evidence of fixture drift, credential,
network, quota, OpenAI response-normalization, retry, or fallback failure.

## 2. Executive verdict

The suite does not have one root cause, and classifying all 12 rows as “model
behavior” would be incorrect.

The retained run proves:

1. **Framework defects directly caused at least two rows.** LIVE-MVP-01
   conversational lost the decisive 25-cent refund and its evidence ID during
   context projection. LIVE-MVP-02 conversational was blocked by false-positive
   SQL validation of valid CTE-derived columns.
2. **Framework interface defects amplified most other model mistakes.** The
   model was shown impossible adapters and write/file tools, source adapter
   identity was omitted, mismatch feedback was misleading, exact validation
   errors were discarded, recent observations were starved by old ones, and
   citation repair lacked a copyable accepted citation.
3. **The model also made genuine semantic and instruction-following errors.**
   Examples include querying gross order-item revenue instead of net paid
   revenue, failing to inspect before guessing schema, selecting folders as
   files, and repeatedly using invalid fields.
4. **All three LIVE-MVP-04 rows failed because of the test configuration, not
   because session compression failed.** The test replaced the frozen
   128,000-token profile with a 5,000-token window that could not hold an
   ordinary current-operation tool exchange. No row reached compression or
   cold reopen.
5. **Several test assertions are invalid or cannot prove their stated claim.**
   The shared helper forbids successful repair history; the graph assertion is
   an out-of-band store query; the comparison oracle and strict comparator do
   not share a type/null contract; positional citation and exact task-count
   checks can reject valid behavior; failed rows emit no required metrics.

LLM-G02 therefore remains **FAIL**. This run is valid failure evidence, but it
is not a clean measurement of the reference model’s autonomous ceiling until
the proven framework and evaluator defects are corrected.

## 3. Row-by-row root-cause matrix

“Primary” means the decisive cause of the observed terminal state. “Latent”
means the row would still fail a later assertion even if it recovered from the
observed terminal cause.

| Row | Observed terminal path | Primary attribution | Contributing or latent cause |
| --- | --- | --- | --- |
| MVP-01 direct | `repair_budget_exhausted`; missing-column guesses, invalid inspect arguments, skipped parallel calls; only one SQL task ran | Model behavior | Framework over-projected tools, hid source routing and exact column/argument corrections; test would reject retained repair observations and absent explicit discovery |
| MVP-01 conversational | Correct joined query ran, but final said four / $162.25 and lacked a usable citation; readiness repair exhausted | **Framework context projection** | The persisted result contains the missing 25-cent refund, but the model-visible projection truncates before it and before the evidence ID; the next readiness correction is only `Runtime correction: …` |
| MVP-01 answerable-ambiguous | `repair_budget_exhausted`; repeated `resource_id` on SQL, missing `source_id`, then table IDs sent to file reads | Model behavior, framework-amplified | Non-strict calls, impossible tool surface, opaque source identity, and generic invalid-argument feedback made correction blind |
| MVP-02 direct | Five failed citation repairs; answer remained Europe / $150 gross | Model behavior | Query omitted payments/refunds and computed gross order-item revenue; readiness gave no literal accepted citation; even a format fix would fail the $125 oracle |
| MVP-02 conversational | `no_progress_action_failure_limit` after the same CTE query was rejected twice | **Framework SQL validator** | Valid CTE output aliases were checked against unrelated base schemas; feedback exposed only issue codes. The query also contained model-level status/date assumptions that would need normal repair after validation |
| MVP-02 answerable-ambiguous | `turn_budget_exhausted` immediately after an exact Europe / 12,500 aggregate executed | Mixed model/framework | The model spent turns on wrong adapter and invented schema. Newest observations had degraded to ellipses, and no model-visible graph tool existed. Raising the turn budget alone would not make synthesis reliable |
| MVP-03 direct | `repair_budget_exhausted`; folder treated as a file, later PostgreSQL plus invalid fields; zero tasks | Mixed framework/model | Singular `customer` search omitted plural `customers-*` files; every adapter/tool was exposed; model ignored search/inspect and invented columns |
| MVP-03 conversational | Catalog searches and file reads succeeded; four DB calls repeated undeclared `resource_id`; no DB/comparison evidence | **Framework repair contract**, with model contribution | Validator knew `unexpected field: resource_id` but controller replaced it with a generic error; PostgreSQL was exposed for SQLite; model skipped file inspections and reread the same file |
| MVP-03 answerable-ambiguous | `repair_budget_exhausted`; folder, revision hash, and DB table IDs were all attempted as files; zero tasks | Mixed framework/model | Natural retrieval hid files and file validation collapsed distinct kind/source/identity mistakes into one error; model never searched |
| MVP-04 direct | Isolation prelude succeeded through a SQLite query, then `context_build_failed` | **Test fixture** | Required context exceeded available context by 176 tokens; parameterized primary prompt never ran |
| MVP-04 conversational | Isolation completed; first primary turn executed four valid parallel reads, then `context_build_failed` | **Test fixture** | Required context exceeded available context by 827 tokens; there was no primary-session history yet, so compression was not involved |
| MVP-04 answerable-ambiguous | Isolation prelude reached a successful query, then `context_build_failed` | **Test fixture** | Required context exceeded available context by 184 tokens; parameterized primary prompt never ran |

## 4. Proven framework defects

### F-01 — Current-operation context destroys recency and essential evidence

Owner: `DataContextBuilder` in `src/daita/domains/data/context.py`.

`_operation_blocks()` has one 12,000-character observation budget, walks model
calls and observations oldest-first, and truncates the whole canonical
observation JSON. Old exploratory rows consume the budget. New accepted rows,
error details, readiness corrections, and `evidence_id` then become ellipses.

The LIVE-MVP-01 conversational database proves direct causation:

- authoritative evidence contains the final row for payment 2008 with
  `refund_amount_cents: 25`;
- the request used for synthesis ends that row at
  `"payment_status":"suc…[truncated]"`;
- the evidence ID is absent from the request;
- the next request contains the readiness feedback only as
  `Runtime correction: …`.

The model therefore calculated $162.25 from the rows it could see instead of
the correct $162.00. This is not an arithmetic error on the visible context.

Required correction:

- allocate projection space newest-first while rendering complete exchanges in
  chronological order;
- preserve a non-truncatable compact envelope containing status/code,
  evidence ID, a ready-to-copy citation, truncation flags, and bounded safe
  repair details;
- compact or omit old payload samples before truncating the newest result;
- keep canonical tool-call/result grouping valid.

### F-02 — SQL validation produces false positives for CTE-derived columns

Owner: SQL validation in `src/daita/domains/data/sql.py`.

The lexical-scope logic records column ownership only when the resolved source
is a base `exp.Table`. CTE-derived sources are skipped. Later validation treats
derived aliases such as `cr.region_name`, `nr.net_paid_cents`,
`qp.amount_cents`, and `qr.total_refund_cents` as if they must exist on the
unrelated base resources and reports missing/ambiguous columns.

The retained LIVE-MVP-02 conversational SQL is structurally valid and should
have reached SQLite. Its business predicates still contain model errors, but
those are normal data-result repairs; the validator must not reject valid SQL
before execution.

Required correction:

- carry CTE projection schemas and base-resource lineage through lexical scope;
- validate derived aliases against the CTE projection;
- retain fail-closed mutation, unknown-table, unknown-derived-column, scope,
  nesting, and shadowing behavior.

### F-03 — Tool projection and source routing violate least surprise

Owners: `DataDomainController.tool_views()`, embedded composition, and catalog
projection.

Every ordinary operation receives eight tools: catalog search/inspect, SQLite
query, PostgreSQL query, two SQLite write tools, file read, and comparison.
This happens even for a SQLite-only read request. Catalog context exposes an
opaque `source_id` but no adapter mapping. All four scenarios contain wrong
PostgreSQL-on-SQLite attempts.

Mismatch feedback is also misleading. It sets `expected_adapter_id` from the
tool the model selected, so PostgreSQL-on-SQLite reports PostgreSQL as
“expected” instead of exposing that the source is SQLite.

Required correction:

- project only tools applicable to attached source adapters and current access
  mode; do not expose write tools during an ordinary read journey;
- expose trusted `source_id -> adapter_id` routing in catalog/search/inspect
  context;
- report both `source_adapter_id` and `selected_tool_adapter_id` with
  unambiguous names;
- preserve this projection through detach/reopen.

### F-04 — Repair feedback discards information the framework already knows

Owners: data-domain validation and readiness.

The capability validator produces safe exact messages such as
`unexpected field: resource_id` and missing required fields. The controller
catches them and returns only `data.invalid_arguments` plus the tool name. SQL
validation computes the missing column, resource, and close candidates, but the
controller emits only issue codes. File validation maps folder, revision hash,
wrong source, table, and absent resource to the same missing-resource result.

Readiness knows every accepted evidence ID but asks for an abstract
`[evidence:<id>]`. Because evidence IDs already start with `evidence-`, the
valid citation is the visually awkward
`[evidence:evidence-<suffix>]`. LIVE-MVP-02 direct tried
`[evidence-<id>]` and then `[evidence:<suffix>]` five times.

Required correction:

- preserve bounded `missing_fields`, `unexpected_fields`, `allowed_fields`,
  missing column, resource, and candidate details;
- distinguish file not found, resource-kind mismatch, revision-vs-resource,
  and source-adapter mismatch;
- include a literal copyable citation, for example
  `"citation": "[evidence:evidence-…]"`, in accepted observations and
  readiness repair;
- never let the compact repair envelope be consumed by old payload samples.

### F-05 — Catalog retrieval and graph/freshness are not usable by the model

Owners: catalog search/storage and catalog capabilities.

- Literal exact-token FTS means singular `customer` does not find plural
  `customers-*.csv`. Initial LIVE-MVP-03 context showed the folder and DB
  tables, but neither file, although the files and their exact `modified_at`
  facets were correctly cataloged.
- Search hits omit adapter identity and file freshness.
- The catalog store supports bounded graph traversal and persists field-pair
  relationships, but model-visible capabilities expose only search and inspect.
  Inspect returns resource/facets, not relationship edges.

Required correction:

- add bounded singular/plural or safe prefix recall;
- make folder-to-child recovery navigable and preserve source/kind bounds;
- expose the authoritative file `modified_at` needed to decide “newest,” either
  in search projection or through a clearly directed inspection path;
- expose bounded catalog-owned relationship traversal or incident edges with
  field-pair provenance. Do not duplicate graph ownership in the data runtime.

### F-06 — Parallel tool advertisement conflicts with fail-stop batching

Owners: model profile/provider boundary and generic loop.

The reference profile advertises parallel tools. The loop processes a batch in
order and, after one rejection, records every later call as
`action.skipped_after_rejection`. That behavior is safe for dependent or
side-effecting calls but wastes independent read-only calls and repairs.

This was an amplifier, not the sole root of a row. The contract should either
disable parallel emission for this domain or prevalidate/continue independent
read-only calls while retaining fail-stop behavior for writes and dependencies.

### F-07 — Context failures are not diagnosable from persisted state

Owner: generic loop context-build boundary.

`RequiredContextOverflow` already carries profile ID, required tokens,
available tokens, tool tokens, and output reserve. The loop catches every
context exception and persists only `context_build_failed`. Exact MVP-04
diagnosis required offline reconstruction.

Persist a typed bounded overflow code/facts. Do not persist raw exceptions or
secrets.

## 5. Proven test, evaluator, and artifact defects

### T-01 — LIVE-MVP-04 uses a nonviable profile and does not test compression

The harness freezes a 128,000-token reference profile with a 2,048-token output
reserve. The test replaces only the context window with 5,000. That leaves an
input limit of 2,952; eight tool schemas cost 762; only 2,190 tokens remain for
system, intent, session, and current-operation messages.

| Variant/stage | Required context | Available after tools | Overflow | Minimum window with current reserves |
| --- | ---: | ---: | ---: | ---: |
| direct isolation | 2,366 | 2,190 | 176 | 5,176 |
| conversational first primary operation | 3,017 | 2,190 | 827 | 5,827 |
| ambiguous isolation | 2,374 | 2,190 | 184 | 5,184 |

No row created a compression checkpoint. Direct and ambiguous failed before
their parameterized primary prompt. Conversational had no history in the
primary session when it failed. The run says nothing about referential
follow-ups, compression, reopen, or post-reopen correctness.

Restoring 128,000 avoids this overflow but raises the default compression
threshold to 94,464 tokens, so it will not reliably exercise compression.
Compression triggering must be decoupled from a false model window, using the
existing `SessionCompressionPolicy` owner or a split deterministic/live proof.
The unrelated isolation prelude must not prevent every primary variant from
running.

### T-02 — The shared inspectability helper forbids successful repair

`assert_inspectable_runtime_state()` requires every observation to be
successful and evidence-linked. The established loop contract intentionally
retains typed action-rejection/readiness observations, and deterministic
acceptance already proves a successful repaired operation with an
`action.rejected` event.

The helper must validate successful observations against accepted evidence and
validate failed observations against their typed repair/event contract. It must
not require a zero-repair trajectory unless zero repair is made an explicit
release policy.

### T-03 — LIVE-MVP-02 does not prove model graph use

After the operation closes, the test opens the store and calls `traverse()`
itself. This proves only that the fixture graph exists. It cannot prove the
model used graph relationships and would pass even if the model guessed every
join.

After a catalog traversal/relationship projection is model-visible, require a
persisted operation task/evidence/provenance record for the resolving path.

### T-04 — LIVE-MVP-03 oracle and comparator disagree on types and NULLs

The production comparator intentionally uses typed canonical JSON:

- CSV IDs are strings while SQLite IDs are integers, so raw keys do not match;
- string/NULL is a `type_mismatch`.

The live oracle coerces DB IDs to strings and DB NULL to `""`, while the result
normalizer accepts only `value_mismatch`. A straightforward successful compare
would therefore fail the oracle, and the uncoached model is implicitly expected
to invent both `CAST(id AS TEXT)` and `COALESCE(email, '')`.

This requires an explicit product-contract decision. The smallest safe option
is to retain strict value semantics, add actionable key-type compatibility
preflight/guidance, and make the oracle preserve NULL/type mismatch. If
null/empty equivalence is desired, it must be a declared comparison policy,
not hidden coercion in the test.

### T-05 — Several assertions overconstrain a valid trajectory

- M01, M02, and M04 require the last query evidence rather than the evidence
  that actually resolves the stated fact.
- M03 requires exactly one file read; two safe reads can still produce one
  correct comparison. Duplicate reads belong in an explicit efficiency ceiling.
- Requiring an explicit `catalog.search` task can reject a model that used the
  automatically selected catalog context and then inspected the right
  resources. Decide whether explicit search is a public invariant. If it is,
  enforce it in the runtime contract; otherwise assert catalog-context
  provenance plus inspection.
- The M03 file names and mtimes order the same way, so selecting by filename can
  masquerade as freshness reasoning. A future fixture should make metadata and
  lexical order disagree if metadata use is the claim being tested.

Assertions should identify evidence by payload/authority and claimed fact, not
by incidental list position or exact harmless read count.

### T-06 — Failed live rows lose required metrics

Metric properties are recorded only after every success assertion. All failed
rows therefore wrote none. The configured/default xUnit2 output is incompatible
with `record_property`; the JUnit contains zero `<properties>` elements. The
plan also requires a redacted JSON summary, which was not emitted.

MVP-04 additionally summarizes only the post-reopen operation, omits provider
metrics from `summarize_live_run()`, and cannot measure the reconstructed
provider with the current recorder.

Move failure-safe summary capture to finalization/hooks, emit the required
redacted sidecar, choose a compatible JUnit property strategy, and aggregate
all scenario operations.

## 6. Genuine model behavior that remains after framework attribution

The framework defects do not excuse every model decision:

- M01 direct guessed schema repeatedly instead of performing discovery and
  complete inspection.
- M01 ambiguous confused SQL, file, resource, and source arguments.
- M02 direct answered $150 gross order-item revenue while the prompt explicitly
  requested net paid revenue after refunds.
- M02 ambiguous spent nearly all turns exploring and correcting guessed schema.
- M03 direct/ambiguous did not search after the initial context failed to reveal
  files and confused folders/revisions/tables with file resources.
- M03 conversational skipped required file freshness inspections and repeated
  a file read.

The current `gpt-4.1-mini` run is therefore a useful baseline, but it is not yet
a fair launch-reference qualification. After framework/evaluator corrections,
run the unchanged corpus against the intended launch-quality model candidates
and freeze the reference from measured hard-pass reliability. Do not switch
models merely to hide known product defects.

## 7. What did not fail

- Runtime validation failed closed; wrong adapters, mutations/PRAGMA, missing
  columns, invalid file resources, and invalid arguments did not reach
  unauthorized I/O.
- Accepted evidence remained source/resource/revision-bound.
- Comparison readiness correctly refused unsupported prose when no comparison
  evidence existed.
- Provider calls, usage, routing, persistence, and synthetic artifact retention
  worked for the paths reached.
- No secret, sentinel, provider retry, fallback, or infrastructure defect was
  observed.

These are meaningful safety positives, but they do not offset zero behavioral
passes.

## 8. Prioritized repair and proof plan

No new framework owner is needed. Extend the current context, SQL validator,
catalog, data-domain controller, comparison, session-compression, loop, and
live-test owners.

### P0-A — Make the evaluator capable of producing a truthful result

1. Repair MVP-04 profile/compression setup and isolate the isolation check from
   each parameterized primary journey.
2. Make inspectability assertions repair-compatible.
3. Bind graph assertions to model-visible operation evidence.
4. Decide and encode the comparison key/null contract.
5. Resolve evidence by supporting payload, not last position; move duplicate
   reads to efficiency policy.
6. Emit failure-path JUnit plus redacted JSON metrics.

### P0-B — Fix directly proven framework contracts

1. Preserve newest observations, evidence IDs, citations, and repair details in
   current-operation context.
2. Add correct CTE projection/lineage validation.
3. Make tool projection source/access-aware and expose correct adapter routing.
4. Return bounded actionable argument, SQL, file, adapter, and citation repair.
5. Make natural catalog retrieval find the intended files and expose bounded
   relationship/freshness authority.
6. Persist typed context-overflow facts.

### P1 — Remove reliability and efficiency amplifiers

1. Reconcile parallel-tool advertisement with fail-stop batch semantics.
2. Add comparison key-type preflight or a declared normalization policy.
3. Coordinate session compression with residual whole-request capacity and
   compact model-visible observation payloads without changing authoritative
   evidence.
4. Define explicit call/token/duplicate-read ceilings only after the first
   correct repeated sample.

### Required deterministic regressions before another paid live run

- A large old observation plus a newest aggregate and readiness correction:
  newest rows, full evidence ID, literal citation, code, and details survive.
- The exact retained chained CTE validates; unknown derived columns, hidden
  mutation, nesting, recursion, and scope violations still fail closed.
- SQLite-only read journeys exclude PostgreSQL/file/write tools as applicable;
  mixed sources expose an exact source-to-adapter map; mismatch feedback is
  correct in both directions.
- Exact retained bad argument/file/citation calls receive actionable bounded
  corrections and recover within the frozen repair budget.
- Catalog retrieval finds both customer exports for all three natural variants;
  graph traversal exposes the exact refunds-to-regions path and field pairs
  through persisted operation evidence.
- Raw string/integer comparison keys receive a deterministic compatibility
  result; normalized keys and NULL semantics match the revised oracle exactly.
- A completed operation with earlier rejected actions passes the corrected
  inspectability helper; malformed observation/evidence correlation still
  fails.
- Four parallel read observations either fit the bounded projection or fail
  with typed numeric overflow facts.
- Session compression triggers under an explicit policy, survives cold reopen,
  and preserves objectives, evidence IDs, resource scope, and isolation.
- A deliberately failed fake live row emits complete JUnit/JSON metrics without
  credential or sentinel material.

Then run the full non-live suite and the unchanged 12-row live corpus once. A
semantic failure remains a failure; do not manually rerun it into a pass.

## 9. Launch decision boundary

Passing these 12 rows would pass only LLM-G02. It would demonstrate the core
data loop and session journey for Wave 1; it would not by itself establish full
MVP launch readiness.

Before a confident launch/cutover claim, the existing plan still requires:

- LLM-G03 stateful autonomy;
- LLM-G04 governance/security;
- LLM-G05 installed product surface;
- LLM-G06 failure/external/provider/PostgreSQL boundaries;
- LLM-G07 repeated reliability, cost, latency, and three consecutive complete
  Tier A runs on the deployment interpreter;
- LLM-G08 complete deterministic, packaging, architecture, security,
  migration/recovery, dual-Python, and root-isolation consolidation; and
- LLM-G09 explicit human authorization.

The next honest milestone is therefore: **correct the P0 framework and evaluator
contracts, prove them deterministically, then rerun the unchanged Wave 1 corpus
once.** Until every required gate passes, the project is not certified ready
for MVP launch or irreversible Phase 10 cutover.
