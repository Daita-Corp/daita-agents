# Live MVP Wave 1 explicit repair specification — 2026-07-20

Status: **normative implementation specification; implementation has not
started and no live rerun is authorized by this document.**

This document is the implementation companion to the
[Wave 1 failure analysis](LIVE_MVP_WAVE1_FAILURE_ANALYSIS_2026-07-20.md). The
failure analysis remains the immutable evidence and attribution record. This
specification resolves the repair choices and defines the contracts that a
coding agent must implement. The
[live LLM production-readiness plan](LIVE_LLM_PRODUCTION_READINESS.md) remains
the authority for release gates and paid rerun discipline.

If those documents appear to conflict:

1. preserve the retained evidence and attribution in the failure analysis;
2. use this specification for implementation behavior;
3. use the production-readiness plan for gate and launch requirements; and
4. stop and report a material conflict instead of choosing a local patch.

Creating this specification does not change
[`../STATUS.md`](../STATUS.md), [`../QUALITY_GATES.md`](../QUALITY_GATES.md),
LLM-G02, or Phase 10 eligibility.

## 1. Objective and scope

Repair the general framework and evaluator contracts exposed by the first
LIVE-MVP-01 through LIVE-MVP-04 run. The implementation must make the same
owners reliable for other resources, schemas, providers, prompts, and Agent
Homes; it must not encode the retained fixture or expected answers in
production behavior.

In scope:

- typed action and readiness repair facts;
- bounded model-visible observation projection;
- whole-request and session budget coordination;
- source-aware tool projection and trusted routing context;
- derived-relation SQL validation;
- catalog retrieval, hierarchy, freshness, and graph projection;
- explicit tabular comparison key/type/NULL semantics;
- request-level parallel-tool policy;
- typed context-build diagnostics;
- evaluator correctness and failure-safe live metrics; and
- deterministic proof followed by one unchanged live rerun.

Out of scope:

- Phase 10, package cutover, source-tree movement, publishing, or release;
- changes to the frozen root-v1 source, tests, or packaging;
- prompt coaching with tools, tables, SQL, IDs, citations, or action order;
- increasing turns, repairs, context windows, or observation limits to hide a
  contract defect;
- a second agent loop, catalog, policy engine, state store, evaluator
  framework, provider registry, or execution path;
- an LLM-as-judge or self-grading semantic verifier; and
- treating a single green 12-row run as complete MVP launch certification.

The Wave 1 prompts and fixture remain frozen. An oracle may change only where
this document corrects a proven evaluator contract, and that change must be
versioned, deterministic, and independently tested.

## 2. Existing owners and non-negotiable invariants

No new top-level framework owner is required.

| Contract | Existing owner to extend |
| --- | --- |
| Capability schema and model tool declaration | `Capability`, `ToolView`, `CapabilityRegistry` |
| Source identity and active configuration | `SourceRegistration` and the source store |
| Runtime authority, task execution, evidence acceptance | `OperationRuntime` and the existing kernel/executor boundary |
| Data validation, observation semantics, readiness | `DataDomainController` |
| Model request construction and context selection | `DataContextBuilder` and `select_context_blocks()` |
| Session history and compression checkpoints | `SessionCompressionService` |
| Structural resource, facet, hierarchy, and relationship truth | catalog models, service, capabilities, and store |
| SQL scope and read-only validation | `src/daita/domains/data/sql.py` |
| Tabular comparison semantics and artifacts | `src/daita/domains/data/comparison.py` |
| Provider request translation | existing provider adapters |
| Live scenario assertions, oracles, and metrics | `tests/live/mvp/` |

Every repair must preserve these invariants:

1. Persisted evidence, observations, transcripts, and checkpoints remain the
   authoritative immutable records. Compaction changes only a model-facing
   projection.
2. `OperationRuntime` remains the execution/evidence boundary. Context,
   catalog presentation, policy, and tests must not perform executor work.
3. Every committed model tool call has exactly one model-visible result, even
   when the call is rejected or skipped. Tool-call/result groups remain valid
   and are rendered chronologically.
4. Identity, disposition, citation, truncation, and the latest actionable
   correction cannot be silently removed from model context.
5. Validator-owned facts remain typed and bounded end to end; downstream code
   must not parse exception strings to reconstruct them.
6. Tool visibility is presentation, not authority. Hidden tools remain subject
   to registry, validation, governance, and executor checks if a malformed
   provider response names them.
7. Catalog structural, freshness, and relationship facts remain catalog-owned.
   The data runtime consumes them and does not build another graph.
8. A context request that cannot fit fails before provider I/O and persists
   safe numeric diagnostic facts.
9. Model profiles describe truthful provider/model capabilities. Per-request
   policy must not be encoded by falsifying a profile.
10. `Readiness.allowed` is a deterministic response-contract decision, not a
    certificate that generated prose is semantically true.
11. Cold reopen reconstructs every behavior-affecting route, default, source
    fact, projection policy, and checkpoint needed for the next operation.
12. Tests assert outcome, authority, and provenance separately from efficiency.
    A safe repaired trajectory is not a failure merely because it contains a
    typed rejection.

## 3. Resolved decisions

The implementation must not reopen these choices without an explicit design
review.

| ID | Decision |
| --- | --- |
| D-01 | Model-visible observations use a mandatory envelope and optional structured body; whole JSON strings are never character-sliced. |
| D-02 | Capability-input and readiness failures carry typed bounded details from their originating owner. |
| D-03 | Tool applicability is declared on model tool views and evaluated against active persisted source registrations; it is never inferred from prompt prose or tool names. |
| D-04 | Source routing is a separate trusted-runtime context block, not catalog data. |
| D-05 | SQL validation models every supported nonrecursive derived relation, not only the retained CTE query. Recursive CTEs remain explicitly unsupported and fail closed. |
| D-06 | Catalog search uses bounded exact-token plus safe prefix recall with exact matches ranked first; no English-only singular/plural special cases are permitted. |
| D-07 | `catalog_inspect` exposes bounded incident relationships and kind-appropriate selection facts; a new catalog-owned traversal capability exposes bounded paths with field-pair provenance. |
| D-08 | For local files, “newest” means the greatest current `FileFacet.modified_at`; equal or missing values are reported as ambiguous, never resolved by filename. |
| D-09 | Comparison keys require an explicit `strict` or `stringify_integral` normalization mode. Compared values remain strict, and NULL is distinct from an empty string. |
| D-10 | Data-domain requests disable parallel tool calls at the provider request boundary for MVP. Profile capability remains truthful and loop fail-stop behavior remains defense in depth. |
| D-11 | Session compression receives the exact residual whole-request allowance. `retain_latest_operations` is a maximum, not a floor. |
| D-12 | `data.ready` is replaced by wording/codes that state only that citation, authority, comparison, and disclosure requirements passed; semantic correctness stays in the hard live oracle. |
| D-13 | The live harness records metrics in one failure-safe finalization path and emits both reviewable test output and the required redacted JSON sidecar. |

## 4. Contract and persistence foundation

Implement shared contract and compatibility changes before behavior changes.
Use the next sequential SQLite migration—migration 18 at the time this document
was written—for schema changes that must be atomic. Do not scatter one repair
across several migration-only special cases.

Required compatibility behavior:

- Add bounded structured readiness repair details to the canonical `Readiness`
  record and persist them. Existing readiness rows decode with empty details;
  no historical decision is reinterpreted.
- Add a provider-neutral `allow_parallel_tool_calls: bool | None` to
  `ModelRequest`. `None` means the legacy/provider-default behavior and is used
  only when decoding an older persisted request or by a domain that has not
  declared a policy. Every new data-domain request sets `False`. Persisted
  started requests replay exactly.
- Add a versioned, restart-stable session compression policy to
  `AgentRuntimeDefaults`/`AgentConfig`. An absent explicit threshold means the
  current deterministic profile-derived default. Existing homes migrate to
  that behavior; new test homes may bind an explicit lower threshold. The
  complete policy participates in the runtime-default fingerprint.
- Add declarative tool applicability to `ToolView` and include non-default
  applicability in extension declaration fingerprints. Omit empty/default
  applicability from fingerprint material so existing source-agnostic
  extension bindings retain their exact prior fingerprint.
- Bump capability output schema versions when catalog or comparison evidence
  shapes change. Existing accepted evidence remains immutable and inspectable;
  it is never upgraded in place or assigned facts it did not contain.
- Extend model-request decoding rather than assuming every retained request was
  written by the new schema.
- Persist typed context-overflow facts through the existing event/failure
  record. Do not add a diagnostic database.

Before adding any column or field, add round-trip, legacy-decode, corruption,
and cold-reopen tests. A migration may preserve a prior default; it may not
invent source authority, evidence, approval, or a model request policy that was
not previously recorded.

## 5. RP-01 — Typed action and readiness repair

### Owners

`CapabilityInputError`, capability schema validation, SQL/file/source
validation, `ActionRejection`, `Readiness`, `DataDomainController`, and
`OperationRuntime`.

### Required implementation

`CapabilityInputError` must become a typed bounded error rather than a bare
message. Its safe projection must support:

- stable violation code;
- `missing_fields`;
- `unexpected_fields`;
- `allowed_fields`;
- `allowed_values` for a declared finite string enum;
- `field_path`;
- `expected_type`;
- `actual_type`; and
- an explicit details-truncated flag.

Field collections are sorted, deterministic, deduplicated, and bounded. Raw
invalid values are never echoed.

Extend the existing capability property-schema subset with a bounded string
`enum` keyword. Validate it in `Capability` construction and
`CapabilityRegistry.validate_arguments()`, and expose the same enum in the
model ToolDefinition. This is an extension of the existing schema owner, not a
second JSON-schema engine. Use it wherever this specification declares a
closed string policy, including `key_normalization`.

`DataDomainController.validate_action()` must:

- catch `CapabilityInputError` separately and transfer its code/details into
  `ActionRejection` without parsing `str(error)`;
- retain a generic safe rejection only for genuinely unclassified malformed
  input;
- project the bounded primary `SqlValidationIssue.details`, including column,
  resource, scope, and close candidates;
- distinguish file-not-found, resource-kind mismatch,
  revision-instead-of-resource, wrong source, and source/tool applicability;
- report `source_adapter_id`, selected capability/tool, and the declared
  applicable adapter IDs with unambiguous names; and
- disclose only facts already visible within the current agent/source scope.

`Readiness` gains bounded `repair_details`. A denied citation decision includes
literal evidence references:

```json
{
  "required_citations": [
    {
      "evidence_id": "evidence-...",
      "citation": "[evidence:evidence-...]"
    }
  ]
}
```

`OperationRuntime.record_readiness()` persists those details and forwards them
unchanged into the correction observation. It does not interpret, regenerate,
or shorten citations.

### Deterministic acceptance

- Missing, unexpected, and wrong-type arguments produce distinct stable codes
  and exact bounded facts.
- Supplying `resource_id` to a SQL capability identifies that field as
  unexpected and lists the actual allowed fields.
- A missing SQL column exposes its resource and bounded candidates.
- Every file/source identity failure has a distinct code.
- Copying a supplied literal citation unchanged satisfies citation syntax.
- Details survive SQLite round-trip, cold reopen, and model projection.
- Oversized details are structurally bounded with an explicit flag.
- No serialized repair contains a traceback, exception class, path outside
  admitted source metadata, raw invalid value, credential, or sentinel.

### Prohibited shortcuts

- Parsing exception strings.
- Per-prompt or per-retained-call error branches.
- Adding repair hints only to system prompts.
- Echoing the submitted argument object.
- Changing evidence IDs to make citations prettier.

## 6. RP-02 — Canonical model-visible observation projection

### Owners

The existing `Observation` record, `DataDomainController.project_observation()`,
and `DataContextBuilder._operation_blocks()`.

### Required implementation

Do not add another persisted observation model. Define a versioned model-facing
projection of the existing record with this logical shape:

```json
{
  "schema_version": 2,
  "code": "...",
  "success": true,
  "message": "...",
  "call_id": "...",
  "task_id": "...",
  "evidence": {
    "id": "evidence-...",
    "citation": "[evidence:evidence-...]"
  },
  "source_truncated": false,
  "projection_truncated": false,
  "repair_details": {},
  "body": {}
}
```

Rules:

1. The envelope—schema version, code, success, message, identities, evidence
   reference, truncation flags, and bounded repair details—is mandatory.
2. Every accepted evidence observation contains its exact evidence ID and
   literal citation. Citation truth is derived from `Evidence.id` once.
3. Successful domain data belongs in the optional structured `body`.
4. Failed observations put their typed facts in `repair_details`; they do not
   masquerade as data bodies.
5. Evidence-kind-specific structural compaction belongs to the data-domain
   projection. The generic context builder must not know about refunds,
   columns, file names, or business fields.
6. A compacted tabular body remains valid JSON and declares row counts, sample
   strategy, omitted counts, and truncation. It is never clipped in the middle
   of a string.
7. `DataContextBuilder` reserves all envelopes first, allocates optional body
   capacity newest-first, and then renders complete exchanges chronologically.
8. The latest actionable correction and newest accepted observation receive
   body/detail capacity before older bodies.
9. Older bodies may be structurally compacted or omitted; their envelopes
   remain present.
10. Assistant/tool exchange groups remain indivisible and every call receives
    exactly one result block.
11. If envelopes alone cannot fit, raise typed `RequiredContextOverflow`
    before provider I/O. Do not erase identities to force a request through.
12. Authoritative `Evidence.payload`, content hashes, observations, and blobs
    remain unchanged.

Replace the current whole-string character slicing. Character limits may
remain as defensive maximums, but allocation uses the shared token estimator
and produces structurally valid projections.

### Deterministic acceptance

- A large old result followed by the retained aggregate/refund result preserves
  the newest complete body, the 25-cent value, full evidence ID, literal
  citation, code, and both truncation flags.
- A readiness correction following large evidence retains its literal
  citation and typed missing facts.
- Multiple older bodies compact before any newer body.
- Rendered exchanges remain chronological and provider-valid.
- Envelope-only overflow produces zero provider calls and exact numeric facts.
- The same tests use unrelated field names and multiple evidence kinds so a
  fixture-specific preservation rule cannot pass.
- Evidence hashes and blob identities are byte-for-byte unchanged.

### Prohibited shortcuts

- Merely reversing the loop over observations.
- Raising `max_observation_characters` or the model window.
- Preserving a named refund row or final array element specially.
- Truncating serialized JSON text.
- Copying evidence into a second context store.

## 7. RP-03 — Declarative tool applicability and trusted source routing

### Owners

`ToolView`, extension declaration fingerprints, `SourceRegistration`,
`DataDomainController.tool_views()`, `DataContextBuilder`, and the existing
source/catalog data view.

### Required implementation

Extend `ToolView` with bounded declarative applicability containing:

- applicable source adapter IDs;
- minimum active source count; and
- required boolean source-configuration flags.

Use one immutable `ToolApplicability` value owned by `ToolView`, with empty
adapter/flag tuples and `minimum_active_sources=0` as its exact default. Do not
add parallel applicability fields directly to individual adapter controllers.

Empty applicability means a global view. Applicability is presentation
metadata and participates in extension declaration fingerprints. Every listed
configuration flag means that at least one matching active source must contain
that exact flag with the boolean value `true`; arbitrary configuration values
are never interpreted as truthy.

Declare the built-in views as follows:

- catalog search, inspect, and traversal: at least one active source;
- SQLite query: at least one active SQLite source;
- PostgreSQL query: at least one active PostgreSQL source;
- local-file read: at least one active local-directory source;
- SQLite update capabilities: at least one active SQLite source whose admitted
  registration has `write_access=true`;
- tabular comparison: at least two active sources; accepted-evidence
  prerequisites remain runtime validated.

Adapter names and configuration flags appear only in declarations and source
registrations, never in prompt classifiers or `if tool_name == ...` routing.

Change the generic `DomainController.tool_views()` protocol to async so the
data controller can read active durable source registrations at each new model
request. The foreground mutation lock keeps attach/detach serialized with a
running operation. A replay of an already-started model call uses the exact
persisted request rather than recomputing tools.

For a new turn, project only registered views whose declared requirements are
satisfied. A model-emitted hidden tool still passes through ordinary registry,
scope, governance, and adapter validation and fails closed.

All model-facing catalog search, context, inspection, and traversal projections
exclude detached sources. Exact historical evidence/resource reads remain
available only to their existing audit and provenance owners; detach does not
rewrite history.

Add a dedicated `ContextKind.SOURCE_ROUTING` and project one small required
`TRUSTED_RUNTIME` source-routing block, separate from
`UNTRUSTED_CATALOG_CONTEXT`. It contains only bounded admitted control facts:

- active `source_id`;
- `adapter_id`;
- declared true/false applicability flags needed by current ToolViews.

Never project raw source configuration, connection strings, paths beyond
already admitted public source metadata, or secret references. Catalog search
and inspect may refer to `source_id`, but they do not become the authority for
adapter routing.

The request's tool definitions remain the sole model-facing list of available
tools; do not duplicate that list or reimplement applicability inside the
source-routing block.

Rewrite static data system instructions so they describe only generic use of
the tools actually projected. They must not name unavailable adapters or write
tools.

The existing monitor-specific ToolView restriction is retained and intersected
with declared source applicability; neither rule may broaden the other.

### Deterministic acceptance

- SQLite-only read configuration projects catalog and SQLite query tools, not
  PostgreSQL or local-file tools.
- Read-only SQLite hides update tools; write-enabled SQLite shows them without
  changing runtime governance.
- A PostgreSQL-only and file-only home each project only applicable views.
- Mixed sources expose the exact trusted source-to-adapter map.
- A detached source removes its views on the next operation; cold reopen
  reconstructs the same projection.
- An arbitrary test extension with a new adapter ID works only through its
  declaration—no data-controller edit is allowed.
- Wrong-adapter repair reports both the actual source adapter and selected
  capability applicability.
- A malicious hidden-tool call fails before executor I/O.
- Static instructions never mention a tool absent from `ModelRequest.tools`.

### Prohibited shortcuts

- Prompt-keyword classification of read versus write intent.
- Hardcoded tool-name filtering in the data controller.
- Treating ToolView filtering as runtime authorization.
- Putting trusted adapter mappings inside untrusted catalog data.
- Caching source state without detach/reopen invalidation.

## 8. RP-04 — Scope-aware derived-relation SQL validation

### Owner

The existing lexical SQL analyzer and validator in
`src/daita/domains/data/sql.py`.

### Required implementation

Replace the base-table-only column map with a lexical environment for every
supported relation in each `sqlglot` scope. Each relation entry carries:

- exposed projection columns;
- qualifier/alias identity;
- base-resource lineage;
- scope identity; and
- whether the relation is a base table, CTE, subquery, or set-operation output.

Validation rules:

- A qualified derived column is validated against that relation's projection,
  not against unrelated base schemas.
- Unqualified columns are resolved only among relations visible in the current
  lexical scope.
- CTE column lists and SELECT aliases determine exposed names where present.
- Nested subqueries, UNION-compatible set outputs, alias shadowing, and CTE
  shadowing obey lexical visibility.
- Base-resource lineage survives through derived relations and remains the
  source of task scope/revision facts.
- Unknown derived columns, ambiguous references, unknown tables, scope leaks,
  and invalid projection counts fail closed with typed issues.
- Mutation detection continues walking the complete AST independently of
  derived-column resolution.
- Recursive CTEs remain unsupported for MVP and receive one explicit bounded
  rejection code; they are not partially validated or executed.

This is an extension of the current validator, not a query planner or second
SQL engine.

### Deterministic acceptance

- The exact retained chained CTE validates and preserves all base-resource
  lineage.
- Equivalent queries using a derived subquery and a nested alias validate.
- Valid set-operation output references validate.
- Unknown CTE/subquery columns, alias shadowing mistakes, ambiguous
  unqualified columns, scope escape, and recursive CTEs fail closed.
- Hidden mutation in a CTE/subquery still fails before execution.
- Tests use several schemas and arbitrary aliases, not only `cr`, `nr`, `qp`,
  or `qr`.
- Existing SQLite/PostgreSQL quoting and case rules remain green.

### Prohibited shortcuts

- Whitelisting retained aliases or SQL text.
- Skipping all validation for CTE-qualified columns.
- Treating every derived column as valid.
- Moving SQL parsing or lineage ownership into catalog or the executor.

## 9. RP-05 — Catalog retrieval, hierarchy, freshness, and graph evidence

### Owners

Catalog models, SQLite catalog storage/search, `CatalogService`, and catalog
capabilities/executors.

### Required implementation

#### 9.1 Search semantics

Keep the current bounded token extraction. For each safe normalized term of at
least three characters, search an exact token and a bounded FTS prefix form.
Rank exact-token matches ahead of prefix-only matches, then use the existing
deterministic score/name/resource-ID order. Record match reasons separately as
`lexical_exact` and `lexical_prefix`.

The model-visible search capability must expose the existing source and
`resource_kinds` bounds. All expansions retain agent, source, kind, result,
term, and query-length limits. Because this changes query expansion rather than
indexed content, do not reindex or migrate homes unless implementation proves
the tokenizer/index itself must change.

Do not implement English stemming or a `customer/customers` exception.

#### 9.2 Inspection and hierarchy

Bump catalog-inspect evidence to a new schema version. In addition to resource
and facets, include:

- bounded current incident relationships;
- direction, endpoints, relationship kind, confidence, provenance, revision,
  and field pairs;
- bounded neighbor summaries containing resource ID, source ID, kind, name,
  and current revision;
- an explicit incident-edge truncation flag; and
- kind-appropriate `selection_facts` derived from current typed facets.

`contains` edges make folder children navigable. The store/service, not the data
controller, owns incident-edge lookup and current-revision filtering.

#### 9.3 Freshness

For a current local-file resource, inspection projects:

```json
{
  "selection_facts": {
    "freshness": {
      "basis": "file.modified_at",
      "value": "...",
      "authority": "connector_metadata",
      "available": true,
      "facet_revision": "sha256:...",
      "sync_id": "...",
      "observed_at": "..."
    }
  }
}
```

“Newest” among candidate files means greatest current `FileFacet.modified_at`.
`CatalogResource.last_observed_at`, filename, revision hash, and discovery order
are not substitutes. Missing freshness makes that candidate ineligible for a
definitive newest claim. Equal greatest timestamps are an ambiguity that must
be disclosed or resolved by an explicit user criterion; resource-ID ordering
may stabilize presentation but may not silently decide the business answer.

Because `modified_at` is deliberately nonstructural, the projection always
binds it to the current facet observation and sync; it is never reconstructed
from the structural facet revision alone.

Freshness remains nested typed catalog data. Do not add a file-only top-level
field to every generic search hit. Search/tool descriptions must direct the
model to inspect freshness-sensitive candidates.

#### 9.4 Traversal capability

Add one model-visible, catalog-owned `catalog_traverse` capability backed by the
existing bounded traversal store. Its input mirrors `CatalogTraversalRequest`
with enforced maxima. Its accepted evidence includes:

- requested start/target resource IDs and relationship-kind bounds;
- reachability and truncation;
- visited node/edge counts;
- each bounded path in order; and
- for every step, full current relationship identity, direction, endpoint
  revisions, provenance, confidence, and field pairs.

The executor expands relationship IDs through catalog-owned records before
creating evidence. Traversal never accepts model-supplied relationship
payloads. Agent/source/current-revision scope remains fail closed.

### Deterministic acceptance

- Prefix recall finds multiple unrelated singular/plural-like stems while
  exact hits rank first and source/kind leakage remains impossible.
- Punctuation, quoting, short tokens, Unicode, FTS operators, and maximum-term
  inputs remain safe and bounded.
- Inspecting a folder exposes current children with truncation facts.
- Inspecting a table exposes reference edges and exact ordered field pairs.
- The two export files expose connector-provided `modified_at`; reversed
  filename/timestamp order still selects by timestamp.
- Missing/equal timestamps produce an explicit ambiguous result.
- Traversal returns the retained refunds-to-regions path with field pairs and
  bounded visit counts through accepted operation evidence.
- Detach/resync/cold reopen never returns stale relationships or facets as
  current.
- Old catalog evidence schemas remain inspectable and are never rewritten.

### Prohibited shortcuts

- Special-casing `customer`, the retained file names, or refunds-to-regions.
- Sorting by filename and calling it freshness.
- Out-of-band graph reads in the live oracle as proof of model use.
- Copying graph ownership into `DataDomainController`.
- Returning relationship IDs without the field-pair authority needed to use
  them.

## 10. RP-06 — Explicit tabular comparison semantics

### Owner

The existing tabular comparison capability, executor, evidence schema, and
artifact writer.

### Required implementation

Add a required `key_normalization` argument with exactly two modes:

1. `strict`: key components match only when canonical JSON type and value are
   equal.
2. `stringify_integral`: strings remain byte-for-byte JSON strings and integers
   become their base-10 string representation. Boolean, float, array, and
   object key types are incompatible. String whitespace, case, Unicode, and
   leading zeroes are not changed, so string `"001"` does not match integer
   `1`.

For both modes, type-domain compatibility is computed from present non-NULL key
components. Missing and NULL components remain deterministic invalid-key rows;
they are never normalized and do not make otherwise compatible non-NULL type
domains fail preflight.

The selected mode and a comparison-policy schema version are persisted in
accepted evidence and the artifact manifest.

The capability input schema declares these two values as a string enum. Any
other value is rejected by the shared typed capability validator with
`allowed_values`; comparison code must not maintain a different accepted list.

Normalization affects only the internal key fingerprint. Comparison evidence
and artifact discrepancy records preserve the original left and right key
values/types and also expose the normalized key used for matching. Never
rewrite source rows or present a normalized key as though both sources stored
that value.

Before a comparison task is materialized, compute bounded
per-column/per-side key type domains from both complete authoritative accepted
datasets through the existing persisted-evidence dataset reader. Put this
algorithm in the comparison owner and call it from the async data-domain
comparison validation. This is read-only validation of persisted evidence; it
does not invoke an executor or bypass `OperationRuntime`.

Embedded composition constructs one existing
`PersistedAcceptedEvidenceDatasetReader` and injects that owner into both the
data controller and comparison executor. The shared comparison owner exposes
one pure preflight implementation; the controller and executor must not grow
separate type-domain algorithms.

If the selected mode cannot make the domains compatible, return typed
`data.compare.incompatible_key_types` rejection details containing the key
column, both evidence IDs, observed type domains, selected mode, and allowed
modes. Do not materialize a task and do not emit a misleading set of
all-left/all-right discrepancies. If normalization creates a same-side key
collision, reject with `data.compare.normalization_collision`, bounded
row/key facts, and no merged rows.

The executor/comparator repeats the same comparison-owned preflight as defense
in depth for direct callers and state changes. Only a successful comparison
creates comparison evidence and an artifact.

Compared values always retain current strict typed semantics:

- integer and string values differ;
- NULL and empty string differ;
- missing and present-NULL differ;
- no trimming, case folding, numeric parsing, date parsing, or hidden coercion
  occurs.

The artifact, inline sample, total counts, truncation, source/evidence
authority, and close/reopen identity contracts remain unchanged.

### Deterministic acceptance

- Strict string/integer keys produce a typed pre-task incompatibility rejection,
  not fourteen misleading one-sided rows.
- `stringify_integral` matches ordinary CSV string IDs to database integer IDs.
- `"001"` versus `1`, boolean/integer, float/integer, whitespace, Unicode,
  composite keys, NULL keys, and normalization collisions follow the declared
  rules.
- A matched string/integer key preserves both original typed keys and the
  normalized match key in evidence and the retained artifact.
- Compared string/NULL produces `type_mismatch`; NULL/empty is never silently
  equal.
- Policy identity and compatibility facts survive evidence/artifact round-trip
  and cold reopen.
- The live oracle calls the same production comparison contract with the same
  policy and expects its exact mismatch kinds.
- Tests cover file/file, database/database, and cross-source inputs.

### Prohibited shortcuts

- Hidden coercion in the oracle.
- Requiring SQL `CAST`/`COALESCE` as the only cross-source repair.
- Treating every key as `str(value)`.
- Normalizing compared values when only keys opted in.
- Silently resolving normalization collisions.

## 11. RP-07 — Residual whole-request budgeting and session compression

### Owners

`DataContextBuilder`, shared token estimation/selection, persisted runtime
defaults, and `SessionCompressionService`.

### Required implementation

Use a two-pass whole-request build:

1. Estimate output reserve and projected tools.
2. Build the minimal required system, trusted routing, intent, and current
   operation envelopes.
3. Allocate structurally bounded current-operation bodies under the remaining
   required-operation allowance.
4. Compute the exact residual available for session history.
5. Pass that residual to `SessionCompressionService.project()`.
6. Add catalog, memory, and skill blocks as optional contributors through the
   final shared selection.
7. Assert the complete request remains within the truthful model profile.

`SessionCompressionService.project()` accepts an explicit maximum projection
token budget. It compresses when either the configured policy threshold or
that residual would be exceeded.

`retain_latest_operations` is a maximum. The service tries that many recent
raw operations and decreases toward zero until the committed summary plus
recent history fits. It advances the existing contiguous checkpoint frontier;
it does not create another summary mechanism.

Rules:

- Current-operation messages are never session-compressed.
- Required historical objectives, corrections, evidence IDs, approvals,
  resource/source revisions, and isolation facts survive in the checkpoint.
- A contributor never budgets against the full model window independently.
- If the minimal required current operation already exceeds capacity, session
  compression is not attempted as a remedy.
- If the minimum safe session summary cannot fit its residual, fail typed
  before provider I/O rather than dropping required session state.
- The real model context window and output reserve remain unchanged.
- A lower compression threshold used by a live test is an explicit immutable
  agent runtime default and reconstructs identically on cold reopen.
- Deterministic compression tests may instantiate the policy directly; a live
  cold-reopen claim may not depend on a nonpersisted injected knob.

### Typed overflow diagnostics

Extend `RequiredContextOverflow` with bounded component facts sufficient to
locate the owner of the overflow:

- profile ID;
- input limit and output reserve;
- tool tokens;
- required system/routing/intent tokens;
- current-operation envelope and body tokens;
- minimum/projected session tokens;
- total required and available tokens; and
- optional omitted-token totals where already known.

The generic loop catches this type before its generic context exception path
and persists code `context.required_overflow` plus safe numeric facts. Unknown
context failures remain generic and never persist raw exception text.

### Deterministic acceptance

- Compression triggers under an explicit low policy while the model profile
  retains its frozen real context window.
- Increasing current-operation size reduces session residual and the number of
  raw recent operations without dropping checkpoint facts.
- Summary plus selected recent history never exceeds the passed residual.
- Four current-operation read results use the observation projection contract
  and never become session history prematurely.
- Current-operation-only overflow fails with zero provider calls.
- Exact component facts persist and remain inspectable after reopen.
- Checkpoint fingerprint corruption, cross-session scope, and impossible
  summary bounds still fail closed.
- Cold reopen produces the same session projection and preserves referential
  follow-ups and sentinel isolation.

### Prohibited shortcuts

- A fake 5,000-token model profile.
- Increasing the false window just enough for the retained test.
- Treating the latest four operations as an unconditional floor.
- Dropping required session blocks after projection.
- Compressing an incomplete current tool exchange.
- Mutating transcript/evidence state during summarization.

## 12. RP-08 — Request-level parallel tool policy

### Owners

`ModelRequest`, provider adapters, `DataContextBuilder`, and the generic loop.

### Required implementation

Separate capability from policy:

- `ModelProfile.supports_parallel_tools` continues to mean that the
  provider/model can emit parallel calls.
- Every new data-domain `ModelRequest` sets
  `allow_parallel_tool_calls=False`.
- A launch-supported provider adapter must translate `False` to its native
  request-level enforcement. It may not silently discard it.
- A route/provider that cannot enforce the requested single-call policy is
  ineligible for the default data loop until batch semantics are implemented.
- `None` preserves legacy/provider-default behavior only for old persisted
  requests or other domains that have made no policy decision.

The loop retains ordered fail-stop processing as defense in depth. If a
provider violates the request and returns several calls, an early rejection
still records later calls as skipped and performs no later I/O.

Parallel execution is a later feature requiring whole-batch prevalidation,
declared dependencies, governance ordering, and restart-safe task semantics.
Read-only access alone does not prove independence.

### Deterministic acceptance

- New data requests persist and replay `allow_parallel_tool_calls=False`.
- The OpenAI adapter emits its supported native false concurrency option.
- A provider that cannot enforce the request is rejected before network I/O.
- A deliberately noncompliant fake provider returning multiple calls still
  causes fail-stop skip observations and no later executor call after rejection.
- Legacy persisted requests decode to `None` and replay their exact prior shape.

### Prohibited shortcuts

- Falsifying `ModelProfile.supports_parallel_tools`.
- Prompt text asking the model to avoid parallelism.
- Inferring independence from read access, tool name, or response order.
- Continuing later writes or evidence-dependent calls after rejection.

## 13. RP-09 — Honest readiness semantics

### Owner

`DataDomainController.evaluate_final_answer()` and its persisted `Readiness`
projection.

### Required implementation

Do not add business-specific answer validation or another LLM judge.

For Wave 1, data readiness decides only whether the deterministic response
contract is satisfied:

- required accepted current-operation evidence exists;
- exact required citations appear;
- comparison answers cite one successful comparison and both accepted inputs;
- required truncation/partial-coverage disclosures appear; and
- denied updates are disclosed and cite accepted impact evidence.

Replace codes/messages that claim semantic grounding with explicit response
contract language:

- denied: `data.response_contract_incomplete`;
- allowed: `data.response_contract_satisfied`.

The allowed message says that evidence-linking and disclosure requirements
passed. It must not state that the answer is factually correct, semantically
entailed, or independently verified.

Semantic correctness remains owned by deterministic fixture oracles and
repeated real-model qualification. An operation completing successfully means
the runtime contract closed; it is not a truth certificate.

### Deterministic acceptance

- A correct answer without citations is denied with literal repair details.
- A deliberately false number with syntactically valid accepted citations may
  satisfy only the runtime response contract, while the hard semantic oracle
  fails it.
- Public and audit messages never call that case semantically verified.
- A rejected comparison preflight creates no comparison evidence and cannot
  satisfy an ordinary discrepancy answer.
- Gross-versus-net remains a hard live failure.

### Prohibited shortcuts

- Numeric substring checks or fixture business rules in production readiness.
- Treating citation presence as proof of entailment.
- A second model's opinion as a hard oracle.
- Weakening the live semantic oracle.

## 14. ER-01 — Evaluator and live-artifact repairs

### Owner

The existing `tests/live/mvp/` harness, assertions, fixture oracles, and
scenario tests.

### Required implementation

#### Repair-compatible inspectability

`assert_inspectable_runtime_state()` distinguishes successful and failed
observations:

- successful task observations resolve to accepted evidence and exact task/
  call/turn scope;
- failed observations resolve to their typed rejection/readiness/event
  contract; and
- malformed, missing, duplicated, or cross-operation correlations still fail.

Zero repairs may be measured as efficiency, but it is not a hard correctness
requirement unless separately adopted as release policy.

#### Graph-use proof

LIVE-MVP-02 must require accepted model-visible catalog relationship evidence,
not a post-operation store call by the test. Correlate the selected traversal
path/field pairs with the resources and join columns used by accepted SQL
evidence. The store may be inspected to validate persistence, but that
inspection cannot substitute for operation provenance.

#### Comparison oracle

The oracle invokes the production comparator with the scenario's declared
`key_normalization`. It preserves strict compared-value and NULL semantics and
accepts every mismatch kind allowed by the production contract. No test-only
coercion is permitted.

#### Outcome-based assertions

- Resolve supporting evidence by payload, authority, and claimed fact—not list
  position.
- Treat one versus two harmless file reads as efficiency, not correctness.
- Do not require explicit `catalog_search` if authoritative automatically
  selected catalog context plus inspection satisfies the declared discovery
  contract. If explicit search becomes a product invariant, declare it first.
- Make freshness metadata order disagree with filename order.
- Keep archive, unauthorized-resource, evidence-scope, and exact oracle checks
  hard.

#### Valid session/compression proof

Use the truthful reference profile. Trigger compression through the explicit
persisted session policy. Isolate the sentinel/session prelude so it cannot
prevent a parameterized primary prompt from running. A deterministic test
proves checkpoint mechanics; the live test proves natural referential behavior,
isolation, and cold reconstruction.

#### Failure-safe metrics

Use one harness-owned finalization path that runs even after an assertion
failure. It aggregates every operation in a scenario, including prelude and
post-reopen operations, and emits:

- JUnit-compatible non-secret identifiers/properties where supported;
- one required redacted JSON sidecar;
- provider/model/interpreter/fixture/route identities;
- operation/session IDs;
- model/tool/action/repair/retry/fallback counts;
- latency and token usage;
- selected/omitted context and observation sizes;
- evidence/task counts; and
- terminal/readiness history.

Reopened-provider metrics come from canonical persisted model-call/route
records or equivalent restart-safe instrumentation. Do not inject the original
live provider object merely to keep a recorder alive.

Run credential and sentinel scans over retained homes, JUnit, JSON, and logs
before reporting artifacts.

### Deterministic acceptance

- A completed repaired operation passes inspectability; malformed correlation
  still fails.
- An out-of-band graph lookup without catalog operation evidence fails the
  graph-use assertion.
- A correct comparison under the declared policy matches the oracle, including
  type/NULL discrepancies.
- Evidence order and a second harmless read do not change semantic pass/fail.
- All three MVP-04 variants reach their primary prompt; compression checkpoint
  and post-reopen state are observed where required.
- A deliberately failed fake live row emits complete redacted JUnit/JSON
  metrics.
- Metrics cover every scenario operation and contain no credential or sentinel.

### Prohibited shortcuts

- Removing hard semantic assertions.
- Turning repairs, retries, or duplicate reads into automatic correctness
  failures without a declared ceiling.
- Counting a test-owned graph query as model graph use.
- Recording metrics only after success assertions.
- Injecting non-reconstructable test state into a cold-open proof.

## 15. Required implementation order and review checkpoints

Implement in this order:

1. **Foundation:** versioned fields, fingerprints, persistence migration,
   legacy decoding, and round-trip tests.
2. **Typed repair:** RP-01 and its unit/runtime regressions.
3. **Observation projection:** RP-02 and provider-valid context regressions.
4. **Tool/routing projection:** RP-03, including detach/reopen and arbitrary
   adapter declarations.
5. **SQL semantics:** RP-04 in the existing validator.
6. **Catalog semantics:** RP-05 search, inspect, traversal, and evidence.
7. **Comparison semantics:** RP-06 and the shared oracle contract.
8. **Whole-request/session budgeting:** RP-07 and typed overflow persistence.
9. **Parallel request policy:** RP-08 across canonical request persistence and
   launch provider translation.
10. **Readiness wording/contract:** RP-09.
11. **Evaluator finalization:** ER-01 aligned with the implemented production
    contracts.
12. **Consolidated deterministic verification.**
13. **One unchanged 12-row paid live rerun.**

For each numbered implementation stage:

- identify the existing owner and smallest coherent change before editing;
- implement one representative path first when standardizing a repeated
  pattern;
- add general owner-level tests plus the retained regression;
- run the focused tests before broadening the pattern;
- review persistence, restart, redaction, and fail-closed behavior; and
- do not proceed by stacking a prompt, retry, budget, or fixture patch over a
  failed invariant.

An implementation agent must stop for design review if a required decision
cannot be implemented inside the listed owner without creating another
framework layer.

## 16. Verification before another live run

At minimum, complete and record:

1. focused unit/contract/acceptance tests for every repair package;
2. migration, legacy decode, corruption, and cold-reopen tests;
3. all Wave 1 deterministic fixture/oracle/harness tests;
4. the complete `next/tests` selection marked
   `not requires_llm and not requires_db`;
5. all architecture tests;
6. Black check and byte compilation;
7. Mypy and Pyright over the affected/full v2 scope;
8. `git diff --check`;
9. generated-disposition and frozen root-v1 isolation checks;
10. both supported Python 3.11 and 3.12 for the complete deterministic suite,
    because production code and persistence will change; and
11. clean-wheel/package checks if request persistence, provider reconstruction,
    CLI configuration, or installed imports change.

Only after those are green may the coding agent run the unchanged 12-row corpus
once with explicit live opt-ins, the same fixture/prompt versions, real OpenAI
inference, and an explicit model. Do not manually rerun one semantic failure
into a pass. A remaining model error remains a failed row and must be recorded
as such.

## 17. Coding-agent directive

Use the failure analysis as evidence and this document as the normative repair
contract. Do not infer implementation choices from the retained prompts. Do
not change production code solely to make a named row green.

For every proposed code change, the handoff must state:

- existing owner extended;
- invariant repaired;
- persisted/schema compatibility impact;
- general deterministic tests added;
- retained failure reproduced by a deterministic regression;
- focused and consolidated command results;
- files changed;
- remaining model behavior or unresolved risk; and
- confirmation that root v1 and Phase 10 remain untouched.

The target milestone is not “make twelve tests green.” It is: **repair the
owner-level contracts, prove them generally and deterministically, then use the
unchanged live corpus to measure the model on a framework that is no longer
misleading or withholding required state.**

## 18. Launch boundary

Completion of this repair specification would make another LLM-G02 measurement
valid. It would not by itself pass LLM-G02, later LLM gates, or authorize MVP
launch. Launch readiness still requires every gate and the explicit human
authorization defined in the production-readiness plan.
