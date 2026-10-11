# Daita harness alignment design

Daita can improve context efficiency and measure model-dependent planning benefits inside its existing architecture. The first implementation should separate frozen admitted context from its per-request presentation, re-fit optional material as a run grows, and expose enough measurements to evaluate the result. Exact transcripts, runtime authentication, provider admission, graph authority, and effect receipts remain with their current owners.

Current-run compaction is a separate contract change. Preserving raw durable evidence does not, by itself, satisfy the current rule that the model receives the exact current transcript. Adopt that change explicitly before experimenting with elision. Keep separate model summarization calls out of the initial implementation.

This design applies to the current working tree as inspected on October 11, 2026. Names described as proposed below are design choices, not existing APIs.

## Architectural ownership

| Owner | Current responsibility | Proposed responsibility |
| --- | --- | --- |
| [Agent](../src/daita/agent.py) | Validate public inputs and delegate to the embedded host | Pass an immutable presentation policy through the existing composition APIs |
| [EmbeddedAgent](../src/daita/hosting/embedded.py) | Compose the model, context builder, runtime, stores, admission and supervisors | Construct the same components with the selected presentation policy |
| [AgentLoop](../src/daita/loop/driver.py) | Preserve exact progression, invoke the model/runtime and enforce run budgets | Continue supplying raw messages to runtime projection and execution; emit bounded context measurements |
| [RunSession](../src/daita/loop/session.py) | Isolate one invocation, cancellation, deadline, writer and mutable evidence | Hold any necessary transient presentation/progress state for that invocation |
| [AgentContextBuilder](../src/daita/context.py) | Freeze classified context and construct each model request | Fit optional blocks per request and, after an explicit contract change, construct deterministic elided views |
| [CapabilityRuntime](../src/daita/capability_runtime.py) | Validate and execute admitted capabilities against exact evidence | Keep the existing raw transcript and contract path; never authenticate projected summaries |
| [ModelRouter](../src/daita/llm/routing.py) and provider adapters | Freeze eligible routes, admit requests, account for attempts and own native I/O | Enforce an explicit counted input ceiling where supported without changing provider ownership or retry semantics |
| [SQLStateStore](../src/daita/storage/sql.py) and RunSessionWriter | Persist canonical records through one state boundary and writer | Own any later durable request-presentation evidence through the same boundary |
| Existing benchmark support | Capture requests, score independent oracles and retain matched episodes | Add Daita component variants, pressure scenarios and paired trajectory metrics |

The execution path remains:

```text
EmbeddedAgent
  -> isolated RunSession
  -> AgentLoop holding the exact current transcript
       -> CapabilityRuntime.project(raw transcript)
       -> AgentContextBuilder.project(frozen context, raw transcript, tool projection)
       -> ModelRouter/provider admission -> model
       -> CapabilityRuntime.execute(raw transcript, exact tool projection)
       -> RunSessionWriter appends exact assistant and ordered tool results
```

The existing split at `loop/driver.py:609-629` and `820-849` is suitable: context construction and runtime execution already receive the same raw messages through separate interfaces. The improvement changes the model-facing branch. Runtime checks continue using the raw branch. Artifact evidence authentication loads persisted transcripts in `domains/data/export_capabilities.py:2087`; native write authentication does the same in `domains/data/controller.py:1399`.

## Findings that determine the design

**Optional context is frozen after its initial fitting.** Preparation fits catalog discovery and prior conversation turns, then collapses them into `RunContextSnapshot.static_messages` at `context.py:1009`. Later requests reuse those messages at `1205`, append every current message at `1216`, and fail if the estimate exceeds the input capacity. A per-request fitter cannot safely remove catalog fields by parsing the rendered system prompt. Preparation must retain explicit bounded blocks and their provenance.

**An avoidable refusal is reproducible offline.** A synthetic request with a 32,000-token profile and a 2,000-token output allowance has a 30,000-token input capacity. After retaining about 3,000 characters of prior history, a later 22,000-character tool observation produces a conservative estimate of 32,570 and fails. Removing only the optional prior messages produces 29,286; the current user/call/result exchange is unchanged and the catalog was read once. The probe demonstrates a fitting opportunity, not native token counts or a model accuracy gain. These figures come from local development probes; the implementation needs a retained deterministic regression case.

**Native counting does not currently replace context-window admission.** `ModelRequest` carries cumulative token/cost allowances but no separate input ceiling. `llm/pricing.py:698` bounds output using cumulative allowance and cost. OpenAI, Anthropic and Gemini count exact native requests before generation, but the context builder can reject earlier using its deliberately conservative estimator. Relaxing that check alone would rely on server errors for window enforcement. OpenAI-compatible and subscription CLI routes have no complete request counter.

**Composite route limits need a joint input and output contract.** `ModelRoute.model_profile` independently takes the minimum context window and minimum output allowance (`llm/routing.py:225`). Lazy providers are constructed with each candidate's own output cap (`llm/factory.py:263`). For synthetic candidates with window/output pairs of 32,768/16,384 and 131,072/2,048, the composite profile advertises 30,720 input tokens, while the first candidate has only 16,384 available with its own output cap. This is a verified configuration inconsistency, not an observed live overflow. A new counted input ceiling must account for the actual wire output ceiling as well.

**A previous request count is not a count of the next projection.** `context.py:1225` uses the maximum of its current estimate and the previous request's measured input. The continuation forecast also uses previous input and non-negative measured growth. Re-fitting or elision can shrink a request, so both mechanisms need review when presentation changes. This is a design dependency, not evidence of an observed production failure under the current fixed window.

**Frozen context audit material is not a durable request trace.** `RunContextSnapshot.audit_context()` exists, but inspection found no production call site that persists it. Model-completed events provide measurements and are explicitly best effort and non-durable. Exact transcripts authenticate execution; they do not establish which optional blocks or elisions a future presentation policy actually showed. Metrics can initially use existing observer/benchmark capture. Durable summarization would need explicit retained presentation evidence.

**Public composition has no context-only injection seam.** `Agent.create/open/from_storage` expose configuration rather than a context-builder override. The embedded custom path requires custom context and tools together (`hosting/embedded.py:6839`). A component experiment must not replace the real runtime or mutate private fields to inject a fitter. Add a narrow typed policy to the existing public composition instead.

## First implementation within the exact transcript contract

### Freeze structured material and fit its presentation

Refactor `RunContextSnapshot` in `context.py` to retain the preparation result in structured, immutable form: mandatory code-owned instruction material; bounded optional catalog and connector-directory candidates; whole prior turns with continuity and full-evidence forms; classified advisory material; and the existing artifact destinations, source/tool bindings and digests.

Retain every mandatory instruction, normalized start message, current exchange, exact callable schema, graph task specification, machine execution binding and unresolved-effect notice. Do not classify retained routine instructions or skills as expendable optional history. Ordinary advisory and discovery blocks can be reduced only where their existing owners already define omission or projection semantics.

The initial fitter and later request fitter should share the existing selection logic. At each step:

1. Authenticate the exact `StepToolProjection` against the raw transcript.
2. Aggregate classification and raw evidence pressure before any presentation changes.
3. Construct the mandatory request with current tool definitions and current procedure guidance.
4. Fit frozen optional discovery and whole prior turns into the remaining presentation target, preserving useful newest continuity before full older evidence.
5. Render explicit omission/coverage markers and produce the final request.
6. Apply the existing hard window and provider token/cost admission boundaries.

The cumulative allowance and continuation forecast can reduce the optional target. They must not reject a mandatory request merely because an advisory two-request forecast is pessimistic. Provider-native admission still decides what can actually be dispatched. Do not subtract conservative character estimates from billed token ledgers.

No catalog, memory or remote refresh occurs during this fitting. Removing displayed discovery does not remove or widen admitted runtime scope. The initial sensitivity floor remains even after classified content is omitted. Preserve the frozen-context digest and add a separate presentation digest covering the actual request and selection policy. Prefer a stable sequence of reductions during a run; any hysteresis state belongs in its `RunSession`, never an ambient dictionary on the shared context builder.

### Expose a narrow immutable policy

Add a small typed presentation policy beside `AgentConfig` in `config.py`, with legacy-equivalent defaults. Initially it needs only optional-context re-fitting and an optional experimental input cap. Introduce elision/progress choices only when those features are implemented. Avoid a generic callback system or policy language.

Thread it through all three public composition paths to `EmbeddedAgent` and `AgentContextBuilder`. Treat it as runtime configuration at first. Persisting it in the agent-home configuration would change the strict stored contract and needs the existing home revision/release workflow. A plain typed runtime option does not automatically require a database migration.

Machine runs need an additional binding decision. `_model_execution_contracts` currently hashes route/profile and model-call policy (`hosting/embedded.py:451`), not a presentation policy. Include the effective new policy in that exact execution contract before enabling non-default policies for graph tasks or routines. A changed policy must revalidate against the retained binding; do not substitute a current digest for an earlier approved one. A foreground-only experiment can leave machine presentation unchanged until this binding is implemented.

For benchmark window sweeps, use a presentation input cap bounded by the real reviewed route capacity. Keep the actual provider/model identity and reviewed profile intact. A test cap is not a claim that the provider's physical context window changed.

### Measure pressure and action economy

Use bounded observations and the existing recording provider to collect the presentation digest, policy digest, raw/current/optional/schema bytes, omitted counts, conservative estimate, native count when available, model turn count, latency, cache usage, failed tool calls and termination reason. Distinguish cumulative charged input from the successful request's context footprint, particularly when a routed call includes failed attempts.

Context-builder refusals happen before the recording provider sees a request. Retain their reason and bounded pressure diagnostics in the benchmark result as well. Unknown native counts or unavailable measurements remain unknown. Observer failure must not change production execution; evaluation must label incomplete captures.

These are existing-owner additions to `context.py`, `loop/driver.py`, `observation.py`, `tests/support/job_benchmarks.py` and the Daita analytics evaluation support. They do not need a telemetry store or event bus.

## Counted input admission

Introduce proposed runtime-only `ModelRequest.max_input_tokens` and `max_output_tokens` ceilings, distinct from the cumulative `max_total_tokens`. The context owner supplies its admitted input ceiling and output reserve. Adapters limit actual wire output to the minimum of the request output ceiling, their configured cap, and the existing cumulative token/cost admission result. Counted adapters enforce the input ceiling against the exact translated native request before generation. Extend the existing shared admission helper rather than adding another counting service.

The preferred route contract uses one output ceiling across the frozen candidates, so input capacity is valid with that same ceiling. Alternatively, retaining different output caps requires taking the minimum of each candidate's window minus its own cap. Do not use the difference of independent minima while allowing larger per-candidate wire output. Add a focused mixed-window/mixed-output regression to the router/provider tests. This also makes the context forecast's stated output allowance consistent with dispatch.

The ceiling is an execution-control fact, not billed usage. Its enforcement must happen under the existing provider permit, deadline, count timeout and cleanup scope. A count failure remains a normalized failure with no generation dispatch. Routed retries retain their usage; adding a window check must not trigger a tool replay or silently move a selected run to another model.

Start by deferring conservative mandatory-input rejection only for routes whose every eligible candidate supports the counted input ceiling. Uncounted and mixed counting routes retain conservative admission and enforce supported output ceilings. Express support through the provider's declared request-policy contract, including lazy preflight, not name checks in `AgentLoop`. Preserve the frozen route's verified common capacity initially, and retain `RunRoute` stickiness after selection. Per-candidate re-projection would expand this design substantially and is unnecessary for the first improvement.

Update both preparation and step projection. `_fit_mandatory_request` can reject before `project`, so changing only the latter is incomplete. Counted-mode pressure estimates can shape optional additions without serving as hard native limits; a complete native count remains the dispatch guard. Keep the existing full output reserve initially.

Do not add a catch-and-regenerate overflow recovery loop. If a future implementation re-fits after a native count, it needs a bounded count-only protocol with positive evidence that generation did not start, and must share the existing logical request's deadline/admission. That is a separate extension.

## Current run elision after the contract changes

The required invariant should explicitly become: the exact transcript remains canonical execution state and durable evidence; model requests may contain bounded, authenticated presentation projections; projections never establish execution authority. `AGENTS.md:219` currently forbids truncating the exact current transcript in the presentation target, so retaining raw storage alone is insufficient.

Implement deterministic elision in `AgentContextBuilder` first. Keep it disabled by default for evaluation. It should target only older bulky successful effect-free observations whose declared contracts support a safe projection. Preserve the start, recent complete exchanges, assistant call identities, small numerical results and active protocol state. Determine eligibility from registered capability metadata and exact result kind, not tool-name heuristics or model suggestions. Unknown result types remain exact.

| Evidence | Initial presentation rule |
| --- | --- |
| Recent exchanges and errors | Preserve exactly so the model can correct the next call |
| Native update/upsert previews, effect results and uncertainty notices | Preserve exact review/binding and effect status information |
| File bindings used for edit/save | Preserve the exact current-run binding information |
| Toolbox load receipts | Preserve activation identity and working-set state |
| Python cell state | Preserve generation/revision, state-loss status and admitted output identifiers |
| Artifact/delivery references and graph terminators | Preserve exact identifiers and lifecycle status |
| Older large read samples or discovery results | Project only under an explicit bounded contract; mark presentation coverage |
| Unrecognized MCP or custom result semantics | Preserve by default |

A projected body must be visibly a projection, retain its original call ID and original-result digest as references, and carry explicit source-versus-presentation coverage. Never replace rows with a sample while preserving a misleading claim of complete presented data. Do not manufacture numerical aggregates by summarizing text. Retain full classification from the raw results regardless of what remains visible.

Do not replace a `ToolResultBlock.output` and leave its old `output_sha256` masquerading as a hash of the new body. Use a clearly identified rendering record with separate original/projected identities. Provider serialization may retain the tool call/result pairing, but that rendering must never enter runtime output validation or evidence authentication.

Compute runtime/toolbox projection digests from raw messages as today. All source/preview/artifact authentication continues loading raw persisted records. Never feed a model-facing elision back into `_loaded_tool_receipt`, preview validation or artifact lineage.

The raw 512 KiB evidence limit and analytical trace budgets remain independent limits. Reducing model input cannot make an over-limit raw run admissible. Provider-specific reasoning/replay metadata also needs translation tests: whole-exchange summarization cannot blindly reuse historical call-ID rewriting or discard signed/replayed provider content.

There is already a recovery path for computation: `analysis_execute.inputs` can bind an exact prior current-run tool-result call ID, and its input owner authenticates the original bytes (`domains/analysis/domain.py:1176`). Measure that path before adding a general recall tool or another store. Small facts needed to answer should remain in presentation; raw storage is not a guarantee that the model will retrieve omitted detail.

## Graph tasks and scheduled runs

Graph model attempts use fresh sessions and no conversation history (`jobs/supervisor.py:944`). Their bounded `TaskContextBundle` includes exact task authority plus untrusted parent results, checkpoints, comments and controls. Its 96 KiB aggregate bound is a storage/presentation bound, not a model token window. `jobs/graph/execution.py:365` already owns parent-payload reduction when that byte bound is exceeded. Any further projection of these handoffs belongs with that owner, preserves the task/specification/binding and control facts, and records omitted result digests. Do not let a generic foreground history fitter rewrite graph authority or assume an omitted parent payload can be retrieved through a tool the task was never granted.

Routine reasoning is self-contained and deliberately omits unrelated conversation history (`hosting/embedded.py:2337`). Keep its exact approved instruction, retained skill bindings, effect constraints and occurrence budgets. A host presentation policy must not silently add mutable foreground memory or a progress list to it. Both machine paths continue using the same `AgentLoop`; graph checkpoints and scheduled occurrence supervision remain separate lifecycle systems.

## Planning and repeated failures

Start with a prompt-only completion/evidence checklist selected through current procedure guidance for genuinely multi-step foreground work. Compare it with the current guidance before adding a tool. Avoid mandatory planning for ordinary lookups and avoid selecting policy from model parameter count alone.

If a maintained progress list has measured value, keep its typed state with `RunSession`. A proposed registered foreground-only capability can author at most a small bounded set of items, with stable item IDs, status, expected revision and a total byte bound. Its adapter uses the ordinary registry/runtime path; it is not a fourth toolbox control or a second planner. The feature would justify one narrow static advisory capability owner, composed by `EmbeddedAgent`, whose state remains session-local.

The model-written list inherits full request classification, is injected as untrusted advisory content, and cannot establish source facts, grants, completion or graph transitions. Compact tool receipts can carry only the accepted revision/digest. Count updates against ordinary step/tool limits and preserve their exact arguments in the transcript. Reopening a home does not resume this list. Graph tasks keep their existing checkpoints and terminators; scheduled runs keep their approved self-contained instruction and retained skills.

Repeated-call instrumentation should fingerprint the exact tool identity, canonical arguments, binding revision and error class without adding raw arguments to payload-free observations. Only consider a one-time code-owned reminder after real trajectories show waste. If early termination is justified, detect a calibrated streak of deterministic effect-free failures, settle the entire current batch, preserve all ordered results, then return a distinct failure reason. Permission changes, changing offsets, legitimate fresh observations and effect uncertainty must not be collapsed into a generic stuck condition. No effect retry or replay is introduced.

## Summarization decision

Separate tool-free model summarization would add model work outside the current one-request-per-step contract. It would need the same route, sensitivity, permits, deadlines, token/cost accounting, partial-usage handling and explicit request-count limits. Treating it as free context-builder I/O would break the budget contract; performing it after exhaustion would break termination semantics.

Keep it out of the first implementation. If deterministic elision still leaves expensive long trajectories, compare an advisory context note authored through an ordinary registered tool in the existing loop with a separately accounted summarization phase. The former can preserve the current model/tool progression; the latter requires an explicit loop contract extension. Both produce untrusted model claims, never source or effect truth.

A production summary must retain its exact text, covered transcript prefix, raw-prefix digest, request/route identity, classifications, policy identity and usage. A digest alone cannot reconstruct what the model saw. Those records belong behind the existing writer and SQL store, with bounded retention and an appropriate current home migration. Do not repurpose `analysis_evidence` for unrelated context summaries or treat summaries as authoritative graph checkpoints.

## Matched evaluation

Use Daita's analytical evaluation fixtures for a separately labeled component track. Preserve the independent Decimal/timezone/grain oracles and add failure-aware scoring, randomized matched episodes, configuration hashes and unknown-cost accounting. Use the real public composition and production runtime throughout.

Freeze provider/model identity, actual output cap, reasoning controls, retry policy, task/data hashes, permissions and budgets. The existing live helper caps provider output at 1,536 tokens even when the reviewed profile advertises more; record that actual cap and keep it constant within pairs. Disable fallback for strict same-model pairs or report actual fallback as a confound. Record cache reads/writes as well as total tokens because changing the prompt prefix can reduce cache efficiency.

| Experiment | Variable | Fixed comparisons |
| --- | --- | --- |
| Optional fitting | Initial-only versus later re-fitting | Exact current transcript, same tools/model/budgets |
| Input admission | Conservative versus counted ceiling | Same request policy and real model capacity |
| Current-run elision | Exact versus deterministic eligible-result projection | Same source scope, raw evidence limits and runtime |
| Foreground planning | Existing guidance, checklist, and later bounded list | Same context policy and action space |
| Computational composition | Direct admitted reads versus reads plus native Python | Same data and output oracle; account for child calls |

Use production defaults first and a separately labeled finite long-run stress regime second. Vary per-request input capacity independently from cumulative token and dollar ceilings. Add tasks with distractor discovery, substantial prior continuity, long bounded reads, dependent Python computations, old evidence used for artifacts and ambiguous definitions. Keep graph/routine regression scenarios separate from foreground accuracy scoring. Variants of one numerical family are clustered observations, not independent new task types.

Report paired successes, failures and cost/latency together, with family-aware uncertainty and a frozen holdout before broad performance claims. Budget stops, context refusals, semantic wrong answers, clarification, unavailable measurements and harness failures remain distinct. Scripted models establish mechanics; real authorized model runs establish reasoning and tool-selection quality.

## Delivery sequence and verification

| Sequence | Small complete change | Acceptance evidence |
| --- | --- | --- |
| 1 | Capture pressure and variant identity in current benchmark owners | No live I/O during collection; unknown counts remain unknown; failed executions cannot pass |
| 2 | Structured frozen context plus optional per-step fitting | A later request sheds optional material before refusal; current exchanges/scope/sensitivity remain exact; catalog read count stays one |
| 3 | Consistent request input/output ceilings and counted admission | Mixed route capacities match actual wire caps; installed SDK transports show counting before generation, exact ceiling refusal and shared deadlines; uncounted/mixed counting routes stay conservative |
| 4 | Explicit exact-input rule revision plus disabled deterministic elision | Raw transcript and digests unchanged; runtime grants/evidence remain exact; provider serializers preserve valid pairing; coverage remains truthful |
| 5 | Measured foreground checklist/list and repeated-failure policy | Isolated sessions, bounded untrusted state, exact revision updates and no graph/routine authority leakage |
| 6 | Summarization only if remaining evidence warrants it | All model work charged; summary text/prefix retained; no post-limit work; durable format/recovery verified |

Extend the existing owner suites rather than adding parallel test frameworks: `tests/context` for fitting, sensitivity and pressure; `tests/loop` for terminal/cancellation/ordered results; `tests/capabilities` for exact toolbox activation; `tests/llm` for counted admission, route stickiness and provider translation; existing artifact/native-write tests for raw lineage; `tests/analysis` for authenticated inputs and state; and graph/routine suites for shared-loop isolation.

Specific regressions must include a classified omitted block retaining its routing floor, an old preview used after unrelated read elision, artifact/Python inputs authenticated from a projected-away read, loaded-tool replacement surviving omitted discovery, raw evidence exhaustion despite tiny presentation, cancellation without an extra model request, stale machine presentation-policy bindings failing normally, and equal deterministic output for identical snapshot/transcript/policy/tool projection.

The first three changes can preserve the exact active-transcript invariant. The fourth changes model presentation and therefore requires the explicit architectural decision. Durable configuration or request-summary records follow the existing release contract: amend an unreleased next revision only when allowed, otherwise append a new revision; released migrations stay immutable.

## Evidence and transfer limits

[An Empirical Study of Harness Design for Coding Agents](https://arxiv.org/pdf/2609.20804) supports model/task/budget-dependent component evaluation and cheap observation elision before expensive summarization. Its experiments concern coding tasks, not Daita's grounded analysis or effect authority. The proposed Daita designs above are code-derived engineering choices whose performance needs matched evaluation.

The earlier context/loop/budget check passed 132 offline tests. The deeper architecture, graph model-task/planner, route-stickiness and routine tool-projection check passed 100 tests in 76.51 seconds; the existing data-operation evaluation harness passed another 62 tests. The synthetic fitting and route probes establish the concrete opportunities above. These checks validate current contracts and the architectural analysis; they do not validate an unimplemented fitter or establish live model gains.
