# Context and source scope

Daita discovers currently admitted readable sources for an ordinary question.
There is no selected-source mode. Every data call still names its exact source
and resource, and current permissions are checked before I/O.

The frozen readable resource set is a candidate ceiling, not an exact selection.
Initial catalog context ranks only within that ceiling and reports separate
`match_outcomes.current_query` and `match_outcomes.prior_query` assessments. The
prior assessment may be null and is continuity evidence only; the two assessments
are never merged into an effective source or persisted as sticky selection state.
An explicit catalog-context `resource_ids` request retains its separate exact-ID
semantics and fails when any requested ID is not currently readable.

Each catalog-owned match outcome reports `binding_status` and `source_status` as
`unique`, `ambiguous`, or `no_match`. It also names the strongest evidence tier,
the exact candidate count, up to 12 `(source_id, resource_id)` bindings, an omitted
count, and deterministic ambiguity reasons. Exact resource evidence outranks broad
catalog metadata, which outranks source hints. Scores order candidates within the
presentation but never turn a multi-candidate tier into a silent winner.
Relationship neighbors and `unmatched_fallback` inventory are not target
candidates. A unique source can still contain several ambiguous resource bindings.
The `assessment_provenance="catalog_service"` field identifies the code owner of
the calculation; it is not authorization. Outcome content remains classified as
untrusted external data.

For a single-target read, a unique current binding can be used by exact ID after
any missing schema is obtained. An ambiguous single-target request should be
clarified instead of choosing the first hit. An explicit comparison may use the
candidate set, with separate exact relational calls grouped by source before
synthesis. A no-match result can be refined with a grounded `catalog_search` query;
fallback inventory is not a match. A prior unique binding is reusable only when the
current wording clearly refers to it. Ambiguous update or upsert intent should be
clarified before preview. These are model-facing presentation rules, not a runtime
clarification gate; exact scope, ID, schema, permission, approval, and receipt checks
remain the execution authority.

Initial context contains a bounded connector directory drawn from current source
registrations, admitted MCP bindings, and eligible skills. A separate compact
manifest lists applicable toolboxes; the directory does not repeat those entries.
An omitted count makes truncation visible. `catalog_search` and `toolbox_search`
return `next_cursor` when more candidates remain. Supply it with the same query
to continue. Matching candidates come first; unmatched fallback candidates are
explicitly labeled. `total_matches` and `total_candidates` describe different
counts. A cursor is valid only for its current prepared scope and discovery
snapshot; a changed query, run, admission, or search snapshot requires a fresh
search. Exact `toolbox_load` names still need no preceding search.
For admitted native updates/upserts, initial guidance names the discoverable
preview/execution tools so one load can prepare both without a search. Revoked
or absent write capabilities are not advertised. Preparation fits this guidance
against the same candidate catalog used by step projection. Loading does not
activate a same-step call or authorize a write.
For grant-requiring capabilities, discovery includes the exact automation contract
used by scheduling. It does not activate execution tools. A contract too large for
the bounded page is omitted in full with `automation_contract_omitted`.
`toolbox_inspect` reads an exact prepared tool contract without loading or executing
it. Search, load and inspection derive their contracts from the same registry and
grant metadata. Load receipts retain compact incomplete references and grant metadata;
they do not copy execution schemas into every load result. Schemas obtained through
inspection remain in the exact transcript after the callable set changes. Exact inspection returns either the complete contract or
bounded fragments; use its `contract_digest`, child JSON Pointer `path`, and
`next_offset` to retrieve omitted portions. Partial data never claims completeness.
`requires_automation_grant` describes scheduled authority. A foreground action
uses the normal exact approval at invocation. For routine authoring, inspect a tool
when its exact contract is missing or incomplete. Inspection checks current local
applicability within the frozen candidates; remote identity/schema checks still
belong to invocation. Framework inbox
destinations concern report delivery; an MCP action's destination is defined by
that action's admitted schema.

The manifest reports the access modes and operational effects of all prepared
candidates, including unloaded tools. A read-only Sources group does not claim
update capability. These facts grant no permission; current revocation, receipt
blocks and approval still apply. Search cannot add an absent effect. Independent
schema/discovery calls or grounded value-read/load calls can share a model step;
loaded execution schemas become callable only on the next step. Preview and write
remain sequential.

Foreground preparation also reads the existing durable external-effect blocking
check. If blocked, mandatory context includes up to 20 receipt IDs and an omitted
count, classified as internal operational metadata. It includes no receipt
payloads or action arguments. Current row values cannot resolve an earlier
operation's uncertainty. This context is frozen for the run, refreshed for the
next run, and grants no execution or recovery authority. Runtime rechecks and
human-only recovery remain unchanged.

Core instructions and admitted context are frozen when a run starts. Each model
step adds procedure guidance for its current callable tools and its remaining
cumulative token allowance. Optional discovery and prior conversation continuity
receive a conservative footprint allowance derived from that run limit, separate
from mandatory instructions, current input, and pinned schemas. The complete
projection still fits the model input window. This bounds optional context without
changing the exact current-run
transcript or its sensitivity floor. Provider-native counting remains the final
request admission check on supported API routes.

`update_source_discovery` and `update_mcp_discovery` edit bounded local `summary`,
`when_to_use`, and normalized `keywords` hints. They preserve execution
permissions and connector identity. Source hints appear in current catalog
searches; MCP declarations use edited hints after reopening, consistent with
their existing activation boundary. Local hints never establish authority,
schema facts, or permission to execute a suggested action.

Use an exact caller filter when a question must stay within particular sources:

```python
result = await agent.run(
    "Compare customer counts in these two databases",
    source_scope_ids=(billing_source.id, application_source.id),
)
```

The default empty tuple admits all readable candidates at run preparation.
Connections or permissions added later cannot enlarge that prepared run;
revocation can narrow it immediately. Source filters apply to catalog access.
The explicit `files_only=True` option excludes both catalog and MCP operations.
Machine runs use their frozen authorization ceilings; an empty machine source
ceiling means no sources.

Conversation continuity is shared across sources and stays bounded. Private
answers retain their classification when summarized, when a later question
narrows its source filter, and after a source is detached or the agent reopens.
The full request classification also includes admitted connector descriptions,
workspace context, current tool results, and retained advisory content. A model
route or MCP outbound ceiling that cannot accept that classification fails
before it receives the content.

Memory, user preferences, and skills carry an owner-written sensitivity label in
their Markdown document. Model-authored replacements inherit the full request
classification. Public reads return the content without the metadata header.
Local imports without a reliable label default to `restricted`. A trusted caller
can explicitly classify local content:

```python
from daita.llm.models import ModelSensitivity

await agent.set_memory(
    "Gross margin is revenue less cost of sales.",
    sensitivity=ModelSensitivity.PUBLIC,
)
await agent.save_skill(
    "compare-counts",
    "Compare counts using explicit definitions.",
    "Inspect both schemas and state any difference in customer definitions.",
    sensitivity=ModelSensitivity.PUBLIC,
)
```

The same typed keyword is available on `set_user_profile`. Local Markdown
classification uses the first line `<!-- daita-sensitivity: restricted -->`
(or another `ModelSensitivity` value). Classification is a local information
handling decision; content that claims permission or describes itself as public
does not override it. Semantic annotations and inactive learning candidates
retain their classification in their existing SQLite records.

Scheduled instructions must be self-contained and retain the originating request's
sensitivity. A scheduled run does not import unrelated conversation history,
mutable memory or semantic recall, or newly saved skills. Its inbox
still belongs to the originating conversation, and only explicitly retained
skill versions are included in its reasoning context.

Budget guidance also reports the remaining model steps and an advisory two-request
forecast from the latest measured input, its recent growth, and two configured
output allowances. It explicitly accounts for needing another request to read tool
results and finish the answer. Unknown input growth is labeled unknown; future
schemas and results can exceed this forecast. The forecast is not a reservation,
a native token count, or permission for extra work. Provider admission still checks
the exact prepared request against the remaining hard allowance, and exhaustion is
terminal. The framework never substitutes a smaller final context or a post-limit
model request. Compact owner-defined results reduce growth without truncating the
exact transcript or weakening the retained contracts.

Procedure guidance follows the authenticated callable working set. When a needed
tool is a prepared candidate but is not loaded, guidance names the exact load and
requires invocation on a later step. Missing access is a scope limitation; search
cannot grant it. Missing columns and keys remain unknown. Current catalog evidence
can be reused without a redundant schema call. Denied, failed or uncertain effects
are reported without retry or replacement; recovery performs no action.

When routine creation or revision is callable, a mandatory code-owned context block
exposes the complete frozen eligible model-route choices and host per-run token/cost
ceilings. These are local configuration facts, not remaining credit or approval.
There is no current-model alias or implied default; a null host cost ceiling does
not remove the routine's own bounded-budget requirement. Machine runs receive no
routine-management facts. Loading another working set removes authoring guidance.
