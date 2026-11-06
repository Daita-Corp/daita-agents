# Context and source scope

Daita discovers currently admitted readable sources for an ordinary question.
There is no selected-source mode. Every data call names its source by display
name and its tables or files by native name, and current permissions are
checked before I/O. The source answers; the catalog only advises.

In practice, Daita can search across sources you have made readable, but it
does not guess which table or connector you meant when a consequential choice
is ambiguous. Source content, prior answers, and saved knowledge can guide an
investigation; they cannot add access or replace what the source reports now.

A source you have made fully readable lets a run read anything PostgreSQL or
SQLite resolves in it, cataloged or not. A source limited to selected tables stays
limited to them, checked from the name in the SQL with no catalog lookup. The
catalog never grants or denies a read. An attached folder works the same way:
under full read access a run reads any data file within the folder's
`subpaths`, decided from the file's format and path when the call runs.

The prompt never carries catalog results ranked for the question. It holds a short
directory of each source's name, type, dialect, schemas and approximate object count
(rounded to two significant figures, so small drift in a source does not change the
prompt), the same for every question. `catalog_search` and `catalog_describe` find tables when
the model needs them, asking the live source as well as the catalog, and show only
what the run may read. Look-alike tables carry an advisory `status` and `evidence` in
those results only; nothing is hidden or reordered.

For a read, the model chooses the best-supported table from the evidence available
(schema, row estimates, keys, comments, a quick query), says which it used, and
answers. It asks only when the evidence cannot separate the options and the choice
would change the answer. Rank or relevance never selects a write target, connector,
account or destination. These are model-facing rules, not a runtime clarification
gate; scope, permission, approval and receipt checks remain the execution authority.

The same material-ambiguity rule applies to admitted MCP connectors and accounts,
recipients, artifact or delivery destinations, and other consequential foreground
choices. Daita asks only when the current request depends on choosing among plausible
alternatives that bounded read-only discovery cannot distinguish. Multiple search
results alone do not require a question, and an explicit request for a set or
comparison can use each exact admitted choice. Clarification is an ordinary completed
assistant turn retained in conversation continuity; it has no separate run status,
pending record, sticky selection, or CLI/TUI path. Approval authorizes an exact effect
but never substitutes for identifying the intended target.

Initial context contains a bounded connector directory drawn from current source
registrations, admitted MCP bindings, and eligible skills. A separate compact
manifest lists applicable toolboxes; the directory does not repeat those entries.
An omitted count makes truncation visible. `toolbox_search` returns `next_cursor` when more candidates remain. Supply it with the same query
to continue. Matching candidates come first; unmatched fallback candidates are
explicitly labeled. `total_matches` and `total_candidates` describe different
counts. A cursor is valid only for its current prepared scope and discovery
snapshot; a changed query, run, admission, or search snapshot requires a fresh
search. Exact `toolbox_load` names still need no preceding search.
For admitted native writes, initial guidance names the discoverable
preview/execution tools so one load can prepare both without a search. Revoked
or absent write capabilities are not advertised. Preparation fits this guidance
against the same candidate catalog used by step projection. Loading does not
activate a same-step call or authorize a write.
`toolbox_inspect` reads an exact prepared tool contract without loading or executing
it. Load and inspection derive their contracts from the same registry. Load
receipts retain compact incomplete references; they do not copy execution
schemas into every load result. Schemas obtained through
inspection remain in the exact transcript after the callable set changes. Exact inspection returns either the complete contract or
bounded fragments; use its `contract_digest`, child JSON Pointer `path`, and
`next_offset` to retrieve omitted portions. Partial data never claims completeness.
An action, in any run, uses the normal exact approval at invocation. Inspection checks current local
applicability within the frozen candidates; remote identity/schema checks still
belong to invocation. Framework inbox
destinations concern report delivery; an MCP action's destination is defined by
that action's admitted schema.

The manifest reports the access modes and operational effects of all prepared
candidates, including unloaded tools. A read-only Sources group does not claim
update capability. These facts grant no permission; current revocation, receipt
blocks and approval still apply. Search cannot add an absent effect. Independent
schema/discovery calls or grounded value-read/load calls can share a model step;
loaded execution schemas become callable only on the next step. A write is one
call: Daita builds its own preview and shows it on the approval card.

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

For an ordinary local foreground run with computer file access, preparation also
freezes a small host-generated map containing the working directory, home, and
the resolved Downloads, Documents, and Desktop locators. It does not inventory
those directories or prove that they exist or are readable. The model can use
working-relative, absolute, or `~/` paths with the Files tools, whose results
report qualified locators and search coverage. Bounded typed-workspace callers
instead retain relative-only guidance. Hosted and machine-originated runs receive
neither this host context nor ambient local-file execution authority.

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
The explicit `files_only=True` option scopes the run to exactly the live session
folder source, which excludes catalog and MCP operations. The session folder
source is never stored, and a narrowed foreground run keeps local files: explicit
`source_scope_ids` narrow catalog sources only.
Machine runs use their frozen authorization ceilings; an empty machine source
ceiling means no sources.

Inside a run the model names sources by display name, tables and files by
native name, and earlier results by short run-local handles. The transcript
keeps the model's text as written, so its tool-call arguments contain those
names, and runs stored before names were used contain run-local source and table
handles. Use `agent.tool_evidence(run_id)` when you need each call as it
executed. It returns `(call, result)` pairs whose arguments are resolved to full
IDs, keeping the tool's own shape,
with the same resolver the runtime used. Approvals and receipts match these
calls. A call that never resolved, such as one rejected for a bad reference,
appears as the model wrote it.

```python
for call, result in await agent.tool_evidence(result.run_id):
    print(call.name, dict(call.arguments), result is not None and result.is_error)
```

Conversation continuity is shared across sources and stays bounded. Private
answers retain their classification when summarized, when a later question
narrows its source filter, and after a source is detached or the agent reopens.
The full request classification also includes admitted connector descriptions,
local-file context, current tool results, and retained advisory content. A model
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
does not override it. Semantic annotations retain their classification in
their existing SQLite records.

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

Procedure guidance follows the authenticated callable working set. The run-start
time and question-specific notes ride on the question, not the instructions, so
repeated questions reuse the provider's prompt cache. When a needed
tool is a prepared candidate but is not loaded, guidance names the exact load and
requires invocation on a later step. Missing access is a scope limitation; search
cannot grant it. Missing columns and keys remain unknown. Current source evidence
can be reused without a redundant schema call. Denied, failed or uncertain effects
are reported without retry or replacement; recovery performs no action.

When routine creation or revision is callable, the context adds one line:
scheduled runs ask a person before any write. A routine needs no model-route,
budget or grant choices; each run uses the agent's current model and normal
run limits. Machine runs receive no routine-management tools.
