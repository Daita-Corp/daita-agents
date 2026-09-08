# Context and source scope

Daita discovers currently admitted readable sources for an ordinary question.
There is no selected-source mode. Every data call still names its exact source
and resource, and current permissions are checked before I/O.

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
For grant-requiring capabilities, discovery includes the exact automation contract
used by scheduling. It does not activate execution tools. A contract too large for
the bounded page is omitted in full with `automation_contract_omitted`; loading
that exact tool returns the complete declaration.

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
