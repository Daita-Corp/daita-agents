# Learning your data systems

Daita can retain what it learns about a business and its data systems, so
later investigations start with better context. Its catalog supplies current
structure; approved knowledge adds definitions, preferences, and procedures.
The catalog and current tool results remain the source of truth for live data.

| Saved knowledge | Use it for |
| --- | --- |
| Agent memory | Business definitions that apply across sources. |
| User profile | Stable preferences about how to work and report. |
| Semantic annotations | Meanings tied to exact current sources, resources, fields, and evidence. |
| Markdown skills | Reusable procedures that Daita can read when relevant. |

In the terminal app, use `/learn <material>` to teach a definition or
procedure, or `/learn` alone right after an answer to keep what worked. Daita
chooses the fitting form, checks current context when needed,
and presents the exact change for approval. Use `/memory` to inspect saved
knowledge and semantic annotations, `/user` for your profile, and `/skills`
to manage procedures. A skill can guide work but cannot connect to a source
or grant an action.

The model reaches this knowledge through a few tools. `memory_set` adds,
replaces or removes one entry of agent memory or the user profile; replace and
remove name the exact current text they change, so the approval card shows only
that edit. `skill_view` reads one skill, and `skill_save` creates or replaces
one; with `action: "delete"` it removes one named skill instead. `semantic_view` lists annotations when no `id`
is given (filtered by source, resource, kind and limit) and loads one
annotation with its `current_sha256` when an `id` is given. `semantic_save`
creates, replaces or supersedes an annotation; with `action: "delete"` it
deletes one. Each save and delete keeps its own exact approval card, and
replacement, supersession and deletion all require the `expected_sha256` from
the latest view.

Daita does not silently promote a model's inference into durable knowledge:
it learns only when you teach it with `/learn`, or ask it to
remember something, and every change waits for your approval.

Semantic annotations remain tied to current catalog identities and evidence.
Metrics referencing a database table or column belong in semantic annotations,
even when requested as “memory.” Daita inspects current catalog structure and
saves the complete definition with exact resource, field, revision and evidence
bindings. The memory tool rejects qualified or backticked references and exact
underscore identifiers matching currently admitted catalog fields; this narrow
structural check does not classify all free-form prose.
If source structure changes, Daita must recheck that grounding before using
the annotation. A similar or renamed column does not prove the approved metric
still applies; confirm the mapping before calculating or replacing its binding.
Saved text is classified for information handling and treated
as untrusted advice. It cannot override current schema, access permissions,
model limits, or approval.

Scheduled routines retain their own approved instructions and exact skill
versions. Later changes to general memory or skills do not silently alter an
already authorized routine. See [context and source scope](CONTEXT_AND_SCOPE.md)
for the detailed retention and sensitivity rules.
