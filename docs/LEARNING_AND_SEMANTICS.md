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
procedure. Daita chooses the fitting form, checks current context when needed,
and presents the exact change for approval. Use `/memory` to inspect saved
knowledge and semantic annotations, `/user` for your profile, and `/skills`
to manage procedures. A skill can guide work but cannot connect to a source
or grant an action.

## Suggestions from completed work

`/review` can inspect completed runs and suggest reusable lessons. This review
is off by default and needs an explicit cost limit when authorized. Suggestions
enter an inactive candidate inbox. Inspect, edit, accept, or reject them through
`/memory`; accepting one candidate starts a fresh foreground run with the
ordinary checks and approval. Daita does not silently promote a model's
inference into durable knowledge.

Semantic annotations remain tied to current catalog identities and evidence.
If source structure changes, Daita must recheck that grounding before using
the annotation. Saved text is classified for information handling and treated
as untrusted advice. It cannot override current schema, access permissions,
model limits, or approval.

Scheduled routines retain their own approved instructions and exact skill
versions. Later changes to general memory or skills do not silently alter an
already authorized routine. See [context and source scope](CONTEXT_AND_SCOPE.md)
for the detailed retention and sensitivity rules.
