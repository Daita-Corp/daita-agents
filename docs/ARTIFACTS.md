# Artifacts

Daita stores generated files as internal artifacts before any optional local
delivery. The existing artifact store is the only boundary that commits
artifact bytes and manifests.

## Exact exports and derived findings

Use the two tabular paths for different guarantees:

- `data_export_tabular` runs one validated relational query directly against
  exact current catalog resources and creates a complete CSV or XLSX artifact
  within fixed source-export bounds. Its provenance is exact source data.
- `artifact_create_tabular` packages bounded model-authored findings as CSV,
  XLSX, or HTML. It requires one or more exact earlier successful tool-call IDs
  from the current run. Those results may come from catalog-backed data,
  workspace files, or admitted MCP tools.

`artifact_create_tabular` does not claim that its rows are a complete or exact
copy of a source. Daita authenticates the referenced results against the
current run transcript and immutable capability registry, inherits their
highest sensitivity, and records their call IDs. Relational evidence also
retains exact current catalog resource revisions.

Use `artifact_create_document` for bounded Markdown or plain-text narrative.
Its optional `evidence_call_ids` receive the same authentication, sensitivity,
and provenance treatment. Use `artifact_snapshot_result` when the requirement
is an exact canonical JSON copy of one validated structured tool result rather
than model-authored analysis.

## Formats and safety

Formats are arguments to stable semantic tools rather than separate tools.
CSV values receive spreadsheet-formula protection, XLSX workbooks are
literal-only fixed packages without formulas or external relationships, and
HTML tables escape all model-authored values and prohibit external content.
Rows, columns, cell text, input bytes, output bytes, execution time, and
per-run artifact totals remain bounded.

`artifact_list` and `artifact_read` expose bounded artifact metadata and
previews. `artifact_convert` converts only a verified exact Daita XLSX snapshot
to CSV without rerunning its source. `artifact_save_local` remains the explicit
approval-gated local delivery path; creating an internal artifact does not
prove that a local file was saved.
