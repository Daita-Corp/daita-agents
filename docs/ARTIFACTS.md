# Artifacts

Daita keeps generated reports and exports as artifacts in the agent home. You
can inspect or reuse them later. The payload and manifest live under
`<state-root>/agents/<agent-name>/artifacts/<run-id>/<artifact-id>/`; the default
state root is `~/.daita`. Saving an additional copy to a user-selected directory
requires explicit approval. Opening the home recovers interrupted artifact
creation and deletion.

## Creating artifacts

Choose a tabular tool by the kind of evidence you need:

- `data_export_tabular` runs one validated relational query directly against
  exact current catalog resources and creates a complete CSV or XLSX artifact
  within fixed source-export bounds. Its provenance is exact source data.
- `artifact_create_tabular` packages bounded model-authored findings as CSV,
  XLSX, or HTML. It requires one or more exact earlier successful tool-call IDs
  from the current run. Those results may come from catalog-backed data,
  local files, or admitted MCP tools.

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

## Reading and saving artifacts

In the TUI, `/artifacts` opens all stored artifacts for the current agent,
across conversations. The list shows short display names, format, size, and
local creation time; full filenames and technical metadata are under **Details**.
Use **↑/↓** to select and **Enter** to preview formatted JSON, Markdown, text,
or CSV/XLSX tables. **Save copy** uses the configured export folder;
**Delete** permanently removes the stored artifact after confirmation.
Use **Previous** and **Next** to browse pages and **Refresh** to reload the
inventory. Saved copies and conversation history remain after deletion.
**PgUp/PgDn** scroll the preview;
**i** toggles Details and **Esc** closes the screen.

The owner API provides the same paged inventory:

```python
artifacts = await agent.list_artifacts(limit=50, offset=0)
```

Hosted listings filter by authenticated `caller_principal_id`. Only ready
artifacts appear; clearing conversation history does not remove them.

`artifact_list` lists metadata for the current conversation. `artifact_read`
previews an exact artifact ID. `artifact_convert` converts a verified exact
Daita XLSX snapshot to CSV without rerunning its source. `artifact_save_local`
saves a copy to an approved local destination.

An edit artifact retains the exact physical anchor, anchor-relative path,
revision, and original content hash authenticated by `file_read`. This applies
equally to the working directory and an external local location. Replacement
approval displays the qualified target, then the delivery path reopens and
revalidates that same anchor and file; it never rebases a failed external edit
onto the working directory or Downloads.

## Deleting an artifact

Remove one committed artifact through the owner API:

```python
newly_deleted = await agent.delete_artifact(artifact_id)
```

In a hosted application, pass the authenticated `caller_principal_id` used for
the producing run; the default is the agent owner. Model tools do not expose
deletion.

Deletion immediately blocks reads and saves, then removes the payload, manifest,
and registry entry. It returns `True` for a new deletion and `False` if the ID is
absent or cleanup was already pending. Errors use `ArtifactError`:

- `artifact_missing`: a read or save targets an unavailable artifact.
- `artifact_busy`: an active job still owns the artifact reservation; wait for
  the attempt to finish before deleting it.
- `artifact_storage_failed` with stage `delete_cleanup`: file cleanup is
  incomplete. Retry deletion or reopen the home to resume cleanup.

Clearing conversation history preserves stored artifacts. Deleting an artifact
preserves historical transcripts, job results, and delivery outcomes, including
embedded text previews. Exported files, application delivery copies, and backups
also remain. Application hosts must delete their own copies and coordinate
deletion with any publication of previously fetched bytes.
