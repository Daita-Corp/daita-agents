# Local workspaces

Every local Daita session admits exactly one workspace. Workspace admission is
a bounded, read-first Files surface; it is not a registered data source and is
not cataloged as SQLite, PostgreSQL, CSV, or JSON. The Files domain never owns
a writer. One existing file can change only through the separate committed
artifact and approved exact-target delivery workflow described below.

## Launching Daita

Pass an explicit absolute directory when predictable selection matters:

```bash
daita --workspace /absolute/path/project
daita --workspace /absolute/path/project --workspace-sensitivity confidential
```

Without `--workspace`, the CLI uses the current directory when it is safe and
does not overlap Daita state. Otherwise it creates or reuses
`~/Daita Workspace`. The allowed sensitivity labels are `internal`,
`confidential`, and `restricted`; `internal` is the default. Workspace
sensitivity applies before
the first model request, including turns that do not ultimately read a file.

The terminal status and `/workspace` command show the admitted workspace. Use
`/files <question>` for a turn that omits attached source, MCP, and source-job
tools. Ordinary user turns may use both the Files tools and the selected data
source.

For headless use:

```bash
daita --root /private/tmp/daita \
  --workspace /absolute/path/project \
  run atlas "Summarize the release notes" --files-only
```

## Python API

Local callers must construct the workspace explicitly:

```python
from pathlib import Path

from daita import Agent, LocalWorkspace

workspace = LocalWorkspace(Path("/absolute/path/project"))
agent = await Agent.open("atlas", workspace=workspace)
```

The workspace and agent-state roots must not overlap in either direction. The
filesystem root, the user's home directory, missing directories, and
non-directories are rejected.

## Read boundary

`file_search` and `file_read` use workspace-relative logical paths and return
bounded results. Daita rejects or skips:

- `..` traversal, absolute paths, path aliases, and symlinks;
- sockets, devices, FIFOs, and other special files;
- secret-like paths such as `.env`, private keys, credential stores, and
  VCS-internal secret material;
- binary content for text reads and content search; and
- files that change while an authenticated cursor or binding is in use.

Search and read results, including file names and excerpts, are untrusted data.
They cannot authorize tool loading, source access, writes, memory changes, or
skill changes. Absolute workspace paths are never placed in model requests,
tool results, transcripts, artifact provenance, or durable state.

## Structured file queries

`file_query` is an on-demand Files tool for direct analysis of one homogeneous
CSV, TSV, JSON-records/NDJSON, or Parquet dataset. Its `path_pattern` is always
workspace-relative and is expanded by Daita without a shell. Daita opens and
revision-binds every exact regular file before execution, rejects mixed formats
or incompatible schemas, and records every input path and physical revision
without persisting a workspace inventory.

The SQL contract is one canonical read-only `SELECT` over exactly one relation
named `data`. It cannot introduce another table, raw path, table/filesystem
function, URL, setting, secret, extension operation, DDL, or DML. DuckDB 1.5.5
runs only in a fresh private one-call worker with extension installation and
autoload, community extensions, persistent secrets, external access, and
network filesystems disabled before configuration is locked. Local file
queries need no S3 configuration.

One call admits at most 1,000 files and 256 MiB of physical input, with a
complete encoded manifest capped at 256 KiB. Results expose at most 100 rows
within a bounded JSON projection; query time is 30 seconds and private spill is
monitored at 2 GiB. The selected 256 MiB DuckDB `memory_limit` is a
buffer-manager target, not a hard process-memory ceiling. Parent-side RSS,
spill, timeout, and cancellation monitoring terminates and reaps the isolated
worker and removes its private scratch state.

## Targeted text edits

For one bounded UTF-8 text file, Daita can perform the cohesive sequence
`file_read` → `artifact_edit_text` → approval → `artifact_save_local`. The edit
tool accepts only the authenticated current-run binding returned by
`file_read`; it does not accept a path, revision, or file bytes. It applies
ordered exact replacements, including exact-anchor insertion and deletion,
then commits the complete replacement as an internal artifact. Preparing the
artifact never changes the workspace.

The final save derives its only target from that committed binding and asks
once for approval with a bounded relative-path change summary. Daita verifies
the exact file identity, revision, content hash, ownership, links, metadata,
and parent-directory safety again after approval. It writes and verifies the
complete output beside the target, preserves safe mode and ownership, fsyncs,
atomically replaces, and records a succeeded, failed, or uncertain receipt.
If the source changes at any point, Daita requires a fresh read and edit; it
does not merge, rebase, redirect, or retry the mutation.

There is no `file_write`, raw byte mutation, terminal execution, binary or rich
document editor, or generic filesystem API. New files continue through a
committed artifact and `artifact_save_local` in `create_new` mode. Hosted and
machine-originated runs receive no ambient local edit authority.

## Development-state compatibility

An unreleased development home containing the removed cataloged file-source
registration is rejected during admission. Delete and recreate that agent;
unreleased state has no compatibility alias or migration.
