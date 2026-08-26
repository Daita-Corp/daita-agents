# Local workspaces

Every local Daita session admits exactly one workspace. The workspace is a
bounded, read-only Files surface; it is not a registered data source and is not
cataloged as SQLite, PostgreSQL, CSV, or JSON.

## Launching Daita

Pass an explicit absolute directory when predictable selection matters:

```bash
daita --workspace /absolute/path/project
daita --workspace /absolute/path/project --workspace-sensitivity confidential
```

Without `--workspace`, the CLI uses the current directory when it is safe and
does not overlap Daita state. Otherwise it creates or reuses `~/Daita
Workspace`. The allowed sensitivity labels are `internal`, `confidential`, and
`restricted`; `internal` is the default. Workspace sensitivity applies before
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

`file_search` and `file_read` are the only Phase 2 workspace tools. They use
workspace-relative logical paths and return bounded results. Daita rejects or
skips:

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

This phase adds no `file_write`, file editing, terminal execution, structured
file query engine, or generic filesystem API. New output continues through a
committed artifact and the existing `artifact_save_local` publication path.

## Pre-production state

Development agent homes created before this workspace slice are disposable.
If a home contains the removed cataloged file-source registration, Daita
rejects it during admission; delete and recreate that development agent. There
is no compatibility alias or migration for unreleased state.
