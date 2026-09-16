# Local computer files

A plain local Daita session provides a read-first Files surface for ordinary
files allowed by the host OS. The launch directory is the working directory,
not the boundary of computer access. Files are not registered data sources and
are not cataloged as SQLite, PostgreSQL, CSV, or JSON. The Files domain never
owns a general writer. One existing text file can change only through the
committed artifact and approved exact-target delivery workflow described below.

## Launching Daita

Run `daita` from a project or from your home directory. Relative paths start at
that frozen working directory. Absolute paths and `~/` paths can address other
ordinary local locations in the same foreground session, including Downloads,
Documents, Desktop, project folders, and mounted volumes. No upload, folder
registration, workspace switch, or restart is required.

Pass an explicit directory to choose a different default working directory:

```bash
daita --workspace /absolute/path/project
daita --workspace /absolute/path/project --workspace-sensitivity confidential
```

Without `--workspace`, the CLI uses the current directory. The user's home is a
valid working directory. If the launch directory is inside private Daita state,
the CLI uses home instead. Daita does not create or select a `~/Daita Workspace`
fallback. The allowed sensitivity labels are `internal`, `confidential`, and
`restricted`; `internal` is the default. Local-file sensitivity applies before
the first model request, including turns that do not ultimately read a file.

The terminal status and `/workspace` command show computer access and the
working directory. Use
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

Typed local callers construct `LocalWorkspace` explicitly. Its default remains
the contained, bounded-workspace contract:

```python
from pathlib import Path

from daita import Agent, LocalFileAccess, LocalWorkspace

workspace = LocalWorkspace(Path("/absolute/path/project"))
agent = await Agent.open("atlas", workspace=workspace)

# Opt in to the same computer-access behavior used by the local CLI.
computer = LocalWorkspace(
    Path("/absolute/path/project"),
    access=LocalFileAccess.COMPUTER,
)
```

For bounded mode, the workspace and agent-state roots must not overlap in either
direction, and filesystem root/home cannot be the workspace. Computer mode can
use home or a directory that contains private state, but direct state access is
rejected and broad searches prune state subtrees. Hosted compositions have no
ambient local backend, and scheduled, follow-up, and other machine-originated
runs cannot use foreground computer access.

## Read boundary

`file_search`, `file_read`, and `file_query` accept working-relative, absolute,
and `~/` paths in computer mode. `file_search` accepts either one `path` or a
`paths` list of one to eight roots. A paths-mode filename search may omit
`query` when it supplies a filename-only glob such as `*.csv`; traversal is
already recursive, so `**/*.csv` is not a valid filename glob. Multi-root
searches share one time, entry, content-byte, depth, and result budget. Results
include qualified locators plus per-root coverage so a missing or inaccessible
root is distinguishable from an empty directory.

Daita rejects or skips:

- `..` traversal, URLs, other-user `~name` forms, control characters, and symlinks;
- sockets, devices, FIFOs, and other special files;
- Daita private state and secret-like paths such as `.env`, `.ssh`, private
  keys, credential stores, and VCS-internal material;
- binary content for text reads and content search; and
- files that change while an authenticated cursor or binding is in use.

OS permissions remain authoritative; Daita does not elevate privileges or
change privacy settings. Search and read results, including path labels and
excerpts, are untrusted data.
They cannot authorize tool loading, source access, writes, memory changes, or
skill changes. Computer-mode context includes only the frozen working directory,
home, and host-resolved Downloads, Documents, and Desktop locators. A resolved
known-folder path is not a claim that the directory exists or was readable.
Bounded-workspace mode continues to expose only relative paths.

## Structured file queries

`file_query` is an on-demand Files tool for direct analysis of one homogeneous
CSV, TSV, JSON-records/NDJSON, or Parquet dataset. Its `path_pattern` follows
the same path rules and is expanded by Daita without a shell. Daita opens and
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
artifact never changes the local file.

The final save derives its only target from that committed binding and asks
once for approval with a qualified target and bounded change summary. Daita verifies
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
