# Artifact byte storage

`AgentHomeArtifactStore` owns publication, read, deletion and recovery. Its
registry owns artifact identity, caller ownership, quotas and the
`creating` → `ready` → `deleting` lifecycle. Applications may supply an
`ArtifactByteStorage` implementation instead of the default local byte layout.
This replaces physical byte I/O without replacing the artifact lifecycle.

## Supplying an adapter

```python
from daita.artifacts.bytes import ArtifactByteStorage
from daita.artifacts.store import AgentHomeArtifactStore


async def open_artifacts(owned_state, storage: ArtifactByteStorage, agent_id: str):
    return await AgentHomeArtifactStore.open(
        agent_id=agent_id,
        registry=owned_state,
        byte_storage=storage,
    )
```

Supply exactly one of `agent_home` or `byte_storage`. External storage does not
expose a fabricated filesystem path. The caller owns credentials, transport,
storage layout and client lifetime; it must drain operations before closing the
client. Methods run in worker threads and must have finite timeouts.

- `publish(agent_id, ref, content)` creates immutable bytes exclusively and must
  reject an existing identity rather than overwrite it.
- `read(agent_id, ref)` returns bounded bytes, or `None` only for confirmed absence.
  Denied access and unavailable storage remain errors. The lifecycle owner checks
  the registered length and checksum before accepting returned bytes.
- `delete(agent_id, ref)` is idempotent for an absent object.

Normalize adapter failures into `ArtifactError`. The registry retains the full
reference, so an object listing or additional inventory cannot grant access.
Concrete service clients, authorization and physical retention stay with the
application implementing the adapter.

## Recovery and ownership

Creation reserves a registry row before publication. Recovery reads back that
exact identity; a lost acknowledgement never triggers automatic publication
replay. Unavailable readback preserves pending state. Deletion hides the entry
before byte cleanup, and unfinished deletion can finish on reopen. Typed state
ownership-loss and unknown-commit failures propagate to the execution owner.

Cancellation drains a dispatched publication before reconciliation. Opening the
store may recover pending rows: its caller must already hold exclusive execution
ownership and fence the previous writer. This component does not provide an
execution lease or external garbage collector.

Public `Agent.create/open` select local homes. `Agent.from_storage` accepts this
byte interface alongside caller-admitted state and advisory documents; see the
[storage contract](STORAGE_CONTRACT.md). Local manifests, released home revisions
and the existing release snapshot remain unchanged. Concrete adapter layout
compatibility and records-plus-bytes restoration require qualification in the
consuming application.

`tests/storage/contracts/test_artifact_bytes.py` exercises the real lifecycle
with a disposable byte dependency: reopen, caller filtering, publication
uncertainty, unavailable readback, failed deletion, state failure, corruption and
cancellation. Those cases do not qualify an external service's transport.
