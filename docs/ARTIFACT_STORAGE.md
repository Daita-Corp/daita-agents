# Artifact byte storage

The artifact registry owns identity, caller ownership, quotas and the
`creating` → `ready` → `deleting` lifecycle. `AgentHomeArtifactStore` remains
the one publication, read, deletion and recovery owner. It can keep bytes in
the existing local home layout or use an explicitly supplied
`ArtifactByteStorage`. Both paths verify the registered length and SHA-256
before accepting bytes. Neither object names nor bucket listings grant access.

This is a composition component. Public `Agent.create/open` still use local
homes. Complete remote agent composition, execution leases and home upgrades
remain required before a deployment can retire its filesystem. Opening this
component can recover pending artifact records, so its caller must already own
the agent's exclusive writer lease and have fenced the prior writer.

## S3 bytes

`daita.artifacts.s3.S3ArtifactByteStorage` accepts a caller-owned S3 SDK client,
bucket and private prefix. The client must use verified TLS, finite connection
and read timeouts, and no automatic retries. The caller owns credential
resolution, endpoint/region selection, IAM scope and client cleanup. The module
does not import an SDK or resolve ambient credentials.

```python
from daita.artifacts.s3 import S3ArtifactByteStorage
from daita.artifacts.store import AgentHomeArtifactStore


async def open_artifacts(owned_state, scoped_s3_client, agent_id):
    return await AgentHomeArtifactStore.open(
        agent_id=agent_id,
        registry=owned_state,
        byte_storage=S3ArtifactByteStorage(
            scoped_s3_client,
            bucket="private-artifacts",
            prefix="homes/example",
        ),
    )
```

The prefix is an administrator-selected logical-home boundary. Scope credentials
to it; a key prefix by itself is not an authorization mechanism. An artifact is
one raw-payload object at
`{prefix}/{agent_id}/{run_id}/{artifact_id}/payload`. The database registry owns
its full reference, including content hash, size, provenance and sensitivity.
No second remote manifest is maintained. Bucket policy must deny public access
and enforce encryption. Writes request SSE-S3 by default; `kms_key_id` selects
SSE-KMS instead. Bucket provisioning and policy changes are external operations.

Publication uses `If-None-Match: *` and cannot overwrite an existing key. A row
is reserved before upload; only readback of the exact bytes promotes it to
ready. A failed upload is not automatically replayed. Lost acknowledgement can
leave a ready artifact even though the caller received an error: recovery reads
the existing registry identity rather than creating another artifact. Unavailable
or denied readback preserves the pending row; only `NoSuchKey` means absence.
Reads reject an unexpected content length before downloading and bound the body
read by the registered length. Response bodies are closed after success/failure.

Deletion first hides the registry entry, then deletes the current object, then
removes the pending row. Failed object cleanup remains hidden and can finish on
reopen. State ownership loss and unknown database commits propagate unchanged;
they are not translated into ordinary retryable artifact failures. In a versioned
bucket, deleting the current object does not purge historical versions. Retention,
backup/restore and physical erasure must include those versions explicitly.

Cancellation drains an already dispatched bounded upload before reconciliation.
The caller must drain the agent before closing the S3 client or handing ownership
to another worker. The byte backend does not supply a distributed lease, garbage
collector or a second artifact inventory. Actual S3 termination/handover and
records-plus-bytes restore still need deployment qualification.

## Compatibility and verification

Local manifests and released home revisions are unchanged. The existing home
release snapshot fingerprints the initial S3 key/payload layout alongside the
PostgreSQL contract. Once released, changing that layout requires the same shared
home revision to advance; there is no S3-specific migration/version sequence.

`tests/storage/contracts/test_artifact_bytes.py` exercises the real lifecycle and
both state implementations with a disposable injected object-service dependency.
It covers reopen, caller filtering, lost upload acknowledgements, unavailable
readback, failed deletion, typed state failures, corruption and cancellation.
`tests/artifacts/test_s3_bytes.py` checks conditional/encrypted/scoped requests,
bounded reads and response-body cleanup. These are not live S3 or SDK transport
qualification. The existing local artifact tests remain in place.
