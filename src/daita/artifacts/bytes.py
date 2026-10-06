"""Physical byte I/O beneath the existing artifact lifecycle owner."""

from typing import Protocol

from .models import ArtifactRef


class ArtifactByteStorage(Protocol):
    """A caller-owned immutable byte store, already scoped by its credentials.

    Methods run in worker threads and must have bounded transport timeouts.
    ``publish`` must create an object exclusively; it must never overwrite one.
    ``read`` returns ``None`` only for a confirmed absent object, never for an
    unavailable service or denied access, and must bound its response by
    ``ref.byte_size``. ``delete`` is idempotent for an absent object.
    Translate storage-specific failures into ``ArtifactError``; denied or
    unavailable reads must never masquerade as missing bytes.

    The artifact registry remains the authority for identity, ownership, quotas
    and lifecycle. This interface does not list objects or recover domain state.
    The caller owns the client's lifetime and closes it after the agent drains.
    """

    def publish(self, agent_id: str, ref: ArtifactRef, content: bytes) -> None: ...

    def read(self, agent_id: str, ref: ArtifactRef) -> bytes | None: ...

    def delete(self, agent_id: str, ref: ArtifactRef) -> None: ...
