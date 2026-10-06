"""In-memory byte dependency for the real registry-owned artifact lifecycle."""

from daita.artifacts.models import ArtifactError, ArtifactRef


class MemoryByteStorage:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str, str], bytes] = {}
        self.publications: list[tuple[str, str, str]] = []
        self.reads: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, str, str]] = []

    def publish(self, agent_id: str, ref: ArtifactRef, content: bytes) -> None:
        key = (agent_id, ref.run_id, ref.artifact_id)
        self.publications.append(key)
        if key in self.objects:
            raise ArtifactError("artifact_storage_failed", "Identity already exists.")
        self.objects[key] = content

    def read(self, agent_id: str, ref: ArtifactRef) -> bytes | None:
        key = (agent_id, ref.run_id, ref.artifact_id)
        self.reads.append(key)
        content = self.objects.get(key)
        return None if content is None else content[: ref.byte_size + 1]

    def delete(self, agent_id: str, ref: ArtifactRef) -> None:
        key = (agent_id, ref.run_id, ref.artifact_id)
        self.deleted.append(key)
        self.objects.pop(key, None)
