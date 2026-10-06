"""Keep advisory semantics in their owners when physical persistence is supplied."""

import asyncio
import threading
from contextlib import contextmanager
from hashlib import sha256

import pytest

from daita.llm.models import ModelSensitivity
from daita.memory import MemoryStore, MemoryValidationError
from daita.skills import SkillPathError, SkillStore, SkillValidationError
from daita.storage.errors import (
    StorageCommitUnknownError,
    StorageError,
    StorageOwnershipLostError,
)

pytestmark = [pytest.mark.contract, pytest.mark.integration]


async def test_memory_roundtrip_labels_bounds_and_revision(advisory_storage):
    memory = MemoryStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        assert await memory.read_context() == ("", "", ModelSensitivity.PUBLIC)
        before = await memory.preflight_replacement("memory", "naïve 🧭")
        await memory.set_memory("naïve 🧭", sensitivity=ModelSensitivity.INTERNAL)
        await memory.set_user_profile(
            "Concise", sensitivity=ModelSensitivity.RESTRICTED
        )
        assert await memory.read_context() == (
            "naïve 🧭",
            "Concise",
            ModelSensitivity.RESTRICTED,
        )
        first = await memory.preflight_replacement("memory", "new")
        await memory.set_memory("temporary")
        await memory.set_memory("naïve 🧭", sensitivity=ModelSensitivity.INTERNAL)
        again = await memory.preflight_replacement("memory", "new")
        assert not before[0] and first[0]
        assert first[1] == again[1] and first[2] != again[2]
        with pytest.raises(MemoryValidationError):
            await memory.set_memory("x" * 2201)
        assert await memory.read_memory() == "naïve 🧭"
    finally:
        await memory.close()
    reopened = MemoryStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        assert await reopened.read_context() == (
            "naïve 🧭",
            "Concise",
            ModelSensitivity.RESTRICTED,
        )
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    "content", [b"\xff", b"<!-- daita-sensitivity: invalid -->\ntext", b"x" * 2201]
)
async def test_memory_corruption_is_not_replaced(advisory_storage, content):
    with advisory_storage.transaction(write=True) as transaction:
        transaction.put("memory", "MEMORY.md", content)
    memory = MemoryStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        with pytest.raises(MemoryValidationError):
            await memory.set_memory("replacement")
        with advisory_storage.transaction() as transaction:
            assert (
                transaction.get("memory", "MEMORY.md", max_bytes=8880).content
                == content
            )
    finally:
        await memory.close()


async def test_skill_preflight_and_retained_content_survive_replacement_and_delete(
    advisory_storage,
):
    skills = SkillStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        assert await skills.save_skill("report", "Report", "Use exact facts.")
        assert not await skills.save_skill("report", "Report", "Use exact facts.")
        original, digest = await skills.read_skill_with_digest("report")
        assert original is not None
        assert (
            await skills.retain_current_skill("report", "sha256:" + digest) == original
        )
        before = await skills.preflight_delete("report")
        await skills.save_skill("report", "Changed", "Read current facts.")
        await skills.save_skill("report", "Report", "Use exact facts.")
        after = await skills.preflight_delete("report")
        assert before[1] == after[1] and before[2] != after[2]
        assert await skills.delete_skill("report")
        assert not await skills.delete_skill("report")
    finally:
        await skills.close()
    reopened = SkillStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        assert await reopened.list_skills() == ()
        assert (
            await reopened.read_retained_skill("report", "sha256:" + digest) == original
        )
        with advisory_storage.transaction(write=True) as transaction:
            transaction.put("retained-skills", digest, b"corrupted")
        with pytest.raises(SkillPathError, match="digest"):
            await reopened.read_retained_skill("report", "sha256:" + digest)
    finally:
        await reopened.close()


async def test_skill_count_is_enforced_inside_serialized_write(advisory_storage):
    first = SkillStore(None, asyncio.Lock(), storage=advisory_storage)
    second = SkillStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        for index in range(31):
            await first.save_skill(f"skill-{index}", "Small", "Instructions.")
        results = await asyncio.gather(
            first.save_skill("last-one", "Small", "Instructions."),
            second.save_skill("last-two", "Small", "Instructions."),
            return_exceptions=True,
        )
        assert sum(value is True for value in results) == 1
        assert sum(isinstance(value, SkillValidationError) for value in results) == 1
        assert len(await first.list_skills()) == 32
    finally:
        await first.close()
        await second.close()


async def test_skill_index_limit_and_invalid_current_documents_fail_closed(
    advisory_storage,
):
    skills = SkillStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        for index in range(15):
            await skills.save_skill(f"skill-{index}", "x" * 240, "Instructions.")
        with pytest.raises(SkillValidationError, match="index"):
            await skills.save_skill("excess", "x" * 240, "Instructions.")
        assert len(await skills.list_skills()) == 15
        with advisory_storage.transaction(write=True) as transaction:
            transaction.put("skills", "broken", b"not a skill")
        with pytest.raises(SkillValidationError):
            await skills.delete_skill("broken")
    finally:
        await skills.close()


async def test_retention_count_is_idempotent_and_bounded(advisory_storage):
    skills = SkillStore(None, asyncio.Lock(), storage=advisory_storage)
    try:
        await skills.save_skill("report", "Report", "Use exact facts.")
        original, digest = await skills.read_skill_with_digest("report")
        with advisory_storage.transaction(write=True) as transaction:
            # Valid immutable documents fill the real owner limit.
            for index in range(256):
                content = f"<!-- daita-sensitivity: restricted -->\n# old-{index}\n\nOld\n\n## Instructions\n\nOld procedure.\n".encode()
                transaction.put("retained-skills", sha256(content).hexdigest(), content)
        with pytest.raises(SkillValidationError, match="count"):
            await skills.retain_current_skill("report", "sha256:" + digest)
        with advisory_storage.transaction(write=True) as transaction:
            current = transaction.get("skills", "report", max_bytes=50000)
            transaction.put("retained-skills", digest, current.content)
        assert (
            await skills.retain_current_skill("report", "sha256:" + digest) == original
        )
    finally:
        await skills.close()


def test_advisory_transaction_rollback_and_bounded_reads(advisory_storage):
    with pytest.raises(RuntimeError, match="abort"):
        with advisory_storage.transaction(write=True) as transaction:
            transaction.put("memory", "MEMORY.md", b"uncommitted")
            raise RuntimeError("abort")
    with advisory_storage.transaction() as transaction:
        assert transaction.get("memory", "MEMORY.md", max_bytes=8880) is None
    with advisory_storage.transaction(write=True) as transaction:
        transaction.put("memory", "MEMORY.md", b"large")
    with advisory_storage.transaction() as transaction:
        with pytest.raises(StorageError):
            transaction.get("memory", "MEMORY.md", max_bytes=1)


@pytest.mark.parametrize(
    "error_type", [StorageOwnershipLostError, StorageCommitUnknownError]
)
@pytest.mark.parametrize("owner", ["memory", "skills"])
async def test_advisory_failure_propagates_without_mutation_replay(
    advisory_storage, error_type, owner
):
    attempts = []

    class FailedStorage:
        @contextmanager
        def transaction(self, *, write=False):
            with advisory_storage.transaction(write=write) as transaction:
                yield transaction
                if write:
                    attempts.append("commit")
                    raise error_type("commit or ownership failed")

    memory = MemoryStore(None, asyncio.Lock(), storage=FailedStorage())
    skills = SkillStore(None, asyncio.Lock(), storage=FailedStorage())
    try:
        with pytest.raises(error_type):
            if owner == "memory":
                await memory.set_memory("replacement")
            else:
                await skills.save_skill("report", "Report", "Read records.")
        assert attempts == ["commit"]
    finally:
        await memory.close()
        await skills.close()


async def test_cancelled_advisory_write_settles_before_close(advisory_storage):
    entered, release = threading.Event(), threading.Event()

    class BlockingStorage:
        @contextmanager
        def transaction(self, *, write=False):
            with advisory_storage.transaction(write=write) as transaction:
                yield transaction
                if write:
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError("test did not release document commit")

    memory = MemoryStore(None, asyncio.Lock(), storage=BlockingStorage())
    writing = asyncio.create_task(memory.set_memory("settled"))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        writing.cancel()
        closing = asyncio.create_task(memory.close())
        await asyncio.sleep(0)
        assert not writing.done() and not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(writing, 3)
        await asyncio.wait_for(closing, 3)
        reopened = MemoryStore(None, asyncio.Lock(), storage=advisory_storage)
        try:
            assert await reopened.read_memory() == "settled"
        finally:
            await reopened.close()
    finally:
        release.set()
        await asyncio.gather(writing, return_exceptions=True)
        if closing is not None:
            await closing


@pytest.mark.parametrize("owner", ["memory", "skills"])
async def test_cancelled_preflight_waits_for_its_storage_snapshot(
    advisory_storage, owner
):
    entered, release = threading.Event(), threading.Event()

    class BlockingRead:
        @contextmanager
        def transaction(self, *, write=False):
            with advisory_storage.transaction(write=write) as transaction:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test did not release preflight")
                yield transaction

    memory = MemoryStore(None, asyncio.Lock(), storage=BlockingRead())
    skills = SkillStore(None, asyncio.Lock(), storage=BlockingRead())
    reading = asyncio.create_task(
        memory.preflight_replacement("memory", "replacement")
        if owner == "memory"
        else skills.preflight_save("report", "Report", "Instructions.")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        reading.cancel()
        await asyncio.sleep(0)
        assert not reading.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reading, 3)
    finally:
        release.set()
        await asyncio.gather(reading, return_exceptions=True)
        await memory.close()
        await skills.close()
