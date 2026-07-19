from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path

import pytest

from daita.skills import (
    Skill,
    SkillActivation,
    SkillActivationConflictError,
    SkillCapabilityUnavailableError,
    SkillDiscoveryError,
    SkillFormatError,
    SkillIndex,
    SkillInspection,
    SkillNotActiveError,
    SkillNotFoundError,
    SkillSelectionBudgetError,
    SkillSelectionReason,
    SkillService,
    SkillSource,
    SkillVersion,
)
import daita.skills.service as service_module

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


class InMemorySkillStore:
    def __init__(self) -> None:
        self.skills: dict[tuple[str, str], Skill] = {}
        self.versions: dict[tuple[str, str], SkillVersion] = {}
        self.index: dict[tuple[str, str], SkillIndex] = {}
        self.activations: dict[tuple[str, str], list[SkillActivation]] = {}
        self.version_loads: list[str] = []

    async def record_discovery(
        self,
        skill: Skill,
        version: SkillVersion,
        index: SkillIndex,
    ) -> SkillIndex:
        key = (skill.agent_id, skill.id)
        existing_skill = self.skills.get(key)
        if existing_skill is not None and (
            existing_skill.stable_name,
            existing_skill.source,
        ) != (skill.stable_name, skill.source):
            raise ValueError("skill identity conflict")
        for stored in self.versions.values():
            if (
                stored.agent_id == version.agent_id
                and stored.skill_id == version.skill_id
                and stored.version == version.version
                and stored.content_hash != version.content_hash
            ):
                raise ValueError("semantic version content conflict")
        self.skills.setdefault(key, skill)
        self.versions.setdefault((version.agent_id, version.id), version)
        current = self.index.get(key)
        if current is None or current.active_version_id is None:
            self.index[key] = index
        return self.index[key]

    async def list_skill_index(self, agent_id: str) -> tuple[SkillIndex, ...]:
        return tuple(
            value for (owner, _), value in self.index.items() if owner == agent_id
        )

    async def load_skill_index(
        self,
        agent_id: str,
        skill_id: str,
    ) -> SkillIndex | None:
        return self.index.get((agent_id, skill_id))

    async def load_skill_version(
        self,
        agent_id: str,
        version_id: str,
    ) -> SkillVersion | None:
        self.version_loads.append(version_id)
        return self.versions.get((agent_id, version_id))

    async def inspect_skill(
        self,
        agent_id: str,
        skill_id: str,
    ) -> SkillInspection | None:
        key = (agent_id, skill_id)
        skill = self.skills.get(key)
        index = self.index.get(key)
        if skill is None or index is None:
            return None
        versions = tuple(
            sorted(
                (
                    value
                    for (owner, _), value in self.versions.items()
                    if owner == agent_id and value.skill_id == skill_id
                ),
                key=lambda value: (value.created_at, value.id),
            )
        )
        activations = tuple(self.activations.get(key, ()))
        return SkillInspection(
            skill=skill,
            index=index,
            versions=versions,
            activations=activations,
        )

    async def activate_skill(
        self,
        activation: SkillActivation,
        *,
        expected_active_version_id: str | None,
    ) -> SkillInspection:
        key = (activation.agent_id, activation.skill_id)
        current = self.index[key]
        if current.active_version_id != expected_active_version_id:
            raise SkillActivationConflictError("stale activation")
        version = self.versions[(activation.agent_id, activation.version_id)]
        self.index[key] = SkillIndex.from_version(
            version,
            active_version_id=version.id,
            updated_at=activation.activated_at,
        )
        self.activations.setdefault(key, []).append(activation)
        inspection = await self.inspect_skill(activation.agent_id, activation.skill_id)
        assert inspection is not None
        return inspection


def _skill_text(
    name: str,
    *,
    version: str = "1.0.0",
    description: str = "Reconcile customer records.",
    mode: str = "on_demand",
    capabilities: tuple[str, ...] = ("data.file.read",),
    instructions: str = "Use accepted evidence and cite every result.",
    extra_metadata: str = "",
) -> str:
    capability_values = ", ".join(f'"{item}"' for item in capabilities)
    return (
        "+++\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        f'description = "{description}"\n'
        'domains = ["data"]\n'
        'resource_kinds = ["table"]\n'
        f"required_capability_ids = [{capability_values}]\n"
        f'activation_mode = "{mode}"\n'
        'sensitivity_notes = "Do not disclose raw rows."\n'
        'policy_notes = "Runtime governance always applies."\n'
        f"{extra_metadata}"
        "+++\n\n"
        f"{instructions}\n"
    )


def _write_skill(
    root: Path,
    name: str,
    *,
    version: str = "1.0.0",
    description: str = "Reconcile customer records.",
    mode: str = "on_demand",
    capabilities: tuple[str, ...] = ("data.file.read",),
    instructions: str = "Use accepted evidence and cite every result.",
    extra_metadata: str = "",
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    path = directory / "SKILL.md"
    path.write_text(
        _skill_text(
            name,
            version=version,
            description=description,
            mode=mode,
            capabilities=capabilities,
            instructions=instructions,
            extra_metadata=extra_metadata,
        ),
        encoding="utf-8",
    )
    return path


def _service(
    root: Path,
    store: InMemorySkillStore,
    *,
    capabilities: frozenset[str] = frozenset({"data.file.read"}),
    source: SkillSource = SkillSource.USER,
    max_skill_bytes: int = 64 * 1_024,
) -> SkillService:
    counter = {"value": 0}

    def make_id(prefix: str) -> str:
        counter["value"] += 1
        return f"{prefix}-{counter['value']:04d}"

    return SkillService(
        agent_id="agent-atlas",
        root=root,
        source=source,
        store=store,
        capability_ids=capabilities,
        max_skill_bytes=max_skill_bytes,
        clock=lambda: NOW,
        id_factory=make_id,
    )


async def _activate(
    service: SkillService,
    index: SkillIndex,
    *,
    expected: str | None = None,
) -> SkillInspection:
    return await service.activate(
        index.skill_id,
        index.version_id,
        expected_active_version_id=expected,
        actor_id="user:test",
        reason="Explicit test activation.",
    )


async def test_refresh_indexes_compact_metadata_and_full_byte_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    path = _write_skill(root, "reconcile-customers")
    raw = path.read_bytes()
    store = InMemorySkillStore()

    index = (await _service(root, store, source=SkillSource.EXTENSION).refresh())[0]
    version = store.versions[("agent-atlas", index.version_id)]

    assert index.skill_id == "skill:reconcile-customers"
    assert index.active_version_id is None
    assert index.source is SkillSource.EXTENSION
    assert not hasattr(index, "instructions")
    assert version.instructions == "Use accepted evidence and cite every result."
    assert version.source_path == "reconcile-customers/SKILL.md"
    assert version.content_hash == "sha256:" + sha256(raw).hexdigest()


async def test_discovery_and_listing_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(root, "zebra-check")
    _write_skill(root, "alpha-check")
    store = InMemorySkillStore()
    service = _service(root, store)

    refreshed = await service.refresh()
    listed = await service.list()

    assert [item.stable_name for item in refreshed] == ["alpha-check", "zebra-check"]
    assert listed == refreshed


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        ('executor_id = "unsafe.execute"\n', "runtime effects"),
        ('runtime_effects = ["skip_policy"]\n', "runtime effects"),
        ('tool_views = ["hidden"]\n', "runtime effects"),
        ('source = "builtin"\n', "unknown fields"),
        ('mystery = "value"\n', "unknown fields"),
    ),
)
async def test_forbidden_and_unknown_metadata_fail_closed(
    tmp_path: Path,
    extra: str,
    message: str,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(root, "unsafe-skill", extra_metadata=extra)

    with pytest.raises(SkillFormatError, match=message):
        await _service(root, InMemorySkillStore()).refresh()


@pytest.mark.parametrize(
    "payload",
    (
        b"\xff\xfe\x00",
        b"+++\nname = 'bad'\n",
        b"+++\nname = 1\nversion = '1.0.0'\ndescription = 'x'\n"
        b"activation_mode = 'explicit'\n+++\nbody\n",
    ),
)
async def test_invalid_encoding_or_front_matter_is_rejected(
    tmp_path: Path,
    payload: bytes,
) -> None:
    root = tmp_path / "skills"
    directory = root / "bad-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_bytes(payload)

    with pytest.raises(SkillFormatError):
        await _service(root, InMemorySkillStore()).refresh()


async def test_directory_and_metadata_name_must_match(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    directory = root / "expected-name"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        _skill_text("different-name"),
        encoding="utf-8",
    )

    with pytest.raises(SkillFormatError, match="exactly match"):
        await _service(root, InMemorySkillStore()).refresh()


async def test_discovery_rejects_relative_and_symlinked_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real-skills"
    real.mkdir()
    _write_skill(real, "safe-skill")
    alias = tmp_path / "skills-alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(SkillDiscoveryError, match="canonical"):
        await _service(alias, InMemorySkillStore()).refresh()

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SkillDiscoveryError, match="canonical"):
        await _service(Path("real-skills"), InMemorySkillStore()).refresh()


async def test_discovery_rejects_symlink_hardlink_and_special_skill_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text(_skill_text("unsafe-skill"), encoding="utf-8")

    for case in ("symlink", "hardlink", "fifo"):
        root = tmp_path / case / "skills"
        directory = root / "unsafe-skill"
        directory.mkdir(parents=True)
        path = directory / "SKILL.md"
        if case == "symlink":
            path.symlink_to(target)
        elif case == "hardlink":
            os.link(target, path)
        else:
            os.mkfifo(path)

        with pytest.raises(SkillDiscoveryError):
            await _service(root, InMemorySkillStore()).refresh()


async def test_discovery_rejects_extra_depth_and_oversized_file(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    path = _write_skill(root, "bounded-skill")
    (path.parent / "extra.md").write_text("not allowed", encoding="utf-8")
    with pytest.raises(SkillDiscoveryError, match="only SKILL.md"):
        await _service(root, InMemorySkillStore()).refresh()

    (path.parent / "extra.md").unlink()
    with pytest.raises(SkillDiscoveryError, match="byte limit"):
        await _service(
            root,
            InMemorySkillStore(),
            max_skill_bytes=32,
        ).refresh()


async def test_post_read_version_change_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "skills"
    path = _write_skill(root, "raced-skill")
    original_read = service_module.os.read
    changed = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = original_read(descriptor, size)
        if chunk and not changed:
            changed = True
            path.write_text(
                _skill_text("raced-skill", instructions="Changed instructions."),
                encoding="utf-8",
            )
        return chunk

    monkeypatch.setattr(service_module.os, "read", racing_read)

    with pytest.raises(SkillDiscoveryError, match="changed during read"):
        await _service(root, InMemorySkillStore()).refresh()


async def test_activation_requires_registered_capability_and_is_explicit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(
        root,
        "warehouse-guide",
        capabilities=("data.sqlite.query",),
        mode="always",
    )
    store = InMemorySkillStore()
    service = _service(root, store, capabilities=frozenset())
    index = (await service.refresh())[0]

    assert index.active_version_id is None
    with pytest.raises(SkillCapabilityUnavailableError, match="data.sqlite.query"):
        await _activate(service, index)
    assert store.activations == {}


async def test_activation_history_supports_auditable_rollback(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    path = _write_skill(root, "versioned-guide", version="1.0.0")
    store = InMemorySkillStore()
    service = _service(root, store)
    first = (await service.refresh())[0]
    first_inspection = await _activate(service, first)

    path.write_text(
        _skill_text("versioned-guide", version="2.0.0", instructions="Version two."),
        encoding="utf-8",
    )
    # Active metadata remains pinned until an explicit activation.
    assert (await service.refresh())[0].active_version_id == first.version_id
    second_version = next(
        version for version in store.versions.values() if version.version == "2.0.0"
    )
    second_inspection = await service.activate(
        first.skill_id,
        second_version.id,
        expected_active_version_id=first.version_id,
        actor_id="user:test",
        reason="Approve version two.",
    )
    rollback = await service.activate(
        first.skill_id,
        first.version_id,
        expected_active_version_id=second_version.id,
        actor_id="user:test",
        reason="Rollback to version one.",
    )

    assert first_inspection.index.active_version_id == first.version_id
    assert second_inspection.index.active_version_id == second_version.id
    assert rollback.index.active_version_id == first.version_id
    assert [item.version_id for item in rollback.activations] == [
        first.version_id,
        second_version.id,
        first.version_id,
    ]
    assert {item.version for item in rollback.versions} == {"1.0.0", "2.0.0"}


async def test_stale_activation_and_unknown_identity_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(root, "exact-guide")
    store = InMemorySkillStore()
    service = _service(root, store)
    index = (await service.refresh())[0]
    await _activate(service, index)

    with pytest.raises(SkillActivationConflictError, match="changed"):
        await service.activate(
            index.skill_id,
            index.version_id,
            expected_active_version_id=None,
            actor_id="user:test",
            reason="Stale request.",
        )
    with pytest.raises(SkillNotFoundError, match="unknown skill"):
        await service.inspect("skill:missing")


async def test_selection_is_deterministic_bounded_and_loads_only_selected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(root, "always-guide", mode="always")
    _write_skill(
        root,
        "customer-reconcile",
        description="Compare customer exports.",
        mode="on_demand",
    )
    _write_skill(root, "manual-guide", mode="explicit")
    _write_skill(root, "unrelated-guide", description="Forecast weather.")
    store = InMemorySkillStore()
    service = _service(root, store)
    indices = await service.refresh()
    for item in indices:
        await _activate(service, item)
    store.version_loads.clear()

    selected = await service.select(
        "Compare the customer export",
        explicit_skill_ids=("skill:manual-guide",),
        limit=3,
    )

    assert [(item.index.skill_id, item.reason) for item in selected] == [
        ("skill:manual-guide", SkillSelectionReason.EXPLICIT),
        ("skill:always-guide", SkillSelectionReason.ALWAYS),
        ("skill:customer-reconcile", SkillSelectionReason.ON_DEMAND),
    ]
    assert len(store.version_loads) == 3
    assert all("unrelated-guide" not in item.index.skill_id for item in selected)


async def test_explicit_selection_rejects_unknown_inactive_missing_caps_and_budget(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(root, "manual-guide", mode="explicit", instructions="Long procedure.")
    store = InMemorySkillStore()
    service = _service(root, store)
    index = (await service.refresh())[0]

    with pytest.raises(SkillNotFoundError, match="unknown explicit"):
        await service.select("Use it", explicit_skill_ids=("skill:missing",))
    with pytest.raises(SkillNotActiveError):
        await service.select("Use it", explicit_skill_ids=(index.skill_id,))

    await _activate(service, index)
    with pytest.raises(SkillSelectionBudgetError):
        await service.select(
            "Use it",
            explicit_skill_ids=(index.skill_id,),
            max_instruction_characters=2,
        )

    reopened_without_capability = _service(root, store, capabilities=frozenset())
    with pytest.raises(SkillCapabilityUnavailableError):
        await reopened_without_capability.select(
            "Use it",
            explicit_skill_ids=(index.skill_id,),
        )
    assert await reopened_without_capability.select("Use manual guide") == ()


def test_records_are_immutable_and_selection_requires_matching_active_version(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    raw = _skill_text("immutable-guide").encode()
    skill, version, index = service_module._parse_skill(
        raw,
        directory_name="immutable-guide",
        agent_id="agent-atlas",
        source=SkillSource.USER,
        created_at=NOW,
    )

    with pytest.raises(FrozenInstanceError):
        skill.stable_name = "changed"  # type: ignore[misc]
    assert index.matches(version)
    with pytest.raises(ValueError, match="active version"):
        service_module.SkillSelection(
            index=index,
            version=version,
            reason=SkillSelectionReason.ON_DEMAND,
        )
