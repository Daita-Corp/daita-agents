from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.learning import (
    LearningDecision,
    LearningProposal,
    LearningProposalState,
    LearningProvenance,
    LearningRejectionCategory,
    LearningSourceOutcome,
    resolve_learning_proposal,
)
from daita.skills import (
    Skill,
    SkillActivation,
    SkillActivationConflictError,
    SkillActivationMode,
    SkillCapabilityUnavailableError,
    SkillChangeAcceptanceResult,
    SkillChangeCandidate,
    SkillChangeCommit,
    SkillChangeConflictError,
    SkillChangeLearningService,
    SkillIndex,
    SkillInspection,
    SkillSelectionReason,
    SkillService,
    SkillSource,
    SkillVersion,
)

NOW = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)


class AdvancingClock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


class AtomicSkillChangeStore:
    def __init__(self) -> None:
        self.proposals: dict[tuple[str, str], LearningProposal] = {}
        self.proposal_keys: dict[tuple[str, str], str] = {}
        self.skills: dict[tuple[str, str], Skill] = {}
        self.versions: dict[tuple[str, str], SkillVersion] = {}
        self.index: dict[tuple[str, str], SkillIndex] = {}
        self.activations: dict[tuple[str, str], list[SkillActivation]] = {}
        self.commits: list[SkillChangeCommit] = []
        self.fail_commit = False

    async def create_proposal(self, proposal: LearningProposal) -> LearningProposal:
        key = (proposal.provenance.agent_id, proposal.idempotency_key)
        existing_id = self.proposal_keys.get(key)
        if existing_id is not None:
            return self.proposals[(proposal.provenance.agent_id, existing_id)]
        self.proposals[(proposal.provenance.agent_id, proposal.id)] = proposal
        self.proposal_keys[key] = proposal.id
        return proposal

    async def load_proposal(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> LearningProposal | None:
        return self.proposals.get((agent_id, proposal_id))

    async def list_proposals(
        self,
        agent_id: str,
        *,
        operation_id: str | None,
        states: tuple[LearningProposalState, ...],
        limit: int,
    ) -> tuple[LearningProposal, ...]:
        return tuple(
            proposal
            for (owner, _), proposal in self.proposals.items()
            if owner == agent_id
            and proposal.state in states
            and (
                operation_id is None or proposal.provenance.operation_id == operation_id
            )
        )[:limit]

    async def resolve_proposal(
        self,
        decision: LearningDecision,
        *,
        expected_state: LearningProposalState,
    ) -> LearningProposal:
        proposal = next(
            value
            for value in self.proposals.values()
            if value.id == decision.proposal_id
        )
        if proposal.state is not expected_state:
            raise SkillChangeConflictError("proposal state changed")
        resolved = resolve_learning_proposal(proposal, decision)
        self.proposals[(proposal.provenance.agent_id, proposal.id)] = resolved
        return resolved

    async def record_discovery(
        self,
        skill: Skill,
        version: SkillVersion,
        index: SkillIndex,
    ) -> SkillIndex:
        key = (skill.agent_id, skill.id)
        self.skills.setdefault(key, skill)
        self.versions.setdefault((version.agent_id, version.id), version)
        current = self.index.get(key)
        active = None if current is None else current.active_version_id
        stored = SkillIndex.from_version(version, active_version_id=active)
        self.index[key] = stored
        return stored

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
                    version
                    for (owner, _), version in self.versions.items()
                    if owner == agent_id and version.skill_id == skill_id
                ),
                key=lambda item: (item.created_at, item.id),
            )
        )
        return SkillInspection(
            skill=skill,
            index=index,
            versions=versions,
            activations=tuple(self.activations.get(key, ())),
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

    async def commit_skill_change(
        self,
        request: SkillChangeCommit,
    ) -> SkillChangeAcceptanceResult:
        self.commits.append(request)
        if self.fail_commit:
            raise SkillChangeConflictError("injected atomic conflict")
        proposal_key = (
            request.proposal.provenance.agent_id,
            request.proposal.id,
        )
        durable = self.proposals[proposal_key]
        if durable.state is LearningProposalState.COMMITTED:
            inspection = await self.inspect_skill(
                request.skill.agent_id,
                request.skill.id,
            )
            assert inspection is not None
            return SkillChangeAcceptanceResult(durable, inspection, replayed=True)
        key = (request.skill.agent_id, request.skill.id)
        current = self.index.get(key)
        actual_active = None if current is None else current.active_version_id
        actual_versions = tuple(
            version
            for (owner, _), version in self.versions.items()
            if owner == request.skill.agent_id and version.skill_id == request.skill.id
        )
        if (
            durable != request.proposal
            or durable.state is not LearningProposalState.PROPOSED
            or actual_active != request.expected_active_version_id
            or len(actual_versions) != request.expected_skill_version_count
        ):
            raise SkillChangeConflictError("guarded skill state changed")
        claimed = next(
            (
                skill
                for (owner, _), skill in self.skills.items()
                if owner == request.skill.agent_id
                and (
                    skill.id == request.skill.id
                    or skill.stable_name == request.skill.stable_name
                )
            ),
            None,
        )
        if claimed is not None and claimed != request.skill:
            raise SkillChangeConflictError("skill identity is already claimed")
        if any(
            item.id == request.version.id or item.version == request.version.version
            for item in actual_versions
        ):
            raise SkillChangeConflictError("skill version is already claimed")

        resolved = resolve_learning_proposal(request.proposal, request.decision)
        final_index = SkillIndex.from_version(
            request.version,
            active_version_id=request.version.id,
            updated_at=request.activation.activated_at,
        )
        self.skills[key] = request.skill
        self.versions[(request.version.agent_id, request.version.id)] = request.version
        self.index[key] = final_index
        self.activations.setdefault(key, []).append(request.activation)
        self.proposals[proposal_key] = resolved
        inspection = await self.inspect_skill(request.skill.agent_id, request.skill.id)
        assert inspection is not None
        return SkillChangeAcceptanceResult(resolved, inspection)


def _provenance(number: int) -> LearningProvenance:
    return LearningProvenance(
        agent_id="agent-atlas",
        operation_id=f"operation-{number}",
        trigger_id=f"trigger-{number}",
        source_outcome=LearningSourceOutcome.SUCCEEDED,
        source_hash="sha256:" + f"{number:064x}",
    )


def _candidate(
    *,
    version: str = "1.0.0",
    instructions: str = "Use accepted evidence and cite each result.",
    capabilities: tuple[str, ...] = ("data.sqlite.query",),
) -> SkillChangeCandidate:
    return SkillChangeCandidate(
        stable_name="reconcile-customers",
        version=version,
        description="Reconcile customer records from bounded evidence.",
        instructions=instructions,
        activation_mode=SkillActivationMode.ALWAYS,
        domains=("data",),
        resource_kinds=("table",),
        required_capability_ids=capabilities,
        sensitivity_notes="Do not expose raw rows.",
    )


def _services(
    tmp_path: Path,
    *,
    capabilities: frozenset[str] = frozenset({"data.sqlite.query"}),
) -> tuple[AtomicSkillChangeStore, SkillService, SkillChangeLearningService]:
    store = AtomicSkillChangeStore()
    root = tmp_path / "skills"
    root.mkdir(parents=True)
    clock = AdvancingClock()
    ids = _ids()
    skills = SkillService(
        agent_id="agent-atlas",
        root=root,
        source=SkillSource.USER,
        store=store,
        capability_ids=capabilities,
        clock=clock,
        id_factory=ids,
    )
    learning = SkillChangeLearningService(
        agent_id="agent-atlas",
        store=store,
        skills=skills,
        clock=clock,
        id_factory=ids,
    )
    return store, skills, learning


async def test_safe_proposal_is_durable_but_version_remains_inert(
    tmp_path: Path,
) -> None:
    store, skills, learning = _services(tmp_path)

    result = await learning.propose(_candidate(), _provenance(1))

    assert result.proposal.state is LearningProposalState.PROPOSED
    assert result.proposed_version is not None
    assert result.proposed_version.source is SkillSource.LEARNED_PROPOSAL
    assert result.proposed_version.source_path is None
    assert result.proposed_version.content_hash == result.proposal.candidate_hash
    assert await store.load_proposal("agent-atlas", result.proposal.id) == (
        result.proposal
    )
    assert await skills.list() == ()
    assert store.skills == {}
    assert store.versions == {}
    assert store.activations == {}
    assert store.commits == []


async def test_natural_operation_request_creates_only_an_inert_proposal(
    tmp_path: Path,
) -> None:
    store, skills, learning = _services(tmp_path)

    result = await learning.propose_natural(
        "Propose a skill named reconcile-status version 1.2.0: "
        "Compare accepted totals and cite discrepancies.",
        _provenance(1),
    )

    assert result is not None
    assert result.proposal.state is LearningProposalState.PROPOSED
    assert result.proposed_version is not None
    assert result.proposed_version.stable_name == "reconcile-status"
    assert result.proposed_version.version == "1.2.0"
    assert result.proposed_version.activation_mode is SkillActivationMode.EXPLICIT
    assert await skills.list() == ()
    assert store.skills == {}
    assert store.versions == {}
    assert store.activations == {}


async def test_natural_executable_skill_request_is_redacted_and_inert(
    tmp_path: Path,
) -> None:
    store, skills, learning = _services(tmp_path)
    raw = "import os and call subprocess.run('unsafe')"

    result = await learning.propose_natural(
        f"Propose a skill called unsafe-runner: {raw}",
        _provenance(1),
    )

    assert result is not None
    assert result.proposal.state is LearningProposalState.REJECTED
    assert result.proposal.rejection_category is (
        LearningRejectionCategory.EXECUTABLE_OR_RUNTIME_EFFECT
    )
    assert result.proposal.candidate_payload is None
    assert result.proposed_version is None
    assert raw not in repr(result.proposal)
    assert await skills.list() == ()
    assert store.skills == {}


async def test_explicit_accept_activates_audited_versions_and_replays(
    tmp_path: Path,
) -> None:
    store, skills, learning = _services(tmp_path)
    first = await learning.propose(_candidate(), _provenance(1))
    accepted_one = await learning.accept(
        first.proposal.id,
        expected_active_version_id=None,
        actor_id="user:owner",
        reason="Approve the reviewed first procedure.",
    )
    first_version_id = accepted_one.inspection.index.active_version_id
    assert first_version_id is not None

    second = await learning.propose(
        _candidate(
            version="2.0.0",
            instructions="Compare accepted evidence, then cite each discrepancy.",
        ),
        _provenance(2),
    )
    before_second_accept = await skills.inspect("skill:reconcile-customers")
    accepted_two = await learning.accept(
        second.proposal.id,
        expected_active_version_id=first_version_id,
        actor_id="user:owner",
        reason="Approve the reviewed second procedure.",
    )
    replay = await learning.accept(
        second.proposal.id,
        expected_active_version_id=None,
        actor_id="user:owner",
        reason="Idempotent retry does not create another activation.",
    )

    assert accepted_one.proposal.result_skill_version == 1
    assert accepted_two.proposal.result_skill_version == 2
    assert accepted_two.proposal.state is LearningProposalState.COMMITTED
    assert before_second_accept.index.active_version_id == first_version_id
    assert len(before_second_accept.versions) == 1
    assert [item.version for item in accepted_two.inspection.versions] == [
        "1.0.0",
        "2.0.0",
    ]
    assert [
        item.previous_version_id for item in accepted_two.inspection.activations
    ] == [None, first_version_id]
    assert [item.actor_id for item in accepted_two.inspection.activations] == [
        "user:owner",
        "user:owner",
    ]
    assert accepted_two.inspection.index.active_version_id == (
        accepted_two.inspection.versions[1].id
    )
    assert replay.replayed is True
    assert len(replay.inspection.activations) == 2
    assert len(store.commits) == 2

    selections = await skills.select("Any query")
    assert len(selections) == 1
    assert selections[0].reason is SkillSelectionReason.ALWAYS
    assert selections[0].version.version == "2.0.0"


async def test_unsafe_change_is_redacted_without_a_version_preview(
    tmp_path: Path,
) -> None:
    store, skills, learning = _services(tmp_path)
    raw = "import os\ndef run(): pass"

    result = await learning.propose(
        _candidate(instructions=raw),
        _provenance(1),
    )

    assert result.proposal.state is LearningProposalState.REJECTED
    assert (
        result.proposal.rejection_category
        is LearningRejectionCategory.EXECUTABLE_OR_RUNTIME_EFFECT
    )
    assert result.proposal.candidate_payload is None
    assert result.proposed_version is None
    assert raw not in repr(result.proposal)
    assert await skills.list() == ()
    assert store.commits == []


async def test_missing_capability_or_stale_pointer_keeps_proposal_inert(
    tmp_path: Path,
) -> None:
    store, _, learning = _services(tmp_path, capabilities=frozenset())
    missing = await learning.propose(_candidate(), _provenance(1))

    with pytest.raises(SkillCapabilityUnavailableError):
        await learning.accept(
            missing.proposal.id,
            expected_active_version_id=None,
            actor_id="user:owner",
            reason="Cannot accept without the declared capability.",
        )

    assert store.commits == []
    assert store.skills == {}
    assert store.proposals[("agent-atlas", missing.proposal.id)].state is (
        LearningProposalState.PROPOSED
    )

    capable_store, _, capable = _services(tmp_path / "capable")
    first = await capable.propose(_candidate(), _provenance(2))
    accepted = await capable.accept(
        first.proposal.id,
        expected_active_version_id=None,
        actor_id="user:owner",
        reason="Initial activation.",
    )
    second = await capable.propose(
        _candidate(version="2.0.0"),
        _provenance(3),
    )
    with pytest.raises(SkillChangeConflictError, match="changed"):
        await capable.accept(
            second.proposal.id,
            expected_active_version_id=None,
            actor_id="user:owner",
            reason="Stale activation guard.",
        )

    assert len(capable_store.commits) == 1
    assert capable_store.index[
        ("agent-atlas", "skill:reconcile-customers")
    ].active_version_id == (accepted.inspection.index.active_version_id)


async def test_atomic_commit_conflict_has_no_partial_skill_or_decision(
    tmp_path: Path,
) -> None:
    store, skills, learning = _services(tmp_path)
    proposed = await learning.propose(_candidate(), _provenance(1))
    store.fail_commit = True

    with pytest.raises(SkillChangeConflictError, match="atomic conflict"):
        await learning.accept(
            proposed.proposal.id,
            expected_active_version_id=None,
            actor_id="user:owner",
            reason="This transaction is forced to conflict.",
        )

    assert await skills.list() == ()
    assert store.skills == {}
    assert store.versions == {}
    assert store.activations == {}
    assert store.proposals[("agent-atlas", proposed.proposal.id)].state is (
        LearningProposalState.PROPOSED
    )
    request = store.commits[0]
    assert request.expected_skill_version_count == 0
    assert request.expected_active_version_id is None
    assert request.staged_index.active_version_id is None
    assert request.activation.version_id == request.version.id
    assert request.decision.result_skill_version == 1


def test_skill_change_learning_has_no_executor_or_execution_path() -> None:
    path = Path(__file__).parents[3] / "src" / "daita" / "skills" / "learning.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    execute_calls: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
        ):
            execute_calls.append(node.lineno)

    assert not any("executor" in name.casefold().split(".") for name in imports)
    assert execute_calls == []
