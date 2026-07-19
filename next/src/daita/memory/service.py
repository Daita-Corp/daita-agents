"""Bounded selection and lifecycle validation for portable agent memory."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import re
import unicodedata

from .._json import canonical_json
from .models import (
    MemoryHistory,
    MemoryInspection,
    MemoryInspectionRequest,
    MemoryListRequest,
    MemoryListResult,
    MemoryQualification,
    MemoryRecallHit,
    MemoryRecallRequest,
    MemoryRecallResult,
    MemoryRestoreRequest,
    MemorySnapshot,
    MemoryState,
    MemorySupersessionRequest,
    QualifiedMemory,
)
from .protocols import MemoryStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryNotFoundError(KeyError):
    def __init__(self, agent_id: str, memory_id: str) -> None:
        self.agent_id = agent_id
        self.memory_id = memory_id
        super().__init__(f"unknown memory for {agent_id}: {memory_id}")


class MemoryServiceContractError(RuntimeError):
    """Raised when a store violates the portable memory contract."""


class MemoryService:
    """Own portable recall semantics without owning concrete persistence."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(store, MemoryStore):
            raise TypeError("store must implement MemoryStore")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._store = store
        self._clock = clock

    async def recall(self, request: MemoryRecallRequest) -> MemoryRecallResult:
        if not isinstance(request, MemoryRecallRequest):
            raise TypeError("request must be a MemoryRecallRequest")
        now = self._now()
        raw = await self._store.recall_candidates(
            query=request.query,
            scope=request.scope,
            states=(MemoryState.ACTIVE,),
            sensitivities=request.allowed_sensitivities,
            unexpired_at=now,
            limit=request.candidate_limit,
        )
        candidates = _candidate_tuple(raw, maximum=request.candidate_limit)

        omitted_scope = 0
        omitted_sensitivity = 0
        omitted_lifecycle = 0
        omitted_revision = 0
        omitted_relevance = 0
        ranked: list[tuple[float, MemorySnapshot]] = []
        for candidate in candidates:
            if not candidate.record.scope.contains(request.scope):
                omitted_scope += 1
                continue
            if candidate.version.sensitivity not in request.allowed_sensitivities:
                omitted_sensitivity += 1
                continue
            qualification = _qualification(
                candidate,
                now=now,
                current_resource_revision=request.current_resource_revision,
            )
            if qualification in {
                MemoryQualification.SUPERSEDED,
                MemoryQualification.REJECTED,
                MemoryQualification.EXPIRED,
            }:
                omitted_lifecycle += 1
                continue
            if qualification in {
                MemoryQualification.STALE_REVISION,
                MemoryQualification.REVISION_UNKNOWN,
            }:
                omitted_revision += 1
                continue
            score = _lexical_score(request.query, candidate)
            if score <= 0.0:
                omitted_relevance += 1
                continue
            ranked.append((score, candidate))

        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1].version.confidence,
                item[1].record.logical_key,
                item[1].record.id,
            )
        )
        hits: list[MemoryRecallHit] = []
        used_characters = 0
        omitted_budget = 0
        omitted_limit = 0
        for score, candidate in ranked:
            if len(hits) >= request.limit:
                omitted_limit += 1
                continue
            cost = _projection_character_cost(candidate, score)
            if used_characters + cost > request.character_budget:
                omitted_budget += 1
                continue
            qualification = _qualification(
                candidate,
                now=now,
                current_resource_revision=request.current_resource_revision,
            )
            hits.append(
                MemoryRecallHit(
                    memory=QualifiedMemory(candidate, qualification),
                    lexical_score=score,
                )
            )
            used_characters += cost

        return MemoryRecallResult(
            hits=tuple(hits),
            candidate_count=len(candidates),
            used_characters=used_characters,
            omitted_by_scope=omitted_scope,
            omitted_by_sensitivity=omitted_sensitivity,
            omitted_by_lifecycle=omitted_lifecycle,
            omitted_by_revision=omitted_revision,
            omitted_by_relevance=omitted_relevance,
            omitted_by_budget=omitted_budget,
            omitted_by_limit=omitted_limit,
            truncated=(
                omitted_budget > 0
                or omitted_limit > 0
                or len(candidates) == request.candidate_limit
            ),
        )

    async def list(self, request: MemoryListRequest) -> MemoryListResult:
        if not isinstance(request, MemoryListRequest):
            raise TypeError("request must be a MemoryListRequest")
        states = [MemoryState.ACTIVE]
        if request.include_superseded:
            states.append(MemoryState.SUPERSEDED)
        if request.include_rejected:
            states.append(MemoryState.REJECTED)
        raw = await self._store.list_candidates(
            scope=request.scope,
            states=tuple(states),
            sensitivities=request.allowed_sensitivities,
            limit=request.limit + 1,
        )
        candidates = _candidate_tuple(raw, maximum=request.limit + 1)
        now = self._now()
        allowed_states = set(states)
        filtered = tuple(
            candidate
            for candidate in candidates
            if candidate.record.state in allowed_states
            and candidate.record.scope.contains(request.scope)
            and candidate.version.sensitivity in request.allowed_sensitivities
        )
        ordered = tuple(
            sorted(
                filtered,
                key=lambda candidate: (
                    candidate.record.logical_key,
                    candidate.record.kind.value,
                    candidate.record.id,
                ),
            )
        )
        selected = ordered[: request.limit]
        return MemoryListResult(
            items=tuple(
                QualifiedMemory(
                    snapshot=candidate,
                    qualification=_qualification(
                        candidate,
                        now=now,
                        current_resource_revision=request.current_resource_revision,
                    ),
                )
                for candidate in selected
            ),
            candidate_count=len(candidates),
            truncated=len(ordered) > request.limit,
        )

    async def inspect(self, request: MemoryInspectionRequest) -> MemoryInspection:
        if not isinstance(request, MemoryInspectionRequest):
            raise TypeError("request must be a MemoryInspectionRequest")
        history = await self._load_history(request.agent_id, request.memory_id)
        snapshot = MemorySnapshot(history.record, history.current)
        return MemoryInspection(
            history=history,
            qualification=_qualification(
                snapshot,
                now=self._now(),
                current_resource_revision=request.current_resource_revision,
            ),
        )

    async def supersede(
        self,
        request: MemorySupersessionRequest,
    ) -> MemoryInspection:
        if not isinstance(request, MemorySupersessionRequest):
            raise TypeError("request must be a MemorySupersessionRequest")
        current = await self._load_history(request.agent_id, request.memory_id)
        _validate_lifecycle_head(current, request.expected_version)
        _validate_replacement_scope(current, request.replacement.resource_revision)
        updated = await self._store.supersede(request)
        _validate_lifecycle_result(current, updated, request.replacement)
        return MemoryInspection(
            history=updated,
            qualification=_qualification(
                MemorySnapshot(updated.record, updated.current),
                now=self._now(),
                current_resource_revision=None,
            ),
        )

    async def restore(self, request: MemoryRestoreRequest) -> MemoryInspection:
        if not isinstance(request, MemoryRestoreRequest):
            raise TypeError("request must be a MemoryRestoreRequest")
        current = await self._load_history(request.agent_id, request.memory_id)
        _validate_lifecycle_head(current, request.expected_version)
        try:
            restored = next(
                version
                for version in current.versions
                if version.version == request.restore_version
            )
        except StopIteration as error:
            raise ValueError(
                f"restore version does not exist: {request.restore_version}"
            ) from error
        _validate_restore_payload(restored, request.replacement)
        _validate_replacement_scope(current, request.replacement.resource_revision)
        updated = await self._store.restore(request)
        _validate_lifecycle_result(current, updated, request.replacement)
        return MemoryInspection(
            history=updated,
            qualification=_qualification(
                MemorySnapshot(updated.record, updated.current),
                now=self._now(),
                current_resource_revision=None,
            ),
        )

    async def _load_history(self, agent_id: str, memory_id: str) -> MemoryHistory:
        history = await self._store.load_history(agent_id, memory_id)
        if history is None:
            raise MemoryNotFoundError(agent_id, memory_id)
        if not isinstance(history, MemoryHistory):
            raise MemoryServiceContractError(
                "memory store load_history returned an unsupported record"
            )
        if history.record.scope.agent_id != agent_id or history.record.id != memory_id:
            raise MemoryServiceContractError(
                "memory store returned history from another identity or agent"
            )
        return history

    def _now(self) -> datetime:
        value = self._clock()
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "memory service clock must return a timezone-aware datetime"
            )
        return value


def _candidate_tuple(
    values: tuple[MemorySnapshot, ...],
    *,
    maximum: int,
) -> tuple[MemorySnapshot, ...]:
    if not isinstance(values, tuple):
        raise MemoryServiceContractError("memory store candidates must be a tuple")
    candidates = values
    if len(candidates) > maximum:
        raise MemoryServiceContractError("memory store exceeded the candidate bound")
    if any(not isinstance(value, MemorySnapshot) for value in candidates):
        raise MemoryServiceContractError(
            "memory store candidates must contain MemorySnapshot records"
        )
    identifiers = [value.record.id for value in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise MemoryServiceContractError("memory store returned duplicate candidates")
    return candidates


def _qualification(
    snapshot: MemorySnapshot,
    *,
    now: datetime,
    current_resource_revision: str | None,
) -> MemoryQualification:
    if snapshot.record.state is MemoryState.SUPERSEDED:
        return MemoryQualification.SUPERSEDED
    if snapshot.record.state is MemoryState.REJECTED:
        return MemoryQualification.REJECTED
    if snapshot.version.expires_at is not None and snapshot.version.expires_at <= now:
        return MemoryQualification.EXPIRED
    bound_revision = snapshot.version.resource_revision
    if bound_revision is None:
        return MemoryQualification.UNBOUND
    if current_resource_revision is None:
        return MemoryQualification.REVISION_UNKNOWN
    if current_resource_revision != bound_revision:
        return MemoryQualification.STALE_REVISION
    return MemoryQualification.CURRENT


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(dict.fromkeys(re.findall(r"\w+", normalized, flags=re.UNICODE)))


def _lexical_score(query: str, snapshot: MemorySnapshot) -> float:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return 0.0
    document = " ".join(
        (
            snapshot.record.logical_key,
            snapshot.version.content,
            canonical_json(snapshot.version.attributes),
        )
    )
    overlap = query_tokens & set(_tokens(document))
    return len(overlap) / len(query_tokens)


def _projection_character_cost(snapshot: MemorySnapshot, score: float) -> int:
    return len(
        canonical_json(
            {
                "confidence": snapshot.version.confidence,
                "content": snapshot.version.content,
                "kind": snapshot.record.kind.value,
                "logical_key": snapshot.record.logical_key,
                "memory_id": snapshot.record.id,
                "provenance": {
                    "content_hash": snapshot.version.provenance.content_hash,
                    "evidence_id": snapshot.version.provenance.evidence_id,
                    "kind": snapshot.version.provenance.kind.value,
                    "operation_id": snapshot.version.provenance.operation_id,
                    "trigger_id": snapshot.version.provenance.trigger_id,
                },
                "score": score,
                "version": snapshot.version.version,
            }
        )
    )


def _validate_lifecycle_head(history: MemoryHistory, expected_version: int) -> None:
    if history.record.state is not MemoryState.ACTIVE:
        raise ValueError("only active memory may be superseded or restored")
    if history.record.current_version != expected_version:
        raise ValueError("memory expected_version does not match the current version")


def _validate_replacement_scope(
    history: MemoryHistory,
    resource_revision: str | None,
) -> None:
    if resource_revision is not None and history.record.scope.resource_id is None:
        raise ValueError("revision-bound replacement requires resource-scoped memory")


def _validate_restore_payload(source: object, replacement: object) -> None:
    from .models import MemoryVersion

    if not isinstance(source, MemoryVersion) or not isinstance(
        replacement, MemoryVersion
    ):
        raise TypeError("restore payload validation requires MemoryVersion records")
    for field_name in (
        "content",
        "attributes",
        "confidence",
        "sensitivity",
        "expires_at",
        "resource_revision",
    ):
        if getattr(source, field_name) != getattr(replacement, field_name):
            raise ValueError(
                f"restore replacement must preserve historical {field_name}"
            )


def _validate_lifecycle_result(
    previous: MemoryHistory,
    updated: MemoryHistory,
    replacement: object,
) -> None:
    from .models import MemoryVersion

    if not isinstance(updated, MemoryHistory):
        raise MemoryServiceContractError(
            "memory lifecycle store returned invalid history"
        )
    if not isinstance(replacement, MemoryVersion):
        raise TypeError("memory lifecycle replacement must be a MemoryVersion")
    if (
        updated.record.id != previous.record.id
        or updated.record.scope.agent_id != previous.record.scope.agent_id
        or updated.record.logical_identity != previous.record.logical_identity
        or updated.current != replacement
    ):
        raise MemoryServiceContractError(
            "memory lifecycle store changed identity or failed to commit replacement"
        )


__all__ = [
    "MemoryNotFoundError",
    "MemoryService",
    "MemoryServiceContractError",
]
