"""Define durable source permission records and update authorization fingerprints."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256

from .._json import FrozenJsonObject, canonical_json
from ..capabilities import EffectEvidenceBasis, EffectObservation, EffectOutcome
from ..llm.models import ModelSensitivity
from ..adapters.models import SourceRegistration
from ..catalog.models import (
    CatalogFacet,
    CatalogResource,
    FacetKind,
    ResourceKind,
    TabularFacet,
)

_SOURCE_PERMISSION_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_PERMISSION_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_SOURCE_PERMISSION_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
_SOURCE_PERMISSION_MAX_RESOURCE_IDS = 10_000
_SOURCE_PERMISSION_MAX_CATALOG_COLUMNS = 512
_SOURCE_PERMISSION_MAX_ASSIGNMENT_COLUMNS = _SOURCE_PERMISSION_MAX_CATALOG_COLUMNS
_SOURCE_PERMISSION_MAX_SUMMARY_EXAMPLES = 5


class SourceReadMode(str, Enum):
    ALL = "all"
    SELECTED = "selected"
    NONE = "none"


class SourcePermissionStateError(RuntimeError):
    """A durable source permission record is missing, foreign, or invalid."""


def _permission_text(value: str, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty text without surrounding space")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def _canonical_permission_texts(
    values: Iterable[str],
    name: str,
    *,
    maximum_items: int,
    maximum_characters: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    items = tuple(values)
    if len(items) > maximum_items:
        raise ValueError(f"{name} exceeds {maximum_items} items")
    for item in items:
        _permission_text(item, name, maximum=maximum_characters)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(items))


@dataclass(frozen=True, slots=True)
class SourceReadScope:
    """One exact fail-closed read scope for an active source."""

    agent_id: str
    source_id: str
    mode: SourceReadMode
    resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _permission_text(self.agent_id, "read scope agent_id")
        if (
            not isinstance(self.source_id, str)
            or _SOURCE_PERMISSION_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("read scope source_id must be a canonical source id")
        if not isinstance(self.mode, SourceReadMode):
            raise TypeError("read scope mode must be a SourceReadMode")
        resource_ids = _canonical_permission_texts(
            self.resource_ids,
            "read scope resource_ids",
            maximum_items=_SOURCE_PERMISSION_MAX_RESOURCE_IDS,
            maximum_characters=256,
        )
        if any(
            _SOURCE_PERMISSION_RESOURCE_ID.fullmatch(resource_id) is None
            for resource_id in resource_ids
        ):
            raise ValueError(
                "read scope resource_ids must be canonical catalog resource ids"
            )
        if self.mode is SourceReadMode.SELECTED and not resource_ids:
            raise ValueError("selected read scope requires resource_ids")
        if self.mode is not SourceReadMode.SELECTED and resource_ids:
            raise ValueError("only selected read scope can contain resource_ids")
        object.__setattr__(self, "resource_ids", resource_ids)

    @classmethod
    def allow_all(cls, *, agent_id: str, source_id: str) -> SourceReadScope:
        return cls(agent_id=agent_id, source_id=source_id, mode=SourceReadMode.ALL)


@dataclass(frozen=True, slots=True)
class RelationalWriteScope:
    """Exact operation, structure, column and row authority for one resource."""

    agent_id: str
    source_id: str
    resource_id: str
    resource_revision: str
    allowed_operations: tuple[str, ...]
    allowed_insert_columns: tuple[str, ...]
    allowed_update_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    generated_identity_columns: tuple[str, ...]
    max_rows: int
    authorization_fingerprint: str

    def __post_init__(self) -> None:
        _permission_text(self.agent_id, "write scope agent_id")
        for value, pattern, name in (
            (self.source_id, _SOURCE_PERMISSION_SOURCE_ID, "source_id"),
            (self.resource_id, _SOURCE_PERMISSION_RESOURCE_ID, "resource_id"),
            (self.resource_revision, _SOURCE_PERMISSION_HASH, "resource_revision"),
            (
                self.authorization_fingerprint,
                _SOURCE_PERMISSION_HASH,
                "authorization_fingerprint",
            ),
        ):
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise ValueError(f"write scope {name} must be canonical")
        for name in (
            "allowed_operations",
            "allowed_insert_columns",
            "allowed_update_columns",
            "key_columns",
            "generated_identity_columns",
        ):
            values = _canonical_permission_texts(
                getattr(self, name),
                name,
                maximum_items=_SOURCE_PERMISSION_MAX_CATALOG_COLUMNS,
                maximum_characters=256,
            )
            object.__setattr__(self, name, values)
        operations = set(self.allowed_operations)
        if not operations or not operations <= {"update", "upsert"}:
            raise ValueError(
                "write scope requires explicit update and/or upsert operations"
            )
        if not self.key_columns or not self.allowed_update_columns:
            raise ValueError("write scope requires exact keys and update columns")
        if set(self.key_columns) & set(self.allowed_update_columns):
            raise ValueError("write scope cannot authorize changing conflict keys")
        if "upsert" not in operations and (
            self.allowed_insert_columns or self.generated_identity_columns
        ):
            raise ValueError(
                "update permission cannot authorize insertion or identity generation"
            )
        if "upsert" in operations and not set(self.key_columns) <= set(
            self.allowed_insert_columns
        ):
            raise ValueError("upsert permission must admit every explicit conflict key")
        if set(self.generated_identity_columns) & (
            set(self.allowed_insert_columns)
            | set(self.allowed_update_columns)
            | set(self.key_columns)
        ):
            raise ValueError(
                "generated identities must be omitted and outside conflict keys"
            )
        if (
            not isinstance(self.max_rows, int)
            or isinstance(self.max_rows, bool)
            or not 1 <= self.max_rows <= 10_000
        ):
            raise ValueError("write scope max_rows must be between 1 and 10000")

    def constraints(self) -> dict[str, object]:
        """Canonical authority shared by permission review, codecs and grants."""
        return {
            "resource_revision": self.resource_revision,
            "allowed_operations": self.allowed_operations,
            "allowed_insert_columns": self.allowed_insert_columns,
            "allowed_update_columns": self.allowed_update_columns,
            "key_columns": self.key_columns,
            "generated_identity_columns": self.generated_identity_columns,
            "max_rows": self.max_rows,
        }


@dataclass(frozen=True, slots=True)
class SourcePermissionResource:
    """One safe complete-catalog choice for the source-permissions control plane."""

    resource_id: str
    display_name: str
    resource_kind: str
    eligible_assignment_columns: tuple[str, ...] = ()
    key_columns: tuple[str, ...] = ()
    upsert_conflict_keys: tuple[tuple[str, ...], ...] = ()
    eligible_insert_columns: tuple[str, ...] = ()
    eligible_upsert_update_columns: tuple[str, ...] = ()
    generated_identity_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resource_id, str)
            or _SOURCE_PERMISSION_RESOURCE_ID.fullmatch(self.resource_id) is None
        ):
            raise ValueError("permission resource_id must be canonical")
        _permission_text(self.display_name, "permission resource display_name")
        _permission_text(
            self.resource_kind,
            "permission resource kind",
            maximum=128,
        )
        columns = _canonical_permission_texts(
            self.eligible_assignment_columns,
            "permission resource eligible_assignment_columns",
            maximum_items=_SOURCE_PERMISSION_MAX_CATALOG_COLUMNS,
            maximum_characters=256,
        )
        object.__setattr__(self, "eligible_assignment_columns", columns)
        for name in (
            "eligible_insert_columns",
            "eligible_upsert_update_columns",
            "generated_identity_columns",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_permission_texts(
                    getattr(self, name),
                    name,
                    maximum_items=512,
                    maximum_characters=256,
                ),
            )
        if (
            not isinstance(self.upsert_conflict_keys, tuple)
            or len(self.upsert_conflict_keys) > 512
        ):
            raise ValueError("upsert conflict keys must be bounded")
        object.__setattr__(
            self,
            "upsert_conflict_keys",
            tuple(
                _canonical_permission_texts(
                    key,
                    "upsert conflict key",
                    maximum_items=512,
                    maximum_characters=256,
                )
                for key in self.upsert_conflict_keys
            ),
        )
        object.__setattr__(
            self,
            "key_columns",
            _canonical_permission_texts(
                self.key_columns,
                "key columns",
                maximum_items=512,
                maximum_characters=256,
            ),
        )

    @property
    def relational_update_eligible(self) -> bool:
        return bool(self.eligible_assignment_columns)


@dataclass(frozen=True, slots=True)
class SourcePermissionState:
    """One exact read/write-scope state returned by the control plane."""

    read_scope: SourceReadScope
    relational_write_scopes: tuple[RelationalWriteScope, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.read_scope, SourceReadScope):
            raise TypeError("permission state read_scope must be SourceReadScope")
        scopes = tuple(self.relational_write_scopes)
        if any(not isinstance(scope, RelationalWriteScope) for scope in scopes):
            raise TypeError(
                "permission state update scopes must be RelationalWriteScope records"
            )
        if len({scope.resource_id for scope in scopes}) != len(scopes):
            raise ValueError("permission state update scopes cannot repeat resources")
        if any(
            scope.agent_id != self.read_scope.agent_id
            or scope.source_id != self.read_scope.source_id
            for scope in scopes
        ):
            raise ValueError("permission state scopes must share one owner")
        object.__setattr__(
            self,
            "relational_write_scopes",
            tuple(sorted(scopes, key=lambda scope: scope.resource_id)),
        )


@dataclass(frozen=True, slots=True)
class SourcePermissionSummary:
    """Bounded terminal-safe before/after summary for one proposed transition."""

    source_display_name: str
    read_mode: SourceReadMode
    selected_read_resource_count: int
    relational_write_table_count: int
    relational_write_table_examples: tuple[str, ...]
    automatic_read_addition_examples: tuple[str, ...]
    dependent_update_revocation_examples: tuple[str, ...]
    postgresql_privilege_status: str = "unknown"
    future_tables_write_enabled: bool = False

    def __post_init__(self) -> None:
        _permission_text(self.source_display_name, "permission summary source name")
        if not isinstance(self.read_mode, SourceReadMode):
            raise TypeError("permission summary read_mode must be SourceReadMode")
        for value, name in (
            (self.selected_read_resource_count, "selected read resource count"),
            (self.relational_write_table_count, "update table count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"permission summary {name} must be non-negative")
        for values, name in (
            (self.relational_write_table_examples, "update table examples"),
            (self.automatic_read_addition_examples, "automatic read examples"),
            (
                self.dependent_update_revocation_examples,
                "dependent revocation examples",
            ),
        ):
            normalized = _canonical_permission_texts(
                values,
                f"permission summary {name}",
                maximum_items=_SOURCE_PERMISSION_MAX_SUMMARY_EXAMPLES,
                maximum_characters=512,
            )
            object.__setattr__(
                self,
                {
                    "update table examples": "relational_write_table_examples",
                    "automatic read examples": "automatic_read_addition_examples",
                    "dependent revocation examples": (
                        "dependent_update_revocation_examples"
                    ),
                }[name],
                normalized,
            )
        if self.postgresql_privilege_status not in {
            "ready",
            "blocked",
            "unknown",
        }:
            raise ValueError("PostgreSQL privilege status is invalid")
        if not isinstance(self.future_tables_write_enabled, bool):
            raise TypeError("future-tables indicator must be boolean")
        if self.future_tables_write_enabled:
            raise ValueError("future PostgreSQL tables cannot be write-enabled")


@dataclass(frozen=True, slots=True)
class SourcePermissionsInspection:
    """Current exact scopes plus safe complete-catalog selection facts."""

    source_id: str
    source_display_name: str
    adapter_id: str
    catalog_generation: str | None
    state: SourcePermissionState
    resources: tuple[SourcePermissionResource, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_id, str)
            or _SOURCE_PERMISSION_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("permission inspection source_id must be canonical")
        _permission_text(
            self.source_display_name,
            "permission inspection source name",
        )
        _permission_text(
            self.adapter_id,
            "permission inspection adapter_id",
            maximum=128,
        )
        if self.catalog_generation is not None:
            _permission_text(
                self.catalog_generation,
                "permission inspection catalog_generation",
            )
        if not isinstance(self.state, SourcePermissionState):
            raise TypeError("permission inspection state must be SourcePermissionState")
        if self.state.read_scope.source_id != self.source_id:
            raise ValueError("permission inspection state belongs to another source")
        resources = tuple(self.resources)
        if len(resources) > _SOURCE_PERMISSION_MAX_RESOURCE_IDS or any(
            not isinstance(resource, SourcePermissionResource) for resource in resources
        ):
            raise ValueError("permission inspection resources are invalid or too large")
        if len({resource.resource_id for resource in resources}) != len(resources):
            raise ValueError("permission inspection resources cannot repeat")
        object.__setattr__(
            self,
            "resources",
            tuple(
                sorted(
                    resources,
                    key=lambda resource: (
                        resource.display_name.casefold(),
                        resource.resource_id,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class SourcePermissionsPreview:
    """One exact in-process confirmed transition; it is never persisted."""

    source_id: str
    catalog_generation: str | None
    before: SourcePermissionState
    after: SourcePermissionState
    automatic_read_additions: tuple[str, ...]
    dependent_write_revocations: tuple[str, ...]
    summary: SourcePermissionSummary
    confirmation_fingerprint: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_id, str)
            or _SOURCE_PERMISSION_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("permission preview source_id must be canonical")
        if self.catalog_generation is not None:
            _permission_text(
                self.catalog_generation,
                "permission preview catalog_generation",
            )
        for state, name in ((self.before, "before"), (self.after, "after")):
            if not isinstance(state, SourcePermissionState):
                raise TypeError(f"permission preview {name} must be state")
            if state.read_scope.source_id != self.source_id:
                raise ValueError(f"permission preview {name} belongs elsewhere")
        for values, name in (
            (self.automatic_read_additions, "automatic read additions"),
            (self.dependent_write_revocations, "dependent update revocations"),
        ):
            normalized = _canonical_permission_texts(
                values,
                f"permission preview {name}",
                maximum_items=_SOURCE_PERMISSION_MAX_RESOURCE_IDS,
                maximum_characters=256,
            )
            if any(
                _SOURCE_PERMISSION_RESOURCE_ID.fullmatch(value) is None
                for value in normalized
            ):
                raise ValueError(f"permission preview {name} must contain resource ids")
            object.__setattr__(
                self,
                (
                    "automatic_read_additions"
                    if name == "automatic read additions"
                    else "dependent_write_revocations"
                ),
                normalized,
            )
        if not isinstance(self.summary, SourcePermissionSummary):
            raise TypeError(
                "permission preview summary must be SourcePermissionSummary"
            )
        if (
            not isinstance(self.confirmation_fingerprint, str)
            or _SOURCE_PERMISSION_HASH.fullmatch(self.confirmation_fingerprint) is None
        ):
            raise ValueError(
                "permission confirmation fingerprint must be a sha256 hash"
            )


def relational_write_authorization_fingerprint(
    *,
    source: SourceRegistration,
    resource: CatalogResource,
    facet: CatalogFacet,
    scope: RelationalWriteScope,
) -> str:
    """Bind exact current structural and permission facts, excluding freshness."""
    if (
        source.adapter_id != "postgresql"
        or not source.active
        or resource.agent_id != source.agent_id
        or resource.source_id != source.id
        or resource.kind is not ResourceKind.TABLE
        or facet.resource_id != resource.id
        or facet.kind is not FacetKind.TABULAR
        or scope.agent_id != source.agent_id
        or scope.source_id != source.id
        or scope.resource_id != resource.id
        or scope.resource_revision != resource.current_revision
    ):
        raise ValueError(
            "write authorization requires exact current owned table structure"
        )
    tabular = TabularFacet.from_payload(facet.payload)
    columns = {column.name: column for column in tabular.columns}
    primary = {
        column.name
        for column in tabular.columns
        if column.primary_key_ordinal is not None
    }
    for name in scope.allowed_update_columns:
        column = columns.get(name)
        if (
            column is None
            or not column.updatable
            or column.identity
            or column.generated
            or name in primary
        ):
            raise ValueError("authorization update column is not eligible")
    if not set(scope.allowed_insert_columns) <= set(columns) or not set(
        scope.key_columns
    ) <= set(columns):
        raise ValueError("authorization column is not eligible")
    if any(not columns[name].identity for name in scope.generated_identity_columns):
        raise ValueError("authorization generated identity is not eligible")
    if "update" in scope.allowed_operations and set(scope.key_columns) != primary:
        raise ValueError("update authority requires the exact primary key")
    if "upsert" in scope.allowed_operations:
        if set(scope.key_columns) not in [
            set(index.columns)
            for index in tabular.indexes
            if index.unique
            and index.predicate is None
            and index.write_conflict_supported
        ]:
            raise ValueError("upsert authority requires a supported conflict key")
        if any(
            columns[name].identity or columns[name].generated
            for name in scope.allowed_insert_columns
        ):
            raise ValueError("explicit generated columns are not eligible")
    return (
        "sha256:"
        + sha256(
            canonical_json(
                {
                    "agent_id": source.agent_id,
                    "adapter_id": source.adapter_id,
                    "source_id": source.id,
                    "resource_id": resource.id,
                    "structure": tabular.structural_payload(),
                    "authority": scope.constraints(),
                }
            ).encode("utf-8")
        ).hexdigest()
    )


class EffectReceiptConflictError(RuntimeError):
    """A reserved operation or immutable terminal observation conflicts."""


class EffectUnresolvedError(RuntimeError):
    """External work is blocked by durable unresolved evidence."""

    def __init__(self, receipt_ids: tuple[str, ...], omitted_count: int) -> None:
        self.receipt_ids = receipt_ids
        self.omitted_count = omitted_count
        super().__init__(
            "Unresolved external effects require explicit foreground recovery."
        )


def effect_receipt_text(value: str, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty text without surrounding space")
    if len(value) > maximum or any(character in "\r\n\x00" for character in value):
        raise ValueError(f"{name} must be bounded single-line text")
    return value


def effect_receipt_aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def effect_receipt_id(
    *, agent_id: str, run_id: str, call_id: str, operation_key: str
) -> str:
    material = {
        "agent_id": effect_receipt_text(agent_id, "receipt agent_id"),
        "run_id": effect_receipt_text(run_id, "receipt run_id"),
        "call_id": effect_receipt_text(call_id, "receipt call_id"),
        "operation_key": operation_key,
    }
    if (
        not isinstance(operation_key, str)
        or _SOURCE_PERMISSION_HASH.fullmatch(operation_key) is None
    ):
        raise ValueError("receipt operation key must be a sha256 digest")
    return (
        "effect-receipt:sha256:"
        + sha256(canonical_json(material).encode("utf-8")).hexdigest()
    )


def validate_effect_receipt_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"effect-receipt:sha256:[0-9a-f]{64}", value) is None
    ):
        raise ValueError("receipt_id must be a canonical effect receipt id")
    return value


class EffectResolutionDecision(str, Enum):
    CLOSE_WITHOUT_RETRY = "close_without_retry"
    ALLOW_FUTURE_WORK = "allow_future_work"


@dataclass(frozen=True, slots=True)
class EffectResolution:
    """One human decision retained separately from the original observation."""

    receipt_id: str
    receipt_digest: str
    decision: EffectResolutionDecision
    approving_principal_id: str
    control_id: str
    resolved_at: datetime
    note: str
    evidence_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_effect_receipt_id(self.receipt_id)
        if (
            not isinstance(self.receipt_digest, str)
            or _SOURCE_PERMISSION_HASH.fullmatch(self.receipt_digest) is None
        ):
            raise ValueError("resolution receipt digest is invalid")
        if not isinstance(self.decision, EffectResolutionDecision):
            raise TypeError("resolution decision is invalid")
        effect_receipt_text(self.approving_principal_id, "resolution principal")
        effect_receipt_text(self.control_id, "resolution control identity")
        effect_receipt_aware(self.resolved_at, "resolution time")
        if (
            not isinstance(self.note, str)
            or not self.note.strip()
            or "\x00" in self.note
            or len(self.note.encode("utf-8")) > 4096
        ):
            raise ValueError("resolution requires a bounded non-empty user note")
        references = _canonical_permission_texts(
            self.evidence_references,
            "resolution evidence references",
            maximum_items=16,
            maximum_characters=512,
        )
        object.__setattr__(self, "evidence_references", references)


@dataclass(frozen=True, slots=True)
class EffectReceipt:
    """One runtime-reserved external operation and its immutable observation."""

    receipt_id: str
    receipt_kind: str
    agent_id: str
    run_id: str
    call_id: str
    capability_id: str
    domain_owner_id: str
    capability_contract_digest: str
    operation_key: str
    argument_fingerprint: str
    sensitivity: ModelSensitivity
    started_at: datetime
    routine_id: str | None = None
    routine_revision: int | None = None
    occurrence_id: str | None = None
    capability_grant_digest: str | None = None
    outcome: EffectOutcome = EffectOutcome.STARTED
    evidence_basis: EffectEvidenceBasis = EffectEvidenceBasis.UNKNOWN
    payload: FrozenJsonObject | None = None
    finished_at: datetime | None = None
    resolution: EffectResolution | None = None

    def __post_init__(self) -> None:
        validate_effect_receipt_id(self.receipt_id)
        for name in (
            "receipt_kind",
            "agent_id",
            "run_id",
            "call_id",
            "capability_id",
            "domain_owner_id",
        ):
            effect_receipt_text(getattr(self, name), f"receipt {name}")
        for name in (
            "capability_contract_digest",
            "operation_key",
            "argument_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _SOURCE_PERMISSION_HASH.fullmatch(value) is None
            ):
                raise ValueError(f"receipt {name} must be a sha256 digest")
        routine_fields = (
            self.routine_id,
            self.routine_revision,
            self.occurrence_id,
            self.capability_grant_digest,
        )
        if any(item is not None for item in routine_fields):
            if any(item is None for item in routine_fields):
                raise ValueError("receipt routine fields must be present together")
            effect_receipt_text(self.routine_id or "", "receipt routine_id")
            effect_receipt_text(self.occurrence_id or "", "receipt occurrence_id")
            if type(self.routine_revision) is not int or self.routine_revision < 1:
                raise ValueError("receipt routine revision must be positive")
            if (
                not isinstance(self.capability_grant_digest, str)
                or _SOURCE_PERMISSION_HASH.fullmatch(self.capability_grant_digest)
                is None
            ):
                raise ValueError("receipt grant digest is invalid")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("receipt sensitivity must be classified")
        effect_receipt_aware(self.started_at, "receipt start time")
        if not isinstance(self.outcome, EffectOutcome) or not isinstance(
            self.evidence_basis, EffectEvidenceBasis
        ):
            raise TypeError("receipt observation classification is invalid")
        if self.receipt_id != effect_receipt_id(
            agent_id=self.agent_id,
            run_id=self.run_id,
            call_id=self.call_id,
            operation_key=self.operation_key,
        ):
            raise ValueError("receipt ID does not match its execution identity")
        if self.outcome is EffectOutcome.STARTED:
            if (
                self.finished_at is not None
                or self.payload is not None
                or self.evidence_basis is not EffectEvidenceBasis.UNKNOWN
            ):
                raise ValueError("started receipt cannot contain terminal evidence")
        else:
            if self.finished_at is None:
                raise ValueError("terminal receipt requires a finish time")
            effect_receipt_aware(self.finished_at, "receipt finish time")
            if self.finished_at < self.started_at:
                raise ValueError("receipt cannot finish before it starts")
            observation = EffectObservation(
                self.outcome, self.evidence_basis, self.payload
            )
            object.__setattr__(self, "payload", observation.payload)
        if self.resolution is not None:
            if (
                not isinstance(self.resolution, EffectResolution)
                or self.outcome is not EffectOutcome.UNCERTAIN
            ):
                raise ValueError("only uncertain receipts can carry a human resolution")
            if (
                self.resolution.receipt_id != self.receipt_id
                or self.resolution.receipt_digest != self.receipt_digest
            ):
                raise ValueError("resolution does not bind this exact observation")
            if (
                self.finished_at is None
                or self.resolution.resolved_at < self.finished_at
            ):
                raise ValueError("resolution cannot precede its terminal observation")

    def material(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "receipt_kind": self.receipt_kind,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "call_id": self.call_id,
            "capability_id": self.capability_id,
            "domain_owner_id": self.domain_owner_id,
            "capability_contract_digest": self.capability_contract_digest,
            "routine_id": self.routine_id,
            "routine_revision": self.routine_revision,
            "occurrence_id": self.occurrence_id,
            "capability_grant_digest": self.capability_grant_digest,
            "operation_key": self.operation_key,
            "argument_fingerprint": self.argument_fingerprint,
            "outcome": self.outcome.value,
            "evidence_basis": self.evidence_basis.value,
            "sensitivity": self.sensitivity.value,
            "payload": self.payload,
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                None if self.finished_at is None else self.finished_at.isoformat()
            ),
        }

    @property
    def receipt_digest(self) -> str:
        return (
            "sha256:"
            + sha256(canonical_json(self.material()).encode("utf-8")).hexdigest()
        )

    @property
    def unresolved(self) -> bool:
        return self.resolution is None and self.outcome in {
            EffectOutcome.STARTED,
            EffectOutcome.UNCERTAIN,
        }

    def finish(
        self, observation: EffectObservation, *, finished_at: datetime
    ) -> EffectReceipt:
        if self.outcome is not EffectOutcome.STARTED:
            raise ValueError("only a started receipt can reach a terminal observation")
        return replace(
            self,
            outcome=observation.outcome,
            evidence_basis=observation.evidence_basis,
            payload=observation.payload,
            finished_at=finished_at,
        )

    def as_started(self) -> EffectReceipt:
        return replace(
            self,
            outcome=EffectOutcome.STARTED,
            evidence_basis=EffectEvidenceBasis.UNKNOWN,
            payload=None,
            finished_at=None,
            resolution=None,
        )


__all__ = [
    "EffectOutcome",
    "EffectReceipt",
    "EffectReceiptConflictError",
    "RelationalWriteScope",
    "SourcePermissionResource",
    "SourcePermissionState",
    "SourcePermissionStateError",
    "SourcePermissionSummary",
    "SourcePermissionsInspection",
    "SourcePermissionsPreview",
    "SourceReadMode",
    "SourceReadScope",
    "effect_receipt_aware",
    "effect_receipt_id",
    "effect_receipt_text",
    "relational_write_authorization_fingerprint",
    "validate_effect_receipt_id",
]
