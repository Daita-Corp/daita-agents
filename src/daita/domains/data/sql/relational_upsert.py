"""Validate one bounded uniform upsert batch against exact catalog structure."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import cast
from uuid import UUID

from ...._json import FrozenJsonObject, canonical_json, freeze_json
from ....capabilities import CapabilityInputError, render_approval_arguments
from .contracts import ResourceSchema
from .relational_update import _literal_issue, _qualified_identity

UPSERT_MAX_ROWS = 1000
UPSERT_MAX_BYTES = 64 * 1024
_KEY_TYPES = frozenset({"bool", "int2", "int4", "int8", "text", "varchar", "uuid"})
_VALUE_TYPES = _KEY_TYPES | {
    "date",
    "timestamp",
    "timestamptz",
    "numeric",
    "float4",
    "float8",
}


def _reject(code: str, message: str) -> None:
    raise CapabilityInputError(code, message)


def _names(raw: object, name: str, *, empty: bool = False) -> tuple[str, ...]:
    if not isinstance(raw, (tuple, list)) or (not raw and not empty):
        raise ValueError(f"{name} requires an array of column names")
    if len(raw) > 512 or any(
        not isinstance(v, str) or not v or len(v) > 256 or "\x00" in v for v in raw
    ):
        raise ValueError(f"{name} contains invalid columns")
    values = cast(tuple[str, ...], tuple(raw))
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class RelationalUpsertIntent:
    source_id: str
    resource_id: str
    key_columns: tuple[str, ...]
    insert_columns: tuple[str, ...]
    update_columns: tuple[str, ...]
    rows: tuple[FrozenJsonObject, ...]
    evidence_call_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, prefix in (
            (self.source_id, "source"),
            (self.resource_id, "catalog-resource"),
        ):
            if (
                not isinstance(value, str)
                or re.fullmatch(prefix + r":sha256:[0-9a-f]{64}", value) is None
            ):
                raise ValueError(
                    "upsert requires exact canonical source and resource IDs"
                )
        for name in ("key_columns", "insert_columns", "update_columns"):
            object.__setattr__(self, name, _names(getattr(self, name), name))
        if not set(self.key_columns) <= set(self.insert_columns) or not set(
            self.update_columns
        ) <= set(self.insert_columns) - set(self.key_columns):
            raise ValueError(
                "upsert requires explicit keys and immutable conflict columns"
            )
        if (
            not isinstance(self.rows, tuple)
            or not 1 <= len(self.rows) <= UPSERT_MAX_ROWS
        ):
            raise ValueError("upsert requires one batch of 1 to 1000 rows")
        for row in self.rows:
            if not isinstance(row, FrozenJsonObject) or set(row) != set(
                self.insert_columns
            ):
                raise ValueError(
                    "upsert rows must have the exact uniform insert-column shape"
                )
            if any(isinstance(value, (Mapping, tuple)) for value in row.values()):
                raise ValueError("upsert accepts scalar literals only")
        evidence = _names(self.evidence_call_ids, "evidence_call_ids", empty=True)
        if len(evidence) > 32:
            raise ValueError("upsert evidence exceeds 32 calls")
        # Preserve evidence order for transcript authentication.
        object.__setattr__(self, "evidence_call_ids", tuple(self.evidence_call_ids))
        if len(canonical_json(self.to_payload()).encode("utf-8")) > UPSERT_MAX_BYTES:
            raise ValueError("upsert batch exceeds 64 KiB")
        if (
            render_approval_arguments(
                {**self.to_payload(), "preview_fingerprint": "sha256:" + "0" * 64}
            )
            is None
        ):
            raise ValueError("upsert batch exceeds the exact approval bound")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RelationalUpsertIntent:
        required = {
            "source_id",
            "resource_id",
            "key_columns",
            "insert_columns",
            "update_columns",
            "rows",
        }
        if not required <= set(value) or set(value) - required - {
            "evidence_call_ids",
            "preview_fingerprint",
        }:
            raise ValueError("upsert arguments have unexpected fields")
        rows = value["rows"]
        if not isinstance(rows, (tuple, list)) or any(
            not isinstance(row, Mapping) for row in rows
        ):
            raise ValueError("upsert rows must be objects")
        return cls(
            source_id=cast(str, value["source_id"]),
            resource_id=cast(str, value["resource_id"]),
            key_columns=_names(value["key_columns"], "key_columns"),
            insert_columns=_names(value["insert_columns"], "insert_columns"),
            update_columns=_names(value["update_columns"], "update_columns"),
            rows=tuple(FrozenJsonObject.from_mapping(row) for row in rows),
            evidence_call_ids=cast(
                tuple[str, ...],
                tuple(cast(tuple[str, ...], value.get("evidence_call_ids", ()))),
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "resource_id": self.resource_id,
            "key_columns": self.key_columns,
            "insert_columns": self.insert_columns,
            "update_columns": self.update_columns,
            "rows": self.rows,
            "evidence_call_ids": self.evidence_call_ids,
        }


def validate_relational_upsert_scope(
    resource: ResourceSchema,
    *,
    key_columns: tuple[str, ...],
    insert_columns: tuple[str, ...],
    update_columns: tuple[str, ...],
    generated_identity_columns: tuple[str, ...],
) -> None:
    """Value-free readiness; no defaults or key equality are inferred from text."""
    if (
        resource.resource_kind != "table"
        or not resource.writable
        or not resource.revision
        or not resource.source_revision
    ):
        _reject(
            "write_resource_not_writable",
            "Upsert requires one current cataloged base table.",
        )
    _qualified_identity(resource)
    keys, inserts, updates, identities = map(
        set, (key_columns, insert_columns, update_columns, generated_identity_columns)
    )
    if not keys or keys not in [set(key) for key in resource.conflict_keys]:
        _reject(
            "upsert_key_unsupported",
            "Conflict columns require a supported immediate, plain, nonpartial unique key.",
        )
    types = {
        column: (namespace, name)
        for column, namespace, name in resource.column_type_provenance
    }
    nulls = dict(resource.column_nullability)
    collations = dict(resource.column_collations)
    for key in keys:
        namespace, name = types.get(key, (None, None))
        if (
            nulls.get(key) is not False
            or namespace != "pg_catalog"
            or name not in _KEY_TYPES
        ):
            _reject(
                "upsert_key_unsupported",
                "Conflict keys require supported non-null built-in equality semantics.",
            )
        if name in {"text", "varchar"} and collations.get(key) not in {"C", "POSIX"}:
            _reject(
                "upsert_key_unsupported",
                "Text conflict keys require exact cataloged C or POSIX collation.",
            )
    if (
        not keys <= inserts
        or not updates
        or not updates <= inserts - keys
        or not inserts <= set(resource.columns)
    ):
        _reject(
            "upsert_column_invalid",
            "Insert and update columns must describe one explicit uniform batch.",
        )
    if resource.generated_columns or (inserts | updates) & set(
        resource.identity_columns
    ):
        _reject(
            "upsert_generated_unsupported",
            "Generated expressions and explicit identity assignments are unsupported.",
        )
    if identities != set(resource.identity_columns) or identities & (
        keys | inserts | updates
    ):
        _reject(
            "upsert_identity_not_admitted",
            "Every omitted identity must be explicitly admitted outside the conflict key.",
        )
    if not updates <= set(resource.updatable_columns) or updates & set(
        resource.primary_key_columns
    ):
        _reject(
            "upsert_column_invalid",
            "Upsert cannot update non-updatable or primary-key columns.",
        )
    defaults = dict(resource.column_defaults)
    for column in resource.columns:
        namespace, name = types.get(column, (None, None))
        if (
            namespace != "pg_catalog"
            or name not in _VALUE_TYPES
            or column not in nulls
            or column not in defaults
        ):
            _reject(
                "upsert_type_unsupported",
                "Upsert requires complete supported scalar type, default and nullability facts.",
            )
        if column in identities:
            if name not in {"int2", "int4", "int8"}:
                _reject(
                    "upsert_identity_not_admitted",
                    "Identity generation requires a built-in integer identity.",
                )
        elif column not in inserts:
            if defaults[column] is not None:
                _reject(
                    "upsert_default_unsupported",
                    "Omitted default expressions are unsupported.",
                )
            if nulls[column] is not True:
                _reject(
                    "upsert_required_value_missing",
                    "An omitted non-null column requires an explicit insertion value.",
                )


@dataclass(frozen=True, slots=True)
class ValidatedRelationalUpsert:
    intent: RelationalUpsertIntent
    resource: ResourceSchema
    rows: tuple[FrozenJsonObject, ...]
    intent_sha256: str


def validate_relational_upsert_intent(
    intent: RelationalUpsertIntent,
    *,
    resource: ResourceSchema,
    generated_identity_columns: tuple[str, ...],
    max_rows: int,
) -> ValidatedRelationalUpsert:
    if (
        resource.resource_id != intent.resource_id
        or resource.source_id != intent.source_id
    ):
        _reject(
            "write_resource_not_writable",
            "Upsert requires the exact admitted resource.",
        )
    validate_relational_upsert_scope(
        resource,
        key_columns=intent.key_columns,
        insert_columns=intent.insert_columns,
        update_columns=intent.update_columns,
        generated_identity_columns=generated_identity_columns,
    )
    if len(intent.rows) > max_rows:
        _reject(
            "write_row_limit",
            "The batch exceeds the admitted row ceiling; chunking is unsupported.",
        )
    types = {
        column: (namespace, name)
        for column, namespace, name in resource.column_type_provenance
    }
    nullable = dict(resource.column_nullability)
    declared = dict(resource.column_declared_types)
    rows: list[FrozenJsonObject] = []
    keys: set[str] = set()
    for original in intent.rows:
        row = original.to_dict()
        for column, value in row.items():
            issue = _literal_issue(
                freeze_json(value),
                column,
                types,
                nullable,
                allow_null=True,
                code="upsert_value_invalid",
            )
            if issue is not None:
                _reject(*issue)
            if value is not None and types[column][1] == "uuid":
                row[column] = str(UUID(cast(str, value)))
            if value is not None and types[column][1] == "varchar":
                match = re.fullmatch(
                    r"character varying\((\d+)\)", declared.get(column, "")
                )
                if match and len(cast(str, value)) > int(match[1]):
                    _reject(
                        "upsert_value_invalid",
                        "A value exceeds the cataloged varchar length.",
                    )
        key = canonical_json(tuple(row[column] for column in intent.key_columns))
        if key in keys:
            _reject(
                "upsert_duplicate_key",
                "Input conflict keys must be distinct under the admitted database equality.",
            )
        keys.add(key)
        rows.append(FrozenJsonObject.from_mapping(row))
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: canonical_json(
                tuple(row[column] for column in intent.key_columns)
            ),
        )
    )
    material = {
        key: value
        for key, value in intent.to_payload().items()
        if key != "evidence_call_ids"
    }
    material["rows"] = ordered
    return ValidatedRelationalUpsert(
        intent,
        resource,
        ordered,
        "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest(),
    )
