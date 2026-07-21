from __future__ import annotations

import json

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    CapabilityInputError,
    CapabilityRegistry,
    ExecutionRequest,
)
from daita.domains.data import (
    TABULAR_COMPARE_CAPABILITY_ID,
    TABULAR_COMPARISON_POLICY_SCHEMA_VERSION,
    TABULAR_COMPARISON_SCHEMA_VERSION,
    TabularComparisonExecutor,
    TabularComparisonPreflightError,
    TabularEvidenceDataset,
    compare_tabular_datasets,
    preflight_tabular_comparison,
    tabular_comparison_extension_declarations,
)

REVISION_A = "sha256:" + "a" * 64
REVISION_B = "sha256:" + "b" * 64


def _frozen_object(value: object) -> FrozenJsonObject:
    assert isinstance(value, FrozenJsonObject)
    return value


def _dataset(
    evidence_id: str,
    source_id: str,
    rows: tuple[dict[str, object], ...],
    *,
    evidence_kind: str = "data.file.read_result",
    revision: str = REVISION_A,
    columns: tuple[str, ...] = ("id", "tenant", "value", "nullable"),
    complete: bool = True,
    truncation_reasons: tuple[str, ...] = (),
) -> TabularEvidenceDataset:
    return TabularEvidenceDataset(
        operation_id="operation-1",
        evidence_id=evidence_id,
        evidence_kind=evidence_kind,
        source_id=source_id,
        source_revision=f"revision:{source_id}",
        resource_revisions=((f"resource-{evidence_id}", revision),),
        columns=columns,
        rows=rows,
        complete=complete,
        truncation_reasons=truncation_reasons,
        row_limit=100,
        byte_limit=65_536,
    )


class _Reader:
    def __init__(self, *datasets: TabularEvidenceDataset) -> None:
        self.datasets = {dataset.evidence_id: dataset for dataset in datasets}

    async def load_dataset(
        self,
        *,
        operation_id: str,
        evidence_id: str,
    ) -> TabularEvidenceDataset:
        dataset = self.datasets[evidence_id]
        assert dataset.operation_id == operation_id
        return dataset


def _request(arguments: dict[str, object]) -> ExecutionRequest:
    return ExecutionRequest(
        operation_id="operation-1",
        task_id="task-1",
        turn_id="turn-1",
        capability_id=TABULAR_COMPARE_CAPABILITY_ID,
        executor_id="data.tabular.compare.executor",
        attempt=1,
        fencing_token=1,
        arguments=arguments,
    )


def test_strict_string_integer_domains_are_typed_preflight_rejection() -> None:
    left = _dataset(
        "evidence-left",
        "source-files",
        ({"id": "1", "value": "left"},),
    )
    right = _dataset(
        "evidence-right",
        "source-db",
        ({"id": 1, "value": "right"},),
        evidence_kind="data.sqlite.query_result",
        revision=REVISION_B,
    )

    with pytest.raises(TabularComparisonPreflightError) as rejected:
        preflight_tabular_comparison(
            left,
            right,
            key_columns=("id",),
            key_normalization="strict",
        )

    assert rejected.value.code == "data.compare.incompatible_key_types"
    assert rejected.value.details.to_dict() == {
        "allowed_modes": ["strict", "stringify_integral"],
        "details_truncated": False,
        "key_column": "id",
        "key_normalization": "strict",
        "left_evidence_id": "evidence-left",
        "left_type_domain": ["string"],
        "right_evidence_id": "evidence-right",
        "right_type_domain": ["integer"],
    }


@pytest.mark.parametrize("unsupported", [True, 1.0, [1], {"id": 1}])
def test_stringify_integral_rejects_every_other_json_key_type(
    unsupported: object,
) -> None:
    left = _dataset(
        "evidence-left",
        "source-left",
        ({"id": unsupported, "value": "left"},),
    )
    right = _dataset(
        "evidence-right",
        "source-right",
        ({"id": 1, "value": "right"},),
        revision=REVISION_B,
    )

    with pytest.raises(TabularComparisonPreflightError) as rejected:
        preflight_tabular_comparison(
            left,
            right,
            key_columns=("id",),
            key_normalization="stringify_integral",
        )

    assert rejected.value.code == "data.compare.incompatible_key_types"


@pytest.mark.parametrize(
    ("left_key", "right_key"),
    [
        ("001", 1),
        (" 1", 1),
        ("A", "a"),
        ("é", "e\u0301"),
    ],
)
def test_string_keys_keep_leading_zero_whitespace_case_and_unicode(
    left_key: str,
    right_key: object,
) -> None:
    result = compare_tabular_datasets(
        _dataset(
            "evidence-left",
            "source-left",
            ({"id": left_key, "value": "same"},),
        ),
        _dataset(
            "evidence-right",
            "source-right",
            ({"id": right_key, "value": "same"},),
            revision=REVISION_B,
        ),
        key_columns=("id",),
        compare_columns=("value",),
        key_normalization="stringify_integral",
    )

    assert _frozen_object(result.payload["counts"])["matched_keys"] == 0
    assert tuple(item["kind"] for item in result.discrepancies) == (
        "left_only",
        "right_only",
    )


def test_composite_keys_normalize_per_component_and_null_missing_stay_invalid() -> None:
    left = _dataset(
        "evidence-left",
        "source-left",
        (
            {"id": "1", "tenant": "north", "value": "left"},
            {"id": None, "tenant": "north", "value": "null"},
            {"tenant": "north", "value": "missing"},
        ),
    )
    right = _dataset(
        "evidence-right",
        "source-right",
        (
            {"id": 1, "tenant": "north", "value": "right"},
            {"id": None, "tenant": "north", "value": "null"},
            {"id": 2, "value": "missing"},
        ),
        revision=REVISION_B,
    )

    result = compare_tabular_datasets(
        left,
        right,
        key_columns=("tenant", "id"),
        compare_columns=("value",),
        key_normalization="stringify_integral",
    )

    counts = _frozen_object(result.payload["counts"])
    assert counts["matched_keys"] == 1
    assert counts["left_invalid_keys"] == 2
    assert counts["right_invalid_keys"] == 2
    mismatch = next(
        item for item in result.discrepancies if item["kind"] == "value_mismatch"
    )
    assert _frozen_object(mismatch["normalized_key"]) == FrozenJsonObject.from_mapping(
        {"tenant": "north", "id": "1"}
    )


def test_preflight_scans_all_retained_rows_while_upstream_partial_stays_explicit() -> (
    None
):
    incompatible_partial = _dataset(
        "evidence-left",
        "source-left",
        (
            {"id": "1", "value": "first"},
            {"id": True, "value": "retained-later-row"},
        ),
        complete=False,
        truncation_reasons=("byte_limit",),
    )
    right = _dataset(
        "evidence-right",
        "source-right",
        ({"id": 1, "value": "first"},),
        revision=REVISION_B,
    )
    with pytest.raises(TabularComparisonPreflightError) as rejected:
        preflight_tabular_comparison(
            incompatible_partial,
            right,
            key_columns=("id",),
            key_normalization="stringify_integral",
        )
    assert rejected.value.details["left_type_domain"] == ("boolean", "string")

    compatible_partial = _dataset(
        "evidence-left-compatible",
        "source-left",
        ({"id": "1", "value": "first"},),
        complete=False,
        truncation_reasons=("byte_limit",),
    )
    result = compare_tabular_datasets(
        compatible_partial,
        right,
        key_columns=("id",),
        compare_columns=("value",),
        key_normalization="stringify_integral",
    )
    assert result.complete is False
    assert result.truncation_reasons == ("left:byte_limit",)


def test_normalization_collision_is_bounded_and_exact_duplicates_are_not_collision() -> (
    None
):
    collision_left = _dataset(
        "evidence-left",
        "source-left",
        (
            {"id": "1", "value": "string"},
            {"id": 1, "value": "integer"},
        ),
    )
    right = _dataset(
        "evidence-right",
        "source-right",
        ({"id": 1, "value": "right"},),
        revision=REVISION_B,
    )

    with pytest.raises(TabularComparisonPreflightError) as rejected:
        compare_tabular_datasets(
            collision_left,
            right,
            key_columns=("id",),
            compare_columns=("value",),
            key_normalization="stringify_integral",
        )
    assert rejected.value.code == "data.compare.normalization_collision"
    details = rejected.value.details
    assert details["side"] == "left"
    assert details["row_indexes"] == (0, 1)
    assert details["details_truncated"] is False

    duplicate_left = _dataset(
        "evidence-left-duplicates",
        "source-left",
        (
            {"id": "1", "value": "first"},
            {"id": "1", "value": "second"},
        ),
    )
    duplicate_result = compare_tabular_datasets(
        duplicate_left,
        right,
        key_columns=("id",),
        compare_columns=("value",),
        key_normalization="stringify_integral",
    )
    assert (
        _frozen_object(duplicate_result.payload["counts"])["left_duplicate_keys"] == 1
    )
    assert duplicate_result.discrepancies[0]["kind"] == "duplicate_key"


def test_compared_values_keep_strict_type_null_and_missing_semantics() -> None:
    left = _dataset(
        "evidence-left",
        "source-left",
        (
            {"id": 1, "value": 1, "nullable": None},
            {"id": 2, "value": "same"},
        ),
    )
    right = _dataset(
        "evidence-right",
        "source-right",
        (
            {"id": 1, "value": "1", "nullable": ""},
            {"id": 2, "value": "same", "nullable": None},
        ),
        revision=REVISION_B,
    )

    result = compare_tabular_datasets(
        left,
        right,
        key_columns=("id",),
        compare_columns=("value", "nullable"),
        key_normalization="strict",
    )

    assert tuple(item["kind"] for item in result.discrepancies) == (
        "type_mismatch",
        "type_mismatch",
        "missing_value",
    )
    assert result.discrepancies[1]["left_type"] == "null"
    assert result.discrepancies[1]["right_type"] == "string"
    assert result.discrepancies[2]["left_present"] is False
    assert result.discrepancies[2]["right_type"] == "null"


@pytest.mark.parametrize(
    ("left_kind", "right_kind"),
    [
        ("data.file.read_result", "data.file.read_result"),
        ("data.sqlite.query_result", "data.postgresql.query_result"),
        ("data.file.read_result", "data.sqlite.query_result"),
    ],
)
def test_comparison_contract_is_source_kind_agnostic(
    left_kind: str,
    right_kind: str,
) -> None:
    result = compare_tabular_datasets(
        _dataset(
            "evidence-left",
            "source-left",
            ({"id": "1", "value": "same"},),
            evidence_kind=left_kind,
        ),
        _dataset(
            "evidence-right",
            "source-right",
            ({"id": 1, "value": "same"},),
            evidence_kind=right_kind,
            revision=REVISION_B,
        ),
        key_columns=("id",),
        compare_columns=("value",),
        key_normalization="stringify_integral",
    )

    assert _frozen_object(result.payload["counts"])["matched_keys"] == 1
    assert result.total_discrepancies == 0


def test_policy_and_compatibility_are_identical_in_evidence_and_artifact_v2() -> None:
    result = compare_tabular_datasets(
        _dataset(
            "evidence-left",
            "source-left",
            ({"id": "1", "value": "left"},),
        ),
        _dataset(
            "evidence-right",
            "source-right",
            ({"id": 1, "value": "right"},),
            revision=REVISION_B,
        ),
        key_columns=("id",),
        compare_columns=("value",),
        key_normalization="stringify_integral",
    )

    artifact = json.loads(result.artifact.content)
    assert artifact["schema_version"] == TABULAR_COMPARISON_SCHEMA_VERSION
    comparison_policy = _frozen_object(result.payload["comparison_policy"])
    assert artifact["comparison_policy"] == comparison_policy.to_dict()
    assert comparison_policy["schema_version"] == (
        TABULAR_COMPARISON_POLICY_SCHEMA_VERSION
    )


async def test_capability_enum_and_executor_repeat_the_same_preflight() -> None:
    left = _dataset(
        "evidence-left",
        "source-left",
        ({"id": "1", "value": "left"},),
    )
    right = _dataset(
        "evidence-right",
        "source-right",
        ({"id": 1, "value": "right"},),
        revision=REVISION_B,
    )
    executor = TabularComparisonExecutor(_Reader(left, right))
    declarations = tabular_comparison_extension_declarations()
    registry = CapabilityRegistry(
        capabilities=declarations.capabilities,
        executors=(executor,),
        tool_views=declarations.tool_views,
    )
    base_arguments: dict[str, object] = {
        "left_evidence_id": "evidence-left",
        "right_evidence_id": "evidence-right",
        "key_columns": ["id"],
        "compare_columns": ["value"],
    }

    with pytest.raises(CapabilityInputError) as missing:
        registry.validate_arguments(TABULAR_COMPARE_CAPABILITY_ID, base_arguments)
    assert missing.value.code == "capability.input.missing_required_fields"

    with pytest.raises(CapabilityInputError) as invalid:
        registry.validate_arguments(
            TABULAR_COMPARE_CAPABILITY_ID,
            {**base_arguments, "key_normalization": "coerce"},
        )
    assert invalid.value.code == "capability.input.enum_mismatch"
    assert invalid.value.details["allowed_values"] == (
        "strict",
        "stringify_integral",
    )

    with pytest.raises(TabularComparisonPreflightError) as executor_rejection:
        await executor.execute(
            _request({**base_arguments, "key_normalization": "strict"})
        )
    assert executor_rejection.value.code == "data.compare.incompatible_key_types"
