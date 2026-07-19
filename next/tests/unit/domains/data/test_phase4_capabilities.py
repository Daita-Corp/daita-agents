from __future__ import annotations

from daita._json import FrozenJsonObject
from daita.capabilities import (
    CapabilityRegistry,
    EvidenceArtifact,
    ExecutionRequest,
)
from daita.domains.data import (
    LOCAL_FILE_READ_CAPABILITY_ID,
    LOCAL_FILE_READ_EVIDENCE_KIND,
    TABULAR_COMPARE_CAPABILITY_ID,
    TABULAR_COMPARE_EVIDENCE_KIND,
    LocalFileReadExecutor,
    LocalFileReadResult,
    TabularComparisonExecutor,
    TabularEvidenceDataset,
    compare_tabular_datasets,
    local_file_read_extension_declarations,
    project_result_rows,
    tabular_comparison_extension_declarations,
)

REVISION_A = "sha256:" + "a" * 64
REVISION_B = "sha256:" + "b" * 64


def _request(capability_id: str, arguments: dict[str, object]) -> ExecutionRequest:
    return ExecutionRequest(
        operation_id="operation-1",
        task_id="task-1",
        turn_id="turn-1",
        capability_id=capability_id,
        executor_id=f"{capability_id}.executor",
        attempt=1,
        fencing_token=1,
        arguments=arguments,
    )


class FileBackend:
    def __init__(self, result: LocalFileReadResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def execute_read(self, **arguments: object) -> LocalFileReadResult:
        self.calls.append(arguments)
        return self.result


async def test_local_file_contract_passes_only_resource_scope_and_fixed_bounds() -> (
    None
):
    artifact = EvidenceArtifact(
        content=b'[{"id":1,"name":"Ada"}]',
        media_type="application/json",
        sensitivity_class="internal",
        retention_class="operation",
    )
    backend = FileBackend(
        LocalFileReadResult(
            source_id="source-files",
            source_revision="tree:1",
            resource_id="resource-export",
            resource_revision=REVISION_A,
            format="csv",
            encoding="utf-8",
            columns=("id", "name"),
            projection=project_result_rows(
                ({"id": 1, "name": "Ada"},),
                max_rows=100,
                max_bytes=65_536,
            ),
            artifact=artifact,
        )
    )
    executor = LocalFileReadExecutor("agent-atlas", backend)
    extension = local_file_read_extension_declarations()
    registry = CapabilityRegistry(
        capabilities=extension.capabilities,
        executors=(executor,),
        tool_views=extension.tool_views,
    )

    candidate = await executor.execute(
        _request(
            LOCAL_FILE_READ_CAPABILITY_ID,
            {
                "source_id": "source-files",
                "resource_id": "resource-export",
            },
        )
    )

    assert [tool.name for tool in registry.tool_definitions()] == ["data_read_file"]
    assert backend.calls == [
        {
            "agent_id": "agent-atlas",
            "source_id": "source-files",
            "resource_id": "resource-export",
            "max_rows": 100,
            "max_bytes": 65_536,
        }
    ]
    assert candidate.kind == LOCAL_FILE_READ_EVIDENCE_KIND
    assert candidate.artifact is artifact
    assert candidate.payload["resource_revision"] == REVISION_A
    assert candidate.payload["complete"] is True
    assert candidate.payload["trust_classification"] == "untrusted_external_data"
    registry.validate_evidence(LOCAL_FILE_READ_CAPABILITY_ID, candidate)


def _dataset(
    evidence_id: str,
    source_id: str,
    revision: str,
    rows: tuple[dict[str, object], ...],
    *,
    complete: bool = True,
    reasons: tuple[str, ...] = (),
) -> TabularEvidenceDataset:
    return TabularEvidenceDataset(
        operation_id="operation-1",
        evidence_id=evidence_id,
        evidence_kind="data.file.read_result",
        source_id=source_id,
        source_revision="source:1",
        resource_revisions=((f"resource-{source_id}", revision),),
        columns=("id", "name", "status"),
        rows=rows,
        complete=complete,
        truncation_reasons=reasons,
        row_limit=100,
        byte_limit=65_536,
    )


def test_tabular_comparison_is_strict_deterministic_and_reports_bad_keys() -> None:
    left = _dataset(
        "evidence-left",
        "source-files",
        REVISION_A,
        (
            {"id": 1, "name": "Ada", "status": "active"},
            {"id": 2, "name": "Grace", "status": "active"},
            {"id": 3, "name": "Linus"},
            {"name": "missing", "status": "active"},
            {"id": None, "name": "null", "status": "active"},
            {"id": 4, "name": "duplicate-a", "status": "active"},
            {"id": 4, "name": "duplicate-b", "status": "active"},
            {"id": True, "name": "Boolean", "status": "active"},
        ),
    )
    right = _dataset(
        "evidence-right",
        "source-database",
        REVISION_B,
        (
            {"id": 1, "name": "Ada", "status": "inactive"},
            {"id": 3, "name": "Linus", "status": "inactive"},
            {"id": 4, "name": "database", "status": "active"},
            {"id": 5, "name": "New", "status": "active"},
            {"id": True, "name": "Boolean", "status": "active"},
        ),
    )

    first = compare_tabular_datasets(
        left,
        right,
        key_columns=("id",),
        compare_columns=("name", "status"),
    )
    second = compare_tabular_datasets(
        left,
        right,
        key_columns=("id",),
        compare_columns=("name", "status"),
    )

    assert first == second
    assert first.complete is True
    artifact_digest = first.payload["artifact_digest"]
    assert isinstance(artifact_digest, str)
    assert artifact_digest.startswith("sha256:")
    assert first.artifact.content == second.artifact.content
    counts = first.payload["counts"]
    assert isinstance(counts, FrozenJsonObject)
    assert counts["left_invalid_keys"] == 2
    assert counts["left_duplicate_keys"] == 1
    assert counts["left_only"] == 1
    assert counts["right_only"] == 1
    assert counts["matched_keys"] == 3
    assert counts["equal_rows"] == 1
    assert counts["different_rows"] == 2
    assert counts["value_mismatches"] == 2
    assert first.total_discrepancies == 7
    kinds = tuple(item["kind"] for item in first.discrepancies)
    assert kinds == (
        "invalid_key",
        "invalid_key",
        "duplicate_key",
        "left_only",
        "right_only",
        "value_mismatch",
        "missing_value",
    )


class DatasetReader:
    def __init__(self, datasets: tuple[TabularEvidenceDataset, ...]) -> None:
        self.datasets = {dataset.evidence_id: dataset for dataset in datasets}
        self.calls: list[tuple[str, str]] = []

    async def load_dataset(
        self,
        *,
        operation_id: str,
        evidence_id: str,
    ) -> TabularEvidenceDataset:
        self.calls.append((operation_id, evidence_id))
        return self.datasets[evidence_id]


async def test_comparison_executor_resolves_only_accepted_evidence_ids() -> None:
    left = _dataset(
        "evidence-left",
        "source-files",
        REVISION_A,
        ({"id": 1, "name": "Ada", "status": "active"},),
    )
    right = _dataset(
        "evidence-right",
        "source-database",
        REVISION_B,
        ({"id": 1, "name": "Ada", "status": "inactive"},),
    )
    reader = DatasetReader((left, right))
    executor = TabularComparisonExecutor(reader)
    extension = tabular_comparison_extension_declarations()
    registry = CapabilityRegistry(
        capabilities=extension.capabilities,
        executors=(executor,),
        tool_views=extension.tool_views,
    )

    candidate = await executor.execute(
        _request(
            TABULAR_COMPARE_CAPABILITY_ID,
            {
                "left_evidence_id": "evidence-left",
                "right_evidence_id": "evidence-right",
                "key_columns": ["id"],
                "compare_columns": ["name", "status"],
            },
        )
    )

    assert reader.calls == [
        ("operation-1", "evidence-left"),
        ("operation-1", "evidence-right"),
    ]
    assert candidate.kind == TABULAR_COMPARE_EVIDENCE_KIND
    assert candidate.artifact is not None
    left_provenance = candidate.payload["left"]
    right_provenance = candidate.payload["right"]
    assert isinstance(left_provenance, FrozenJsonObject)
    assert isinstance(right_provenance, FrozenJsonObject)
    assert left_provenance["source_id"] == "source-files"
    assert right_provenance["source_id"] == "source-database"
    registry.validate_evidence(TABULAR_COMPARE_CAPABILITY_ID, candidate)


def test_partial_inputs_and_discrepancy_limit_are_explicit() -> None:
    left = _dataset(
        "evidence-left",
        "source-files",
        REVISION_A,
        tuple(
            {"id": index, "name": f"left-{index}", "status": "active"}
            for index in range(4)
        ),
        complete=False,
        reasons=("row_limit",),
    )
    right = _dataset(
        "evidence-right",
        "source-database",
        REVISION_B,
        tuple(
            {"id": index, "name": f"right-{index}", "status": "active"}
            for index in range(4)
        ),
    )

    result = compare_tabular_datasets(
        left,
        right,
        key_columns=("id",),
        compare_columns=("name",),
        max_discrepancies=2,
        max_inline_discrepancies=1,
    )

    assert result.complete is False
    assert result.total_discrepancies == 4
    assert result.payload["stored_discrepancies"] == 2
    assert result.payload["truncated"] is True
    assert result.truncation_reasons == ("left:row_limit", "discrepancy_limit")
    discrepancy_sample = result.payload["discrepancy_sample"]
    assert isinstance(discrepancy_sample, tuple)
    assert len(discrepancy_sample) == 1
