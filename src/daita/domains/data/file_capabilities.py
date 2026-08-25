"""Declare and execute bounded reads of catalog-admitted local data files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    Executor,
    ToolDiscoveryMetadata,
    ToolExecution,
    ToolExposureClass,
    ToolOutput,
    ToolView,
)
from .results import BoundedResultProjection

LOCAL_FILE_READ_CAPABILITY_ID = "data.file.read"
LOCAL_FILE_READ_EVIDENCE_KIND = "data.file.read_result"
LOCAL_FILE_READ_EXECUTOR_ID = "data.file.read.executor"
LOCAL_FILE_READ_TOOL_NAME = "data_read_file"


def _required_text(value: str, field_name: str, *, maximum: int = 2_048) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")


def _sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must use sha256")


def _column_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("columns must be a sequence of strings")
    result = tuple(values)
    if len(result) > 512:
        raise ValueError("columns exceed 512 items")
    for value in result:
        _required_text(value, "column", maximum=256)
    if len(result) != len(set(result)):
        raise ValueError("columns cannot contain duplicates")
    return result


def _limitation_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("limitation_reasons must be a sequence of strings")
    result = tuple(values)
    if len(result) > 16:
        raise ValueError("limitation_reasons exceed 16 items")
    for value in result:
        _required_text(value, "limitation reason", maximum=128)
    if len(result) != len(set(result)):
        raise ValueError("limitation_reasons cannot contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class LocalFileReadResult:
    """One source-validated tabular file read and its bounded projection."""

    source_id: str
    source_revision: str
    resource_id: str
    resource_revision: str
    format: str
    encoding: str
    columns: tuple[str, ...]
    projection: BoundedResultProjection
    complete: bool = True
    limitation_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, name, maximum in (
            (self.source_id, "source_id", 512),
            (self.source_revision, "source_revision", 1_024),
            (self.resource_id, "resource_id", 512),
            (self.format, "format", 64),
            (self.encoding, "encoding", 128),
        ):
            _required_text(value, name, maximum=maximum)
        _sha256(self.resource_revision, "resource_revision")
        columns = _column_tuple(self.columns)
        if not isinstance(self.projection, BoundedResultProjection):
            raise TypeError("projection must be a BoundedResultProjection")
        if not isinstance(self.complete, bool):
            raise TypeError("complete must be a boolean")
        limitations = _limitation_tuple(self.limitation_reasons)
        if self.complete == bool(limitations):
            raise ValueError("complete must agree with limitation_reasons")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "limitation_reasons", limitations)

    @property
    def truncated(self) -> bool:
        return not self.complete or self.projection.truncated

    @property
    def truncation_reasons(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*self.limitation_reasons, *self.projection.truncation_reasons)
            )
        )

    def tool_data(self) -> FrozenJsonObject:
        projection = self.projection.to_payload().to_dict()
        return FrozenJsonObject.from_mapping(
            {
                **projection,
                "columns": self.columns,
                "complete": self.complete,
                "encoding": self.encoding,
                "format": self.format,
                "resource_id": self.resource_id,
                "resource_revision": self.resource_revision,
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "truncated": self.truncated,
                "truncation_reasons": self.truncation_reasons,
            }
        )


class LocalFileReadBackend(Protocol):
    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        max_rows: int,
        max_bytes: int,
    ) -> LocalFileReadResult: ...


@dataclass(frozen=True, slots=True)
class LocalFileReadDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class LocalFileReadExecutor:
    executor_id = LOCAL_FILE_READ_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backend: LocalFileReadBackend,
        *,
        max_rows: int = 100,
        max_bytes: int = 65_536,
    ) -> None:
        _required_text(agent_id, "agent_id", maximum=512)
        if not callable(getattr(backend, "execute_read", None)):
            raise TypeError("backend must provide execute_read")
        for value, name, minimum in (
            (max_rows, "max_rows", 1),
            (max_bytes, "max_bytes", 2),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        self._agent_id = agent_id
        self._backend = backend
        self._max_rows = max_rows
        self._max_bytes = max_bytes

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = request.arguments["source_id"]
        resource_id = request.arguments["resource_id"]
        assert isinstance(source_id, str)
        assert isinstance(resource_id, str)
        result = await self._backend.execute_read(
            agent_id=self._agent_id,
            source_id=source_id,
            resource_id=resource_id,
            max_rows=self._max_rows,
            max_bytes=self._max_bytes,
        )
        if not isinstance(result, LocalFileReadResult):
            raise TypeError("local-file backend must return LocalFileReadResult")
        if result.source_id != source_id or result.resource_id != resource_id:
            raise ValueError("local-file backend returned different source scope")
        return ToolOutput(
            kind=LOCAL_FILE_READ_EVIDENCE_KIND,
            data=result.tool_data(),
        )


def local_file_read_declarations(
    agent_id: str,
    backend: LocalFileReadBackend,
) -> LocalFileReadDeclarations:
    executor = LocalFileReadExecutor(agent_id, backend)
    declarations = local_file_read_capability_declarations()
    return LocalFileReadDeclarations(
        capabilities=declarations.capabilities,
        executors=(executor,),
        tool_views=declarations.tool_views,
    )


def local_file_read_capability_declarations() -> CapabilityDeclarations:
    """Advertise the stable file-read contract without opening a local root."""

    capability = Capability(
        id=LOCAL_FILE_READ_CAPABILITY_ID,
        description="Read one attached CSV or JSON resource within fixed bounds.",
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "resource_id": {"type": "string"},
            },
            "required": ["source_id", "resource_id"],
            "additionalProperties": False,
        },
        output_kind=LOCAL_FILE_READ_EVIDENCE_KIND,
        output_schema=_file_read_output_schema(),
        executor_id=LOCAL_FILE_READ_EXECUTOR_ID,
        access_mode=AccessMode.READ,
    )
    view = ToolView(
        name=LOCAL_FILE_READ_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        discovery=ToolDiscoveryMetadata(
            summary="Read bounded rows from one attached CSV or JSON resource.",
            when_to_use="Use for validated values from an attached local data file.",
            keywords=("data", "file", "csv", "json", "read"),
            exposure_class=ToolExposureClass.CORE,
            eager_priority=900,
        ),
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(LOCAL_FILE_READ_EXECUTOR_ID,),
        tool_views=(view,),
    )


def _file_read_output_schema() -> dict[str, object]:
    properties = {
        "byte_limit": {"type": "integer"},
        "columns": {"type": "array"},
        "complete": {"type": "boolean"},
        "encoding": {"type": "string"},
        "format": {"type": "string"},
        "resource_id": {"type": "string"},
        "resource_revision": {"type": "string"},
        "returned_rows": {"type": "integer"},
        "row_limit": {"type": "integer"},
        "rows": {"type": "array"},
        "source_id": {"type": "string"},
        "source_revision": {"type": "string"},
        "total_rows": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "truncation_reasons": {"type": "array"},
        "trust_classification": {"type": "string"},
        "utf8_bytes": {"type": "integer"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


__all__ = [
    "LOCAL_FILE_READ_CAPABILITY_ID",
    "LOCAL_FILE_READ_EVIDENCE_KIND",
    "LOCAL_FILE_READ_EXECUTOR_ID",
    "LOCAL_FILE_READ_TOOL_NAME",
    "LocalFileReadBackend",
    "LocalFileReadDeclarations",
    "LocalFileReadExecutor",
    "LocalFileReadResult",
    "local_file_read_declarations",
    "local_file_read_capability_declarations",
]
