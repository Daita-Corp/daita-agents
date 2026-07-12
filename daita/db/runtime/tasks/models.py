"""Data models for DB runtime task planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from daita.runtime import Task, TaskDependency

from .common import _json_dict


@dataclass(frozen=True)
class DbTaskSpec:
    """Runtime-owned description of DB work before a persisted task exists."""

    capability_id: str
    task_id: str | None = None
    owner: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = "planner"
    sequence: int = 1
    dependencies: tuple[TaskDependency, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    deterministic_key: str | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if not self.capability_id:
            raise ValueError("capability_id is required")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        object.__setattr__(self, "input", _json_dict(self.input))
        object.__setattr__(
            self,
            "dependencies",
            tuple(
                (
                    dependency
                    if isinstance(dependency, TaskDependency)
                    else TaskDependency.from_dict(dependency)
                )
                for dependency in self.dependencies
            ),
        )
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "task_id": self.task_id,
            "owner": self.owner,
            "input": self.input,
            "reason": self.reason,
            "sequence": self.sequence,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "metadata": self.metadata,
            "deterministic_key": self.deterministic_key,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbTaskSpec":
        values = dict(data)
        values["dependencies"] = tuple(
            TaskDependency.from_dict(item) for item in values.get("dependencies", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class DbTaskPlan:
    """Materialization result for one or more DB task specs."""

    tasks: tuple[Task, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "diagnostics": self.diagnostics,
        }
