"""Immutable, bounded evidence presented to one graph model-task attempt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from ..._json import FrozenJsonObject, canonical_json
from ...capabilities import GraphTaskBinding

MAX_TASK_CONTEXT_BYTES = 96 * 1024
MAX_PARENT_RESULTS = 16
MAX_PRIOR_ATTEMPTS = 2
MAX_CONTEXT_CHECKPOINTS = 8
MAX_CONTEXT_COMMENTS = 16
MAX_CONTEXT_CONTROLS = 16


def _frozen_records(
    values: tuple[Mapping[str, object], ...], *, maximum: int, name: str
) -> tuple[FrozenJsonObject, ...]:
    material = tuple(FrozenJsonObject.from_mapping(value) for value in values)
    if len(material) > maximum:
        raise ValueError(f"task context {name} exceeds its count bound")
    return material


@dataclass(frozen=True, slots=True)
class TaskContextBundle:
    """The only durable graph evidence visible to a model task.

    No conversation transcript or mutable advisory state is accepted here.
    Parent results, prior attempts, checkpoints, and comments are explicitly
    labeled as untrusted when rendered into a model request.
    """

    binding: GraphTaskBinding
    root_objective: str
    outcome_contract: Mapping[str, object]
    task_specification: Mapping[str, object]
    parent_results: tuple[Mapping[str, object], ...] = ()
    prior_attempts: tuple[Mapping[str, object], ...] = ()
    checkpoints: tuple[Mapping[str, object], ...] = ()
    comments: tuple[Mapping[str, object], ...] = ()
    controls: tuple[Mapping[str, object], ...] = ()
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, GraphTaskBinding):
            raise TypeError("task context requires GraphTaskBinding")
        if (
            not isinstance(self.root_objective, str)
            or not self.root_objective.strip()
            or len(self.root_objective.encode("utf-8")) > 16 * 1024
        ):
            raise ValueError("task context root objective is invalid")
        if self.created_at is not None and (
            not isinstance(self.created_at, datetime)
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("task context created_at must be timezone-aware")
        object.__setattr__(
            self,
            "outcome_contract",
            FrozenJsonObject.from_mapping(self.outcome_contract),
        )
        object.__setattr__(
            self,
            "task_specification",
            FrozenJsonObject.from_mapping(self.task_specification),
        )
        for field_name, maximum in (
            ("parent_results", MAX_PARENT_RESULTS),
            ("prior_attempts", MAX_PRIOR_ATTEMPTS),
            ("checkpoints", MAX_CONTEXT_CHECKPOINTS),
            ("comments", MAX_CONTEXT_COMMENTS),
            ("controls", MAX_CONTEXT_CONTROLS),
        ):
            object.__setattr__(
                self,
                field_name,
                _frozen_records(
                    tuple(getattr(self, field_name)),
                    maximum=maximum,
                    name=field_name,
                ),
            )
        if (
            len(canonical_json(self.material()).encode("utf-8"))
            > MAX_TASK_CONTEXT_BYTES
        ):
            raise ValueError("task context exceeds its aggregate byte bound")

    def material(self) -> dict[str, object]:
        return {
            "binding_digest": self.binding.digest,
            "root_objective": self.root_objective,
            "outcome_contract": self.outcome_contract,
            "task_specification": self.task_specification,
            "untrusted_parent_results": self.parent_results,
            "untrusted_prior_attempts": self.prior_attempts,
            "untrusted_checkpoints": self.checkpoints,
            "untrusted_comments": self.comments,
            "untrusted_controls": self.controls,
            "created_at": (
                None if self.created_at is None else self.created_at.isoformat()
            ),
        }

    @property
    def digest(self) -> str:
        return (
            "sha256:"
            + sha256(canonical_json(self.material()).encode("utf-8")).hexdigest()
        )


__all__ = ["TaskContextBundle"]
