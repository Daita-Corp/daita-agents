"""Define validated public configuration for an agent instance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .llm.models import ModelCallPolicy
from .llm.routing import ModelRoute
from .loop.models import LoopLimits


@dataclass(frozen=True, slots=True)
class AnalysisLimits:
    cell_seconds: float = 30.0
    cpu_seconds: float = 120.0
    memory_bytes: int = 1_536 * 1024 * 1024
    scratch_bytes: int = 64 * 1024 * 1024
    scratch_files: int = 256
    max_threads: int = 32
    sample_seconds: float = 0.02
    cleanup_seconds: float = 5.0
    revocation_seconds: float = 0.25
    trace_bytes: int = 4 * 1024 * 1024
    worker_count: int = 1
    generations: int = 16
    code_bytes: int = 65_536
    log_bytes: int = 16_384
    child_calls: int = 16
    rpc_bytes: int = 1_048_576
    input_bytes: int = 64 * 1024 * 1024
    file_bytes: int = 16 * 1024 * 1024
    outputs_per_cell: int = 4
    outputs_per_run: int = 16
    output_bytes_per_run: int = 64 * 1024 * 1024
    trace_records: int = 256
    decoded_bytes: int = 128 * 1024 * 1024
    table_rows: int = 1_000_000
    table_columns: int = 256
    result_bytes: int = 65_536
    parser_seconds: float = 10.0

    def __post_init__(self) -> None:
        ceilings = {
            "cell_seconds": 30,
            "cpu_seconds": 120,
            "memory_bytes": 1_536 * 1024 * 1024,
            "scratch_bytes": 64 * 1024 * 1024,
            "scratch_files": 256,
            "max_threads": 32,
            "sample_seconds": 0.1,
            "cleanup_seconds": 10,
            "revocation_seconds": 1,
            "trace_bytes": 4 * 1024 * 1024,
            "worker_count": 2,
            "generations": 16,
            "code_bytes": 65_536,
            "log_bytes": 16_384,
            "child_calls": 16,
            "rpc_bytes": 1_048_576,
            "input_bytes": 64 * 1024 * 1024,
            "file_bytes": 16 * 1024 * 1024,
            "outputs_per_cell": 4,
            "outputs_per_run": 16,
            "output_bytes_per_run": 64 * 1024 * 1024,
            "trace_records": 256,
            "decoded_bytes": 128 * 1024 * 1024,
            "table_rows": 1_000_000,
            "table_columns": 256,
            "result_bytes": 65_536,
            "parser_seconds": 10,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if (
                name
                not in {
                    "cell_seconds",
                    "cpu_seconds",
                    "sample_seconds",
                    "cleanup_seconds",
                    "revocation_seconds",
                    "parser_seconds",
                }
                and type(value) is not int
            ):
                raise TypeError(f"{name} must be an integer")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= ceiling
            ):
                raise ValueError(
                    f"{name} must be finite and within its analytical bound"
                )
        if (
            self.sample_seconds < 0.01
            or self.revocation_seconds < 0.02
            or self.cleanup_seconds < 1
            or self.trace_bytes < 1024 * 1024
            or self.trace_records < 4
            or self.result_bytes < 2048
        ):
            raise ValueError(
                "Analytical polling and cleanup intervals are below their supported bounds"
            )


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model_route: ModelRoute | None = None
    limits: LoopLimits = LoopLimits()
    model_call_policy: ModelCallPolicy = field(default_factory=ModelCallPolicy)
    analysis_limits: AnalysisLimits = field(default_factory=AnalysisLimits)

    def __post_init__(self) -> None:
        if self.model_route is not None and not isinstance(
            self.model_route, ModelRoute
        ):
            raise TypeError("model_route must be ModelRoute or None")
        if not isinstance(self.model_call_policy, ModelCallPolicy):
            raise TypeError("model_call_policy must be ModelCallPolicy")
        if not isinstance(self.limits, LoopLimits):
            raise TypeError("limits must be LoopLimits")
        if not isinstance(self.analysis_limits, AnalysisLimits):
            raise TypeError("analysis_limits must be AnalysisLimits")


__all__ = ["AgentConfig", "AnalysisLimits"]
