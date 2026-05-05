"""Public eval suite API."""

from __future__ import annotations

from pathlib import Path


from .config import EvalSuiteConfig
from .models import EvalReport
from .runner import run_suite


class EvalSuite:
    """A configured eval suite for a runnable Daita target."""

    def __init__(self, config: EvalSuiteConfig, config_path: str | None = None):
        self.config = config
        self.config_path = config_path

    @classmethod
    def from_file(cls, path: str | Path) -> "EvalSuite":
        config = EvalSuiteConfig.from_file(path)
        return cls(config, config_path=str(path))

    async def run(
        self,
        *,
        output_dir: str | Path | None = None,
        write_artifacts: bool = True,
        compare_baseline: bool = False,
        record_baseline: bool = False,
        baseline_path: str | Path | None = None,
    ) -> EvalReport:
        return await run_suite(
            self.config,
            config_path=self.config_path,
            output_dir=output_dir,
            write_artifacts=write_artifacts,
            compare_baseline=compare_baseline,
            record_baseline=record_baseline,
            baseline_path=baseline_path,
        )
