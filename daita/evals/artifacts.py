"""Artifact writer for eval reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import ArtifactConfig
from .models import EvalReport
from .reporters import render_junit, render_markdown


class ArtifactWriter:
    """Writes canonical eval artifacts to disk."""

    def __init__(self, output_root: Path, artifact_config: ArtifactConfig):
        self.output_root = output_root
        self.config = artifact_config

    def write(self, report: EvalReport) -> Path:
        run_dir = self.output_root / report.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report.artifact_path = str(run_dir)

        self._write_json(run_dir / "report.json", report.model_dump(mode="json"))
        (run_dir / "summary.md").write_text(render_markdown(report))
        (run_dir / "junit.xml").write_text(render_junit(report))

        cases_dir = run_dir / "cases"
        for case in report.cases:
            case_dir = cases_dir / _safe_path(case.case_id)
            case_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(case_dir / "case.json", case.model_dump(mode="json"))
            for run in case.runs:
                self._write_json(
                    case_dir / f"{run.run_id}.json",
                    run.model_dump(mode="json"),
                )
                judge_run_dir = case_dir / run.run_id
                if run.judges:
                    judge_run_dir.mkdir(parents=True, exist_ok=True)
                for judge in run.judges:
                    if judge.input_payload is not None and judge.input_artifact_path:
                        self._write_json(
                            judge_run_dir / judge.input_artifact_path,
                            judge.input_payload,
                        )
                    if judge.output_payload is not None and judge.output_artifact_path:
                        self._write_json(
                            judge_run_dir / judge.output_artifact_path,
                            judge.output_payload,
                        )
            if case.runs_requested > 1:
                self._write_json(
                    case_dir / "diff.json",
                    {
                        "case_id": case.case_id,
                        "stability": case.stability.model_dump(mode="json"),
                    },
                )
        return run_dir

    def _write_json(self, path: Path, value: Any) -> None:
        redacted = self._redact(self._apply_privacy_policy(value))
        path.write_text(json.dumps(redacted, indent=2, sort_keys=True, default=str))

    def _apply_privacy_policy(self, value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if key == "final_answer" and not self.config.include_full_answers:
                    result[key] = None
                elif key == "payload" and not self.config.include_evidence_payloads:
                    result[key] = None
                else:
                    result[key] = self._apply_privacy_policy(child)
            return result
        if isinstance(value, list):
            return [self._apply_privacy_policy(child) for child in value]
        return value

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._redact(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self._redact(child) for child in value]
        if isinstance(value, str):
            text = value
            for pattern in self.config.redact_patterns:
                text = re.sub(pattern, "[REDACTED]", text)
            if len(text) > self.config.max_chars:
                text = text[: self.config.max_chars] + "...[truncated]"
            return text
        return value


def _safe_path(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "case"
