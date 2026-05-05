"""Dataset loading and expansion for eval suites."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config import DatasetConfig, EvalCaseConfig, EvalSuiteConfig


def expand_cases(
    config: EvalSuiteConfig, *, config_path: str | None = None
) -> list[EvalCaseConfig]:
    """Return explicit cases plus dataset-derived cases."""
    cases = list(config.cases)
    if config.dataset is None:
        return cases

    base_dir = Path(config_path).parent if config_path else Path.cwd()
    dataset_path = Path(config.dataset.path)
    if not dataset_path.is_absolute():
        dataset_path = base_dir / dataset_path

    records = load_dataset_records(dataset_path)
    cases.extend(_record_to_case(record, config.dataset, config) for record in records)
    return cases


def load_dataset_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    if suffix == ".json":
        data = json.loads(path.read_text())
    else:
        data = yaml.safe_load(path.read_text()) or []

    if isinstance(data, dict):
        data = data.get("records", [])
    if not isinstance(data, list):
        raise ValueError("Eval dataset must be a list or an object with a records list")
    return data


def _record_to_case(
    record: dict[str, Any],
    dataset: DatasetConfig,
    suite: EvalSuiteConfig,
) -> EvalCaseConfig:
    if dataset.input_field not in record:
        raise ValueError(f"Dataset record missing input field: {dataset.input_field}")

    template = suite.case_template.model_dump(mode="python", exclude_none=True)
    case_data: dict[str, Any] = {
        **template,
        "id": str(record.get(dataset.id_field) or _stable_record_id(record)),
        "prompt": str(record[dataset.input_field]),
    }

    expected = record.get(dataset.expected_field)
    if expected:
        case_data["expectations"] = _merge_expectations(
            case_data.get("expectations") or {},
            _expected_to_expectations(expected),
        )
    return EvalCaseConfig.model_validate(case_data)


def _expected_to_expectations(expected: Any) -> dict[str, Any]:
    if not isinstance(expected, dict):
        return {"answer": {"contains": [str(expected)]}}
    if any(key in expected for key in ("answer", "tools", "sql", "operations")):
        return expected

    answer_keys = {"equals", "contains", "not_contains", "regex", "numeric"}
    answer = {key: value for key, value in expected.items() if key in answer_keys}
    return {"answer": answer} if answer else expected


def _merge_expectations(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_expectations(result[key], value)
        else:
            result[key] = value
    return result


def _stable_record_id(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, default=str).encode()
    return "case-" + hashlib.sha256(payload).hexdigest()[:12]
