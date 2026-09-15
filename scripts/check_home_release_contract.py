#!/usr/bin/env python3
"""Generate and verify release evidence for the durable agent-home contract."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from typing import Any

from daita.llm.models import ModelCallPolicy
from daita.storage.home_migrations import (
    CURRENT_HOME_REVISION,
    HOME_MIGRATIONS,
    MINIMUM_SUPPORTED_HOME_REVISION,
)
from daita.storage.sqlite_schema import CURRENT_SCHEMA, SQLiteSchema

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "release" / "agent-home-contract.json"
SNAPSHOT_FORMAT = 1


class HomeReleaseContractError(ValueError):
    """Raised when release evidence does not match the current durable contract."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _schema_contract(schema: SQLiteSchema) -> dict[str, object]:
    return {
        "foreign_keys": {
            table: [list(item) for item in foreign_keys]
            for table, foreign_keys in sorted(schema.foreign_keys.items())
        },
        "named_indexes": {
            name: [table, unique, list(columns)]
            for name, (table, unique, columns) in sorted(schema.named_indexes.items())
        },
        "required_sql_fragments": {
            table: list(fragments)
            for table, fragments in sorted(schema.required_sql_fragments.items())
        },
        "tables": {
            table: [list(column) for column in columns]
            for table, columns in sorted(schema.tables.items())
        },
        "unique_constraints": {
            table: [list(columns) for columns in sorted(constraints)]
            for table, constraints in sorted(schema.unique_constraints.items())
        },
    }


def _literal_value(
    node: ast.expr,
    constants: dict[str, object],
) -> object:
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    return ast.literal_eval(node)


def _module_constants(tree: ast.Module) -> dict[str, object]:
    constants: dict[str, object] = {}
    for statement in tree.body:
        assigned_value: ast.expr | None
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            assigned_value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            assigned_value = statement.value
        else:
            continue
        if assigned_value is None:
            continue
        try:
            literal = ast.literal_eval(assigned_value)
        except (ValueError, TypeError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = literal
    return constants


def _record_field_contracts(codec_root: Path) -> dict[str, list[str]]:
    contracts: dict[str, tuple[str, ...]] = {}
    call_count = 0
    for path in sorted(codec_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _module_constants(tree)
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "record_fields"
            ):
                continue
            call_count += 1
            if len(node.args) != 3:
                raise HomeReleaseContractError(
                    f"{path.relative_to(ROOT)}:{node.lineno} uses an unsupported "
                    "record_fields call shape"
                )
            try:
                record_name = _literal_value(node.args[1], constants)
                raw_fields = _literal_value(node.args[2], constants)
            except (ValueError, TypeError, KeyError) as error:
                raise HomeReleaseContractError(
                    f"{path.relative_to(ROOT)}:{node.lineno} must declare literal "
                    "record name and field contracts"
                ) from error
            if (
                not isinstance(record_name, str)
                or not isinstance(raw_fields, tuple)
                or not raw_fields
                or not all(isinstance(item, str) and item for item in raw_fields)
                or len(set(raw_fields)) != len(raw_fields)
            ):
                raise HomeReleaseContractError(
                    f"{path.relative_to(ROOT)}:{node.lineno} has an invalid "
                    "record field contract"
                )
            prior = contracts.setdefault(record_name, raw_fields)
            if prior != raw_fields:
                raise HomeReleaseContractError(
                    f"stored record {record_name!r} has conflicting field contracts"
                )
    if call_count == 0:
        raise HomeReleaseContractError("no stored record field contracts were found")
    return {name: list(contracts[name]) for name in sorted(contracts)}


def _find_function(tree: ast.Module, name: str, path: Path) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise HomeReleaseContractError(
            f"{path.relative_to(ROOT)} must define exactly one {name} function"
        )
    return matches[0]


def _dictionary_field_sets(
    path: Path, function_names: tuple[str, ...]
) -> list[list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    contracts: set[tuple[str, ...]] = set()
    for function_name in function_names:
        function = _find_function(tree, function_name, path)
        for node in ast.walk(function):
            if not isinstance(node, ast.Dict) or not node.keys:
                continue
            keys: list[str] = []
            for key in node.keys:
                if key is None:
                    break
                try:
                    value = ast.literal_eval(key)
                except (ValueError, TypeError):
                    break
                if not isinstance(value, str) or not value:
                    break
                keys.append(value)
            else:
                contracts.add(tuple(sorted(keys)))
    if not contracts:
        raise HomeReleaseContractError(
            f"no dictionary field contracts found in {path.relative_to(ROOT)}"
        )
    return [list(contract) for contract in sorted(contracts)]


def _literal_string_sets(path: Path, function_name: str) -> list[list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = _find_function(tree, function_name, path)
    contracts: set[tuple[str, ...]] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Set) or not node.elts:
            continue
        try:
            values = tuple(sorted(ast.literal_eval(item) for item in node.elts))
        except (ValueError, TypeError):
            continue
        if all(isinstance(value, str) and value for value in values):
            contracts.add(values)
    if not contracts:
        raise HomeReleaseContractError(
            f"no literal field set found in {path.relative_to(ROOT)}:{function_name}"
        )
    return [list(contract) for contract in sorted(contracts)]


def _skill_grammar_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    literals: set[str] = set()
    for function_name in ("_render_skill", "_parse_skill"):
        function = _find_function(tree, function_name, path)
        for node in ast.walk(function):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if "daita-sensitivity" in value or "Instructions" in value or "# " in value:
                literals.add(value)
    if not literals:
        raise HomeReleaseContractError("retained skill grammar literals were not found")
    return sorted(literals)


def _home_file_contract() -> dict[str, object]:
    embedded = ROOT / "src" / "daita" / "hosting" / "embedded.py"
    delivery = ROOT / "src" / "daita" / "artifacts" / "delivery.py"
    skills = ROOT / "src" / "daita" / "skills" / "store.py"
    return {
        "agent_manifest": {
            "fields": _literal_string_sets(embedded, "_read_manifest"),
            "path": "agent.toml",
        },
        "artifact_layout": {
            "delivery_configuration": "artifacts/delivery-config.json",
            "manifest": "artifacts/<run-id>/<artifact-id>/manifest.json",
            "payload": "artifacts/<run-id>/<artifact-id>/payload",
        },
        "delivery_configuration": {
            "field_sets": _dictionary_field_sets(
                delivery,
                ("_config_mapping", "_grant_to_mapping"),
            ),
            "path": "artifacts/delivery-config.json",
        },
        "model_configuration": {
            "field_sets": _dictionary_field_sets(
                embedded,
                ("_encode_agent_config", "_persisted_profile_limits"),
            ),
            "model_call_policy_fields": sorted(
                item.name for item in fields(ModelCallPolicy)
            ),
            "path": "config.json",
        },
        "owned_documents": ["MEMORY.md", "USER.md"],
        "retained_skills": {
            "document": "skills/<skill-name>/SKILL.md",
            "grammar_literals": _skill_grammar_literals(skills),
        },
        "state_database": "state.db",
    }


def _migration_contract() -> list[dict[str, object]]:
    return [
        {
            "affected_paths": list(migration.affected_paths),
            "checksum": migration.checksum,
            "definition": migration.definition,
            "migration_id": migration.migration_id,
            "revision": migration.revision,
        }
        for migration in HOME_MIGRATIONS
    ]


def build_contract() -> dict[str, object]:
    """Return the semantic durable-format material guarded at release."""

    codec_root = ROOT / "src" / "daita" / "storage" / "sqlite_codecs"
    return {
        "home_files": _home_file_contract(),
        "migrations": _migration_contract(),
        "sqlite_record_fields": _record_field_contracts(codec_root),
        "sqlite_schema": _schema_contract(CURRENT_SCHEMA),
    }


def make_snapshot(
    contract: dict[str, object],
    *,
    home_revision: int,
    minimum_supported_home_revision: int,
) -> dict[str, object]:
    return {
        "contract": contract,
        "contract_sha256": sha256(
            _canonical_json(contract).encode("utf-8")
        ).hexdigest(),
        "home_revision": home_revision,
        "minimum_supported_home_revision": minimum_supported_home_revision,
        "snapshot_format": SNAPSHOT_FORMAT,
    }


def build_snapshot() -> dict[str, object]:
    """Build the current release snapshot from production persistence owners."""

    return make_snapshot(
        build_contract(),
        home_revision=CURRENT_HOME_REVISION,
        minimum_supported_home_revision=MINIMUM_SUPPORTED_HOME_REVISION,
    )


def render_snapshot(snapshot: dict[str, object]) -> str:
    return json.dumps(snapshot, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _validated_snapshot(value: Any, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "contract",
        "contract_sha256",
        "home_revision",
        "minimum_supported_home_revision",
        "snapshot_format",
    }:
        raise HomeReleaseContractError(f"{label} has an invalid snapshot shape")
    contract = value["contract"]
    digest = value["contract_sha256"]
    revision = value["home_revision"]
    minimum = value["minimum_supported_home_revision"]
    if value["snapshot_format"] != SNAPSHOT_FORMAT:
        raise HomeReleaseContractError(f"{label} uses an unsupported snapshot format")
    if not isinstance(contract, dict) or not contract:
        raise HomeReleaseContractError(f"{label} has no durable contract")
    if digest != sha256(_canonical_json(contract).encode("utf-8")).hexdigest():
        raise HomeReleaseContractError(f"{label} contract digest is invalid")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 1
        or not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum < 1
        or minimum > revision
    ):
        raise HomeReleaseContractError(f"{label} revision policy is invalid")
    return value


def load_snapshot(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HomeReleaseContractError(
            f"cannot read snapshot {path}: {error}"
        ) from error
    return _validated_snapshot(value, label=str(path))


def check_snapshot(path: Path = DEFAULT_SNAPSHOT) -> None:
    committed = load_snapshot(path)
    expected = build_snapshot()
    if committed != expected or path.read_text(encoding="utf-8") != render_snapshot(
        expected
    ):
        raise HomeReleaseContractError(
            f"{path} does not match the current durable contract; run "
            f"python scripts/check_home_release_contract.py write --snapshot {path}"
        )


def compare_snapshots(
    base: dict[str, object],
    current: dict[str, object],
) -> None:
    base = _validated_snapshot(base, label="released snapshot")
    current = _validated_snapshot(current, label="candidate snapshot")
    base_revision = base["home_revision"]
    current_revision = current["home_revision"]
    assert isinstance(base_revision, int)
    assert isinstance(current_revision, int)
    if current_revision < base_revision:
        raise HomeReleaseContractError(
            "candidate home revision is older than the latest tagged snapshot"
        )
    if (
        current["contract_sha256"] != base["contract_sha256"]
        and current_revision <= base_revision
    ):
        raise HomeReleaseContractError(
            "the durable agent-home contract changed without a new home revision"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in ("check", "write"):
        command = subparsers.add_parser(command_name)
        command.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--current", type=Path, default=DEFAULT_SNAPSHOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "write":
            snapshot = build_snapshot()
            arguments.snapshot.write_text(render_snapshot(snapshot), encoding="utf-8")
            print(
                f"wrote agent-home contract revision {CURRENT_HOME_REVISION} "
                f"to {arguments.snapshot}"
            )
        elif arguments.command == "check":
            check_snapshot(arguments.snapshot)
            print(
                f"verified agent-home contract revision {CURRENT_HOME_REVISION} "
                f"in {arguments.snapshot}"
            )
        elif arguments.command == "compare":
            compare_snapshots(
                load_snapshot(arguments.base),
                load_snapshot(arguments.current),
            )
            print(
                f"verified candidate {arguments.current} against tagged "
                f"snapshot {arguments.base}"
            )
        else:  # pragma: no cover - argparse owns command admission
            raise AssertionError("unreachable release-contract command")
    except HomeReleaseContractError as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
