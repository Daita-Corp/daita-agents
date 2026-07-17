from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib

NEXT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = NEXT_ROOT.parent
MATRIX_PATH = NEXT_ROOT / "PARITY_MATRIX.md"

ALLOWED_CLASSIFICATIONS = {"MVP", "cutover", "post-MVP"}
ALLOWED_DISPOSITIONS = {
    "port",
    "replace",
    "defer (documented)",
    "external integration",
    "proposed removal requiring Phase 10 approval",
}


def _literal_all(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"No literal __all__ found in {path}")


def _matrix_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for line in MATRIX_PATH.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| (?:MB|API|EXT|SURF)-[A-Z0-9-]+ \|", line):
            continue
        rows.append([part.strip() for part in line.strip().strip("|").split("|")])
    return rows


def test_mandatory_inventory_has_all_51_unique_section_17_6_rows() -> None:
    rows = _matrix_rows()
    mandatory_ids = [row[0] for row in rows if row[0].startswith("MB-")]

    assert len(mandatory_ids) == 51
    assert len(mandatory_ids) == len(set(mandatory_ids))
    assert {identifier.split("-")[1] for identifier in mandatory_ids} == {
        "RR",
        "LG",
        "AR",
        "CR",
        "ML",
        "MO",
        "SP",
    }


def test_every_matrix_row_has_an_allowed_classification_and_disposition() -> None:
    rows = _matrix_rows()
    stable_ids = [row[0] for row in rows]

    assert len(stable_ids) == len(set(stable_ids))
    assert all(row[-2] in ALLOWED_CLASSIFICATIONS for row in rows)
    assert all(row[-1] in ALLOWED_DISPOSITIONS for row in rows)


def test_every_baseline_public_export_is_named_in_the_matrix() -> None:
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    export_files = (
        REPOSITORY_ROOT / "daita" / "__init__.py",
        REPOSITORY_ROOT / "daita" / "db" / "__init__.py",
        REPOSITORY_ROOT / "daita" / "llm" / "__init__.py",
    )
    exports = {name for path in export_files for name in _literal_all(path)}

    missing = sorted(name for name in exports if f"`{name}`" not in matrix)
    assert missing == []
    assert len(_literal_all(export_files[0])) == 61
    assert len(_literal_all(export_files[1])) == 23
    assert len(_literal_all(export_files[2])) == 13


def test_every_baseline_optional_extra_is_named_in_the_matrix() -> None:
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    extras = set(configuration["project"]["optional-dependencies"])

    missing = sorted(extra for extra in extras if f"`{extra}`" not in matrix)
    assert missing == []
    assert len(extras) == 44


def test_mvp_cutover_and_post_mvp_are_documented_separately() -> None:
    matrix = MATRIX_PATH.read_text(encoding="utf-8")

    assert "**MVP** means the architecture MVP" in matrix
    assert "**cutover** means the replacement-candidate gate" in matrix
    assert "**post-MVP** means deliberately deferred" in matrix
    assert "does not authorize Phase 10" in matrix
