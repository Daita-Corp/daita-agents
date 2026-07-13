from xml.etree import ElementTree

from scripts.run_from_db_bucket3_gate import (
    _consecutive_clean_runs,
    _junit_counts,
)


def test_consecutive_clean_runs_reset_after_a_failed_run():
    records = [
        {"counts_toward_consecutive_requirement": True},
        {"counts_toward_consecutive_requirement": False},
        {"counts_toward_consecutive_requirement": True},
        {"counts_toward_consecutive_requirement": True},
    ]

    assert _consecutive_clean_runs(records) == 2


def test_junit_counts_keep_skips_separate_from_passes(tmp_path):
    root = ElementTree.Element("testsuites")
    ElementTree.SubElement(
        root,
        "testsuite",
        tests="7",
        failures="1",
        errors="1",
        skipped="2",
    )
    path = tmp_path / "junit.xml"
    ElementTree.ElementTree(root).write(path)

    assert _junit_counts(path) == {
        "passed": 3,
        "failed": 1,
        "skipped": 2,
        "errors": 1,
    }
