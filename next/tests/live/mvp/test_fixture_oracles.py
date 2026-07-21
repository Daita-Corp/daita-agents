from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3

import pytest

from daita.domains.data import compare_tabular_datasets

from .fixture_oracles import (
    CROSS_SOURCE_KEY_NORMALIZATION,
    EXPECTED_FIXTURE_VERSION,
    EXPECTED_MANIFEST_SHA256,
    CommerceFixture,
    Discrepancy,
    build_commerce_fixture,
    build_customer_comparison_datasets,
    compute_oracles,
    load_manifest,
    manifest_digest,
    normalize_comparison_discrepancies,
)
from .harness import (
    ArtifactLeakError,
    LiveMvpUnavailable,
    REFERENCE_PROVIDER,
    WAVE1_BUDGETS,
    assert_artifacts_redacted,
    load_live_mvp_configuration,
)
from .prompt_corpus import (
    MVP_SESSION_FOLLOW_UPS,
    PROMPTS_BY_SCENARIO,
    PROMPT_CORPUS_VERSION,
    SESSION_FOLLOW_UPS,
)

pytestmark = pytest.mark.unit


def test_manifest_digest_and_fixture_shape_are_versioned(
    commerce_fixture: CommerceFixture,
) -> None:
    manifest = load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["fixture_version"] == EXPECTED_FIXTURE_VERSION
    assert commerce_fixture.manifest_digest == EXPECTED_MANIFEST_SHA256
    assert manifest_digest(manifest) == EXPECTED_MANIFEST_SHA256
    assert commerce_fixture.database_path.is_file()
    assert len(tuple(commerce_fixture.files_root.iterdir())) == 3

    with sqlite3.connect(commerce_fixture.database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        foreign_keys = sum(
            len(connection.execute(f'PRAGMA foreign_key_list("{table}")').fetchall())
            for table in tables
        )

    assert {
        "customers",
        "customers_archive_2025",
        "regions",
        "orders",
        "orders_archive_2025",
        "order_items",
        "products",
        "payments",
        "refunds",
        "support_cases",
        "monitor_metrics",
    } <= tables
    assert foreign_keys >= 7


def test_oracles_compute_exact_aggregates_freshness_and_discrepancies(
    commerce_fixture: CommerceFixture,
) -> None:
    oracles = compute_oracles(commerce_fixture)

    assert oracles.aggregate.customer_count == 4
    assert oracles.aggregate.net_revenue_cents == 16_200
    assert oracles.leading_region.region_name == "Europe"
    assert oracles.leading_region.net_revenue_cents == 12_500
    assert {
        plan: (value.customer_count, value.net_revenue_cents)
        for plan, value in oracles.plan_breakdown.items()
    } == {"enterprise": (2, 10_500), "growth": (2, 5_700)}
    assert oracles.enterprise_refunded_customer_count == 2
    assert oracles.enterprise_refunded_cents == 2_823
    assert oracles.newest_customer_export.name == "customers-2026-03-15.csv"
    assert oracles.comparison_key_normalization == CROSS_SOURCE_KEY_NORMALIZATION
    assert CROSS_SOURCE_KEY_NORMALIZATION == "stringify_integral"
    assert oracles.comparison_policy_schema_version == 1
    assert oracles.discrepancies == (
        Discrepancy(
            "value_mismatch",
            "2",
            "plan",
            "enterprise",
            "growth",
            file_present=True,
            database_present=True,
            file_type="string",
            database_type="string",
        ),
        Discrepancy(
            "value_mismatch",
            "4",
            "lifecycle_status",
            "inactive",
            "active",
            file_present=True,
            database_present=True,
            file_type="string",
            database_type="string",
        ),
        Discrepancy(
            "type_mismatch",
            "5",
            "email",
            "chloe@example.test",
            None,
            file_present=True,
            database_present=True,
            file_type="string",
            database_type="null",
        ),
        Discrepancy("right_only", "7"),
        Discrepancy("left_only", "8"),
    )


def test_aggregate_oracle_recomputes_from_mutated_fixture(
    commerce_fixture: CommerceFixture,
) -> None:
    before = compute_oracles(commerce_fixture).aggregate.net_revenue_cents
    with sqlite3.connect(commerce_fixture.database_path) as connection:
        connection.execute(
            "UPDATE refunds SET amount_cents = amount_cents + 100 WHERE id = 3001"
        )

    after = compute_oracles(commerce_fixture).aggregate.net_revenue_cents

    assert after == before - 100


def test_newest_export_oracle_uses_actual_fixture_metadata(
    commerce_fixture: CommerceFixture,
) -> None:
    older = commerce_fixture.files_root / "customers-2026-02-01.csv"
    older.touch()
    changed = compute_oracles(commerce_fixture)
    assert changed.newest_customer_export == older


def test_newest_export_oracle_fails_closed_on_equal_greatest_mtime(
    commerce_fixture: CommerceFixture,
) -> None:
    newest = commerce_fixture.files_root / "customers-2026-03-15.csv"
    older = commerce_fixture.files_root / "customers-2026-02-01.csv"
    greatest = newest.stat().st_mtime_ns
    os.utime(older, ns=(greatest, greatest))

    with pytest.raises(AssertionError, match="freshness is ambiguous"):
        compute_oracles(commerce_fixture)


def test_comparison_oracle_handles_every_kind_with_strict_values_and_orientation(
    commerce_fixture: CommerceFixture,
) -> None:
    newest = commerce_fixture.files_root / "customers-2026-03-15.csv"
    file_dataset, database_dataset = build_customer_comparison_datasets(
        commerce_fixture,
        newest,
    )
    assert isinstance(file_dataset.rows[0]["id"], str)
    assert isinstance(database_dataset.rows[0]["id"], int)

    file_dataset = replace(
        file_dataset,
        rows=(
            {"id": "1", "email": ""},
            {"id": "2", "email": "file"},
            {"id": "3"},
            {"id": "4", "email": "file-only"},
            {"email": "missing-key"},
            {"id": "6", "email": "duplicate-a"},
            {"id": "6", "email": "duplicate-b"},
        ),
    )
    database_dataset = replace(
        database_dataset,
        rows=(
            {"id": 1, "email": None},
            {"id": 2, "email": "database"},
            {"id": 3, "email": None},
            {"id": 5, "email": "database-only"},
            {"email": "missing-key"},
        ),
    )

    forward = compare_tabular_datasets(
        file_dataset,
        database_dataset,
        key_columns=("id",),
        compare_columns=("email",),
        key_normalization=CROSS_SOURCE_KEY_NORMALIZATION,
    )
    expected = normalize_comparison_discrepancies(
        forward,
        file_evidence_id=file_dataset.evidence_id,
    )
    reverse = compare_tabular_datasets(
        replace(database_dataset, rows=tuple(reversed(database_dataset.rows))),
        replace(file_dataset, rows=tuple(reversed(file_dataset.rows))),
        key_columns=("id",),
        compare_columns=("email",),
        key_normalization=CROSS_SOURCE_KEY_NORMALIZATION,
    )
    actual = normalize_comparison_discrepancies(
        reverse.payload,
        file_evidence_id=file_dataset.evidence_id,
    )

    assert actual == expected
    assert {item.kind for item in expected} == {
        "duplicate_key",
        "invalid_key",
        "left_only",
        "missing_value",
        "right_only",
        "type_mismatch",
        "value_mismatch",
    }
    type_mismatch = next(item for item in expected if item.kind == "type_mismatch")
    assert (
        type_mismatch.file_present,
        type_mismatch.file_type,
        type_mismatch.file_value,
    ) == (True, "string", "")
    assert (
        type_mismatch.database_present,
        type_mismatch.database_type,
        type_mismatch.database_value,
    ) == (True, "null", None)
    missing_value = next(item for item in expected if item.kind == "missing_value")
    assert missing_value.file_present is False
    assert missing_value.database_present is True
    assert missing_value.database_type == "null"
    assert {item.source for item in expected if item.kind == "invalid_key"} == {
        "file",
        "database",
    }


def test_prompt_corpus_has_three_natural_uncoached_variants_per_scenario() -> None:
    prohibited = (
        "catalog_search",
        "catalog_inspect",
        "data_query_",
        "data_read_file",
        "data_compare_tabular",
        "source_id",
        "resource_id",
        "evidence_id",
        "SELECT ",
    )

    assert PROMPT_CORPUS_VERSION == "wave1-prompts-v1"
    assert set(PROMPTS_BY_SCENARIO) == {
        "LIVE-MVP-01",
        "LIVE-MVP-02",
        "LIVE-MVP-03",
        "LIVE-MVP-04",
    }
    for prompts in PROMPTS_BY_SCENARIO.values():
        assert len(prompts) == 3
        assert len(set(prompts)) == 3
        assert all(prompt.strip().endswith(("?", ".")) for prompt in prompts)
        assert all(token not in prompt for prompt in prompts for token in prohibited)
    assert MVP_SESSION_FOLLOW_UPS == (SESSION_FOLLOW_UPS[1],)
    assert len(SESSION_FOLLOW_UPS) == 5


def test_live_configuration_requires_every_explicit_non_secret_setting() -> None:
    with pytest.raises(LiveMvpUnavailable) as captured:
        load_live_mvp_configuration({})

    assert captured.value.requirements == (
        "DAITA_RUN_LIVE_LLM=1",
        "DAITA_RUN_LIVE_MVP=1",
        "DAITA_LIVE_MVP_PROVIDER=openai",
        "DAITA_LIVE_MVP_MODEL=<explicit-model>",
        "OPENAI_API_KEY",
    )
    assert "secret" not in repr(captured.value.__dict__).casefold()

    configuration = load_live_mvp_configuration(
        {
            "DAITA_RUN_LIVE_LLM": "1",
            "DAITA_RUN_LIVE_MVP": "1",
            "DAITA_LIVE_MVP_PROVIDER": "openai",
            "DAITA_LIVE_MVP_MODEL": "gpt-test-explicit",
            "OPENAI_API_KEY": "configured-but-not-retained",
        }
    )
    assert configuration.provider == REFERENCE_PROVIDER
    assert configuration.model == "gpt-test-explicit"
    assert "configured-but-not-retained" not in repr(configuration)
    assert WAVE1_BUDGETS.max_turns == 12
    assert WAVE1_BUDGETS.max_actions == 24


def test_artifact_redaction_scanner_detects_secret_and_sentinel(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "safe.json").write_text('{"provider":"openai"}', encoding="utf-8")
    assert_artifacts_redacted(root, ("secret-value", "session-sentinel"))

    (root / "failed.txt").write_text("contains session-sentinel", encoding="utf-8")
    with pytest.raises(ArtifactLeakError, match="prohibited configured material"):
        assert_artifacts_redacted(root, ("secret-value", "session-sentinel"))


def test_fixture_builder_rejects_reuse_of_existing_root(tmp_path: Path) -> None:
    root = tmp_path / "already-exists"
    root.mkdir()
    with pytest.raises(FileExistsError):
        build_commerce_fixture(root)
