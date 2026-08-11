from pathlib import Path

FIXTURE = Path("tests/fixtures/postgresql")


def test_phase_four_fixture_declares_disposable_least_privileged_writer_and_canary():
    sql = (FIXTURE / "init.sql").read_text(encoding="utf-8")
    compose = (FIXTURE / "compose.yaml").read_text(encoding="utf-8")

    assert "tmpfs:" in compose
    assert "/var/lib/postgresql/data" in compose
    for attribute in (
        "NOSUPERUSER",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
        "NOBYPASSRLS",
    ):
        assert attribute in sql
    assert "CREATE ROLE daita_writer" in sql
    assert "CREATE SCHEMA write_canary" in sql
    assert "REVOKE ALL PRIVILEGES ON DATABASE daita_fixture FROM PUBLIC" in sql
    assert "REVOKE ALL PRIVILEGES ON DATABASE postgres FROM PUBLIC" in sql
    assert "REVOKE ALL PRIVILEGES ON DATABASE template1 FROM PUBLIC" in sql
    assert "GRANT CONNECT ON DATABASE daita_fixture TO daita_writer" in sql
    assert "GRANT USAGE ON SCHEMA write_canary TO daita_writer" in sql
    assert "GRANT SELECT ON write_canary.accounts TO daita_writer" in sql
    assert "UPDATE (\n    status," in sql
    assert "canary-42" in sql and "constraint peer" in sql
    assert "CREATE TABLE write_canary.permission_denied" in sql
    assert "CREATE TABLE write_canary.no_primary_key" in sql
    assert "CREATE TABLE write_canary.composite_primary_key" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "CREATE TRIGGER reject_trigger_update" in sql


def test_fixture_role_administration_is_not_owned_by_production_daita():
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("src/daita").rglob("*.py"))
    )

    assert "CREATE ROLE daita_writer" not in production
    assert "GRANT CONNECT ON DATABASE" not in production
    assert "administrator_password" not in production


def test_phase_four_rollout_document_covers_required_operational_response():
    guide = Path("docs/POSTGRESQL_ONE_ROW_UPDATE.md").read_text(encoding="utf-8")

    for required in (
        "Phase 4 release note",
        "External least-privilege setup",
        "Credential handling",
        "point-in-time",
        "recovery",
        "Non-production canary procedure",
        "set_source_write_access",
        "source-write-access disable",
        "Unknown-outcome response",
        "outcome_unknown",
        "do not retry automatically",
        "administrator credential",
        "tests/test_postgresql_update_certification_live.py",
        "tests/live/test_postgresql_update_acceptance_live.py",
    ):
        assert required in guide
    assert "/Users/jendala/.daita" in guide
    assert "Never substitute production" in guide
