from pathlib import Path

READ_FIXTURE = Path("tests/fixtures/postgresql")
UPDATE_FIXTURE = Path("tests/fixtures/postgres-large")


def test_postgres_large_is_the_only_update_fixture():
    read_sql = (READ_FIXTURE / "init.sql").read_text(encoding="utf-8")
    update_sql = (UPDATE_FIXTURE / "init.sql").read_text(encoding="utf-8")

    assert "daita_writer" not in read_sql
    assert "CREATE ROLE daita_large_writer" in update_sql
    assert "GRANT SELECT ON support.tickets TO daita_large_writer" in update_sql
    assert (
        "GRANT UPDATE (priority) ON support.tickets TO daita_large_writer" in update_sql
    )


def test_update_fixture_role_is_least_privileged_and_externally_owned():
    sql = (UPDATE_FIXTURE / "init.sql").read_text(encoding="utf-8")
    for attribute in (
        "NOSUPERUSER",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
        "NOBYPASSRLS",
    ):
        assert attribute in sql

    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("src/daita").rglob("*.py"))
    )
    assert "CREATE ROLE daita_large_writer" not in production
    assert "GRANT UPDATE (priority)" not in production


def test_fixture_docs_route_update_testing_to_postgres_large():
    readme = (READ_FIXTURE / "README.md").read_text(encoding="utf-8")
    update_readme = (UPDATE_FIXTURE / "README.md").read_text(encoding="utf-8")
    assert "dedicated PostgreSQL update fixture" in readme
    assert "daita_large_writer" in update_readme
    assert "/source permissions" in update_readme


def test_release_canary_is_isolated_from_existing_reader_and_update_fixture():
    sql = (UPDATE_FIXTURE / "init.sql").read_text(encoding="utf-8")
    release_sql = sql.split("CREATE SCHEMA write_acceptance;", 1)[1]
    release_sql = release_sql.split("CREATE TABLE private.fixture_status", 1)[0]
    assert "CREATE ROLE daita_large_write_tester" in release_sql
    assert "GENERATED ALWAYS AS IDENTITY PRIMARY KEY" in release_sql
    assert 'domain text COLLATE "C" NOT NULL UNIQUE' in release_sql
    assert 'evidence_url text COLLATE "C" NOT NULL UNIQUE' in release_sql
    for attribute in (
        "NOSUPERUSER",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
        "NOBYPASSRLS",
    ):
        assert attribute in release_sql
    assert "GRANT SELECT, UPDATE ON write_acceptance.companies" in release_sql
    assert "GRANT INSERT (domain, name, evidence_url, notes)" in release_sql
    assert "GRANT ALL" not in release_sql
    assert "support.tickets" not in release_sql
    assert "TO daita_large_reader" not in release_sql
    assert "TO daita_large_writer;" not in release_sql
