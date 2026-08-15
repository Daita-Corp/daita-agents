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
