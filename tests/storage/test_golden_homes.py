"""Open one immutable whole-home fixture for every production revision."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

from daita import Agent
from daita.llm.models import TextBlock
from daita.storage.home_migrations import (
    CURRENT_HOME_REVISION,
    HOME_MIGRATIONS,
    MINIMUM_SUPPORTED_HOME_REVISION,
)
from tests.support.paths import REPO_ROOT
from tests.support.workspace import workspace_for

FIXTURES = REPO_ROOT / "tests/fixtures/agent-home-revisions"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_fixture(fixture: Path, home: Path) -> None:
    home.mkdir(parents=True)
    for source in fixture.iterdir():
        if source.name == "state.sql":
            with sqlite3.connect(home / "state.db") as connection:
                connection.executescript(source.read_text(encoding="utf-8"))
                if fixture.name == "revision-2":
                    assert connection.execute(
                        "PRAGMA journal_mode = WAL"
                    ).fetchone() == ("wal",)
        elif source.is_dir():
            shutil.copytree(source, home / source.name)
        else:
            shutil.copyfile(source, home / source.name)


def test_every_production_home_revision_has_one_golden_fixture() -> None:
    assert {path.name for path in FIXTURES.iterdir() if path.is_dir()} == {
        f"revision-{migration.revision}" for migration in HOME_MIGRATIONS
    }


async def test_revision_1_golden_whole_home_upgrades_once_and_preserves_content(
    tmp_path: Path,
) -> None:
    fixture = FIXTURES / "revision-1"
    home = tmp_path / "agents/golden"
    _materialize_fixture(fixture, home)
    before = _sha256(home / "state.db")

    status = await Agent.inspect_home("golden", root=tmp_path)
    assert status.current_revision == 2
    assert status.found_revision == 1
    assert status.minimum_supported_revision == 1
    assert status.upgrade_required
    assert not status.recovery_required

    agent = await Agent.open(
        "golden",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        assert agent.id == "agent-golden-revision-1"
        assert await agent.read_memory() == "Golden durable memory.\n"
        assert await agent.read_user_profile() == "Golden durable user profile.\n"
        skill = await agent.read_skill("golden-skill")
        assert skill is not None
        assert skill.description == "A fixed revision-one procedure."
        assert skill.instructions == "Use current evidence only."
        transcript = await agent.transcript("run-golden-revision-1")
        assert transcript.messages[-1].content == (TextBlock("Golden answer."),)
    finally:
        await agent.close()

    assert _sha256(home / "state.db") != before
    assert len(tuple((home / ".home-rollbacks").iterdir())) == 1
    upgraded = _sha256(home / "state.db")
    reopened = await Agent.open(
        "golden",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    await reopened.close()
    assert _sha256(home / "state.db") == upgraded
    assert len(tuple((home / ".home-rollbacks").iterdir())) == 1


@pytest.mark.parametrize(
    "source_revision",
    tuple(
        migration.revision
        for migration in HOME_MIGRATIONS
        if migration.revision >= MINIMUM_SUPPORTED_HOME_REVISION
    ),
)
async def test_every_supported_golden_home_reaches_current_revision(
    tmp_path: Path,
    source_revision: int,
) -> None:
    fixture = FIXTURES / f"revision-{source_revision}"
    home = tmp_path / "agents/golden"
    _materialize_fixture(fixture, home)
    before = _sha256(home / "state.db")

    initial = await Agent.inspect_home("golden", root=tmp_path)
    assert initial.found_revision == source_revision
    assert initial.upgrade_required is (source_revision < CURRENT_HOME_REVISION)

    agent = await Agent.open(
        "golden",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    await agent.close()

    current = await Agent.inspect_home("golden", root=tmp_path)
    assert current.found_revision == CURRENT_HOME_REVISION
    assert not current.upgrade_required
    assert not current.recovery_required
    if source_revision == CURRENT_HOME_REVISION:
        assert _sha256(home / "state.db") == before
        assert not (home / ".home-rollbacks").exists()
    else:
        assert len(tuple((home / ".home-rollbacks").iterdir())) == 1
