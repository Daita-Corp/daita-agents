"""Create the immutable revision-2 whole-home fixture from revision 1 once."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from daita.storage.home_migrations.revision_0002 import REVISION_2
from daita.storage.home_migrations.revision_0002_conversion import (
    stage_revision_2_home,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "agent-home-revisions"


def _materialize(source: Path, target: Path) -> None:
    target.mkdir()
    for item in source.iterdir():
        if item.name == "state.sql":
            with sqlite3.connect(target / "state.db") as connection:
                connection.executescript(item.read_text(encoding="utf-8"))
        elif item.is_dir():
            shutil.copytree(item, target / item.name)
        else:
            shutil.copyfile(item, target / item.name)


def main() -> int:
    destination = FIXTURES / "revision-2"
    if destination.exists():
        raise RuntimeError("revision-2 golden fixture already exists")
    with tempfile.TemporaryDirectory(prefix="daita-revision-2-golden-") as raw:
        work = Path(raw)
        source = work / "source"
        target = work / "target"
        _materialize(FIXTURES / "revision-1", source)
        stage_revision_2_home(
            source,
            target,
            now=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        )
        with sqlite3.connect(target / "state.db") as connection:
            connection.execute(
                "INSERT INTO agent_home_migrations VALUES (?, ?, ?)",
                (REVISION_2.revision, REVISION_2.migration_id, REVISION_2.checksum),
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            dump = "\n".join(connection.iterdump()) + "\n"
        destination.mkdir()
        for item in target.iterdir():
            if item.name in {"state.db", "state.db-wal", "state.db-shm"}:
                continue
            if item.is_dir():
                shutil.copytree(item, destination / item.name)
            else:
                shutil.copyfile(item, destination / item.name)
        (destination / "state.sql").write_text(dump, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
