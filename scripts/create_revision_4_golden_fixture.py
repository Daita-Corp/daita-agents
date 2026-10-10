"""Create revision 4 from the immutable released revision-3 home."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

from daita.storage.home_migrations.revision_0004 import REVISION_4

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/agent-home-revisions"


def main() -> int:
    destination = FIXTURES / "revision-4"
    if destination.exists():
        raise RuntimeError("The revision-4 fixture already exists")
    with tempfile.TemporaryDirectory(prefix="daita-revision-4-golden-") as temporary:
        home = Path(temporary)
        for item in (FIXTURES / "revision-3").iterdir():
            if item.name == "state.sql":
                with sqlite3.connect(home / "state.db") as connection:
                    connection.executescript(item.read_text())
            elif item.is_dir():
                shutil.copytree(item, home / item.name)
            else:
                shutil.copyfile(item, home / item.name)
        REVISION_4.apply(home, None)
        with sqlite3.connect(home / "state.db") as connection:
            connection.execute(
                "INSERT INTO agent_home_migrations VALUES (?, ?, ?)",
                (4, REVISION_4.migration_id, REVISION_4.checksum),
            )
            connection.commit()
            dump = "\n".join(connection.iterdump()) + "\n"
        destination.mkdir()
        for item in home.iterdir():
            if item.name.startswith("state.db"):
                continue
            if item.is_dir():
                shutil.copytree(item, destination / item.name)
            else:
                shutil.copyfile(item, destination / item.name)
        (destination / "state.sql").write_text(dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
