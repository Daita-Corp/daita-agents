"""A real host process whose PostgreSQL write hangs at commit, to be killed there.

Run as ``python -m tests.support.killed_write_host <root> <batch.json> <marker>``
from the repository root. Its scripted model answers one foreground turn, then
writes the native fixture's one-row upsert when the routine fires. Once the
approved write reaches its commit, the process writes ``<marker>`` and waits,
so a test can SIGKILL it while the outcome is genuinely unknown.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from daita.host.paths import host_paths
from daita.host.server import serve_host
from tests.data.writes._upsert_support import Connection, Database
from tests.support.mcp import MemoryKeychain
from tests.support.native_writes import (
    ScriptedResearchModel,
    companies_structure,
    measured_step,
    patch_fake_postgresql,
    response,
    scheduled_write_script,
)


def main() -> None:
    root, batch_file, marker = (Path(item) for item in sys.argv[1:4])
    db = Database()
    patch = pytest.MonkeyPatch()
    patch_fake_postgresql(patch, db, companies_structure(db))
    commit = Connection.commit

    async def hang_at_commit(connection: Connection) -> None:
        if connection.readonly:
            return await commit(connection)
        marker.write_text("committing", encoding="utf-8")
        await asyncio.Event().wait()

    patch.setattr(Connection, "commit", hang_at_commit)
    model = ScriptedResearchModel()
    model.replace_script(
        [
            measured_step(response(text="Noted.")),
            *scheduled_write_script(json.loads(batch_file.read_text())),
        ]
    )
    options = {
        "model": model,
        "model_profile": model.model_profile,
        "keychain": MemoryKeychain(),
    }
    asyncio.run(serve_host(host_paths(root), agent_options=options))


if __name__ == "__main__":
    main()
