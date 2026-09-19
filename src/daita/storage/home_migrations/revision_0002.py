"""Atomically convert revision-1 jobs into the revision-2 task graph."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..graph_schema import REVISION_2_DATABASE_SQL
from ..sqlite_schema import SCHEMA_REVISION_2
from .models import HomeMigration

_IMPLEMENTATION_FILES = (
    "revision_0002_conversion.py",
    "revision_0002_legacy_jobs.py",
    "revision_0002_legacy_job_codecs.py",
    "revision_0002_legacy_autonomy.py",
    "revision_0002_legacy_autonomy_codecs.py",
    "revision_0002_legacy_delivery.py",
)


def _implementation_source(name: str) -> str:
    return Path(__file__).with_name(name).read_text(encoding="utf-8")


def _apply_revision_2(home: Path, source_shape: str | None) -> None:
    from .revision_0002_conversion import stage_revision_2_home

    if source_shape is not None:
        raise ValueError("revision 2 accepts only a released revision-1 home")
    state_path = home / "state.db"
    if not state_path.is_file() or state_path.is_symlink():
        raise ValueError("revision-1 state database is unavailable")
    work = Path(tempfile.mkdtemp(prefix=".revision-2-", dir=home.parent))
    target = work / "target"
    try:
        stage_revision_2_home(
            home,
            target,
            now=datetime.now(UTC),
        )
        converted = target / "state.db"
        if not converted.is_file() or converted.is_symlink():
            raise ValueError("revision-2 staged database is unavailable")
        os.replace(converted, state_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)


REVISION_2 = HomeMigration(
    revision=2,
    migration_id="0002_durable_adaptive_task_graph",
    definition=(
        "Atomically replace revision-1 single-job and autonomous-follow-up state "
        "with the ratified durable adaptive task graph, preserving terminal "
        "evidence and converting only provably safe queued profiles to runnable "
        "graph work."
    ),
    affected_paths=("state.db",),
    target_schema=SCHEMA_REVISION_2,
    apply=_apply_revision_2,
    implementation_material=(
        REVISION_2_DATABASE_SQL,
        *(_implementation_source(name) for name in _IMPLEMENTATION_FILES),
    ),
)

__all__ = ["REVISION_2"]
