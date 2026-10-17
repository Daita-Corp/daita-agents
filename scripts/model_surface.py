"""Check or rewrite the committed model-facing surface snapshot.

    .venv/bin/python scripts/model_surface.py check    # exit 1 when stale
    .venv/bin/python scripts/model_surface.py update   # rewrite the snapshot
    .venv/bin/python scripts/model_surface.py digest   # print the current digest

The snapshot is computed offline from a scripted fixture agent; no model,
network or credential is used. See docs/TESTING.md, "Model-surface gate".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(_ROOT / "src"), str(_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from tests.support.model_surface import (
    DIGEST_PATH,
    SNAPSHOT_PATH,
    UPDATE_COMMAND,
    committed_digest,
    compute_surface_text,
    describe_changes,
    digest_text,
    write_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "update", "digest"))
    arguments = parser.parse_args(argv)
    text = compute_surface_text()
    digest = digest_text(text)
    if arguments.command == "digest":
        print(digest)
        return 0
    if arguments.command == "update":
        write_snapshot(text)
        print(f"wrote {SNAPSHOT_PATH.relative_to(_ROOT)} ({len(text.encode())} bytes)")
        print(f"digest {digest}")
        return 0
    try:
        committed = SNAPSHOT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"missing snapshot; run {UPDATE_COMMAND}", file=sys.stderr)
        return 1
    if committed != text or committed_digest() != digest:
        for line in describe_changes(committed, text):
            print(f"- {line}", file=sys.stderr)
        print(f"model-facing surface changed; run {UPDATE_COMMAND}", file=sys.stderr)
        return 1
    print(f"model surface current: {digest} ({DIGEST_PATH.relative_to(_ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
