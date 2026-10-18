"""Delete authenticated private scratch after its only writer has been reaped."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


def remove_owned_scratch(root: Path, device: int, inode: int) -> None:
    original = root.lstat()
    if (
        not stat.S_ISDIR(original.st_mode)
        or original.st_uid != os.getuid()
        or (original.st_dev, original.st_ino) != (device, inode)
    ):
        raise OSError("Scratch identity changed")
    # Generated code can remove directory permissions or set user flags such as
    # UF_IMMUTABLE. Restore only entries inside this authenticated tree, without
    # following symlinks, after reaping.
    _clear_flags(root, original)
    root.chmod(0o700)

    def failed(error: OSError) -> None:
        raise error

    for current, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in directories:
            path = Path(current, name)
            facts = path.lstat()
            if stat.S_ISDIR(facts.st_mode):
                if facts.st_uid != os.getuid():
                    raise OSError("Scratch directory ownership changed")
                _clear_flags(path, facts)
                path.chmod(0o700)
            else:
                _clear_flags(path, facts)
        for name in files:
            path = Path(current, name)
            _clear_flags(path, path.lstat())
    final = root.lstat()
    if (final.st_dev, final.st_ino) != (device, inode):
        raise OSError("Scratch identity changed")
    shutil.rmtree(root)


def _clear_flags(path: Path, facts: os.stat_result) -> None:
    # st_flags and chflags exist only on BSD-derived platforms such as macOS.
    if getattr(facts, "st_flags", 0):
        os.chflags(path, 0, follow_symlinks=False)
