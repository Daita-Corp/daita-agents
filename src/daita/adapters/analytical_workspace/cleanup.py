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
    # Generated code can remove directory permissions. Restore only directories
    # inside this authenticated tree, without following symlinks, after reaping.
    root.chmod(0o700)

    def failed(error: OSError) -> None:
        raise error

    for current, directories, _files in os.walk(
        root, followlinks=False, onerror=failed
    ):
        for name in directories:
            path = Path(current, name)
            facts = path.lstat()
            if stat.S_ISDIR(facts.st_mode):
                if facts.st_uid != os.getuid():
                    raise OSError("Scratch directory ownership changed")
                path.chmod(0o700)
    final = root.lstat()
    if (final.st_dev, final.st_ino) != (device, inode):
        raise OSError("Scratch identity changed")
    shutil.rmtree(root)
