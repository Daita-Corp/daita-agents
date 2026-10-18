"""Delete authenticated private scratch after its only writer has been reaped."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import struct
import sys
from functools import cache
from pathlib import Path

_ACL_TYPE_EXTENDED = 0x00000100
_ATTR_CMN_EXTENDED_SECURITY = 0x00400000
_FSOPT_NOFOLLOW = 0x00000001
_KAUTH_FILESEC_MAGIC = 0x012CC16D


def remove_owned_scratch(root: Path, device: int, inode: int) -> None:
    original = _lstat(root)
    if (
        not stat.S_ISDIR(original.st_mode)
        or original.st_uid != os.getuid()
        or (original.st_dev, original.st_ino) != (device, inode)
    ):
        raise OSError("Scratch identity changed")
    # Generated code can remove directory permissions, set user flags such as
    # UF_IMMUTABLE, or add deny ACL entries. Restore only entries inside this
    # authenticated tree, without following symlinks, after reaping.
    _clear_flags(root, original)
    root.chmod(0o700)

    def failed(error: OSError) -> None:
        raise error

    for current, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in directories:
            path = Path(current, name)
            facts = _lstat(path)
            if stat.S_ISDIR(facts.st_mode):
                if facts.st_uid != os.getuid():
                    raise OSError("Scratch directory ownership changed")
                _clear_flags(path, facts)
                path.chmod(0o700)
            else:
                _clear_flags(path, facts)
        for name in files:
            path = Path(current, name)
            _clear_flags(path, _lstat(path))
    final = root.lstat()
    if (final.st_dev, final.st_ino) != (device, inode):
        raise OSError("Scratch identity changed")
    shutil.rmtree(root)


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except PermissionError:
        if sys.platform != "darwin":
            raise
        # A deny ACL can hide metadata even from the owner, and user flags can
        # block ACL edits; restore both without following symlinks, then retry.
        os.chflags(path, 0, follow_symlinks=False)
        _remove_acl(path)
        return path.lstat()


def _clear_flags(path: Path, facts: os.stat_result) -> None:
    # User flags and ACLs exist only on BSD-derived platforms such as macOS.
    if sys.platform == "darwin":
        if facts.st_flags:
            os.chflags(path, 0, follow_symlinks=False)
        libc = _libc()
        ctypes.set_errno(0)
        existing = libc.acl_get_link_np(os.fsencode(path), _ACL_TYPE_EXTENDED)
        if existing:
            libc.acl_free(existing)
        if existing or ctypes.get_errno() != errno.ENOENT:
            _remove_acl(path)


@cache
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.acl_get_link_np.restype = ctypes.c_void_p
    libc.acl_get_link_np.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.acl_free.argtypes = [ctypes.c_void_p]
    libc.setattrlist.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    return libc


def _remove_acl(path: Path) -> None:
    # setattrlist(ATTR_CMN_EXTENDED_SECURITY, FSOPT_NOFOLLOW) needs only ownership,
    # even when deny entries block reading, writing, or listing the entry itself.
    # The kauth_filesec carries null owner/group GUIDs (unchanged) and
    # KAUTH_FILESEC_NOACL, which removes the ACL.
    attributes = struct.pack("<HHIIIII", 5, 0, _ATTR_CMN_EXTENDED_SECURITY, 0, 0, 0, 0)
    filesec = struct.pack("<I16s16sII", _KAUTH_FILESEC_MAGIC, b"", b"", 0xFFFFFFFF, 0)
    value = struct.pack("<iI", 8, len(filesec)) + filesec
    if _libc().setattrlist(
        os.fsencode(path), attributes, value, len(value), _FSOPT_NOFOLLOW
    ):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(path))
