"""Define immutable caller intent and host-known local file locations."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .llm.models import ModelSensitivity


class LocalFileAccess(str, Enum):
    """Select the runtime-only local file boundary for one host session."""

    WORKSPACE = "workspace"
    COMPUTER = "computer"


@dataclass(frozen=True, slots=True)
class LocalWorkspace:
    """One canonical working directory admitted again for each agent session.

    This record owns no descriptor and grants no durable authority. The
    composition root performs the final state-root overlap and physical
    identity checks before it constructs local file capabilities. Existing
    typed callers remain bounded unless they explicitly select computer access.
    """

    root: Path
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL
    access: LocalFileAccess = LocalFileAccess.WORKSPACE

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("workspace root must be pathlib.Path")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("workspace sensitivity must be ModelSensitivity")
        if not isinstance(self.access, LocalFileAccess):
            raise TypeError("workspace access must be LocalFileAccess")
        if self.sensitivity is ModelSensitivity.PUBLIC:
            raise ValueError("workspace sensitivity must be internal or stricter")
        raw = os.fspath(self.root)
        if not raw or "\x00" in raw:
            raise ValueError("workspace root must be an existing directory")
        try:
            canonical = self.root.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("workspace root must be an existing directory") from error
        if not canonical.is_dir():
            raise ValueError("workspace root must be an existing directory")
        try:
            user_home = Path.home().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("the user home directory could not be admitted") from error
        if canonical == Path(canonical.anchor):
            raise ValueError("the filesystem root cannot be a working directory")
        if self.access is LocalFileAccess.WORKSPACE:
            if canonical == user_home:
                raise ValueError("the user home directory cannot be a workspace")
        object.__setattr__(self, "root", canonical)


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two canonical paths contain one another in either direction."""

    return left == right or left in right.parents or right in left.parents


_KNOWN_DIRECTORY_NAMES = frozenset({"Downloads", "Documents", "Desktop"})


def resolve_os_known_directory(name: str, *, user_home: Path | None = None) -> Path:
    """Resolve one supported OS-known user directory without invoking a shell."""

    if name not in _KNOWN_DIRECTORY_NAMES:
        raise ValueError("known directory name is unsupported")
    home = Path.home() if user_home is None else user_home
    if not isinstance(home, Path):
        raise TypeError("user_home must be pathlib.Path")
    home = home.resolve(strict=True)
    if sys.platform == "win32":
        return _windows_known_directory(name)
    if sys.platform == "darwin":
        return _macos_known_directory(name)
    return _xdg_known_directory(name, home)


def resolve_os_downloads_directory() -> Path:
    """Resolve the OS Downloads directory for artifact delivery."""

    return resolve_os_known_directory("Downloads")


def _windows_known_directory(name: str) -> Path:
    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    values = {
        "Downloads": (
            0x374DE290,
            0x123F,
            0x4565,
            (0x91, 0x64, 0x39, 0xC4, 0x92, 0x5E, 0x46, 0x7B),
        ),
        "Documents": (
            0xFDD39AD0,
            0x238F,
            0x46AF,
            (0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7),
        ),
        "Desktop": (
            0xB4BFCC3A,
            0xDB2C,
            0x424C,
            (0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41),
        ),
    }
    data1, data2, data3, data4 = values[name]
    folder = _GUID(data1, data2, data3, (ctypes.c_ubyte * 8)(*data4))
    result = ctypes.c_wchar_p()
    windows_libraries = getattr(ctypes, "windll")
    status = windows_libraries.shell32.SHGetKnownFolderPath(
        ctypes.byref(folder), 0, None, ctypes.byref(result)
    )
    if status != 0 or not result.value:
        raise OSError(f"{name} known folder is unavailable")
    try:
        return Path(result.value)
    finally:
        windows_libraries.ole32.CoTaskMemFree(result)


def _macos_known_directory(name: str) -> Path:
    directory = {"Documents": 9, "Desktop": 12, "Downloads": 15}[name]
    foundation = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/Foundation.framework/Foundation"
    )
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    foundation.NSSearchPathForDirectoriesInDomains.argtypes = (
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_bool,
    )
    foundation.NSSearchPathForDirectoriesInDomains.restype = ctypes.c_void_p
    array = foundation.NSSearchPathForDirectoriesInDomains(directory, 1, True)
    if not array:
        raise OSError(f"{name} search path is unavailable")
    objc.sel_registerName.argtypes = (ctypes.c_char_p,)
    objc.sel_registerName.restype = ctypes.c_void_p
    message_address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    if message_address is None:
        raise OSError("Objective-C runtime is unavailable")
    object_at_index = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong
    )(message_address)
    utf8_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(
        message_address
    )
    string = object_at_index(array, objc.sel_registerName(b"objectAtIndex:"), 0)
    encoded = utf8_string(string, objc.sel_registerName(b"UTF8String"))
    if not encoded:
        raise OSError(f"{name} search path is unavailable")
    return Path(encoded.decode("utf-8"))


def _xdg_known_directory(name: str, home: Path) -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    document = config_home / "user-dirs.dirs"
    key = f"XDG_{name.upper()}_DIR="
    try:
        lines = document.read_text(encoding="utf-8").splitlines()
    except OSError:
        return home / name
    for line in lines:
        if not line.startswith(key):
            continue
        raw = line[len(key) :].strip()
        if len(raw) < 2 or raw[0] != '"' or raw[-1] != '"':
            break
        value = raw[1:-1]
        if value == "$HOME":
            return home
        prefix = "$HOME/"
        if value.startswith(prefix):
            return home / value[len(prefix) :]
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        break
    return home / name


__all__ = [
    "LocalFileAccess",
    "LocalWorkspace",
    "resolve_os_downloads_directory",
    "resolve_os_known_directory",
]
