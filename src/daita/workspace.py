"""Define immutable caller intent for one required local workspace."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .llm.models import ModelSensitivity


@dataclass(frozen=True, slots=True)
class LocalWorkspace:
    """One canonical local directory admitted again for each agent session.

    This record owns no descriptor and grants no durable authority. The
    composition root performs the final state-root overlap and physical
    identity checks before it constructs local file capabilities.
    """

    root: Path
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("workspace root must be pathlib.Path")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("workspace sensitivity must be ModelSensitivity")
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
        if canonical == Path(canonical.anchor):
            raise ValueError("the filesystem root cannot be a workspace")
        try:
            user_home = Path.home().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("the user home directory could not be admitted") from error
        if canonical == user_home:
            raise ValueError("the user home directory cannot be a workspace")
        object.__setattr__(self, "root", canonical)


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two canonical paths contain one another in either direction."""

    return left == right or left in right.parents or right in left.parents


__all__ = ["LocalWorkspace"]
