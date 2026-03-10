"""Backend registry — maps data types to the right FocusBackend."""

from __future__ import annotations

from typing import Any, List

from .backends.base import FocusBackend
from .backends.dict import DictBackend
from .backends.pandas import PandasBackend

# Order matters: more specific backends must come before DictBackend (the catch-all)
_BACKENDS: List[FocusBackend] = [
    PandasBackend(),
    DictBackend(),
]


def get_backend(data: Any) -> FocusBackend:
    for backend in _BACKENDS:
        if backend.supports(data):
            return backend
    return DictBackend()


def register_backend(backend: FocusBackend, position: int = 0) -> None:
    """Insert a custom backend at the given position in the detection chain."""
    _BACKENDS.insert(position, backend)
