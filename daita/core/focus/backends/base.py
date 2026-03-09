"""Abstract base for Focus backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Set, Tuple

from ..ast import FocusQuery


class FocusBackend(ABC):

    @abstractmethod
    def supports(self, data: Any) -> bool:
        """Return True if this backend can handle the given data type."""
        ...

    def apply(self, data: Any, query: FocusQuery) -> Any:
        """Apply the FocusQuery. Native clauses run first; evaluator handles the rest."""
        data, applied = self._native_apply(data, query)
        from ..evaluator import evaluate_remaining
        return evaluate_remaining(data, query, applied)

    def _native_apply(self, data: Any, query: FocusQuery) -> Tuple[Any, Set[str]]:
        """
        Apply supported clauses natively against the data source.
        Returns (modified_data, set_of_clause_names_already_applied).
        Subclasses override this; default: no native handling.
        """
        return data, set()
