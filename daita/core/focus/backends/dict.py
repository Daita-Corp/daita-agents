"""
Dict / list-of-dicts backend.

No native clause handling — the universal evaluator does all the work.
This backend exists purely to claim ownership of dict and list[dict] data types.
"""

from __future__ import annotations

from typing import Any

from .base import FocusBackend


class DictBackend(FocusBackend):

    def supports(self, data: Any) -> bool:
        if isinstance(data, dict):
            return True
        if isinstance(data, list) and (not data or isinstance(data[0], dict)):
            return True
        return False
