"""Task mixin composition for ``DbRuntime``."""

from __future__ import annotations

from .catalog import DbRuntimeTaskCatalogMixin
from .evidence import DbRuntimeTaskEvidenceMixin
from .execution import DbRuntimeTaskExecutionMixin
from .inputs import DbRuntimeTaskInputMixin
from .planning import DbRuntimeTaskPlanningMixin
from .readiness import DbRuntimeTaskReadinessMixin
from .synthesis import DbRuntimeTaskSynthesisMixin


class DbRuntimeTasksMixin(
    DbRuntimeTaskExecutionMixin,
    DbRuntimeTaskPlanningMixin,
    DbRuntimeTaskReadinessMixin,
    DbRuntimeTaskInputMixin,
    DbRuntimeTaskCatalogMixin,
    DbRuntimeTaskEvidenceMixin,
    DbRuntimeTaskSynthesisMixin,
):
    pass
