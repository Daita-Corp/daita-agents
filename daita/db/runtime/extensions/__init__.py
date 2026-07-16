"""Runtime extension declarations and executors for DB runtime planning."""

from .analysis import (
    DbAnalysisCheckpointExecutor,
    DbAnalysisPlanExecutor,
    DbAnalysisPlanValidationExecutor,
    DbAnalysisReplanExecutor,
    DbAnalysisSummarizeExecutor,
)
from .monitor_create import (
    DbMonitorCommitCreateExecutor,
    DbMonitorPlanCreateExecutor,
)
from .memory_update import (
    DbMemoryCommitUpdateExecutor,
    DbMemoryPlanUpdateExecutor,
)
from .memory_learning import (
    DbMemoryLearningEnqueueExecutor,
    DbMemoryLearningRunExecutor,
)
from .monitor_lifecycle import (
    DbMonitorCommitLifecycleExecutor,
    DbMonitorLocalDeliveryExecutor,
    DbMonitorPlanLifecycleExecutor,
)
from .monitor_read import (
    DbMonitorReadExecutor,
    DbMonitorResolveApprovalExecutor,
)
from .hosted_delivery import (
    HostedInAppMonitorDeliveryExecutor,
    HostedInAppMonitorDeliveryPlugin,
)
from .plugin import DbRuntimePlanningPlugin

__all__ = [
    "DbAnalysisCheckpointExecutor",
    "DbAnalysisPlanExecutor",
    "DbAnalysisPlanValidationExecutor",
    "DbAnalysisReplanExecutor",
    "DbAnalysisSummarizeExecutor",
    "DbMemoryCommitUpdateExecutor",
    "DbMemoryLearningEnqueueExecutor",
    "DbMemoryLearningRunExecutor",
    "DbMemoryPlanUpdateExecutor",
    "DbMonitorCommitCreateExecutor",
    "DbMonitorCommitLifecycleExecutor",
    "DbMonitorLocalDeliveryExecutor",
    "DbMonitorPlanCreateExecutor",
    "DbMonitorPlanLifecycleExecutor",
    "DbMonitorReadExecutor",
    "DbMonitorResolveApprovalExecutor",
    "DbPlanningContextExecutor",
    "DbQueryPlanValidationExecutor",
    "DbRuntimePlanningPlugin",
    "HostedInAppMonitorDeliveryExecutor",
    "HostedInAppMonitorDeliveryPlugin",
]


def __getattr__(name: str):
    """Keep legacy query executors lazy for unsupported connector runtimes."""

    if name in {"DbPlanningContextExecutor", "DbQueryPlanValidationExecutor"}:
        from .query import (
            DbPlanningContextExecutor,
            DbQueryPlanValidationExecutor,
        )

        return {
            "DbPlanningContextExecutor": DbPlanningContextExecutor,
            "DbQueryPlanValidationExecutor": DbQueryPlanValidationExecutor,
        }[name]
    raise AttributeError(name)
