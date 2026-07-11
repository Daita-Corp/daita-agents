import asyncio
from types import SimpleNamespace

import pytest

from daita.db import DbRuntime
from daita.db.evidence import evidence_in_task_plan_order
from daita.db.loop import DbAgentLoop
from daita.db.loop.execution import DbLoopTaskBatchExecutor
from daita.db.models import DbExecutionConfig, DbRuntimeConfig
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.db.runtime.types import DbRuntimeGovernanceBlocked
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import (
    AccessMode,
    ApprovalStatus,
    Capability,
    Evidence,
    EvidenceSchema,
    GovernanceResult,
    Operation,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    Task,
    TaskDependency,
    TaskDependencyKind,
    TaskStatus,
)


def _capability(
    capability_id="db.sql.execute_read",
    *,
    access=AccessMode.READ,
    concurrent_safe=True,
    side_effecting=False,
):
    return Capability(
        id=capability_id,
        owner="test",
        description="Test capability.",
        domains=frozenset({"db"}),
        operation_types=frozenset({"data.query"}),
        access=access,
        risk=RiskLevel.LOW,
        input_schema={"type": "object"},
        output_evidence=frozenset({"query.result"}),
        executor=f"test.{capability_id.rsplit('.', 1)[-1]}",
        side_effecting=side_effecting,
        concurrent_safe=concurrent_safe,
    )


def _task(index, capability=None, *, dependencies=(), status=TaskStatus.PENDING):
    capability = capability or _capability()
    return Task(
        id=f"task-{index}",
        operation_id="op-1",
        capability_id=capability.id,
        executor_id=capability.executor,
        input={"index": index},
        status=status,
        required_evidence=capability.output_evidence,
        dependencies=dependencies,
        metadata={"owner": capability.owner},
    )


class _Registry:
    def __init__(self, capabilities):
        self.capabilities = tuple(capabilities)

    def get_capability(self, capability_id, owner=None):
        matches = [
            item
            for item in self.capabilities
            if item.id == capability_id and (owner is None or item.owner == owner)
        ]
        if not matches:
            raise KeyError(capability_id)
        return matches[0]


class _Runtime:
    def __init__(self, capabilities, *, max_concurrency=1, execute=None):
        self.config = DbRuntimeConfig(
            execution=DbExecutionConfig(max_read_concurrency=max_concurrency)
        )
        self.registry = _Registry(capabilities)
        self._execute = execute or self._default_execute
        self.governance = GovernanceResult(True, False, False)
        self.not_ready = set()
        self.calls = []

    async def _default_execute(self, task, operation):
        return (
            Evidence(
                kind="query.result",
                owner="test",
                operation_id=operation.id,
                task_id=task.id,
                payload={"index": task.input["index"]},
            ),
        )

    async def execute_task(self, task, operation):
        self.calls.append(task.id)
        return await self._execute(task, operation)

    async def task_readiness(self, task, operation):
        return {"ready": task.id not in self.not_ready}

    async def evaluate_governance_persistence(
        self, operation, *, task, capability, stage
    ):
        return SimpleNamespace(result=self.governance)


class _ControlledReadExecutor:
    id = "concurrency_test.read"
    capability_ids = frozenset({"concurrency.read"})

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.second_returned = asyncio.Event()

    async def execute(self, task, operation, context):
        self.calls.append(task.id)
        index = int(task.input["index"])
        if index == 0:
            self.first_started.set()
            await self.release_first.wait()
        else:
            await self.first_started.wait()
            self.second_returned.set()
        return [
            Evidence(
                kind="query.result",
                owner="concurrency_test",
                operation_id=operation.id,
                task_id=task.id,
                payload={"index": index},
            )
        ]


class _ConcurrentReadPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="concurrency_test",
        display_name="Concurrency Test",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self) -> None:
        self.executor = _ControlledReadExecutor()

    def declare_capabilities(self):
        return (
            Capability(
                id="concurrency.read",
                owner="concurrency_test",
                description="Execute one controlled read.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor=self.executor.id,
                runtime_only=True,
                side_effecting=False,
                concurrent_safe=True,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def declare_evidence_schemas(self):
        return (
            EvidenceSchema(
                kind="query.result",
                owner="concurrency_test",
                json_schema={"type": "object"},
            ),
        )


class _BlockingReadPolicy:
    id = "block_concurrent_read"
    owner = "concurrency_policy"

    def __init__(self, effect: PolicyEffect) -> None:
        self.effect = effect

    def applies_to(self, request, operation_type):
        return True

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation):
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=self.effect,
            reason="Controlled concurrency policy.",
            severity=RiskLevel.LOW,
            required_approvals=(
                ("controlled_read",)
                if self.effect is PolicyEffect.REQUIRE_APPROVAL
                else ()
            ),
        )


class _ConcurrencyPolicyPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="concurrency_policy",
        display_name="Concurrency Policy",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self, effect: PolicyEffect) -> None:
        self.policy = _BlockingReadPolicy(effect)

    def declare_policies(self):
        return (self.policy,)


class _NeverPlanner:
    async def plan(self, state):
        raise AssertionError("planner should not run")


OPERATION = Operation(id="op-1", operation_type="db.run")


async def _real_concurrent_runtime(*, policy_effect=None):
    read_plugin = _ConcurrentReadPlugin()
    plugins = [read_plugin]
    if policy_effect is not None:
        plugins.append(_ConcurrencyPolicyPlugin(policy_effect))
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            plugins=tuple(plugins),
            execution=DbExecutionConfig(max_read_concurrency=2),
        )
    )
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_id="real-concurrent-operation",
        operation_type="data.query",
        request={"prompt": "controlled reads"},
        evaluate_governance=False,
    )
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id="concurrency.read",
                owner="concurrency_test",
                task_id="real-task-0",
                input={"index": 0},
                sequence=1,
            ),
            DbTaskSpec(
                capability_id="concurrency.read",
                owner="concurrency_test",
                task_id="real-task-1",
                input={"index": 1},
                sequence=2,
            ),
        ),
    )
    return runtime, read_plugin, operation, plan.tasks


async def _wait_for_task_evidence(runtime, operation_id, task_id):
    async with asyncio.timeout(1):
        while True:
            evidence = await runtime.store.list_evidence(operation_id)
            if any(item.task_id == task_id for item in evidence):
                return
            await asyncio.sleep(0)


def test_execution_config_defaults_are_serial_and_validate_bounds():
    config = DbRuntimeConfig()
    assert config.execution == DbExecutionConfig(1, 1)
    assert config.metadata["from_db_options"]["analysis_max_concurrency"] == 1
    configured = DbRuntimeConfig(
        execution=DbExecutionConfig(analysis_max_concurrency=3)
    )
    assert configured.metadata["from_db_options"]["analysis_max_concurrency"] == 3
    with pytest.raises(ValueError, match="max_read_concurrency"):
        DbExecutionConfig(max_read_concurrency=0)
    with pytest.raises(ValueError, match="analysis_max_concurrency"):
        DbExecutionConfig(analysis_max_concurrency=0)


def test_evidence_projection_preserves_non_task_positions_and_task_plan_order():
    capability = _capability()
    tasks = (_task(0, capability), _task(1, capability))
    planner = Evidence(kind="planner.decision", payload={"turn": 1})
    observation = Evidence(kind="planner.observation", payload={"turn": 1})
    second = Evidence(kind="query.result", task_id="task-1", payload={"index": 1})
    first = Evidence(kind="query.result", task_id="task-0", payload={"index": 0})

    ordered = evidence_in_task_plan_order(
        (planner, second, observation, first),
        tasks,
    )

    assert ordered == (planner, first, observation, second)


async def test_max_one_preserves_serial_plan_order():
    capability = _capability()
    active = 0
    order = []

    async def execute(task, operation):
        nonlocal active
        active += 1
        assert active == 1
        order.append(task.id)
        await asyncio.sleep(0)
        active -= 1
        return ()

    runtime = _Runtime((capability,), execute=execute)
    outcomes = await DbLoopTaskBatchExecutor(runtime).execute(
        tuple(_task(index, capability) for index in range(3)), OPERATION
    )
    assert order == ["task-0", "task-1", "task-2"]
    assert [item.task_index for item in outcomes] == [0, 1, 2]


async def test_eligible_reads_overlap_without_exceeding_bound():
    capability = _capability()
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum = 0

    async def execute(task, operation):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            started.set()
        await release.wait()
        active -= 1
        return await _Runtime((capability,))._default_execute(task, operation)

    runtime = _Runtime((capability,), max_concurrency=2, execute=execute)
    running = asyncio.create_task(
        DbLoopTaskBatchExecutor(runtime).execute(
            tuple(_task(index, capability) for index in range(3)), OPERATION
        )
    )
    await started.wait()
    assert maximum == 2
    release.set()
    outcomes = await running
    assert maximum == 2
    assert [item.task_index for item in outcomes] == [0, 1, 2]


async def test_serial_validation_wave_unlocks_concurrent_read_wave():
    validation = _capability(
        "db.sql.validate",
        access=AccessMode.METADATA_READ,
        concurrent_safe=False,
    )
    read = _capability()
    release = asyncio.Event()
    reads_started = asyncio.Event()
    completed = set()
    active_reads = 0

    async def execute(task, operation):
        nonlocal active_reads
        if task.capability_id == validation.id:
            completed.add(task.id)
            return ()
        active_reads += 1
        if active_reads == 2:
            reads_started.set()
        await release.wait()
        active_reads -= 1
        return ()

    runtime = _Runtime((validation, read), max_concurrency=2, execute=execute)

    async def readiness(task, operation):
        producer_ids = {
            dependency.producer_task_id
            for dependency in task.dependencies
            if dependency.producer_task_id is not None
        }
        return {"ready": producer_ids <= completed}

    runtime.task_readiness = readiness
    tasks = (
        _task(0, validation),
        _task(1, validation),
        _task(
            2,
            read,
            dependencies=(
                TaskDependency(
                    kind=TaskDependencyKind.EVIDENCE,
                    evidence_kind="sql.validation",
                    producer_task_id="task-0",
                ),
            ),
        ),
        _task(
            3,
            read,
            dependencies=(
                TaskDependency(
                    kind=TaskDependencyKind.EVIDENCE,
                    evidence_kind="sql.validation",
                    producer_task_id="task-1",
                ),
            ),
        ),
    )
    batch = DbLoopTaskBatchExecutor(runtime)
    running = asyncio.create_task(batch.execute(tasks, OPERATION))
    await reads_started.wait()
    assert active_reads == 2
    release.set()
    outcomes = await running
    assert runtime.calls[:2] == ["task-0", "task-1"]
    assert [item.task_index for item in outcomes] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "capabilities,dependencies",
    [
        (
            (_capability(),),
            (
                TaskDependency(
                    kind=TaskDependencyKind.EVIDENCE,
                    evidence_kind="query.result",
                    producer_task_id="task-0",
                ),
            ),
        ),
        (
            (
                _capability(),
                _capability(
                    "db.sql.execute_write", access=AccessMode.WRITE, side_effecting=True
                ),
            ),
            (),
        ),
        (
            (
                _capability(
                    "db.sql.execute_write",
                    access=AccessMode.WRITE,
                    side_effecting=True,
                ),
            ),
            (),
        ),
        ((_capability(side_effecting=True),), ()),
        ((_capability(concurrent_safe=False),), ()),
        (
            (_capability(),),
            (
                TaskDependency(
                    kind=TaskDependencyKind.APPROVAL,
                    approval_id="approval-1",
                    approval_status=ApprovalStatus.APPROVED,
                ),
            ),
        ),
    ],
    ids=[
        "dependency",
        "mixed-read-write",
        "writes",
        "side-effecting",
        "not-opted-in",
        "approval-dependency",
    ],
)
async def test_ineligible_batches_remain_serial(capabilities, dependencies):
    active = 0
    maximum = 0

    async def execute(task, operation):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1
        return ()

    runtime = _Runtime(capabilities, max_concurrency=2, execute=execute)
    first = _task(0, capabilities[0])
    second_capability = capabilities[-1]
    second = _task(1, second_capability, dependencies=dependencies)
    await DbLoopTaskBatchExecutor(runtime).execute((first, second), OPERATION)
    assert maximum == 1
    assert runtime.calls == ["task-0", "task-1"]


async def test_not_ready_batch_remains_serial():
    capability = _capability()
    runtime = _Runtime((capability,), max_concurrency=2)
    runtime.not_ready.add("task-1")
    await DbLoopTaskBatchExecutor(runtime).execute(
        (_task(0, capability), _task(1, capability)), OPERATION
    )
    assert runtime.calls == ["task-0", "task-1"]


async def test_earlier_not_ready_task_remains_a_serial_plan_order_barrier():
    capability = _capability()
    runtime = _Runtime((capability,), max_concurrency=2)
    runtime.not_ready.add("task-0")

    await DbLoopTaskBatchExecutor(runtime).execute(
        (_task(0, capability), _task(1, capability)),
        OPERATION,
    )

    assert runtime.calls == ["task-0", "task-1"]


@pytest.mark.parametrize(
    "governance",
    [
        GovernanceResult(False, True, False),
        GovernanceResult(False, False, True),
    ],
    ids=["denied", "approval-required"],
)
async def test_governance_block_is_not_launched_concurrently(governance):
    capability = _capability()

    async def execute(task, operation):
        raise DbRuntimeGovernanceBlocked(
            operation=operation, task=task, governance=governance
        )

    runtime = _Runtime((capability,), max_concurrency=2, execute=execute)
    runtime.governance = governance
    outcomes = await DbLoopTaskBatchExecutor(runtime).execute(
        (_task(0, capability), _task(1, capability)), OPERATION
    )
    assert runtime.calls == ["task-0"]
    assert outcomes[0].governance_error is not None


async def test_child_failure_isolated_and_results_restore_plan_order():
    capability = _capability()
    release_first = asyncio.Event()

    async def execute(task, operation):
        if task.id == "task-0":
            await release_first.wait()
            return (Evidence(kind="query.result", payload={"index": 0}),)
        if task.id == "task-1":
            release_first.set()
            raise ValueError("boom")
        return (Evidence(kind="query.result", payload={"index": 2}),)

    runtime = _Runtime((capability,), max_concurrency=3, execute=execute)
    outcomes = await DbLoopTaskBatchExecutor(runtime).execute(
        tuple(_task(index, capability) for index in range(3)), OPERATION
    )
    assert [item.task_index for item in outcomes] == [0, 1, 2]
    assert outcomes[1].error is not None
    assert outcomes[0].evidence[0].payload["index"] == 0
    assert outcomes[2].evidence[0].payload["index"] == 2


async def test_evidence_outcomes_are_deterministic_across_repeated_runs():
    capability = _capability()
    observed = []
    for _ in range(5):
        release_first = asyncio.Event()

        async def execute(task, operation):
            if task.id == "task-0":
                await release_first.wait()
            else:
                release_first.set()
            return (
                Evidence(kind="query.result", payload={"index": task.input["index"]}),
            )

        runtime = _Runtime((capability,), max_concurrency=2, execute=execute)
        outcomes = await DbLoopTaskBatchExecutor(runtime).execute(
            (_task(0, capability), _task(1, capability)), OPERATION
        )
        observed.append([item.evidence[0].payload["index"] for item in outcomes])
    assert observed == [[0, 1]] * 5


async def test_real_runtime_orders_completion_evidence_and_resume_skips_tasks():
    runtime, plugin, operation, tasks = await _real_concurrent_runtime()
    running = asyncio.create_task(
        DbLoopTaskBatchExecutor(runtime).execute(tasks, operation)
    )
    try:
        await plugin.executor.second_returned.wait()
        await _wait_for_task_evidence(runtime, operation.id, "real-task-1")
        plugin.executor.release_first.set()
        outcomes = await running

        raw_evidence = await runtime.store.list_evidence(operation.id)
        assert [item.task_id for item in raw_evidence] == [
            "real-task-1",
            "real-task-0",
        ]
        assert [item.task.id for item in outcomes] == [
            "real-task-0",
            "real-task-1",
        ]

        loop = DbAgentLoop(runtime, _NeverPlanner())
        observation = await loop._observation_after_execution(
            operation.id,
            executed=tuple(
                evidence for outcome in outcomes for evidence in outcome.evidence
            ),
            execution_errors=(),
            turn=1,
        )
        await loop._persist_observation(operation, observation, turn=1)
        snapshot = await runtime.inspect_operation(operation.id)
        assert [
            item.task_id for item in snapshot.evidence if item.task_id is not None
        ] == [
            "real-task-0",
            "real-task-1",
        ]
        persisted_observation = next(
            item for item in snapshot.evidence if item.kind == "planner.observation"
        )
        assert [
            item["task_id"]
            for item in persisted_observation.payload["observation"][
                "accepted_evidence_summaries"
            ]
        ] == ["real-task-0", "real-task-1"]
        loop_result = await loop._result(
            operation,
            "finished",
            warnings=(),
            diagnostics={},
        )
        assert [item["task_id"] for item in loop_result.task_refs] == [
            "real-task-0",
            "real-task-1",
        ]
        assert [item["id"] for item in loop_result.evidence_refs[:2]] == [
            snapshot.evidence[0].id,
            snapshot.evidence[1].id,
        ]

        calls_before_resume = tuple(plugin.executor.calls)
        resumed = await runtime.resume_operation(operation.id)
        assert tuple(plugin.executor.calls) == calls_before_resume
        assert [
            item.task_id for item in resumed.evidence if item.task_id is not None
        ] == [
            "real-task-0",
            "real-task-1",
        ]
    finally:
        plugin.executor.release_first.set()
        if not running.done():
            await running
        await runtime.teardown()


@pytest.mark.parametrize(
    "effect,expects_approval",
    [
        (PolicyEffect.DENY, False),
        (PolicyEffect.REQUIRE_APPROVAL, True),
    ],
    ids=["denied", "approval-required"],
)
async def test_real_governance_blocks_batch_before_any_read(effect, expects_approval):
    runtime, plugin, operation, tasks = await _real_concurrent_runtime(
        policy_effect=effect
    )
    try:
        outcomes = await DbLoopTaskBatchExecutor(runtime).execute(tasks, operation)
        snapshot = await runtime.inspect_operation(operation.id)
    finally:
        await runtime.teardown()

    assert plugin.executor.calls == []
    assert len(outcomes) == 1
    assert outcomes[0].governance_error is not None
    assert [task.status for task in snapshot.tasks] == [
        TaskStatus.BLOCKED,
        TaskStatus.PENDING,
    ]
    assert bool(snapshot.approval_requests) is expects_approval


async def test_parent_cancellation_cancels_and_awaits_children():
    capability = _capability()
    both_started = asyncio.Event()
    blocker = asyncio.Event()
    active = 0
    finished = 0

    async def execute(task, operation):
        nonlocal active, finished
        active += 1
        if active == 2:
            both_started.set()
        try:
            await blocker.wait()
        finally:
            active -= 1
            finished += 1
        return ()

    runtime = _Runtime((capability,), max_concurrency=2, execute=execute)
    running = asyncio.create_task(
        DbLoopTaskBatchExecutor(runtime).execute(
            (_task(0, capability), _task(1, capability)), OPERATION
        )
    )
    await both_started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert active == 0
    assert finished == 2


async def test_terminal_resume_task_is_not_executed_twice_and_writes_stay_serial():
    read = _capability()
    write = _capability(
        "db.sql.execute_write", access=AccessMode.WRITE, side_effecting=True
    )
    runtime = _Runtime((read, write), max_concurrency=4)
    completed = _task(0, read, status=TaskStatus.SUCCEEDED)
    pending_write = _task(1, write)
    await DbLoopTaskBatchExecutor(runtime).execute(
        (completed, pending_write), OPERATION
    )
    assert runtime.calls == ["task-1"]
