"""Production-composition proof for the current durable graph entry point."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from daita import (
    Agent,
    AgentConfig,
    GraphInspection,
    GraphState,
    HostCapacityLimits,
    LoopLimits,
    SQLiteSource,
)
from daita._json import canonical_json
from daita.domains.data import (
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_EVIDENCE_KIND,
    DATA_QUERY_TOOL_NAME,
)
from daita.jobs.graph.models import TaskRole
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind
from daita.storage.home_migrations import CURRENT_HOME_REVISION
from tests.support.job_benchmarks import create_probe_home
from tests.support.workspace import workspace_for

pytestmark = [pytest.mark.integration, pytest.mark.acceptance]


async def _open_count_agent(
    tmp_path, model=None, tables=2, no_model=False, graph_parallelism=1
):
    import sqlite3

    model = None if no_model else model or MockModelProvider(())
    agent = await Agent.create(
        "counts",
        root=tmp_path,
        model=model,
        model_profile=None if model is None else model.model_profile,
        workspace=workspace_for(tmp_path),
        config=AgentConfig(
            host_capacity=HostCapacityLimits(graph_parallelism=graph_parallelism)
        ),
    )
    sources = []
    for index in range(2):
        path = tmp_path / f"source-{index}.sqlite"
        with sqlite3.connect(path) as connection:
            for table in range(tables):
                connection.execute(f'CREATE TABLE "table_{table}" (id INTEGER)')
                connection.executemany(
                    f'INSERT INTO "table_{table}" VALUES (?)',
                    ((i,) for i in range(index * 5)),
                )
        sources.append(await agent.attach(SQLiteSource(path)))
    return agent, tuple(sources)


@pytest.mark.parametrize(
    "authoring,fault",
    [
        ("reference", fault)
        for fault in [
            None,
            "extra",
            "mixed",
            "forged",
            "cross_run",
            "duplicate_source",
            "partial",
            "scope",
            "files_only",
        ]
    ]
    + [
        ("source", fault)
        for fault in [
            None,
            "extra",
            "mixed",
            "forged",
            "duplicate_source",
            "scope",
            "files_only",
        ]
    ],
)
async def test_count_comparison_model_admission_authenticates_required_coverage(
    tmp_path, monkeypatch, fault, authoring
):
    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.llm.models import ToolResultBlock

    sources: tuple[Any, ...] = ()
    reads = []
    original_read = SQLiteQueryBackend.execute_read

    async def read(self, **kwargs):
        reads.append(kwargs)
        return await original_read(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)

    class AdmissionModel(MockModelProvider):
        async def generate(self, request):
            self._requests.append(request)
            results = {
                b.call_id: b
                for m in request.messages
                for b in m.content
                if isinstance(b, ToolResultBlock)
            }
            if not results:
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "load",
                            "toolbox_load",
                            {"tool_names": ("start_count_comparison",)},
                        ),
                        *(
                            ToolCall(
                                f"selection-{i}",
                                "catalog_schema",
                                {
                                    "source_id": (
                                        sources[0]
                                        if fault == "duplicate_source"
                                        else source
                                    ).id,
                                    "detail": "identity",
                                    "limit": 1 if fault == "partial" else 50,
                                },
                            )
                            for i, source in enumerate(sources)
                            if authoring == "reference"
                        ),
                    ),
                )
            if "start" not in results:
                refs = (
                    tuple(
                        cast(Any, results[f"selection-{i}"].output)["data"][
                            "selection"
                        ]["count_selection_ref"]
                        for i in range(2)
                    )
                    if authoring == "reference" and fault not in {"files_only", "scope"}
                    else ("sel_" + "1" * 16, "sel_" + "2" * 16)
                )
                if fault == "forged":
                    refs = (refs[0], "sel_" + "a" * 16)
                if fault == "cross_run":
                    refs = old_refs
                arguments: dict[str, Any] = {"selection_refs": refs}
                if authoring == "source":
                    arguments = {
                        "source_ids": (
                            sources[0].id,
                            (
                                sources[0].id
                                if fault == "duplicate_source"
                                else (
                                    "source:unknown"
                                    if fault == "forged"
                                    else sources[1].id
                                )
                            ),
                        ),
                        "coverage": "complete_admitted_base_tables",
                    }
                if fault == "extra":
                    if authoring == "reference":
                        arguments["source_ids"] = tuple(s.id for s in sources)
                    else:
                        arguments["selection_refs"] = refs
                if fault == "mixed":
                    arguments["initial_task"] = {}
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall("start", "start_count_comparison", arguments),
                    ),
                )
            assert results["start"].is_error is (fault is not None)
            return ModelResponse(
                finish_reason=FinishReason.STOP, text="Admission reviewed."
            )

    model = AdmissionModel(())
    agent, sources = await _open_count_agent(tmp_path, model)
    old_refs = ()
    try:
        if fault == "cross_run":
            model._script = ()
            # Obtain a real prior run's references, without granting new-run authority.
            original_generate = model.generate

            async def prior(request):
                model._requests.append(request)
                if len(model._requests) == 1:
                    return ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=tuple(
                            ToolCall(
                                f"old-{i}",
                                "catalog_schema",
                                {"source_id": s.id, "detail": "identity", "limit": 50},
                            )
                            for i, s in enumerate(sources)
                        ),
                    )
                return ModelResponse(
                    finish_reason=FinishReason.STOP, text="Discovered."
                )

            monkeypatch.setattr(model, "generate", prior)
            previous = await agent.run("Discover source identities.")
            previous_transcript = await agent.transcript(previous.run_id)
            old_refs = tuple(
                cast(Any, r.output)["data"]["selection"]["count_selection_ref"]
                for c, r in previous_transcript.tool_pairs
                if c.name == "catalog_schema" and r is not None
            )
            monkeypatch.setattr(model, "generate", original_generate)
        result = await agent.run(
            "Explicitly start a durable count comparison.",
            source_scope_ids=(
                ()
                if fault == "files_only"
                else (
                    (sources[0].id,)
                    if fault == "scope"
                    else tuple(s.id for s in sources)
                )
            ),
            files_only=fault == "files_only",
        )
        assert result.kind is LoopExitKind.COMPLETED
        jobs = await agent.list_jobs()
        if fault is not None:
            assert jobs == () and reads == []
        else:
            assert len(jobs) == 1
            inspection = await _wait_terminal(agent, jobs[0].job_id)
            assert inspection.job.state is GraphState.SUCCEEDED
            assert len(reads) == 2 and len(model.requests) == 3
            assert cast(Any, inspection.count_progress)["winner_source_ids"] == (
                sources[1].id,
            )
    finally:
        await agent.close()


class _ProductionGraphProvider(MockModelProvider):
    """Route deterministic answers by the public request's authenticated shape."""

    def __init__(
        self,
        *,
        source_id: str,
        resources: tuple[tuple[str, str], ...],
    ) -> None:
        super().__init__((), provider_id="mock:production-graph", complete_pricing=True)
        self._source_id = source_id
        self._resource_ids = tuple(resource_id for _name, resource_id in resources)
        self._sql = "SELECT " + ", ".join(
            f"(SELECT COUNT(*) FROM {name}) AS count_{index}"
            for index, (name, _resource_id) in enumerate(resources)
        )
        self._captured: list[ModelRequest] = []
        self._capability_id = DATA_QUERY_CAPABILITY_ID
        self._tool_name = DATA_QUERY_TOOL_NAME
        self._result_kind = DATA_QUERY_EVIDENCE_KIND
        self._arguments = {
            "source_id": self._source_id,
            "resource_ids": self._resource_ids,
            "sql": self._sql,
            "parameters": (),
        }

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        return tuple(self._captured)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self._captured.append(request)
        tool_names = {item.name for item in request.tools}
        result_call_ids = {
            block.call_id
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
        if "task_complete" in tool_names:
            if self._tool_name not in tool_names:
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "graph-load-operation",
                            "toolbox_load",
                            {"tool_names": (self._tool_name,)},
                        ),
                    ),
                    usage=usage,
                )
            if "graph-data-query" not in result_call_ids:
                arguments = self._arguments
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id="graph-data-query",
                            name=self._tool_name,
                            arguments=arguments,
                        ),
                    ),
                    usage=usage,
                )
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="graph-task-complete",
                        name="task_complete",
                        arguments={
                            "result_kind": self._result_kind,
                            "summary": "The exact bounded query completed.",
                            "payload": {
                                "queried_resource_count": len(self._resource_ids)
                            },
                            "evidence_call_ids": ("graph-data-query",),
                            "artifact_ids": (),
                            "residual_risk": None,
                            "downstream_constraints": {},
                        },
                    ),
                ),
                usage=usage,
            )
        if "start_graph_job" in tool_names:
            if "public-start-graph" in result_call_ids:
                return ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="The durable graph was admitted.",
                    usage=usage,
                )
            initial_arguments = self._arguments
            references = {
                "source_ids": (self._source_id,),
                "resource_ids": self._resource_ids,
                "connector_binding_ids": (),
            }
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="public-start-graph",
                        name="start_graph_job",
                        arguments={
                            "objective": "Count the rows in the exact admitted table.",
                            "outcome_contract": {
                                "required_result_kind": self._result_kind
                            },
                            "deadline_seconds": 300,
                            "initial_task": {
                                "capability_id": self._capability_id,
                                "arguments": initial_arguments,
                                "expected_result_contract": {
                                    "result_kind": self._result_kind
                                },
                                "retained_references": references,
                            },
                        },
                    ),
                ),
                usage=usage,
            )
        assert "toolbox_load" in tool_names
        return ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="public-load-graph",
                    name="toolbox_load",
                    arguments={"tool_names": ("start_graph_job",)},
                ),
            ),
            usage=usage,
        )


async def _wait_terminal(agent: Agent, job_id: str) -> GraphInspection:
    # The expanded fixture exercises ten child reads and repeated contract
    # validation. Avoid saturating its state store with full inspection polls.
    deadline = asyncio.get_running_loop().time() + 30
    while asyncio.get_running_loop().time() < deadline:
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.job.state in {
            GraphState.SUCCEEDED,
            GraphState.FAILED,
            GraphState.CANCELLED,
            GraphState.NEEDS_ATTENTION,
        }:
            return inspection
        await asyncio.sleep(0.05)
    assert inspection is not None
    raise AssertionError(
        f"production graph did not reach a terminal state: {inspection.job.state}; "
        f"attempts={[a.state for a in inspection.attempts]}"
    )


@pytest.mark.parametrize(
    "fault",
    [None, "source", "batch", "binding", "views", "queued_revocation", "queued_drift"],
)
async def test_internal_count_scope_binds_exact_arguments_and_rechecks_queued_reads(
    tmp_path, monkeypatch, fault
):
    from dataclasses import replace
    from datetime import UTC, datetime
    from hashlib import sha256

    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.capabilities import AccessMode, CapabilityInputError
    from daita.capability_runtime import InternalCapabilityRequest
    from daita.domains.data.controller import DataCapabilityDomain
    from daita.jobs.graph.execution import (
        count_task_execution_scope,
        graph_task_binding,
        prepare_graph_attempt,
    )
    from daita.jobs.graph.guard import TaskAttemptStateGuard
    from daita.jobs.graph.models import BudgetAmount
    from daita.loop.models import (
        InstructionAuthority,
        RunInput,
        RunOrigin,
        RunStartEnvelope,
    )
    from daita.storage.sqlite_records import SourceReadMode, SourceReadScope

    reads = []
    original_read = SQLiteQueryBackend.execute_read

    async def observed(self, **kwargs):
        reads.append(kwargs)
        return await original_read(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", observed)
    provider = MockModelProvider(())
    agent, sources = await _open_count_agent(tmp_path, provider)
    await agent._embedded._job_supervisor.close()
    try:
        job = await agent.start_count_comparison(
            source_ids=tuple(s.id for s in sources)
        )
        inspection = await agent.inspect_job(job.job_id)
        assert inspection is not None
        task = next(t for t in inspection.tasks if t.role is TaskRole.INTERNAL)
        store = agent._embedded._store
        now = datetime.now(UTC)
        prep = prepare_graph_attempt(inspection, task, claimed_at=now)
        attempt = await store.claim_graph_task(
            agent.id,
            job.job_id,
            task.task_id,
            attempt_id="attempt:exact",
            claim_token="claim:exact",
            run_id="run:exact",
            executor_id=prep.executor_id,
            claimed_at=now,
            lease_seconds=30,
            absolute_deadline_at=prep.absolute_deadline_at,
            budget_reservations=(BudgetAmount("work_units", 1),),
        )
        assert attempt is not None
        attempt = await store.start_graph_attempt(
            agent.id,
            job.job_id,
            task.task_id,
            attempt.attempt_id,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            started_at=now,
        )
        assert attempt is not None
        binding = graph_task_binding(inspection, task, attempt)
        scope = count_task_execution_scope(inspection, task, attempt, binding)
        from daita.storage.sqlite_codecs.execution_scope import (
            decode_execution_scope,
            encode_execution_scope,
        )

        assert decode_execution_scope(encode_execution_scope(scope)) == scope

        for changed in (
            {"per_run_max_tokens": 1},
            {"eligible_model_routes": ("mock:any",)},
            {"allowed_access_modes": frozenset({AccessMode.WRITE})},
            {"graph_task_binding": replace(binding, task_role="worker")},
        ):
            with pytest.raises(ValueError):
                replace(scope, **changed)
        instruction = "Execute frozen count batch."
        run = RunInput(
            "run:exact",
            agent.id,
            instruction,
            now,
            start=RunStartEnvelope(
                origin=RunOrigin.JOB_TASK,
                instruction_authority=InstructionAuthority.CODE_OWNED,
                trusted_instruction_id="count:exact",
                trusted_instruction=instruction,
                instruction_digest="sha256:" + sha256(instruction.encode()).hexdigest(),
                payload_digest="sha256:" + sha256(b"{}").hexdigest(),
                execution_scope=scope,
            ),
        )
        with pytest.raises(ValueError, match="zero-model"):
            await agent._embedded._loop.prepare(run)
        if fault is None:
            from daita.jobs.graph.execution import task_context_bundle
            from daita.loop.driver import PreparedLoopRun
            from daita.loop.session import (
                RunCancellationToken,
                RunSession,
                RunSessionOptions,
            )
            from daita.loop.transcripts import RunSessionWriter

            machine = replace(run, conversation_id="count:isolated")
            options = RunSessionOptions(
                task_context=task_context_bundle(
                    inspection, task, attempt, binding, created_at=now
                ),
                task_attempt_guard=TaskAttemptStateGuard(
                    store, binding, attempt.claim_token, attempt.run_id
                ),
            )
            prepared_run = PreparedLoopRun(
                machine, (), None, None, None, LoopLimits(), options
            )
            session = RunSession(
                machine,
                RunSessionWriter(store, machine),
                asyncio.get_running_loop().time() + 30,
                RunCancellationToken(),
                options=options,
            )
            with pytest.raises(ValueError, match="zero-model"):
                await agent._embedded._loop.run(session, prepared=prepared_run)
        arguments = dict(cast(Mapping[str, Any], prep.contract["arguments"]))
        if fault == "source":
            arguments["source_id"] = next(
                s.id for s in sources if s.id != arguments["source_id"]
            )
        if fault == "batch":
            arguments["resource_ids"] = tuple(reversed(arguments["resource_ids"]))
        if fault == "views":
            arguments["include_views"] = True
        if fault == "binding":
            arguments["selection_binding"] = {
                **arguments["selection_binding"],
                "source_revision": "changed",
            }
        request = InternalCapabilityRequest(
            run=run,
            call_id="call:exact",
            capability_id="data.count_batch",
            contract_digest=str(prep.contract["contract_digest"]),
            arguments=arguments,
            sensitivity=scope.sensitivity_ceiling,
            task_attempt_guard=TaskAttemptStateGuard(
                store, binding, attempt.claim_token, attempt.run_id
            ),
        )
        if fault in {"queued_revocation", "queued_drift"}:
            coordinator = agent._embedded._admission_coordinator
            permit = await coordinator.source_resource_permit(
                f"source:{arguments['source_id']}"
            )
            bound = asyncio.Event()
            original = DataCapabilityDomain.prepare_call

            async def prepared(self, *args, **kwargs):
                value = await original(self, *args, **kwargs)
                bound.set()
                return value

            monkeypatch.setattr(DataCapabilityDomain, "prepare_call", prepared)
            async with permit:
                pending = asyncio.create_task(
                    agent._embedded._capability_runtime.execute_internal(request)
                )
                await asyncio.wait_for(bound.wait(), 3)
                if fault == "queued_revocation":
                    await store.replace_source_permission_scopes(
                        SourceReadScope(
                            agent_id=agent.id,
                            source_id=arguments["source_id"],
                            mode=SourceReadMode.NONE,
                            resource_ids=(),
                        ),
                        (),
                    )
                else:
                    source = next(s for s in sources if s.id == arguments["source_id"])
                    import sqlite3

                    with sqlite3.connect(source.configuration["path"]) as connection:
                        connection.execute(
                            "ALTER TABLE table_0 ADD COLUMN changed INTEGER"
                        )
                    await agent.refresh_source(source.id)
            with pytest.raises(CapabilityInputError):
                await pending
        elif fault:
            with pytest.raises(CapabilityInputError):
                await agent._embedded._capability_runtime.execute_internal(request)
        else:
            output = await agent._embedded._capability_runtime.execute_internal(request)
            assert (
                output.output.kind == "data.row_counts" and output.artifact_ref is None
            )
        assert len(reads) == (1 if fault is None else 0)
        assert provider.requests == ()
        with pytest.raises(KeyError):
            await agent.transcript(attempt.run_id)
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "operation", ["complete_graph_attempt", "finalize_graph_attempt"]
)
@pytest.mark.parametrize("committed", [False, True])
async def test_count_output_acknowledgment_loss_preserves_exact_results_and_budgets(
    tmp_path, monkeypatch, operation, committed
):
    from daita.adapters.sqlite_query import SQLiteQueryBackend

    provider = MockModelProvider(())
    agent, sources = await _open_count_agent(tmp_path, provider)
    store = agent._embedded._store
    original = getattr(store, operation)
    original_read = SQLiteQueryBackend.execute_read
    lost = False
    submitted = []
    reads = []

    async def read(self, **kwargs):
        reads.append(kwargs)
        return await original_read(self, **kwargs)

    async def loss(*args, **kwargs):
        nonlocal lost
        submitted.append(args[0])
        if not lost:
            lost = True
            if committed:
                await original(*args, **kwargs)
            raise ConnectionError("Injected lost storage acknowledgment")
        return await original(*args, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
    monkeypatch.setattr(store, operation, loss)
    try:
        job = await agent.start_count_comparison(
            source_ids=tuple(s.id for s in sources)
        )
        inspection = await _wait_terminal(agent, job.job_id)
        assert lost and provider.requests == ()
        root = next(l for l in inspection.budget_ledgers if l.task_id is None)
        assert root.reserved == 0
        assert all(
            not r.artifact_ids and not r.effect_receipt_ids for r in inspection.results
        )
        if operation == "complete_graph_attempt" and not committed:
            assert inspection.job.state is GraphState.FAILED
            assert len(reads) == 1 and root.settled == 1
            assert inspection.results == () and inspection.delivery_ids == ()
            assert await agent.read_job_result(job.job_id) is None
        else:
            assert inspection.job.state is GraphState.SUCCEEDED
            assert len(reads) == 2 and len(inspection.delivery_ids) == 1
            assert root.settled == (4 if not committed else 3)
            if committed:
                assert (
                    next(
                        r
                        for r in inspection.results
                        if r.result_id == submitted[0].result_id
                    )
                    == submitted[0]
                )
            assert (
                len(await store.list_graph_deliveries(agent.id, job_id=job.job_id)) == 1
            )
    finally:
        await agent.close()


@pytest.mark.parametrize("after_reduction", [False, True])
async def test_count_finalizer_restarts_from_accepted_outputs_without_source_reads(
    tmp_path, monkeypatch, after_reduction
):
    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.domains.data.capabilities import DataFinalizeCountComparisonExecutor

    agent, sources = await _open_count_agent(tmp_path, no_model=True)
    original = DataFinalizeCountComparisonExecutor.execute
    original_read = SQLiteQueryBackend.execute_read
    entered = asyncio.Event()
    reads = []

    async def read(self, **kwargs):
        reads.append(kwargs)
        return await original_read(self, **kwargs)

    async def interrupted(self, request):
        if after_reduction:
            await original(self, request)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
    monkeypatch.setattr(DataFinalizeCountComparisonExecutor, "execute", interrupted)
    try:
        assert agent._embedded._loop is None
        job = await agent.start_count_comparison(
            source_ids=tuple(s.id for s in sources)
        )
        await asyncio.wait_for(entered.wait(), 5)
        current = await agent.inspect_job(job.job_id)
        assert len(current.results) == 2 and len(reads) == 2
        assert cast(Any, current.count_progress)["status"] == "awaiting_finalization"
        assert cast(Any, current.count_progress)["winner_source_ids"] == ()
        await agent.close()
        monkeypatch.setattr(DataFinalizeCountComparisonExecutor, "execute", original)
        agent = await Agent.open(
            "counts", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        finished = await _wait_terminal(agent, job.job_id)
        assert finished.job.state is GraphState.SUCCEEDED
        assert len(reads) == 2 and len(finished.attempts) == 4
        assert len(finished.results) == 3 and len(finished.delivery_ids) == 1
        assert all(r in finished.results for r in current.results)
        assert (
            next(l for l in finished.budget_ledgers if l.task_id is None).settled == 4
        )
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "interruption",
    [
        "close",
        "cancel",
        "lease",
        "deadline",
        "after_response",
        "native_drift",
        "binding_drift",
    ],
)
@pytest.mark.parametrize("tables", [2, 5])
async def test_interrupted_count_reads_are_never_replayed_and_progress_is_honest(
    tmp_path, monkeypatch, interruption, tables
):
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta

    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.jobs.graph.models import AttemptState

    model = MockModelProvider(())
    agent, sources = await _open_count_agent(tmp_path, model, tables=tables)
    original = SQLiteQueryBackend.execute_read
    blocked = asyncio.Event()
    reads: list[dict[str, Any]] = []

    if interruption == "binding_drift":
        from daita.capabilities import CapabilityInputError
        from daita.domains.data.controller import DataCapabilityDomain

        validate = DataCapabilityDomain._validate_count_binding

        async def changed_binding(self, *args, **kwargs):
            if reads:
                raise CapabilityInputError(
                    "count_binding_stale",
                    "Frozen binding changed before dispatch.",
                    {"execution_disposition": "not_dispatched"},
                )
            return await validate(self, *args, **kwargs)

        monkeypatch.setattr(
            DataCapabilityDomain, "_validate_count_binding", changed_binding
        )

    async def read(self, **kwargs):
        reads.append(kwargs)
        # Retain one accepted observation, then interrupt an unaccepted read.
        if len(reads) == 2:
            if interruption == "native_drift":
                import sqlite3

                source = next(s for s in sources if s.id == kwargs["source_id"])
                with sqlite3.connect(source.configuration["path"]) as connection:
                    connection.execute("ALTER TABLE table_0 ADD COLUMN changed INTEGER")
                return await original(self, **kwargs)
            if interruption == "after_response":
                await original(self, **kwargs)
                raise ConnectionError("Response lost before durable acceptance")
            blocked.set()
            await asyncio.Event().wait()
        return await original(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
    try:
        if interruption == "deadline":
            from daita.jobs.graph.admission import GraphAdmissionBuilder

            build = GraphAdmissionBuilder.build_count_comparison

            def short(self, **kwargs):
                admission = build(self, **kwargs)
                tasks = tuple(
                    (
                        replace(
                            t,
                            specification=replace(
                                t.specification, max_wall_time_seconds=1
                            ),
                            task_spec_digest=replace(
                                t.specification, max_wall_time_seconds=1
                            ).digest,
                        )
                        if t.role is TaskRole.INTERNAL
                        else t
                    )
                    for t in admission.tasks
                )
                # Bind the changed fixture task digest before ordinary admission.
                from daita.jobs.graph.models import canonical_digest, topology_digest

                mapping = tuple(
                    {
                        **m,
                        "task_spec_digest": next(
                            t.task_spec_digest
                            for t in tasks
                            if t.task_id == m["task_id"]
                        ),
                    }
                    for m in cast(
                        tuple[Mapping[str, Any], ...],
                        admission.job.specification.outcome_contract["batch_tasks"],
                    )
                )
                spec = replace(
                    admission.job.specification,
                    outcome_contract={
                        **admission.job.specification.outcome_contract,
                        "batch_tasks": mapping,
                    },
                )
                return replace(
                    admission,
                    tasks=tasks,
                    job=replace(
                        admission.job,
                        specification=spec,
                        specification_digest=spec.digest,
                    ),
                    graph=replace(
                        admission.graph,
                        topology_digest=topology_digest(tasks, admission.dependencies),
                    ),
                )

            monkeypatch.setattr(GraphAdmissionBuilder, "build_count_comparison", short)
        job = await agent.start_count_comparison(
            source_ids=tuple(s.id for s in sources)
        )
        if interruption not in {"after_response", "native_drift", "binding_drift"}:
            await asyncio.wait_for(blocked.wait(), 5)
        if interruption == "cancel":
            await agent.cancel_job(job.job_id)
        elif interruption == "lease":
            current = await agent.inspect_job(job.job_id)
            active = next(
                a for a in current.attempts if a.state is AttemptState.RUNNING
            )
            await agent._embedded._store.fence_graph_attempt(
                agent.id,
                job.job_id,
                active.task_id,
                active.attempt_id,
                fencing_epoch=active.fencing_epoch,
                fenced_at=datetime.now(UTC),
                requeue=True,
                reason_code="lease_expired",
            )
        if interruption in {"close", "cancel", "lease"}:
            await agent.close()
            agent = await Agent.open(
                "counts",
                root=tmp_path,
                model=model,
                model_profile=model.model_profile,
                workspace=workspace_for(tmp_path),
            )
        inspection = await _wait_terminal(agent, job.job_id)
        assert inspection.job.state in {
            GraphState.FAILED,
            GraphState.CANCELLED,
            GraphState.NEEDS_ATTENTION,
        }
        assert len(reads) == (1 if interruption == "binding_drift" else 2)
        assert model.requests == ()
        assert len(inspection.results) == 1
        assert (
            await agent._embedded._store.list_graph_deliveries(
                agent.id, job_id=job.job_id
            )
            == ()
        )
        if interruption in {"native_drift", "binding_drift"}:
            assert inspection.job.state is GraphState.NEEDS_ATTENTION
            assert inspection.controls[-1].kind.value == "source_or_contract_drift"
        progress = cast(Any, inspection.count_progress)
        if interruption == "binding_drift":
            assert (
                progress["controls"][-1]["details"]["details"]["execution_disposition"]
                == "not_dispatched"
            )
            assert progress["attempts"][-1]["execution_disposition"] == "not_dispatched"
        observed = next(s for s in progress["sources"] if s["counted_resources"])
        missing = next(s for s in progress["sources"] if not s["counted_resources"])
        assert observed["observed_total_rows"] == observed["counted_resources"] * (
            5 if observed["source_id"] == sources[1].id else 0
        )
        if tables == 2:
            assert (
                observed["counted_resources"] == 2
                and observed["total_rows"] is not None
            )
        else:
            assert (
                0 < observed["counted_resources"] < tables
                and observed["total_rows"] is None
            )
        assert missing["counted_resources"] == 0 and missing["observed_total_rows"] == 0
        assert (
            len(missing["missing_resource_ids"]) == tables
            and progress["winner_source_ids"] == ()
        )
        assert await agent.read_job_result(job.job_id) is None
        assert (
            sum(l.settled for l in inspection.budget_ledgers if l.task_id is None) == 2
        )
    finally:
        await agent.close()


async def test_public_agent_executes_graph_model_task_in_current_home(
    tmp_path: Path,
) -> None:
    home = await create_probe_home(
        tmp_path,
        "public-production-graph",
        distractor_tables=3,
    )
    resources = tuple(sorted(home.resource_ids.items()))
    provider = _ProductionGraphProvider(
        source_id=home.source_id,
        resources=resources,
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        workspace=workspace_for(home.root),
    )
    try:
        status = await Agent.inspect_home(home.name, root=home.root)
        assert status.found_revision == status.current_revision == CURRENT_HOME_REVISION
        foreground = await agent.run(
            "Start a durable graph query over all five exact resources.",
            source_scope_ids=(home.source_id,),
        )
        assert foreground.kind is LoopExitKind.COMPLETED
        jobs = await agent.list_jobs()
        transcript = await agent.transcript(foreground.run_id)
        outputs = [
            dict(block.output)
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert len(jobs) == 1, canonical_json(outputs)
        inspection = await _wait_terminal(agent, jobs[0].job_id)
        result = await agent.read_job_result(jobs[0].job_id)
        diagnostic: list[dict[str, object]] = []
        for attempt in inspection.attempts:
            try:
                attempt_transcript = await agent.transcript(attempt.run_id)
            except KeyError:
                continue
            diagnostic.extend(
                dict(block.output)
                for message in attempt_transcript.messages
                for block in message.content
                if isinstance(block, ToolResultBlock)
            )
        assert inspection.job.state is GraphState.SUCCEEDED, canonical_json(
            {
                "diagnostic": diagnostic,
                "tasks": [
                    task.specification.digest_material() for task in inspection.tasks
                ],
            }
        )
        assert inspection.job.origin_run_id == foreground.run_id
        assert len(inspection.delivery_ids) == 1
        assert result is not None
        assert result.result_kind == "graph.result_finalized"
        assert result.payload["accepted_result_count"] == 1
        assert any(
            item.result_kind == provider._result_kind for item in inspection.results
        )
        assert any(
            "task_complete" in {item.name for item in request.tools}
            for request in provider.requests
        )
    finally:
        await agent.close()


async def test_configured_graph_capacity_overlaps_four_model_attempts_and_preserves_root_limits(
    tmp_path,
):
    home = await create_probe_home(
        tmp_path, "parallel-public-graphs", distractor_tables=3
    )
    resources = tuple(sorted(home.resource_ids.items()))
    ready = asyncio.Event()
    release = asyncio.Event()

    class ParallelProvider(_ProductionGraphProvider):
        active = 0
        peak = 0

        async def generate(self, request):
            if "task_complete" in {tool.name for tool in request.tools}:
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active == 4:
                    ready.set()
                try:
                    await release.wait()
                    return await super().generate(request)
                finally:
                    self.active -= 1
            return await super().generate(request)

    provider = ParallelProvider(source_id=home.source_id, resources=resources)
    config = AgentConfig(
        host_capacity=HostCapacityLimits(
            execution_capacity=8, provider_capacity=5, graph_parallelism=4
        )
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        config=config,
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        workspace=workspace_for(home.root),
    )
    try:
        for _ in range(4):
            result = await agent.run(
                "Start a durable query graph.", source_scope_ids=(home.source_id,)
            )
            assert result.kind is LoopExitKind.COMPLETED
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert provider.active == 4
        jobs = await agent.list_jobs()
        assert len(jobs) == 4
        for job in jobs:
            inspection = await agent.inspect_job(job.job_id)
            assert inspection is not None
            active = [
                attempt
                for attempt in inspection.attempts
                if attempt.state.value in {"claimed", "running"}
            ]
            assert len(active) == 1
            assert len(active) <= inspection.job.specification.limits.max_parallelism
        release.set()
        terminals = await asyncio.gather(
            *(_wait_terminal(agent, job.job_id) for job in jobs)
        )
        assert all(
            inspection.job.state is GraphState.SUCCEEDED for inspection in terminals
        )
        assert provider.peak == 4
    finally:
        release.set()
        await agent.close()
    assert (
        agent.admission_diagnostics().active_leases
        == agent.admission_diagnostics().active_permits
        == 0
    )


@pytest.mark.parametrize("started", [False, True])
async def test_startup_fences_unaccepted_count_attempts_without_source_replay(
    tmp_path, monkeypatch, started
):
    from datetime import UTC, datetime

    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.jobs.graph.execution import prepare_graph_attempt
    from daita.jobs.graph.models import AttemptState

    agent, sources = await _open_count_agent(tmp_path, no_model=True)
    await agent._embedded._job_supervisor.close()

    async def forbidden_read(self, **kwargs):
        raise AssertionError(
            "startup must never replay an active unaccepted count read"
        )

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", forbidden_read)
    try:
        job = await agent.start_count_comparison(
            source_ids=tuple(s.id for s in sources)
        )
        current = await agent.inspect_job(job.job_id)
        task = next(t for t in current.tasks if t.role is TaskRole.INTERNAL)
        now = datetime.now(UTC)
        prep = prepare_graph_attempt(current, task, claimed_at=now)
        store = agent._embedded._store
        attempt = await store.claim_graph_task(
            agent.id,
            job.job_id,
            task.task_id,
            attempt_id="attempt:crash",
            claim_token="claim:crash",
            run_id="run:crash",
            executor_id=prep.executor_id,
            claimed_at=now,
            lease_seconds=30,
            absolute_deadline_at=prep.absolute_deadline_at,
            budget_reservations=prep.budget_reservations,
        )
        assert attempt is not None
        if started:
            await store.start_graph_attempt(
                agent.id,
                job.job_id,
                task.task_id,
                attempt.attempt_id,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                started_at=now,
            )
        # Leave durable authority active without a registered worker, as process
        # loss would, then exercise the ordinary host's actual startup recovery.
        await agent.close()
        agent = await Agent.open(
            "counts", root=tmp_path, workspace=workspace_for(tmp_path)
        )
        recovered = await agent.inspect_job(job.job_id)
        assert recovered is not None
        assert recovered.job.state is GraphState.FAILED
        assert (
            len(recovered.attempts) == 1
            and recovered.attempts[0].state is AttemptState.FENCED
        )
        assert (
            recovered.results == () and await agent.read_job_result(job.job_id) is None
        )
        assert (
            next(l for l in recovered.budget_ledgers if l.task_id is None).settled == 1
        )
        assert (
            next(l for l in recovered.budget_ledgers if l.task_id is None).reserved == 0
        )
        assert all(
            s["total_rows"] is None
            for s in cast(Any, recovered.count_progress)["sources"]
        )
    finally:
        await agent.close()


@pytest.mark.parametrize("failure", [False, True])
async def test_parallel_count_batches_hold_only_source_permits_and_settle_after_failure(
    tmp_path, monkeypatch, failure
):
    import contextvars
    from collections import Counter

    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.jobs.graph.models import ACTIVE_ATTEMPT_STATES

    agent, sources = await _open_count_agent(
        tmp_path, tables=5, no_model=True, graph_parallelism=4
    )
    store_call = contextvars.ContextVar("count_graph_store_call", default=False)
    supervisor = agent._embedded._job_supervisor
    original_store = supervisor._graph_store_call
    original_read = SQLiteQueryBackend.execute_read
    both = asyncio.Event()
    release = asyncio.Event()
    active: Counter[str] = Counter()
    maximum: Counter[str] = Counter()
    calls = []
    global_maximum = 0

    async def observed_store(operation):
        token = store_call.set(True)
        try:
            return await original_store(operation)
        finally:
            store_call.reset(token)

    async def read(self, **kwargs):
        nonlocal global_maximum
        assert not store_call.get()
        source = kwargs["source_id"]
        calls.append((source, tuple(kwargs["expected_resource_revisions"])))
        active[source] += 1
        maximum[source] = max(maximum[source], active[source])
        global_maximum = max(global_maximum, sum(active.values()))
        if sum(active.values()) == 2:
            both.set()
        try:
            await release.wait()
            if failure and source == sources[0].id:
                raise ConnectionError("Injected bounded read failure")
            return await original_read(self, **kwargs)
        finally:
            active[source] -= 1

    monkeypatch.setattr(supervisor, "_graph_store_call", observed_store)
    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
    try:
        job = await agent.start_count_comparison(
            source_ids=tuple(s.id for s in sources)
        )
        await asyncio.wait_for(both.wait(), 5)
        async with asyncio.timeout(5):
            while len((await agent.inspect_job(job.job_id)).attempts) < 4:
                await asyncio.sleep(0.01)
        release.set()
        terminal = await _wait_terminal(agent, job.job_id)
        async with asyncio.timeout(5):
            while supervisor._graph_workers:
                await asyncio.sleep(0.01)
        terminal = await agent.inspect_job(job.job_id)
        assert supervisor.status.state == "running"
        assert all(a.state not in ACTIVE_ATTEMPT_STATES for a in terminal.attempts)
        assert (
            next(l for l in terminal.budget_ledgers if l.task_id is None).reserved == 0
        )
        assert global_maximum == 2 and set(maximum.values()) == {1}
        if failure:
            assert terminal.job.state is GraphState.FAILED
            assert 2 <= len(calls) <= 4 and len(set(calls)) == len(calls)
            assert len(terminal.attempts) == 4
            assert (
                next(l for l in terminal.budget_ledgers if l.task_id is None).settled
                == 4
            )
            assert await agent.read_job_result(job.job_id) is None
        else:
            assert terminal.job.state is GraphState.SUCCEEDED
            assert len(calls) == 4 and len(terminal.attempts) == 5
            assert len(terminal.delivery_ids) == 1
    finally:
        release.set()
        await agent.close()
