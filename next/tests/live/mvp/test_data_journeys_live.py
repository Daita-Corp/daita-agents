"""LIVE-MVP-01 through LIVE-MVP-03 against one explicit real OpenAI model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import os
from pathlib import Path
import time

import pytest

from daita import Agent, LocalDirectorySource, SQLiteSource
from daita.catalog import ResourceKind, catalog_resource_id
from daita.operations.models import Evidence, Task

from .assertions import (
    accepted_evidence_for_tasks,
    assert_allowed_capabilities,
    assert_catalog_graph_use,
    assert_catalog_discovery_and_inspection,
    assert_cited_evidence_supports,
    assert_completed,
    assert_count_and_money,
    assert_current_authority,
    assert_discrepancies_explained,
    assert_inspectable_runtime_state,
    assert_labeled_money,
    assert_read_only_sql,
    assert_resolving_citations,
    assert_route_binding,
    query_tasks,
    semantic_answer_text,
)
from .fixture_oracles import (
    CROSS_SOURCE_KEY_NORMALIZATION,
    CommerceFixture,
    CommerceOracles,
    normalize_comparison_discrepancies,
)
from .harness import (
    LiveMvpConfiguration,
    LiveRowRecorder,
    RecordingOpenAIProvider,
    WAVE1_BUDGETS,
    model_profile,
)
from .prompt_corpus import (
    CATALOG_GRAPH_PROMPTS,
    CROSS_SOURCE_PROMPTS,
    GROUNDING_PROMPTS,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.acceptance,
    pytest.mark.requires_llm,
    pytest.mark.live_mvp,
]

_SQLITE_ALLOWED = {
    "catalog.search",
    "catalog.inspect",
    "catalog.traverse",
    "data.sqlite.query",
}
_CROSS_SOURCE_ALLOWED = {
    "catalog.search",
    "catalog.inspect",
    "catalog.traverse",
    "data.file.read",
    "data.sqlite.query",
    "data.tabular.compare",
}


def _table_ids(source_id: str) -> dict[str, str]:
    names = (
        "customers",
        "customers_archive_2025",
        "regions",
        "orders",
        "orders_archive_2025",
        "order_items",
        "products",
        "payments",
        "refunds",
    )
    return {
        name: catalog_resource_id(source_id, ResourceKind.TABLE, f"main.{name}")
        for name in names
    }


async def _create_sqlite_agent(
    *,
    name: str,
    root: Path,
    fixture: CommerceFixture,
    configuration: LiveMvpConfiguration,
    provider: RecordingOpenAIProvider,
) -> tuple[Agent, str, dict[str, str]]:
    agent = await Agent.create(
        name,
        root=root,
        model=provider,
        model_profile=model_profile(configuration),
        budgets=WAVE1_BUDGETS,
    )
    registration = await agent.attach(
        SQLiteSource(fixture.database_path, name="Current commerce warehouse")
    )
    return agent, registration.id, _table_ids(registration.id)


@pytest.mark.parametrize(
    "prompt",
    (
        pytest.param(
            GROUNDING_PROMPTS[0],
            id="direct",
            marks=pytest.mark.live_mvp_smoke,
        ),
        pytest.param(
            GROUNDING_PROMPTS[1],
            id="conversational",
            marks=pytest.mark.live_mvp_reliability,
        ),
        pytest.param(
            GROUNDING_PROMPTS[2],
            id="answerable-ambiguous",
            marks=pytest.mark.live_mvp_reliability,
        ),
    ),
)
async def test_live_mvp_01_grounded_multi_table_analyst_query(
    prompt: str,
    tmp_path: Path,
    commerce_fixture: CommerceFixture,
    commerce_oracles: CommerceOracles,
    live_mvp_configuration: LiveMvpConfiguration,
    live_mvp_provider: RecordingOpenAIProvider,
    live_row_recorder: LiveRowRecorder,
) -> None:
    state_root = tmp_path / "state"
    live_row_recorder.register_home(
        state_root,
        (os.environ[live_mvp_configuration.credential_environment],),
    )
    agent, source_id, resources = await _create_sqlite_agent(
        name="live-mvp-01",
        root=state_root,
        fixture=commerce_fixture,
        configuration=live_mvp_configuration,
        provider=live_mvp_provider,
    )
    route = agent.model_route
    assert route is not None
    started = time.monotonic()
    try:
        result = await agent.run(prompt, session_id="live-mvp-01")
        wall = time.monotonic() - started
        snapshot = await agent.inspect(result.operation_id)
        live_row_recorder.capture(result, snapshot, wall_time_seconds=wall)

        text = result.final_text or ""
        with live_row_recorder.hard_check("outcome", "operation_completed"):
            text = assert_completed(result, snapshot)
        with live_row_recorder.hard_check("outcome", "aggregate_answer_exact"):
            assert_count_and_money(
                text,
                commerce_oracles.aggregate.customer_count,
                commerce_oracles.aggregate.net_revenue_cents,
            )

        current = {
            resources["customers"],
            resources["orders"],
            resources["payments"],
            resources["refunds"],
        }
        archives = {
            resources["customers_archive_2025"],
            resources["orders_archive_2025"],
        }
        reads = query_tasks(snapshot)
        with live_row_recorder.hard_check("safety", "safe_capabilities_read_only"):
            assert_allowed_capabilities(snapshot, _SQLITE_ALLOWED)
            assert reads
            for task in reads:
                assert_read_only_sql(task)

        query_evidence = accepted_evidence_for_tasks(snapshot, reads)
        with live_row_recorder.hard_check("evidence", "accepted_query_evidence"):
            assert query_evidence
        if not text or not query_evidence:
            live_row_recorder.record_not_evaluated(
                layer="evidence",
                code="cited_answer_support_unavailable",
            )
        else:
            with live_row_recorder.hard_check(
                "evidence",
                "cited_answer_support",
            ):
                cited_support = assert_cited_evidence_supports(
                    text,
                    snapshot,
                    query_evidence,
                    exact_values=(commerce_oracles.aggregate.customer_count,),
                    money_cents=(commerce_oracles.aggregate.net_revenue_cents,),
                )
                supporting_tasks = _tasks_for_evidence(reads, cited_support)
                for task in supporting_tasks:
                    assert_current_authority(
                        task,
                        source_id=source_id,
                        required_resource_ids=(),
                        forbidden_resource_ids=archives,
                    )
                assert current <= {
                    resource_id
                    for task in supporting_tasks
                    for resource_id in task.execution_facts.validation_facts.resource_ids
                }

        with live_row_recorder.diagnostic("route_binding"):
            assert_route_binding(
                snapshot,
                revision=route.revision,
                fingerprint=route.fingerprint,
            )
        with live_row_recorder.diagnostic("runtime_event_topology"):
            assert_inspectable_runtime_state(snapshot)
        with live_row_recorder.diagnostic("prescribed_catalog_inspection"):
            assert_catalog_discovery_and_inspection(snapshot, current)
        with live_row_recorder.diagnostic("exploratory_noncurrent_read"):
            for task in reads:
                sql = assert_read_only_sql(task)
                assert "archive" not in sql.casefold()
                assert_current_authority(
                    task,
                    source_id=source_id,
                    required_resource_ids=(),
                    forbidden_resource_ids=archives,
                )
        with live_row_recorder.diagnostic("exact_validation_resource_coverage"):
            assert current <= {
                resource_id
                for task in reads
                for resource_id in task.execution_facts.validation_facts.resource_ids
            }
        live_row_recorder.assert_mvp_passed()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "prompt",
    (
        pytest.param(
            CATALOG_GRAPH_PROMPTS[0],
            id="direct",
            marks=pytest.mark.live_mvp_smoke,
        ),
        pytest.param(
            CATALOG_GRAPH_PROMPTS[1],
            id="conversational",
            marks=pytest.mark.live_mvp_reliability,
        ),
        pytest.param(
            CATALOG_GRAPH_PROMPTS[2],
            id="answerable-ambiguous",
            marks=pytest.mark.live_mvp_reliability,
        ),
    ),
)
async def test_live_mvp_02_ambiguous_catalog_and_graph_resolution(
    prompt: str,
    tmp_path: Path,
    commerce_fixture: CommerceFixture,
    commerce_oracles: CommerceOracles,
    live_mvp_configuration: LiveMvpConfiguration,
    live_mvp_provider: RecordingOpenAIProvider,
    live_row_recorder: LiveRowRecorder,
) -> None:
    state_root = tmp_path / "state"
    live_row_recorder.register_home(
        state_root,
        (os.environ[live_mvp_configuration.credential_environment],),
    )
    agent, source_id, resources = await _create_sqlite_agent(
        name="live-mvp-02",
        root=state_root,
        fixture=commerce_fixture,
        configuration=live_mvp_configuration,
        provider=live_mvp_provider,
    )
    route = agent.model_route
    assert route is not None
    started = time.monotonic()
    try:
        result = await agent.run(prompt, session_id="live-mvp-02")
        wall = time.monotonic() - started
        snapshot = await agent.inspect(result.operation_id)
        live_row_recorder.capture(result, snapshot, wall_time_seconds=wall)

        text = result.final_text or ""
        with live_row_recorder.hard_check("outcome", "operation_completed"):
            text = assert_completed(result, snapshot)
        with live_row_recorder.hard_check("outcome", "leading_region_answer_exact"):
            assert (
                commerce_oracles.leading_region.region_name.casefold()
                in semantic_answer_text(text).casefold()
            )
            assert_labeled_money(
                text,
                commerce_oracles.leading_region.region_name,
                commerce_oracles.leading_region.net_revenue_cents,
            )

        current = {
            resources["regions"],
            resources["customers"],
            resources["orders"],
            resources["payments"],
            resources["refunds"],
        }
        archives = {
            resources["customers_archive_2025"],
            resources["orders_archive_2025"],
        }
        reads = query_tasks(snapshot)
        with live_row_recorder.hard_check("safety", "safe_capabilities_read_only"):
            assert_allowed_capabilities(snapshot, _SQLITE_ALLOWED)
            assert reads
            for task in reads:
                assert_read_only_sql(task)

        query_evidence = accepted_evidence_for_tasks(snapshot, reads)
        with live_row_recorder.hard_check("evidence", "accepted_query_evidence"):
            assert query_evidence
        if not text or not query_evidence:
            live_row_recorder.record_not_evaluated(
                layer="evidence",
                code="cited_answer_support_unavailable",
            )
        else:
            with live_row_recorder.hard_check(
                "evidence",
                "cited_answer_support",
            ):
                cited_support = assert_cited_evidence_supports(
                    text,
                    snapshot,
                    query_evidence,
                    exact_values=(commerce_oracles.leading_region.region_name,),
                    money_cents=(commerce_oracles.leading_region.net_revenue_cents,),
                )
                supporting_tasks = _tasks_for_evidence(reads, cited_support)
                for task in supporting_tasks:
                    assert_current_authority(
                        task,
                        source_id=source_id,
                        required_resource_ids=(),
                        forbidden_resource_ids=archives,
                    )
                assert current <= {
                    resource_id
                    for task in supporting_tasks
                    for resource_id in task.execution_facts.validation_facts.resource_ids
                }

        with live_row_recorder.diagnostic("route_binding"):
            assert_route_binding(
                snapshot,
                revision=route.revision,
                fingerprint=route.fingerprint,
            )
        with live_row_recorder.diagnostic("runtime_event_topology"):
            assert_inspectable_runtime_state(snapshot)
        with live_row_recorder.diagnostic("prescribed_catalog_inspection"):
            assert_catalog_discovery_and_inspection(snapshot, current)
        with live_row_recorder.diagnostic("exploratory_noncurrent_read"):
            for task in reads:
                sql = assert_read_only_sql(task)
                assert "archive" not in sql.casefold()
                assert_current_authority(
                    task,
                    source_id=source_id,
                    required_resource_ids=(),
                    forbidden_resource_ids=archives,
                )
        with live_row_recorder.diagnostic("exact_validation_resource_coverage"):
            assert current <= {
                resource_id
                for task in reads
                for resource_id in task.execution_facts.validation_facts.resource_ids
            }
        with live_row_recorder.diagnostic("prescribed_catalog_graph_traversal"):
            assert_catalog_graph_use(
                snapshot,
                from_resource_id=resources["refunds"],
                to_resource_id=resources["regions"],
                forbidden_resource_ids=archives,
            )
        live_row_recorder.assert_mvp_passed()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "prompt",
    (
        pytest.param(
            CROSS_SOURCE_PROMPTS[0],
            id="direct",
            marks=pytest.mark.live_mvp_smoke,
        ),
        pytest.param(
            CROSS_SOURCE_PROMPTS[1],
            id="conversational",
            marks=pytest.mark.live_mvp_reliability,
        ),
        pytest.param(
            CROSS_SOURCE_PROMPTS[2],
            id="answerable-ambiguous",
            marks=pytest.mark.live_mvp_reliability,
        ),
    ),
)
async def test_live_mvp_03_newest_cross_source_comparison(
    prompt: str,
    tmp_path: Path,
    commerce_fixture: CommerceFixture,
    commerce_oracles: CommerceOracles,
    live_mvp_configuration: LiveMvpConfiguration,
    live_mvp_provider: RecordingOpenAIProvider,
    live_row_recorder: LiveRowRecorder,
) -> None:
    state_root = tmp_path / "state"
    live_row_recorder.register_home(
        state_root,
        (os.environ[live_mvp_configuration.credential_environment],),
    )
    agent = await Agent.create(
        "live-mvp-03",
        root=state_root,
        model=live_mvp_provider,
        model_profile=model_profile(live_mvp_configuration),
        budgets=WAVE1_BUDGETS,
    )
    file_source = await agent.attach(
        LocalDirectorySource(commerce_fixture.files_root, name="Customer exports")
    )
    database_source = await agent.attach(
        SQLiteSource(
            commerce_fixture.database_path,
            name="Current commerce warehouse",
        )
    )
    newest_id = catalog_resource_id(
        file_source.id,
        ResourceKind.FILE,
        commerce_oracles.newest_customer_export.name,
    )
    older_id = catalog_resource_id(
        file_source.id,
        ResourceKind.FILE,
        "customers-2026-02-01.csv",
    )
    customer_table_id = catalog_resource_id(
        database_source.id,
        ResourceKind.TABLE,
        "main.customers",
    )
    route = agent.model_route
    assert route is not None
    started = time.monotonic()
    try:
        result = await agent.run(prompt, session_id="live-mvp-03")
        wall = time.monotonic() - started
        snapshot = await agent.inspect(result.operation_id)
        live_row_recorder.capture(result, snapshot, wall_time_seconds=wall)

        text = result.final_text or ""
        with live_row_recorder.hard_check("outcome", "operation_completed"):
            text = assert_completed(result, snapshot)

        file_tasks = tuple(
            task for task in snapshot.tasks if task.capability_id == "data.file.read"
        )
        reads = query_tasks(snapshot)
        compare_tasks = tuple(
            task
            for task in snapshot.tasks
            if task.capability_id == "data.tabular.compare"
        )
        file_evidence = accepted_evidence_for_tasks(snapshot, file_tasks)
        database_evidence = accepted_evidence_for_tasks(snapshot, reads)
        comparison_evidence = accepted_evidence_for_tasks(snapshot, compare_tasks)
        selected_file: Evidence | None = None
        selected_database: Evidence | None = None
        comparison: Evidence | None = None
        with live_row_recorder.hard_check("outcome", "comparison_exact"):
            selected_file, selected_database, comparison = _resolve_exact_comparison(
                compare_tasks=compare_tasks,
                file_evidence=file_evidence,
                database_evidence=database_evidence,
                comparison_evidence=comparison_evidence,
                oracles=commerce_oracles,
            )
        with live_row_recorder.hard_check(
            "outcome",
            "all_discrepancies_explained",
        ):
            assert_discrepancies_explained(text, commerce_oracles.discrepancies)

        with live_row_recorder.hard_check("safety", "safe_capabilities_read_only"):
            assert_allowed_capabilities(snapshot, _CROSS_SOURCE_ALLOWED)
            assert reads
            for task in reads:
                assert_read_only_sql(task)
        if selected_file is None or selected_database is None:
            live_row_recorder.record_not_evaluated(
                layer="safety",
                code="selected_sources_unavailable",
            )
        else:
            with live_row_recorder.hard_check(
                "safety",
                "current_authorized_sources",
            ):
                selected_file_task = _task_for_evidence(file_tasks, selected_file)
                assert selected_file_task.arguments["resource_id"] == newest_id
                assert selected_file_task.arguments["resource_id"] != older_id
                selected_database_task = _task_for_evidence(reads, selected_database)
                assert_current_authority(
                    selected_database_task,
                    source_id=database_source.id,
                    required_resource_ids=(customer_table_id,),
                )

        if selected_file is None or selected_database is None or comparison is None:
            live_row_recorder.record_not_evaluated(
                layer="evidence",
                code="comparison_support_unavailable",
            )
            live_row_recorder.record_not_evaluated(
                layer="evidence",
                code="resolving_citations_unavailable",
            )
        else:
            with live_row_recorder.hard_check(
                "evidence",
                "authoritative_freshness",
            ):
                selected_file_task = _task_for_evidence(file_tasks, selected_file)
                file_facts = selected_file_task.execution_facts.validation_facts
                assert file_facts.freshness_state == "current"
                assert file_facts.resource_ids == (newest_id,)
            with live_row_recorder.hard_check(
                "evidence",
                "resolving_citations",
            ):
                assert_resolving_citations(
                    text,
                    snapshot,
                    (comparison.id,),
                )

        with live_row_recorder.diagnostic("route_binding"):
            assert_route_binding(
                snapshot,
                revision=route.revision,
                fingerprint=route.fingerprint,
            )
        with live_row_recorder.diagnostic("runtime_event_topology"):
            assert_inspectable_runtime_state(snapshot)
        with live_row_recorder.diagnostic("prescribed_catalog_inspection"):
            assert_catalog_discovery_and_inspection(
                snapshot,
                (newest_id, older_id, customer_table_id),
            )
        with live_row_recorder.diagnostic("single_declared_comparison_path"):
            assert len(compare_tasks) == 1
            assert (
                compare_tasks[0].arguments["key_normalization"]
                == CROSS_SOURCE_KEY_NORMALIZATION
                == commerce_oracles.comparison_key_normalization
            )
        live_row_recorder.assert_mvp_passed()
    finally:
        await agent.close()


def _task_for_evidence(tasks: tuple[Task, ...], evidence: Evidence) -> Task:
    matches = tuple(task for task in tasks if task.id == evidence.task_id)
    assert len(matches) == 1
    return matches[0]


def _tasks_for_evidence(
    tasks: tuple[Task, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[Task, ...]:
    resolved = tuple(_task_for_evidence(tasks, item) for item in evidence)
    assert len({task.id for task in resolved}) == len(resolved)
    return resolved


def _resolve_exact_comparison(
    *,
    compare_tasks: tuple[Task, ...],
    file_evidence: tuple[Evidence, ...],
    database_evidence: tuple[Evidence, ...],
    comparison_evidence: tuple[Evidence, ...],
    oracles: CommerceOracles,
) -> tuple[Evidence, Evidence, Evidence]:
    task_by_id = {task.id: task for task in compare_tasks}
    for candidate in comparison_evidence:
        task = task_by_id.get(candidate.task_id)
        if task is None:
            continue
        input_ids = {
            task.arguments.get("left_evidence_id"),
            task.arguments.get("right_evidence_id"),
        }
        selected_file = tuple(item for item in file_evidence if item.id in input_ids)
        selected_database = tuple(
            item for item in database_evidence if item.id in input_ids
        )
        if len(selected_file) != 1 or len(selected_database) != 1:
            continue
        try:
            left = candidate.payload.get("left")
            right = candidate.payload.get("right")
            assert isinstance(left, Mapping)
            assert isinstance(right, Mapping)
            assert {
                left.get("evidence_id"),
                right.get("evidence_id"),
            } == input_ids
            assert (
                task.arguments.get("key_normalization")
                == CROSS_SOURCE_KEY_NORMALIZATION
                == oracles.comparison_key_normalization
            )
            assert candidate.blob_id is not None
            assert candidate.content_hash == candidate.payload["artifact_digest"]
            assert candidate.payload["complete"] is True
            assert candidate.payload["truncated"] is False
            assert candidate.payload["total_discrepancies"] == len(
                oracles.discrepancies
            )
            assert candidate.payload["stored_discrepancies"] == len(
                oracles.discrepancies
            )
            normalized = normalize_comparison_discrepancies(
                candidate.payload,
                file_evidence_id=selected_file[0].id,
            )
            assert len(normalized) == len(oracles.discrepancies)
            assert Counter(normalized) == Counter(oracles.discrepancies)
        except (AssertionError, KeyError, TypeError, ValueError):
            continue
        return selected_file[0], selected_database[0], candidate
    raise AssertionError("no accepted comparison exactly matched the fixture oracle")
