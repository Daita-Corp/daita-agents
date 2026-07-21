"""Hard, provider-independent assertions shared by Wave 1 live scenarios."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
import re

from daita._json import FrozenJsonObject
from daita.catalog import ResourceKind, catalog_resource_id
from daita.loop.models import LoopExit, LoopExitKind
from daita.operations.checkpoints import ModelCallStatus, OperationSnapshot
from daita.operations.models import Evidence, OperationStatus, Task, TaskStatus
from sqlglot import exp, parse_one
from sqlglot.expressions.core import Expression

from .fixture_oracles import Discrepancy

_CITATION = re.compile(r"\[evidence:([^\]\r\n]+)\]")
_MUTATION = re.compile(
    r"\b(ALTER|ATTACH|CREATE|DELETE|DETACH|DROP|INSERT|PRAGMA|REPLACE|UPDATE|VACUUM)\b",
    re.IGNORECASE,
)


def assert_completed(result: LoopExit, snapshot: OperationSnapshot) -> str:
    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text is not None
    assert snapshot.operation.status is OperationStatus.SUCCEEDED
    assert snapshot.operation.final_text == result.final_text
    assert snapshot.operation.terminal_reason == result.reason
    assert snapshot.readiness
    assert snapshot.readiness[-1].allowed is True
    assert snapshot.model_calls
    assert all(
        call.status is ModelCallStatus.COMPLETED for call in snapshot.model_calls
    )
    assert all(call.response is not None for call in snapshot.model_calls)
    assert (
        sum(
            call.response.usage.total_tokens
            for call in snapshot.model_calls
            if call.response is not None
        )
        > 0
    )
    return result.final_text


def assert_inspectable_runtime_state(snapshot: OperationSnapshot) -> None:
    assert snapshot.tasks
    assert snapshot.evidence
    assert snapshot.observations
    assert snapshot.readiness
    assert snapshot.events
    assert all(turn.operation_id == snapshot.operation.id for turn in snapshot.turns)
    assert all(
        call.operation_id == snapshot.operation.id for call in snapshot.model_calls
    )
    assert all(task.operation_id == snapshot.operation.id for task in snapshot.tasks)
    assert all(
        evidence.operation_id == snapshot.operation.id for evidence in snapshot.evidence
    )
    assert all(
        observation.operation_id == snapshot.operation.id
        for observation in snapshot.observations
    )
    assert all(event.operation_id == snapshot.operation.id for event in snapshot.events)
    assert all(
        task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        for task in snapshot.tasks
    )
    turn_by_id = {turn.id: turn for turn in snapshot.turns}
    model_call_by_id = {call.id: call for call in snapshot.model_calls}
    model_call_by_turn = {call.turn_id: call for call in snapshot.model_calls}
    task_by_id = {task.id: task for task in snapshot.tasks}
    evidence_by_id = {item.id: item for item in snapshot.evidence}
    assert len(turn_by_id) == len(snapshot.turns)
    assert len(model_call_by_id) == len(snapshot.model_calls)
    assert len(model_call_by_turn) == len(snapshot.model_calls)
    assert len(task_by_id) == len(snapshot.tasks)
    assert len(evidence_by_id) == len(snapshot.evidence)
    assert set(model_call_by_turn) == set(turn_by_id)
    tool_call_ids_by_model_call: dict[str, set[str]] = {}
    for model_call in snapshot.model_calls:
        assert model_call.turn_id in turn_by_id
        assert model_call.response is not None
        tool_call_ids = {call.id for call in model_call.response.tool_calls}
        assert len(tool_call_ids) == len(model_call.response.tool_calls)
        tool_call_ids_by_model_call[model_call.id] = tool_call_ids
        assert (
            sum(
                event.type == "model_call.started"
                and event.turn_id == model_call.turn_id
                and event.model_call_id == model_call.id
                for event in snapshot.events
            )
            == 1
        )
        assert (
            sum(
                event.type == "model_response.recorded"
                and event.turn_id == model_call.turn_id
                and event.model_call_id == model_call.id
                for event in snapshot.events
            )
            == 1
        )
    for task in snapshot.tasks:
        assert task.turn_id in model_call_by_turn
        model_call = model_call_by_turn[task.turn_id]
        assert task.call_id in tool_call_ids_by_model_call[model_call.id]
        matching_created_events = tuple(
            event
            for event in snapshot.events
            if event.type == "task.created"
            and event.turn_id == task.turn_id
            and event.model_call_id == model_call.id
            and event.call_id == task.call_id
            and event.task_id == task.id
            and event.capability_id == task.capability_id
            and event.executor_id == task.executor_id
            and event.payload.get("task_id") == task.id
            and event.payload.get("capability_id") == task.capability_id
            and event.payload.get("executor_id") == task.executor_id
        )
        assert len(matching_created_events) == 1
    accepted_ids = {
        item.id for item in snapshot.evidence if item.accepted and item.applicable
    }
    assert accepted_ids
    for evidence_id in accepted_ids:
        evidence = evidence_by_id[evidence_id]
        assert evidence.task_id in task_by_id
        task = task_by_id[evidence.task_id]
        assert task.turn_id in model_call_by_turn
        model_call = model_call_by_turn[task.turn_id]
        assert evidence.turn_id == task.turn_id
        assert evidence.capability_id == task.capability_id
        assert evidence.executor_id == task.executor_id
        matching_accepted_events = tuple(
            event
            for event in snapshot.events
            if event.type == "evidence.accepted"
            and event.turn_id == task.turn_id
            and event.model_call_id == model_call.id
            and event.call_id == task.call_id
            and event.task_id == task.id
            and event.evidence_id == evidence.id
            and event.capability_id == task.capability_id
            and event.executor_id == task.executor_id
            and event.payload.get("task_id") == task.id
            and event.payload.get("evidence_id") == evidence.id
        )
        assert len(matching_accepted_events) == 1
    successful_evidence_ids: set[str] = set()
    failed_correlations: set[tuple[object, ...]] = set()
    for observation in snapshot.observations:
        assert observation.operation_id == snapshot.operation.id
        assert observation.turn_id in model_call_by_turn
        model_call = model_call_by_turn[observation.turn_id]
        if observation.task_id is not None:
            assert observation.task_id in task_by_id
        linked_task = (
            None if observation.task_id is None else task_by_id[observation.task_id]
        )
        correlated_call_id = (
            linked_task.call_id if linked_task is not None else observation.call_id
        )
        if correlated_call_id is not None:
            assert correlated_call_id in tool_call_ids_by_model_call[model_call.id]
        matching_observation_events = tuple(
            event
            for event in snapshot.events
            if event.type == "observation.recorded"
            and event.turn_id == observation.turn_id
            and event.model_call_id == model_call.id
            and event.call_id == correlated_call_id
            and event.task_id == observation.task_id
            and event.evidence_id == observation.evidence_id
            and event.payload.get("code") == observation.code
        )
        assert len(matching_observation_events) == 1
        if observation.success:
            assert observation.task_id is not None
            assert observation.evidence_id is not None
            assert linked_task is not None
            task = linked_task
            assert observation.evidence_id in evidence_by_id
            evidence = evidence_by_id[observation.evidence_id]
            assert task.status is TaskStatus.SUCCEEDED
            assert task.operation_id == snapshot.operation.id
            assert task.turn_id == observation.turn_id
            if observation.call_id is not None:
                assert task.call_id == observation.call_id
            assert evidence.id in accepted_ids
            assert evidence.task_id == task.id
            assert evidence.turn_id == observation.turn_id
            assert evidence.capability_id == task.capability_id
            assert evidence.executor_id == task.executor_id
            assert evidence.id in task.evidence_ids
            assert evidence.id not in successful_evidence_ids
            successful_evidence_ids.add(evidence.id)
            continue

        assert observation.evidence_id is None
        correlation = (
            observation.turn_id,
            correlated_call_id,
            observation.task_id,
            observation.code,
        )
        assert correlation not in failed_correlations
        failed_correlations.add(correlation)
        if observation.task_id is not None:
            assert linked_task is not None
            task = linked_task
            assert task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}
            assert task.turn_id == observation.turn_id
            if observation.call_id is not None:
                assert task.call_id == observation.call_id
            expected_event_type = (
                "task.failed" if task.status is TaskStatus.FAILED else "task.cancelled"
            )
            if task.status is TaskStatus.FAILED:
                assert task.error_code == observation.code
            else:
                assert task.error_code is None
                assert observation.code == "approval_cancelled"
            assert any(
                event.type == expected_event_type
                and event.turn_id == observation.turn_id
                and event.model_call_id == model_call.id
                and event.call_id == task.call_id
                and event.task_id == observation.task_id
                and event.capability_id == task.capability_id
                and event.executor_id == task.executor_id
                and event.payload.get("error_code") == observation.code
                for event in snapshot.events
            )
        elif observation.call_id is not None:
            if observation.code == "action.skipped_after_rejection":
                assert any(
                    event.type == "action.skipped"
                    and event.turn_id == observation.turn_id
                    and event.model_call_id == model_call.id
                    and event.call_id == observation.call_id
                    and event.payload.get("blocked_by_call_id")
                    == observation.payload.get("blocked_by_call_id")
                    and event.payload.get("blocked_by_code")
                    == observation.payload.get("blocked_by_code")
                    for event in snapshot.events
                )
            else:
                assert any(
                    event.type == "action.rejected"
                    and event.turn_id == observation.turn_id
                    and event.model_call_id == model_call.id
                    and event.call_id == observation.call_id
                    and event.payload.get("code") == observation.code
                    for event in snapshot.events
                )
        else:
            assert observation.task_id is None
            assert any(
                event.type == "readiness.recorded"
                and event.turn_id == observation.turn_id
                and event.model_call_id == model_call.id
                and event.payload.get("allowed") is False
                and event.payload.get("code") == observation.code
                for event in snapshot.events
            )
            assert any(
                readiness.allowed is False and readiness.code == observation.code
                for readiness in snapshot.readiness
            )
    assert successful_evidence_ids == accepted_ids
    for task in snapshot.tasks:
        if task.status is TaskStatus.SUCCEEDED:
            assert task.evidence_ids
            assert set(task.evidence_ids) <= successful_evidence_ids
            continue
        matching_failures = tuple(
            observation
            for observation in snapshot.observations
            if not observation.success and observation.task_id == task.id
        )
        assert len(matching_failures) == 1
        failure = matching_failures[0]
        assert failure.turn_id == task.turn_id
        if failure.call_id is not None:
            assert failure.call_id == task.call_id
        if task.status is TaskStatus.FAILED:
            assert failure.code == task.error_code
        else:
            assert failure.code == "approval_cancelled"
    event_types = {event.type for event in snapshot.events}
    assert {
        "model_call.started",
        "model_response.recorded",
        "task.created",
        "evidence.accepted",
        "observation.recorded",
        "readiness.recorded",
        "operation.succeeded",
    } <= event_types


def assert_allowed_capabilities(
    snapshot: OperationSnapshot,
    allowed: Iterable[str],
) -> None:
    allowed_ids = set(allowed)
    assert {task.capability_id for task in snapshot.tasks} <= allowed_ids


def assert_route_binding(
    snapshot: OperationSnapshot,
    *,
    revision: int,
    fingerprint: str,
) -> None:
    assert snapshot.operation.model_route_revision == revision
    assert snapshot.operation.model_route_fingerprint == fingerprint


def assert_catalog_discovery_and_inspection(
    snapshot: OperationSnapshot,
    required_resource_ids: Iterable[str],
) -> None:
    accepted_task_ids = {
        evidence.task_id
        for evidence in snapshot.evidence
        if evidence.accepted and evidence.applicable
    }
    explicit_search = any(
        task.capability_id == "catalog.search" and task.id in accepted_task_ids
        for task in snapshot.tasks
    )
    selected_catalog_context = any(
        record.get("kind") == "catalog"
        for call in snapshot.model_calls
        for record in frozen_mappings(
            call.request.context_selection.get("selected_blocks", ())
        )
    )
    assert explicit_search or selected_catalog_context
    inspected = {
        resource_id
        for task in snapshot.tasks
        if task.capability_id == "catalog.inspect" and task.id in accepted_task_ids
        for resource_id in (task.arguments.get("resource_id"),)
        if isinstance(resource_id, str)
    }
    assert set(required_resource_ids) <= inspected


def accepted_evidence_for_tasks(
    snapshot: OperationSnapshot,
    tasks: Iterable[Task],
) -> tuple[Evidence, ...]:
    """Resolve accepted evidence through exact task authority, not list position."""

    task_by_id = {task.id: task for task in tasks}
    return tuple(
        evidence
        for evidence in snapshot.evidence
        if evidence.accepted
        and evidence.applicable
        and evidence.operation_id == snapshot.operation.id
        and evidence.task_id in task_by_id
        and evidence.capability_id == task_by_id[evidence.task_id].capability_id
        and evidence.executor_id == task_by_id[evidence.task_id].executor_id
        and evidence.id in task_by_id[evidence.task_id].evidence_ids
    )


def assert_cited_evidence_supports(
    text: str,
    snapshot: OperationSnapshot,
    candidates: Iterable[Evidence],
    *,
    exact_values: Iterable[object] = (),
    money_cents: Iterable[int] = (),
) -> tuple[Evidence, ...]:
    candidate_by_id = {item.id: item for item in candidates}
    candidate_ids = set(candidate_by_id)
    assert candidate_ids
    assert_resolving_citations(text, snapshot, ())
    cited_candidates = tuple(
        candidate_by_id[evidence_id]
        for evidence_id in dict.fromkeys(_CITATION.findall(text))
        if evidence_id in candidate_by_id
    )
    assert cited_candidates
    cited_scalars = tuple(
        scalar
        for evidence in cited_candidates
        for scalar in _result_row_scalars(evidence)
    )
    assert all(
        any(_semantic_scalar_match(expected, actual) for actual in cited_scalars)
        for expected in exact_values
    )
    assert all(
        any(_money_scalar_match(cents, actual) for actual in cited_scalars)
        for cents in money_cents
    )
    return cited_candidates


def assert_session_cited_evidence_supports(
    text: str,
    snapshots: Sequence[OperationSnapshot],
    candidates: Iterable[Evidence],
    *,
    exact_values: Iterable[object] = (),
    money_cents: Iterable[int] = (),
) -> tuple[Evidence, ...]:
    """Resolve one session answer against accepted evidence from prior operations."""

    accepted = {
        evidence.id: evidence
        for snapshot in snapshots
        for evidence in snapshot.evidence
        if evidence.accepted and evidence.applicable
    }
    candidate_by_id = {item.id: item for item in candidates if item.id in accepted}
    assert candidate_by_id
    cited = tuple(dict.fromkeys(_CITATION.findall(text)))
    assert cited
    assert set(cited) <= set(accepted)
    cited_candidates = tuple(
        candidate_by_id[evidence_id]
        for evidence_id in cited
        if evidence_id in candidate_by_id
    )
    assert cited_candidates
    cited_scalars = tuple(
        scalar
        for evidence in cited_candidates
        for scalar in _result_row_scalars(evidence)
    )
    assert all(
        any(_semantic_scalar_match(expected, actual) for actual in cited_scalars)
        for expected in exact_values
    )
    assert all(
        any(_money_scalar_match(cents, actual) for actual in cited_scalars)
        for cents in money_cents
    )
    return cited_candidates


def assert_catalog_graph_use(
    snapshot: OperationSnapshot,
    *,
    from_resource_id: str,
    to_resource_id: str,
    forbidden_resource_ids: Iterable[str] = (),
) -> None:
    """Prove an accepted traversal informed a later accepted SQL join."""

    traversal_tasks = tuple(
        task
        for task in snapshot.tasks
        if task.capability_id == "catalog.traverse"
        and task.status is TaskStatus.SUCCEEDED
    )
    traversal_evidence = accepted_evidence_for_tasks(snapshot, traversal_tasks)
    if not traversal_evidence:
        raise AssertionError("accepted catalog traversal evidence is required")
    observed_evidence_ids = {
        observation.evidence_id
        for observation in snapshot.observations
        if observation.success
    }
    turn_positions = {turn.id: index for index, turn in enumerate(snapshot.turns)}
    forbidden = set(forbidden_resource_ids)
    sql_tasks = tuple(
        task
        for task in query_tasks(snapshot)
        if task.status is TaskStatus.SUCCEEDED
        and accepted_evidence_for_tasks(snapshot, (task,))
    )
    assert sql_tasks
    for evidence in traversal_evidence:
        if evidence.id not in observed_evidence_ids:
            continue
        payload = evidence.payload
        if (
            payload.get("reachable") is not True
            or payload.get("truncated") is not False
        ):
            continue
        request = frozen_mapping(payload.get("request"))
        from_ids = _string_tuple(request.get("from_resource_ids"))
        to_ids = _string_tuple(request.get("to_resource_ids"))
        if from_resource_id not in from_ids or to_resource_id not in to_ids:
            continue
        traversal_task = next(
            task for task in traversal_tasks if task.id == evidence.task_id
        )
        for path in frozen_mappings(payload.get("paths")):
            resource_ids = _string_tuple(path.get("resource_ids"))
            resource_set = set(resource_ids)
            if (
                from_resource_id not in resource_set
                or to_resource_id not in resource_set
                or forbidden & resource_set
            ):
                continue
            required_pairs = {
                frozenset(
                    (
                        (
                            str(step["from_resource_id"]),
                            str(pair["source_field"]).casefold(),
                        ),
                        (
                            str(step["to_resource_id"]),
                            str(pair["target_field"]).casefold(),
                        ),
                    )
                )
                for step in frozen_mappings(path.get("steps"))
                for pair in frozen_mappings(step.get("field_pairs"))
            }
            if not required_pairs:
                continue
            for sql_task in sql_tasks:
                if (
                    turn_positions[traversal_task.turn_id]
                    >= turn_positions[sql_task.turn_id]
                ):
                    continue
                validation_resources = set(
                    sql_task.execution_facts.validation_facts.resource_ids
                )
                if not resource_set <= validation_resources:
                    continue
                source_id = sql_task.execution_facts.validation_facts.source_id
                if source_id is None:
                    continue
                if required_pairs <= _sql_join_pairs(
                    assert_read_only_sql(sql_task),
                    source_id=source_id,
                ):
                    return
    raise AssertionError(
        "accepted catalog traversal did not inform a later accepted SQL join"
    )


def query_tasks(snapshot: OperationSnapshot) -> tuple[Task, ...]:
    return tuple(
        task
        for task in snapshot.tasks
        if task.capability_id in {"data.sqlite.query", "data.postgresql.query"}
    )


def assert_read_only_sql(task: Task) -> str:
    sql = task.arguments.get("sql")
    assert isinstance(sql, str)
    assert sql.lstrip().upper().startswith(("SELECT", "WITH"))
    assert _MUTATION.search(sql) is None
    assert ";" not in sql.rstrip().rstrip(";")
    assert task.execution_facts.access_mode.value == "read"
    assert task.execution_facts.side_effecting is False
    return sql


def assert_current_authority(
    task: Task,
    *,
    source_id: str,
    required_resource_ids: Iterable[str],
    forbidden_resource_ids: Iterable[str] = (),
) -> None:
    facts = task.execution_facts.validation_facts
    required = set(required_resource_ids)
    forbidden = set(forbidden_resource_ids)
    assert facts.schema_version == 1
    assert facts.validation_passed is True
    assert facts.in_scope is True
    assert facts.destructive is False
    assert facts.source_ids == (source_id,)
    assert facts.source_id == source_id
    assert facts.freshness_state == "current"
    assert required <= set(facts.resource_ids)
    assert forbidden.isdisjoint(facts.resource_ids)
    assert {resource for resource, _ in facts.resource_revisions} == set(
        facts.resource_ids
    )
    assert dict(facts.source_revisions).keys() == {source_id}


def assert_resolving_citations(
    text: str,
    snapshot: OperationSnapshot,
    required_evidence_ids: Iterable[str],
) -> None:
    cited = tuple(_CITATION.findall(text))
    accepted = {
        evidence.id
        for evidence in snapshot.evidence
        if evidence.accepted
        and evidence.applicable
        and evidence.operation_id == snapshot.operation.id
    }
    assert cited
    assert set(cited) <= accepted
    assert set(required_evidence_ids) <= set(cited)


def assert_count_and_money(text: str, count: int, cents: int) -> None:
    semantic = semantic_answer_text(text)
    count_token = rf"(?<!\d){count}(?!\d)"
    customer = r"(?:active[-\s]+)?customers?"
    assert re.search(
        rf"(?:{count_token}\s+{customer}|{customer}(?:\s+count)?\s*(?:is|was|:|=)?\s*{count_token})",
        semantic,
        flags=re.IGNORECASE,
    )
    assert_money(semantic, cents)


def assert_money(text: str, cents: int) -> None:
    text = semantic_answer_text(text)
    value = Decimal(cents) / Decimal(100)
    whole = int(value)
    decimal = f"{value:.2f}"
    alternatives = {decimal, f"{value:,.2f}"}
    if value == whole:
        alternatives.add(str(whole))
        alternatives.add(f"{whole:,}")
    assert any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for candidate in alternatives
        for pattern in _money_patterns(candidate)
    )


def assert_labeled_money(text: str, label: str, cents: int) -> None:
    semantic = semantic_answer_text(text)
    segments = tuple(
        segment
        for segment in _semantic_segments(semantic)
        if re.search(re.escape(label), segment, flags=re.IGNORECASE)
    )
    assert segments
    for segment in segments:
        try:
            assert_money(segment, cents)
        except AssertionError:
            continue
        return
    raise AssertionError("money value is not bound to its answer label")


def _money_patterns(candidate: str) -> tuple[str, ...]:
    token = rf"(?<!\d){re.escape(candidate)}(?!\d)"
    patterns = (
        rf"\$\s*{token}",
        rf"\bUSD\s*{token}",
        rf"{token}\s*(?:USD|dollars?)\b",
        rf"\b(?:net\s+(?:paid\s+)?revenue|revenue|refund(?:ed)?(?:\s+total)?|amount|total)"
        rf"(?:\s+(?:is|was|of|totaled|equals?))?\s*[:=]?\s*{token}",
    )
    if "." in candidate:
        return (*patterns, token)
    return patterns


def assert_text_identities(text: str, identities: Iterable[str]) -> None:
    text = semantic_answer_text(text)
    for identity in identities:
        assert re.search(rf"(?<!\d){re.escape(identity)}(?!\d)", text)


def assert_discrepancies_explained(
    text: str,
    discrepancies: Iterable[Discrepancy],
) -> None:
    """Require material strict comparison facts without prescribing prose."""

    semantic = semantic_answer_text(text)
    expected = tuple(discrepancies)
    identities = tuple(
        item.customer_id for item in expected if item.customer_id is not None
    )
    for discrepancy in expected:
        if discrepancy.kind == "invalid_key":
            assert discrepancy.customer_id is None
            assert any(
                _invalid_key_segment_matches(segment, discrepancy)
                for segment in _semantic_segments(semantic)
            )
            continue
        customer_id = discrepancy.customer_id
        assert customer_id is not None
        matched = False
        for segment in _identity_segments(
            semantic,
            customer_id,
            identities=identities,
        ):
            try:
                _assert_discrepancy_segment(segment, discrepancy)
            except AssertionError:
                continue
            matched = True
            break
        assert matched


def semantic_answer_text(text: str) -> str:
    return _CITATION.sub("", text)


def _assert_discrepancy_segment(
    segment: str,
    discrepancy: Discrepancy,
) -> None:
    if discrepancy.kind in {"left_only", "right_only"}:
        _assert_one_sided_direction(segment, discrepancy.kind)
        return
    if discrepancy.kind == "duplicate_key":
        assert discrepancy.source in {"file", "database"}
        assert discrepancy.duplicate_count is not None
        assert discrepancy.duplicate_count > 1
        assert re.search(
            _comparison_source_pattern(discrepancy.source),
            segment,
            flags=re.IGNORECASE,
        )
        assert re.search(
            r"\b(?:duplicate|duplicated|repeated)\b",
            segment,
            flags=re.IGNORECASE,
        )
        assert re.search(
            rf"(?<!\d){discrepancy.duplicate_count}(?!\d)",
            segment,
        )
        return
    assert discrepancy.column is not None
    column_terms = {
        discrepancy.column.casefold(),
        discrepancy.column.replace("_", " ").casefold(),
    }
    assert any(term in segment.casefold() for term in column_terms)
    _assert_oriented_value(
        segment,
        discrepancy.file_value,
        source_pattern=r"(?:csv|export|file)",
        present=discrepancy.file_present,
    )
    _assert_oriented_value(
        segment,
        discrepancy.database_value,
        source_pattern=r"(?:current\s+(?:data|record)|database|warehouse)",
        present=discrepancy.database_present,
    )
    if discrepancy.kind == "type_mismatch":
        assert any(
            marker in segment.casefold()
            for marker in ("null", "none", "type", "missing")
        )


def _invalid_key_segment_matches(segment: str, discrepancy: Discrepancy) -> bool:
    try:
        assert discrepancy.source in {"file", "database"}
        assert re.search(
            _comparison_source_pattern(discrepancy.source),
            segment,
            flags=re.IGNORECASE,
        )
        missing = discrepancy.missing_columns
        null = discrepancy.null_columns
        assert missing or null
        lowered = segment.casefold()
        for column in missing:
            _assert_column_mentioned(segment, column)
            assert any(
                marker in lowered
                for marker in (
                    "absent",
                    "lacks",
                    "missing",
                    "not present",
                    "omitted",
                    "without",
                )
            )
        for column in null:
            _assert_column_mentioned(segment, column)
            assert any(marker in lowered for marker in ("null", "none"))
        assert any(
            marker in lowered
            for marker in (
                "cannot be matched",
                "cannot match",
                "invalid key",
                "key is invalid",
                "key unusable",
                "unmatchable",
                "unusable key",
            )
        )
    except AssertionError:
        return False
    return True


def _assert_column_mentioned(segment: str, column: str) -> None:
    terms = {column.casefold(), column.replace("_", " ").casefold()}
    assert any(term in segment.casefold() for term in terms)


def _comparison_source_pattern(source: str) -> str:
    if source == "file":
        return r"(?:csv|export|file)"
    if source == "database":
        return r"(?:current\s+(?:data|record)|database|warehouse)"
    raise AssertionError("comparison discrepancy source is invalid")


def _semantic_segments(text: str) -> tuple[str, ...]:
    return tuple(
        segment
        for segment in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", text)
        if segment.strip()
    )


def _identity_segments(
    text: str,
    identity: str,
    *,
    identities: Iterable[str],
) -> tuple[str, ...]:
    matches = tuple(re.finditer(rf"(?<!\d){re.escape(identity)}(?!\d)", text))
    assert matches
    identity_matches = tuple(
        sorted(
            (
                match
                for candidate in dict.fromkeys(identities)
                for match in re.finditer(
                    rf"(?<!\d){re.escape(candidate)}(?!\d)",
                    text,
                )
            ),
            key=lambda match: (match.start(), match.end()),
        )
    )
    separators = tuple(re.finditer(r"(?<=[.!?])\s+|(?:\r?\n)+", text))
    return tuple(
        text[
            max(
                (
                    separator.end()
                    for separator in separators
                    if separator.end() <= match.start()
                ),
                default=max(0, match.start() - 240),
            ) : next(
                (
                    other.start()
                    for other in identity_matches
                    if other.start() >= match.end()
                    and (other.start(), other.end()) != (match.start(), match.end())
                ),
                min(len(text), match.end() + 480),
            )
        ]
        for match in matches
    )


def _assert_one_sided_direction(segment: str, kind: str) -> None:
    file_source = r"(?:csv|export|file)"
    database_source = r"(?:current\s+(?:data|record)|database|warehouse)"
    present, absent = (
        (file_source, database_source)
        if kind == "left_only"
        else (database_source, file_source)
    )
    absence = r"(?:absent|missing|not\s+(?:found|in|present)|does\s+not\s+appear)"
    direction_patterns = (
        rf"(?:only\s+(?:in|on)|present\s+(?:in|on)|appears\s+(?:in|on)).{{0,32}}{present}",
        rf"{present}.{{0,80}}{absence}.{{0,40}}{absent}",
        rf"{absence}.{{0,40}}{absent}.{{0,80}}{present}",
    )
    assert any(
        re.search(pattern, segment, flags=re.IGNORECASE) is not None
        for pattern in direction_patterns
    )


def _assert_material_value(segment: str, value: object | None) -> None:
    lowered = segment.casefold()
    if value is None:
        assert any(marker in lowered for marker in ("null", "none", "missing"))
        return
    if value == "":
        assert any(marker in lowered for marker in ("blank", "empty", '""', "''"))
        return
    rendered = str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        assert re.search(rf"(?<!\d){re.escape(rendered)}(?!\d)", segment)
    else:
        assert rendered.casefold() in lowered


def _assert_oriented_value(
    segment: str,
    value: object | None,
    *,
    source_pattern: str,
    present: bool | None,
) -> None:
    clauses = tuple(
        clause
        for clause in re.split(
            r"\s*(?:[;,]|\b(?:and|but|versus|vs\.?|whereas|while)\b)\s*",
            segment,
            flags=re.IGNORECASE,
        )
        if re.search(source_pattern, clause, flags=re.IGNORECASE) is not None
    )
    assert clauses
    for clause in clauses:
        try:
            if present is False:
                assert re.search(
                    r"\b(?:absent|missing|not\s+(?:found|present))\b",
                    clause,
                    flags=re.IGNORECASE,
                )
            else:
                _assert_material_value(clause, value)
        except AssertionError:
            continue
        return
    raise AssertionError("comparison value is not bound to its authoritative source")


def _scalars(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(scalar for item in value.values() for scalar in _scalars(item))
    if isinstance(value, tuple):
        return tuple(scalar for item in value for scalar in _scalars(item))
    return (value,)


def _result_row_scalars(evidence: Evidence) -> tuple[object, ...]:
    rows = evidence.payload.get("rows")
    if not isinstance(rows, tuple) or any(not isinstance(row, Mapping) for row in rows):
        return ()
    return tuple(scalar for row in rows for scalar in _scalars(row))


def _semantic_scalar_match(expected: object, actual: object) -> bool:
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.casefold() == expected.casefold()
    if isinstance(expected, int) and not isinstance(expected, bool):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    return type(actual) is type(expected) and actual == expected


def _money_scalar_match(cents: int, actual: object) -> bool:
    assert isinstance(cents, int) and not isinstance(cents, bool)
    dollars = Decimal(cents) / Decimal(100)
    if isinstance(actual, bool) or actual is None:
        return False
    if isinstance(actual, (int, float, Decimal)):
        rendered = Decimal(str(actual))
    elif isinstance(actual, str):
        normalized = actual.strip().removeprefix("$").replace(",", "")
        try:
            rendered = Decimal(normalized)
        except ArithmeticError:
            return False
    else:
        return False
    return rendered in {Decimal(cents), dollars}


def _string_tuple(value: object) -> tuple[str, ...]:
    assert isinstance(value, tuple)
    assert all(isinstance(item, str) for item in value)
    return tuple(value)


def _sql_join_pairs(
    sql: str,
    *,
    source_id: str,
) -> set[frozenset[tuple[str, str]]]:
    parsed = parse_one(sql)
    cte_lineage: dict[str, set[str]] = {}
    for cte in parsed.find_all(exp.CTE):
        cte_lineage[cte.alias_or_name.casefold()] = _physical_resource_ids(
            cte.this,
            source_id=source_id,
            cte_names={
                item.alias_or_name.casefold() for item in parsed.find_all(exp.CTE)
            },
        )
    qualifiers: dict[str, set[str]] = {}
    for table in parsed.find_all(exp.Table):
        name = table.name.casefold()
        qualifier = table.alias_or_name.casefold()
        if name in cte_lineage:
            qualifiers[qualifier] = cte_lineage[name]
            continue
        native_identity = _table_native_identity(table)
        qualifiers[qualifier] = {
            catalog_resource_id(source_id, ResourceKind.TABLE, native_identity)
        }
    pairs: set[frozenset[tuple[str, str]]] = set()
    for equality in parsed.find_all(exp.EQ):
        left = equality.this
        right = equality.expression
        if (
            isinstance(left, exp.Column)
            and isinstance(right, exp.Column)
            and left.table
            and right.table
        ):
            left_resources = qualifiers.get(left.table.casefold(), set())
            right_resources = qualifiers.get(right.table.casefold(), set())
            pairs.update(
                frozenset(
                    (
                        (left_resource, left.name.casefold()),
                        (right_resource, right.name.casefold()),
                    )
                )
                for left_resource in left_resources
                for right_resource in right_resources
            )
    return pairs


def _physical_resource_ids(
    expression: Expression,
    *,
    source_id: str,
    cte_names: set[str],
) -> set[str]:
    return {
        catalog_resource_id(
            source_id,
            ResourceKind.TABLE,
            _table_native_identity(table),
        )
        for table in expression.find_all(exp.Table)
        if table.name.casefold() not in cte_names
    }


def _table_native_identity(table: exp.Table) -> str:
    schema = table.db or "main"
    return f"{schema}.{table.name}"


def frozen_mapping(value: object) -> FrozenJsonObject:
    assert isinstance(value, FrozenJsonObject)
    return value


def frozen_mappings(value: object) -> tuple[FrozenJsonObject, ...]:
    assert isinstance(value, tuple)
    assert all(isinstance(item, FrozenJsonObject) for item in value)
    return tuple(value)


def assert_no_text_leak(
    snapshots: Sequence[OperationSnapshot],
    prohibited: str,
) -> None:
    for snapshot in snapshots:
        for call in snapshot.model_calls:
            if prohibited in repr(call.request):
                raise AssertionError("session sentinel crossed model-request scope")
            if call.response is not None and prohibited in repr(call.response):
                raise AssertionError("session sentinel crossed model-response scope")


__all__ = [
    "assert_allowed_capabilities",
    "accepted_evidence_for_tasks",
    "assert_catalog_graph_use",
    "assert_catalog_discovery_and_inspection",
    "assert_completed",
    "assert_cited_evidence_supports",
    "assert_count_and_money",
    "assert_current_authority",
    "assert_discrepancies_explained",
    "assert_inspectable_runtime_state",
    "assert_labeled_money",
    "assert_money",
    "assert_no_text_leak",
    "assert_read_only_sql",
    "assert_resolving_citations",
    "assert_route_binding",
    "assert_session_cited_evidence_supports",
    "assert_text_identities",
    "frozen_mapping",
    "frozen_mappings",
    "query_tasks",
    "semantic_answer_text",
]
