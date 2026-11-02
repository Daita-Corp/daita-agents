"""Encode the one current checklist-job record shape."""

from __future__ import annotations

from ...jobs.models import (
    ChecklistItem,
    ChecklistStatus,
    Delegation,
    DelegationState,
    Job,
    JobState,
    UserAnswer,
)
from ...llm.models import ModelSensitivity
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    integer,
    load_payload,
    optional_text,
    record,
    record_fields,
    sequence,
    text,
)

_JOB_FIELDS = (
    "job_id",
    "agent_id",
    "conversation_id",
    "objective",
    "source_ids",
    "state",
    "checklist",
    "delegations",
    "question",
    "answers",
    "current_run_id",
    "run_count",
    "tokens_used",
    "run_limit",
    "token_limit",
    "final_summary",
    "failure_code",
    "sensitivity",
    "revision",
    "created_at",
    "updated_at",
)
_ITEM_FIELDS = ("id", "title", "status", "after", "note")
_DELEGATION_FIELDS = ("id", "goal", "state", "run_id", "summary", "item")
_ANSWER_FIELDS = ("question", "text", "answered_at")
MAX_ENCODED_JOB_BYTES = 1024 * 1024


def encode_job(value: Job) -> str:
    if not isinstance(value, Job):
        raise TypeError("job codec requires Job")
    encoded = dump_payload(
        record(
            "Job",
            {
                "job_id": value.job_id,
                "agent_id": value.agent_id,
                "conversation_id": value.conversation_id,
                "objective": value.objective,
                "source_ids": list(value.source_ids),
                "state": value.state.value,
                "checklist": [
                    record(
                        "ChecklistItem",
                        {
                            "id": item.id,
                            "title": item.title,
                            "status": item.status.value,
                            "after": list(item.after),
                            "note": item.note,
                        },
                    )
                    for item in value.checklist
                ],
                "delegations": [
                    record(
                        "Delegation",
                        {
                            "id": item.id,
                            "goal": item.goal,
                            "state": item.state.value,
                            "run_id": item.run_id,
                            "summary": item.summary,
                            "item": item.item,
                        },
                    )
                    for item in value.delegations
                ],
                "question": value.question,
                "answers": [
                    record(
                        "UserAnswer",
                        {
                            "question": item.question,
                            "text": item.text,
                            "answered_at": datetime_encode(item.answered_at),
                        },
                    )
                    for item in value.answers
                ],
                "current_run_id": value.current_run_id,
                "run_count": value.run_count,
                "tokens_used": value.tokens_used,
                "run_limit": value.run_limit,
                "token_limit": value.token_limit,
                "final_summary": value.final_summary,
                "failure_code": value.failure_code,
                "sensitivity": value.sensitivity.value,
                "revision": value.revision,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
            },
        )
    )
    if len(encoded.encode("utf-8")) > MAX_ENCODED_JOB_BYTES:
        raise ValueError("job record exceeds its encoded byte bound")
    return encoded


def _item(value: JsonValue) -> ChecklistItem:
    fields = record_fields(value, "ChecklistItem", _ITEM_FIELDS)
    return ChecklistItem(
        id=text(fields["id"], "checklist item id"),
        title=text(fields["title"], "checklist item title"),
        status=ChecklistStatus(text(fields["status"], "checklist item status")),
        after=tuple(
            text(item, "checklist dependency")
            for item in sequence(fields["after"], "checklist dependencies")
        ),
        note=optional_text(fields["note"], "checklist item note"),
    )


def _delegation(value: JsonValue) -> Delegation:
    fields = record_fields(value, "Delegation", _DELEGATION_FIELDS)
    return Delegation(
        id=text(fields["id"], "delegation id"),
        goal=text(fields["goal"], "delegation goal"),
        state=DelegationState(text(fields["state"], "delegation state")),
        run_id=text(fields["run_id"], "delegation run"),
        summary=optional_text(fields["summary"], "delegation summary"),
        item=optional_text(fields["item"], "delegation item"),
    )


def _answer(value: JsonValue) -> UserAnswer:
    fields = record_fields(value, "UserAnswer", _ANSWER_FIELDS)
    return UserAnswer(
        question=text(fields["question"], "answered question"),
        text=text(fields["text"], "answer"),
        answered_at=datetime_decode(fields["answered_at"]),
    )


def decode_job(value: str) -> Job:
    if len(value.encode("utf-8")) > MAX_ENCODED_JOB_BYTES:
        raise ValueError("job record exceeds its encoded byte bound")
    fields = record_fields(load_payload(value), "Job", _JOB_FIELDS)
    return Job(
        job_id=text(fields["job_id"], "job id"),
        agent_id=text(fields["agent_id"], "job agent"),
        conversation_id=text(fields["conversation_id"], "job conversation"),
        objective=text(fields["objective"], "job objective"),
        source_ids=tuple(
            text(item, "job source")
            for item in sequence(fields["source_ids"], "job sources")
        ),
        state=JobState(text(fields["state"], "job state")),
        checklist=tuple(
            _item(item) for item in sequence(fields["checklist"], "job checklist")
        ),
        delegations=tuple(
            _delegation(item)
            for item in sequence(fields["delegations"], "job delegations")
        ),
        question=optional_text(fields["question"], "job question"),
        answers=tuple(
            _answer(item) for item in sequence(fields["answers"], "job answers")
        ),
        current_run_id=optional_text(fields["current_run_id"], "job current run"),
        run_count=integer(fields["run_count"], "job run count"),
        tokens_used=integer(fields["tokens_used"], "job tokens used"),
        run_limit=integer(fields["run_limit"], "job run limit"),
        token_limit=integer(fields["token_limit"], "job token limit"),
        final_summary=optional_text(fields["final_summary"], "job final summary"),
        failure_code=optional_text(fields["failure_code"], "job failure code"),
        sensitivity=ModelSensitivity(text(fields["sensitivity"], "job sensitivity")),
        revision=integer(fields["revision"], "job revision"),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
    )


__all__ = ["decode_job", "encode_job"]
