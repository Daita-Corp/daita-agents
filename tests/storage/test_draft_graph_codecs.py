from __future__ import annotations

import json
from datetime import timedelta

import pytest

from daita.jobs.graph.models import (
    AttemptState,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    canonical_digest,
)
from daita.llm.models import ModelSensitivity
from daita.storage.sqlite_codecs.graph import (
    decode_graph_job,
    decode_graph_task,
    decode_job_graph,
    decode_task_attempt,
    decode_task_checkpoint,
    decode_task_comment,
    decode_task_dependency,
    decode_task_result,
    encode_graph_job,
    encode_graph_task,
    encode_job_graph,
    encode_task_attempt,
    encode_task_checkpoint,
    encode_task_comment,
    encode_task_dependency,
    encode_task_result,
)
from tests.support.graph import GRAPH_NOW, graph_admission, task_result

pytestmark = pytest.mark.unit


def test_primary_graph_records_round_trip_exactly():
    admission = graph_admission()
    attempt = TaskAttempt(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        attempt_id="attempt-1",
        ordinal=1,
        fencing_epoch=1,
        state=AttemptState.RUNNING,
        claim_token="claim-1",
        run_id="run-1",
        lease_expires_at=GRAPH_NOW + timedelta(seconds=30),
        absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
        started_at=GRAPH_NOW,
        heartbeat_at=GRAPH_NOW,
        ended_at=None,
        execution_scope_digest=admission.tasks[0].task_scope_digest,
        executor_id="executor-1",
    )
    checkpoint = TaskCheckpoint(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        fencing_epoch=1,
        ordinal=1,
        milestone="validated",
        payload={"rows": 10},
        created_at=GRAPH_NOW,
        payload_digest=canonical_digest({"rows": 10}),
    )
    comment = TaskComment(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        comment_id="comment-1",
        author_kind="system",
        author_id="test",
        sensitivity=ModelSensitivity.RESTRICTED,
        body="Bounded review comment.",
        created_at=GRAPH_NOW,
        body_digest=canonical_digest({"body": "Bounded review comment."}),
    )
    result = task_result()

    assert decode_graph_job(encode_graph_job(admission.job)) == admission.job
    assert decode_job_graph(encode_job_graph(admission.graph)) == admission.graph
    assert (
        decode_graph_task(encode_graph_task(admission.tasks[0])) == admission.tasks[0]
    )
    assert (
        decode_task_dependency(encode_task_dependency(admission.dependencies[0]))
        == admission.dependencies[0]
    )
    assert decode_task_attempt(encode_task_attempt(attempt)) == attempt
    assert decode_task_checkpoint(encode_task_checkpoint(checkpoint)) == checkpoint
    assert decode_task_comment(encode_task_comment(comment)) == comment
    assert decode_task_result(encode_task_result(result)) == result


def test_codec_rejects_unknown_and_missing_fields():
    encoded = encode_graph_job(graph_admission().job)
    payload = json.loads(encoded)
    payload["fields"]["unknown"] = True
    with pytest.raises(ValueError):
        decode_graph_job(json.dumps(payload))

    payload = json.loads(encoded)
    del payload["fields"]["job_id"]
    with pytest.raises(ValueError):
        decode_graph_job(json.dumps(payload))


def test_codec_revalidates_record_digests():
    encoded = encode_graph_task(graph_admission().tasks[0])
    payload = json.loads(encoded)
    payload["fields"]["task_spec_digest"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="digest"):
        decode_graph_task(json.dumps(payload))
