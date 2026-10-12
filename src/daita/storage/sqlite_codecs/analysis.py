"""Strict current analytical evidence serialization."""

from ..._json import FrozenJsonObject
from ...loop.analysis import AnalysisEvidence
from .common import (
    dump_payload,
    load_payload,
    plain_encode,
    record,
    record_fields,
    text,
)


def encode_analysis_evidence(value: AnalysisEvidence) -> str:
    return dump_payload(
        record(
            "AnalysisEvidence",
            {
                "run_id": value.run_id,
                "evidence_id": value.evidence_id,
                "kind": value.kind,
                "facts": plain_encode(value.facts),
            },
        )
    )


def decode_analysis_evidence(value: str) -> AnalysisEvidence:
    fields = record_fields(
        load_payload(value),
        "AnalysisEvidence",
        ("run_id", "evidence_id", "kind", "facts"),
    )
    if not isinstance(fields["facts"], dict):
        raise ValueError("Analysis evidence facts must be an object")
    return AnalysisEvidence(
        run_id=text(fields["run_id"], "run id"),
        evidence_id=text(fields["evidence_id"], "evidence id"),
        kind=text(fields["kind"], "kind"),
        facts=FrozenJsonObject.from_mapping(fields["facts"]),
    )
