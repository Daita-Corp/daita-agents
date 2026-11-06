"""Measures the fan-out detector against an independently labelled SQL corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daita.domains.data.sql.query_checks import fan_out_warnings

pytestmark = pytest.mark.contract

_CORPUS = Path(__file__).parents[1] / "fixtures" / "fan-out-corpus" / "corpus.json"


def test_fan_out_detector_recall_and_false_positive_rate() -> None:
    corpus: dict[str, Any] = json.loads(_CORPUS.read_text())
    keys = corpus["keys"]
    flagged = {
        query["id"]: bool(fan_out_warnings(query["sql"], "postgresql", keys))
        for query in corpus["queries"]
    }
    positives = [q["id"] for q in corpus["queries"] if q["fan_out"]]
    negatives = [q["id"] for q in corpus["queries"] if not q["fan_out"]]
    missed = [i for i in positives if not flagged[i]]
    false_alarms = [i for i in negatives if flagged[i]]
    recall = 1 - len(missed) / len(positives)
    false_positive_rate = len(false_alarms) / len(negatives)
    assert recall >= 0.90 and false_positive_rate <= 0.05, (
        f"recall={recall:.2f} (missed {missed}); "
        f"false_positive_rate={false_positive_rate:.2f} (flagged {false_alarms})"
    )
