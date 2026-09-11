"""Shared helpers extracted from ``test_effect_contracts.py``."""

from __future__ import annotations

from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationGrantPolicy,
    Capability,
    EffectEvidenceBasis,
    EffectReceiptPolicy,
    OperationalEffect,
)

_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "string", "minLength": 1, "maxLength": 64}},
    "required": ["target"],
    "additionalProperties": False,
}
_RECEIPT = EffectReceiptPolicy(
    "test.action", _SCHEMA, EffectEvidenceBasis.SERVER_REPORTED
)
_GRANT = AutomationGrantPolicy("test.target", _SCHEMA)


def _capability() -> Capability:
    return Capability(
        id="test.action",
        description="One unrelated external action.",
        input_schema=_SCHEMA,
        output_kind="test.action.result",
        output_schema=_SCHEMA,
        executor_id="test.action.executor",
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.EXTERNAL_ACTION,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        automation_grant_policy=_GRANT,
        effect_receipt_policy=_RECEIPT,
    )
