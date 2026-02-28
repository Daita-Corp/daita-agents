"""
Contradiction checker for remember() in MemoryPlugin.

Checks whether a new important fact conflicts with existing high-importance
memories. Three outcomes:

- no_conflict: proceed to store normally
- evolution: real-world change (schema update, status change) — auto-replace
- contradiction: reasoning error — block storage, return explanation

Difference from the semantic dedup guard (0.92 cosine):
  Dedup catches near-identical wording (same fact rephrased).
  This checker catches logically conflicting facts that may be semantically
  distant — e.g. "organization has no incoming FKs" vs
  "FK: api_keys.org_id -> organization.id" (low cosine, but logically incompatible).

Schema changes are classified as EVOLUTION, not CONTRADICTION:
  Storing "FK: orders.customer_id -> customers.id was dropped" after the old
  "FK: orders.customer_id -> customers.id" memory triggers an auto-replace,
  so the memory stays accurate without blocking the agent.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CONFLICT_CHECK_PROMPT = """You are checking whether two memory entries conflict.

EXISTING MEMORY: {existing}
(importance: {existing_importance:.1f})

NEW FACT: {new_fact}
(importance: {new_importance:.1f})

Classify the relationship. Default to NO_CONFLICT when in doubt.

NO_CONFLICT — use this when:
  - Both statements can be true simultaneously, even partially.
  - The facts are about DIFFERENT entities (different tables, columns, or systems).
  - Multiple facts of the same TYPE are expected to coexist (e.g. multiple FK
    constraints, multiple jsonb columns in different tables, row counts for
    different tables, multiple root tables).
  EXAMPLES of NO_CONFLICT:
    "FK: orders.customer_id → customers.id" and "FK: orders.product_id → products.id"
    "Table users: ~100 rows" and "Table orders: ~500 rows"
    "Table A has jsonb column config" and "Table B has jsonb column metadata"
    "Table X is a root table" and "Table Y is a root table"

EVOLUTION — use this ONLY when:
  - The NEW FACT explicitly describes a change to the SAME SPECIFIC thing the
    existing memory describes (same table, same column, same constraint).
  - Both cannot be true about the SAME entity at the same time.
  - The new fact represents the current state after a real-world change.
  EXAMPLES of EVOLUTION (same entity, changed state):
    Existing: "FK: orders.customer_id → customers.id"
    New: "The orders.customer_id FK was dropped in migration v5"
    (same constraint, new state: it no longer exists)

CONTRADICTION — use this ONLY when:
  - The new fact conflicts with an established fact about the SAME SPECIFIC entity.
  - The conflict cannot be a real-world change (both cannot be true simultaneously).
  - The new fact is almost certainly a reasoning error.
  EXAMPLES of CONTRADICTION:
    Existing: "FK: api_keys.org_id → organization.id" (importance 0.9)
    New: "organization is a root table — no FK references point to it"
    (directly contradicts the FK that points TO organization)

Respond with JSON only, no other text:
{{"result": "NO_CONFLICT"|"EVOLUTION"|"CONTRADICTION", "reason": "one sentence"}}"""


@dataclass
class ConflictResult:
    status: str  # "no_conflict", "evolution", "contradiction"
    conflicting_chunk_id: Optional[str] = None
    conflicting_content: Optional[str] = None
    conflict_reason: Optional[str] = None


class ContradictionChecker:
    """
    Checks whether a new fact conflicts with existing high-importance memories.

    Only fires when importance >= threshold (default 0.7) to limit LLM overhead.
    Uses the same mini-LLM pattern as CloudMemoryCurator._check_supersedes():
    one call at temperature=0.0, max_tokens=120.
    """

    def __init__(self, llm, recall_fn, importance_threshold: float = 0.7):
        """
        Args:
            llm: LLM provider instance returned by create_llm_provider()
            recall_fn: Async callable matching backend.recall() signature
            importance_threshold: Only check new facts at or above this importance
        """
        self._llm = llm
        self._recall = recall_fn
        self._threshold = importance_threshold
        self._last_reason = ""

    async def check(
        self,
        new_content: str,
        new_importance: float,
    ) -> ConflictResult:
        """
        Check whether new_content conflicts with any existing important memory.

        Args:
            new_content: The fact about to be stored
            new_importance: Its importance score

        Returns:
            ConflictResult with status "no_conflict", "evolution", or "contradiction"
        """
        if new_importance < self._threshold:
            return ConflictResult(status="no_conflict")

        # Threshold rationale:
        # - 0.55 lets BM25 surface same-entity facts even when semantic format
        #   differs (e.g. "org is a root table" vs "FK: agents → organization").
        #   The shared entity name (organization) boosts BM25, pushing the hybrid
        #   score above 0.55 even if pure cosine is ~0.45-0.55.
        # - The improved LLM prompt handles same-template/different-entity pairs
        #   (e.g. "Table A jsonb col X" vs "Table B jsonb col Y") correctly as
        #   NO_CONFLICT, so false positives from lower threshold are filtered out.
        candidates = await self._recall(
            query=new_content,
            limit=5,
            score_threshold=0.55,
            min_importance=max(0.0, self._threshold - 0.1),
        )

        if not candidates:
            return ConflictResult(status="no_conflict")

        for candidate in candidates:
            existing_importance = candidate.get("metadata", {}).get("importance", 0.5)
            verdict = await self._classify(
                new_content=new_content,
                new_importance=new_importance,
                existing_content=candidate["content"],
                existing_importance=existing_importance,
            )

            if verdict in ("EVOLUTION", "CONTRADICTION"):
                return ConflictResult(
                    status=verdict.lower(),
                    conflicting_chunk_id=candidate.get("chunk_id"),
                    conflicting_content=candidate["content"],
                    conflict_reason=self._last_reason,
                )

        return ConflictResult(status="no_conflict")

    async def _classify(
        self,
        new_content: str,
        new_importance: float,
        existing_content: str,
        existing_importance: float,
    ) -> str:
        """Returns 'NO_CONFLICT', 'EVOLUTION', or 'CONTRADICTION'."""
        self._last_reason = ""
        prompt = CONFLICT_CHECK_PROMPT.format(
            existing=existing_content,
            existing_importance=existing_importance,
            new_fact=new_content,
            new_importance=new_importance,
        )
        try:
            response = await self._llm.generate(
                messages=prompt,
                max_tokens=120,
                temperature=0.0,
            )
            # Strip markdown code fences if the model wraps its JSON output
            raw = response.strip()
            fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
            if fence_match:
                raw = fence_match.group(1).strip()
            data = json.loads(raw)
            self._last_reason = data.get("reason", "")
            return data.get("result", "NO_CONFLICT").upper()
        except Exception as e:
            logger.debug("Contradiction check failed: %s — defaulting to no_conflict", e)
            return "NO_CONFLICT"
