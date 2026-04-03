"""
LLM-based reranker for memory recall results.

After hybrid search returns candidates, the reranker makes a single LLM call to
reorder them by actual relevance to the query. This catches cases where formula-
based scoring ranks a superficially similar but semantically wrong memory above a
deeply relevant one.

Follows the same LLM call pattern as contradiction.py:
  - temperature=0.0 for deterministic output
  - JSON output, graceful fallback on any failure
  - Single call over a small candidate set (no parallel overhead)
"""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are ranking memory retrieval results by relevance to a query.

QUERY: {query}

CANDIDATES (numbered 0-{last_idx}):
{candidates}

Return a JSON array of candidate indices ordered from MOST to LEAST relevant.
Include all {count} indices. Do not include any explanation — JSON only.

Example for 3 candidates: [2, 0, 1]"""


class MemoryReranker:
    """
    Reranks a list of memory recall candidates using a single LLM call.

    Takes the top_n results from hybrid search and asks the LLM to reorder
    them by true relevance. Falls back to the original order on any failure.
    """

    def __init__(self, llm, top_n: int = 15):
        """
        Args:
            llm: LLM provider instance (same interface as ContradictionChecker)
            top_n: Maximum candidates to send to the LLM for reranking
        """
        self._llm = llm
        self._top_n = top_n

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        final_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates by LLM judgment and return the top final_limit.

        Args:
            query: The original recall query
            candidates: Memory results from hybrid search (dicts with 'content' key)
            final_limit: How many results to return after reranking

        Returns:
            Reranked and truncated list. Falls back to original order on failure.
        """
        if not candidates:
            return candidates

        pool = candidates[: self._top_n]

        if len(pool) == 1:
            return pool[:final_limit]

        candidate_lines = "\n".join(
            f"[{i}] {c['content'][:300]}" for i, c in enumerate(pool)
        )
        prompt = RERANK_PROMPT.format(
            query=query,
            last_idx=len(pool) - 1,
            candidates=candidate_lines,
            count=len(pool),
        )

        try:
            response = await self._llm.generate(
                messages=prompt,
                max_tokens=200,
                temperature=0.0,
            )
            raw = response.strip()
            # Strip markdown code fences if present
            fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if fence:
                raw = fence.group(1).strip()

            order = json.loads(raw)
            if not isinstance(order, list):
                raise ValueError("Expected JSON array")

            # Validate indices and rebuild ordered list
            seen = set()
            reranked = []
            for idx in order:
                if isinstance(idx, int) and 0 <= idx < len(pool) and idx not in seen:
                    reranked.append(pool[idx])
                    seen.add(idx)

            # Append any candidates the LLM omitted (safety net)
            for i, c in enumerate(pool):
                if i not in seen:
                    reranked.append(c)

            return reranked[:final_limit]

        except Exception as e:
            logger.debug("Memory reranking failed: %s — returning original order", e)
            return candidates[:final_limit]
