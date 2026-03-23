"""
Query-type routing for memory recall.

Classifies incoming queries into temporal, contextual, or factual types and
returns search parameters tuned to that type. No LLM call required — purely
regex-based classification.

Types:
  temporal   — questions about time, change, history, recency
  contextual — questions about reasons, explanations, context
  factual    — direct lookups (default)
"""

import re
from dataclasses import dataclass


@dataclass
class QueryRoute:
    query_type: str          # "temporal", "contextual", "factual"
    score_threshold: float   # minimum score to include a result
    semantic_weight: float   # weight for cosine similarity in hybrid scoring
    keyword_weight: float    # weight for BM25 in hybrid scoring
    temporal_boost: float    # extra score bonus for recent memories (0 = disabled)


class QueryRouter:
    """Classify a query string and return retrieval routing parameters."""

    # Temporal: queries about change, history, recency, or time-bounded facts
    _TEMPORAL_PATTERNS = re.compile(
        r"\b(before|after|when did|last time|previously|earlier|ago|"
        r"recent|recently|latest|history|changed|updated|used to|"
        r"was once|became|no longer|anymore|still|switched|replaced|"
        r"old|former|previous|prior|originally|at the time|back then)\b",
        re.IGNORECASE,
    )

    # Contextual: queries about reasoning, explanation, motivation
    _CONTEXTUAL_PATTERNS = re.compile(
        r"\b(why did|why does|why is|why was|why were|how does|how did|"
        r"explain|explanation|reason|because|context|background|motivation|"
        r"purpose|rationale|what caused|what led|understand|clarify|"
        r"chosen|selected|decided|opted)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def classify(query: str) -> QueryRoute:
        """
        Classify a query string and return routing parameters.

        Temporal queries use a lower threshold and recency boost so that
        recently-updated facts rank higher.

        Contextual queries use a higher semantic weight and lower threshold
        to cast a wider semantic net.

        Factual queries (default) use the current hybrid weights.
        """
        if QueryRouter._TEMPORAL_PATTERNS.search(query):
            return QueryRoute(
                query_type="temporal",
                score_threshold=0.5,
                semantic_weight=0.6,
                keyword_weight=0.4,
                temporal_boost=0.15,
            )

        if QueryRouter._CONTEXTUAL_PATTERNS.search(query):
            return QueryRoute(
                query_type="contextual",
                score_threshold=0.45,
                semantic_weight=0.75,
                keyword_weight=0.25,
                temporal_boost=0.0,
            )

        # Default: factual — preserve existing behaviour exactly
        return QueryRoute(
            query_type="factual",
            score_threshold=0.6,
            semantic_weight=0.6,
            keyword_weight=0.4,
            temporal_boost=0.0,
        )
