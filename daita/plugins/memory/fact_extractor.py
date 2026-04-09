"""
Structured fact extraction for memory ingestion.

When storing a new memory, a single LLM call extracts (entity, relation, value,
temporal_context) tuples from the raw text. These are stored in the memory's
metadata JSON under the key "extracted_facts", enabling richer temporal and
relational reasoning during recall.

Follows the same LLM call pattern as contradiction.py:
  - temperature=0.0 for deterministic output
  - JSON output, graceful fallback on any failure
  - Lazy import of LLM dependencies
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured facts from the memory text below.

MEMORY: {content}

Return a JSON array of fact objects. Each object must have:
  "entity"           — the subject (person, company, technology, material, table, etc.)
  "relation"         — the relationship or property (e.g. "prefers", "has FK", "changed to")
  "value"            — the object or value (specific and concrete)
  "temporal_context" — time qualifier if present, or null (e.g. "as of 2024-01", "before migration v5")

Rules:
- Extract every distinct fact; a single sentence may yield multiple objects.
- Entities must be specific nouns — companies, technologies, materials, standards, people. \
NOT descriptive phrases or clauses.
- Values must be specific attributes or states, NOT raw dollar amounts or large numbers \
on their own.
- Time expressions ("as of 2023", "before Q2", "since March") go in temporal_context, \
NEVER in entity or value.
- Generic words like "challenges", "efforts", "companies", "advantages" are NOT entities. \
Use the specific name instead.
- Be concise: entity and value should be 1-3 words, not full sentences.
- If there is no time qualifier, set temporal_context to null.
- Return [] if the text contains no extractable facts (e.g. it is purely procedural).
- Return JSON only, no explanation.

Example input: "User prefers dark mode. As of January the API rate limit was raised to 1000/min."
Example output:
[
  {{"entity": "user", "relation": "prefers", "value": "dark mode", "temporal_context": null}},
  {{"entity": "API rate limit", "relation": "raised to", "value": "1000/min", "temporal_context": "as of January"}}
]

WRONG — do NOT produce entities like these:
  "true", "false", "null", "enabled"        → boolean/status literals, not entities
  "unauthorized access"                     → vague state description, use the specific system or resource
  "error handling", "best practices"        → abstract process nouns, not entities
  "as of 2023"                              → this is a temporal_context
  "challenges and limitations"              → too generic
  "technical hurdles like dendrite formation"→ too long, use "dendrite formation"
  "USD 1,359.18 million"                    → raw number, not an entity"""


@dataclass
class ExtractedFact:
    entity: str
    relation: str
    value: str
    temporal_context: Optional[str] = None


class FactExtractor:
    """
    Extracts structured (entity, relation, value, temporal_context) tuples
    from raw memory content using a single LLM call.

    Falls back to an empty list on any failure so the main remember() path
    is never blocked.
    """

    def __init__(self, llm):
        """
        Args:
            llm: LLM provider instance (same interface as ContradictionChecker)
        """
        self._llm = llm

    async def extract(self, content: str) -> List[ExtractedFact]:
        """
        Extract structured facts from memory content.

        Args:
            content: Raw memory text to analyse

        Returns:
            List of ExtractedFact dataclasses, or [] on failure
        """
        if not content or not content.strip():
            return []

        prompt = EXTRACTION_PROMPT.format(content=content)

        try:
            response = await self._llm.generate(
                messages=prompt,
                max_tokens=400,
                temperature=0.0,
            )
            raw = response.strip()
            fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if fence:
                raw = fence.group(1).strip()

            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("Expected JSON array")

            facts = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entity", "")).strip()
                relation = str(item.get("relation", "")).strip()
                value = str(item.get("value", "")).strip()
                if not entity or not relation or not value:
                    continue
                temporal = item.get("temporal_context")
                facts.append(
                    ExtractedFact(
                        entity=entity,
                        relation=relation,
                        value=value,
                        temporal_context=str(temporal).strip() if temporal else None,
                    )
                )
            return facts

        except Exception as e:
            logger.debug("Fact extraction failed: %s — skipping", e)
            return []

    @staticmethod
    def facts_to_metadata(facts: List[ExtractedFact]) -> List[dict]:
        """Serialize a list of ExtractedFact into JSON-safe dicts."""
        return [asdict(f) for f in facts]
