"""
Content preprocessor for memory ingestion.

Splits raw content into two representations:
  - **storage_content**: the original text, stored in the DB and daily log
  - **index_content**: cleaned text used for embedding, dedup, fact extraction,
    and contradiction checking

The index representation strips structural noise — code blocks, inline code,
markdown formatting, bullet prefixes — so the embedding captures the factual
signal rather than the formatting template.  This prevents structurally
identical but factually different memories (e.g. two table schemas) from
appearing as near-duplicates to the embedding model.

Pure string processing — no LLM calls, no external dependencies.
"""

import re

# Fenced code blocks: ```lang\n...\n```
_FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)

# Inline code: `some code`
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")

# Markdown bold/italic: **text**, __text__, *text*, _text_
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_UNDERLINE_BOLD_RE = re.compile(r"__(.+?)__")
_ITALIC_RE = re.compile(r"\*(.+?)\*")
_UNDERLINE_ITALIC_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")

# Markdown headers: ## Header text
_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Bullet/numbered list prefixes: - item, * item, 1. item
_BULLET_RE = re.compile(r"^[\s]*[-*]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)

# Consecutive whitespace (but preserve single newlines for readability)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def preprocess_content(content: str) -> tuple:
    """Split content into storage and index representations.

    Args:
        content: Raw memory text from the agent.

    Returns:
        ``(storage_content, index_content)`` — storage is the original text,
        index is the cleaned version for embedding / dedup / fact extraction.
    """
    if not content or not content.strip():
        return (content, content)

    cleaned = content

    # 1. Remove fenced code blocks entirely — SQL, YAML, JSON, etc.
    #    These are implementation details, not facts to index.
    cleaned = _FENCED_CODE_RE.sub("", cleaned)

    # 2. Replace inline code with its text content (keep the identifier,
    #    drop the backticks — `orders.total` → orders.total)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)

    # 3. Strip markdown formatting but keep the text
    cleaned = _BOLD_RE.sub(r"\1", cleaned)
    cleaned = _UNDERLINE_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _UNDERLINE_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _HEADER_RE.sub("", cleaned)

    # 4. Strip bullet/list prefixes
    cleaned = _BULLET_RE.sub("", cleaned)
    cleaned = _NUMBERED_RE.sub("", cleaned)

    # 5. Collapse excessive whitespace
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()

    # If stripping removed everything (content was purely code), fall back
    # to original so we don't embed an empty string.
    if not cleaned:
        cleaned = content.strip()

    return (content, cleaned)
