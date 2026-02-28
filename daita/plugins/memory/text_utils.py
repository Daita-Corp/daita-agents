"""
Text utilities for keyword extraction and phrase matching.

Provides utilities for extracting important terms from queries and
detecting exact phrase matches in content.
"""

import re
from typing import List, Set


# Common stop words to exclude from keyword extraction
STOP_WORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'don',
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
    'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'might',
    'more', 'most', 'must', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off',
    'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 'that',
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves',
    # Question words (common in queries)
    'tell', 'remember', 'recall', 'find', 'show', 'get', 'give'
}


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Converts to lowercase and removes extra whitespace while preserving
    basic punctuation for version numbers and technical terms.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Collapse multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_keywords(query: str) -> List[str]:
    """
    Extract important terms from query for keyword matching.

    Prioritizes:
    1. Version numbers (e.g., "0.7.0", "v2.1", "2.0.0")
    2. Proper nouns and technical terms (capitalized words)
    3. Significant words (length >= 3)
    4. Removes common stop words

    Args:
        query: Natural language query

    Returns:
        List of important keywords
    """
    keywords = []

    # First pass: find version numbers with special regex
    # Matches: 0.7.0, v2.1, 2.0.0, version 1.2, etc.
    version_pattern = r'\b(?:v(?:ersion)?\s*)?(\d+\.\d+(?:\.\d+)?(?:-[a-z0-9]+)?)\b'
    versions = re.findall(version_pattern, query, re.IGNORECASE)
    keywords.extend(versions)

    # Second pass: tokenize and filter
    # Split on whitespace and common punctuation (but preserve dots in versions)
    tokens = re.findall(r'\b[\w.-]+\b', query)

    for token in tokens:
        token_lower = token.lower()

        # Skip if already captured as version
        if token in versions:
            continue

        # Skip stop words
        if token_lower in STOP_WORDS:
            continue

        # Skip very short words (unless they're uppercase acronyms)
        if len(token) < 3 and not token.isupper():
            continue

        # Keep significant terms
        keywords.append(token_lower)

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords


def contains_exact_phrase(query: str, content: str) -> bool:
    """
    Check if content contains exact phrases from query.

    Looks for multi-word sequences (2+ words) that appear verbatim in content.
    Case-insensitive matching.

    Args:
        query: Query string to extract phrases from
        content: Content to search within

    Returns:
        True if at least one exact phrase (2+ words) is found
    """
    # Normalize both texts
    query_norm = normalize_text(query)
    content_norm = normalize_text(content)

    # Extract potential phrases (2+ consecutive words)
    # Split on punctuation but keep words together
    query_tokens = re.findall(r'\b[\w.-]+\b', query_norm)

    # Generate all 2-word, 3-word, and 4-word phrases
    for phrase_len in [4, 3, 2]:  # Check longer phrases first
        for i in range(len(query_tokens) - phrase_len + 1):
            phrase = ' '.join(query_tokens[i:i + phrase_len])

            # Skip if phrase is mostly stop words
            phrase_words = phrase.split()
            non_stop_count = sum(1 for w in phrase_words if w not in STOP_WORDS)
            if non_stop_count < 2:  # Need at least 2 meaningful words
                continue

            # Check if phrase exists in content
            if phrase in content_norm:
                return True

    return False


def get_exact_phrases(query: str, content: str) -> List[str]:
    """
    Get list of exact phrases from query that appear in content.

    Useful for debugging and transparency.

    Args:
        query: Query string
        content: Content to search within

    Returns:
        List of matching phrases
    """
    matches = []

    query_norm = normalize_text(query)
    content_norm = normalize_text(content)

    query_tokens = re.findall(r'\b[\w.-]+\b', query_norm)

    # Generate all 2-word, 3-word, and 4-word phrases
    for phrase_len in [4, 3, 2]:
        for i in range(len(query_tokens) - phrase_len + 1):
            phrase = ' '.join(query_tokens[i:i + phrase_len])

            # Skip if phrase is mostly stop words
            phrase_words = phrase.split()
            non_stop_count = sum(1 for w in phrase_words if w not in STOP_WORDS)
            if non_stop_count < 2:
                continue

            if phrase in content_norm:
                matches.append(phrase)

    return matches


def clean_memory_content(text: str) -> str:
    """
    Strip LLM-generated meta prefixes from memory content.

    Removes common LLM narrative prefixes that waste embedding space
    and reduce semantic quality. Examples: "Today I learned", "I discovered",
    "Note that", etc.

    This saves ~20% embedding costs and improves recall accuracy.

    Args:
        text: Raw memory content from LLM

    Returns:
        Cleaned content with prefixes removed
    """
    # Common LLM prefixes to strip (case-insensitive)
    prefixes = [
        r'^today i learned that\s*',
        r'^today i learned\s*',
        r'^i learned that\s*',
        r'^i learned\s*',
        r'^i discovered that\s*',
        r'^i discovered\s*',
        r'^note that\s*',
        r'^it\'?s important to (?:note|remember) that\s*',
        r'^it\'?s important to (?:note|remember)\s*',
        r'^remember that\s*',
        r'^(?:please )?(?:note|remember):\s*',
        r'^the user (?:said|mentioned|told me) that\s*',
        r'^the user (?:said|mentioned|told me)\s*',
        r'^user preference:\s*',
        r'^key (?:point|insight|learning):\s*',
        r'^important:\s*',
    ]

    cleaned = text.strip()

    # Try each prefix pattern
    for pattern in prefixes:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove leading punctuation artifacts (colons, dashes)
    cleaned = re.sub(r'^[\s:,-]+', '', cleaned)

    # Capitalize first letter if needed
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned.strip()
