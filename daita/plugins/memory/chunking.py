"""
Markdown chunking utilities for memory system.

Chunks markdown files intelligently by paragraphs and headers.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    content: str
    start_line: int
    end_line: int


def chunk_markdown(content: str, max_chunk_size: int = 400) -> List[Chunk]:
    """
    Chunk markdown content by paragraphs and headers.

    Args:
        content: Markdown content to chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of chunks with line numbers
    """
    if not content or not content.strip():
        return []

    lines = content.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0
    start_line = 0

    for i, line in enumerate(lines, start=1):
        line_len = len(line)

        # Check if adding this line would exceed max size
        if current_size + line_len > max_chunk_size and current_chunk:
            # Save current chunk
            chunk_content = "\n".join(current_chunk).strip()
            if chunk_content:
                chunks.append(
                    Chunk(content=chunk_content, start_line=start_line, end_line=i - 1)
                )

            # Start new chunk
            current_chunk = [line]
            current_size = line_len
            start_line = i
        else:
            # Add line to current chunk
            current_chunk.append(line)
            current_size += line_len

            # Set start line for first chunk
            if not start_line:
                start_line = i

    # Save final chunk
    if current_chunk:
        chunk_content = "\n".join(current_chunk).strip()
        if chunk_content:
            chunks.append(
                Chunk(content=chunk_content, start_line=start_line, end_line=len(lines))
            )

    return chunks
