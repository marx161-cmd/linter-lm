"""Splits text into embedding-sized windows.

This exists for exactly one reason: embedding models have an input ceiling,
and a transcript/article/long-code-file can exceed it. Chunking here is
purely an embedding-construction detail -- chunks are never stored,
displayed, or injected on their own. A match on chunk N of a file always
resolves to "inject the whole file," never "inject chunk N."

If a file fits in one embed call, it produces exactly one chunk (itself) --
chunking is a no-op for short files, not a forced minimum granularity.
"""
from __future__ import annotations

from typing import List

from . import embeddings

DEFAULT_OVERLAP_CHARS = 200


def split_to_limit(
    text: str,
    max_chars: int = None,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[str]:
    """Split text into chunks no longer than max_chars, with overlap so a
    relevant passage straddling a chunk boundary isn't cut in half and
    diluted in both neighbors.

    Splits on paragraph/line boundaries where possible rather than mid-word,
    falling back to a hard cut only if a single "paragraph" itself exceeds
    max_chars (e.g. a minified code file with no newlines).
    """
    if max_chars is None:
        max_chars = embeddings.MAX_CHARS_PER_EMBED_CALL

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # try to break at the last paragraph or line boundary inside the window
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind("\n", start, end)
            if boundary != -1 and boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)  # ensure forward progress
    return chunks
