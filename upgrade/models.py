"""Data models for contextstore.

Deliberately small. This is not a SillyTavern-World-Info-style activation
engine (no keys/cooldown/sticky/probability/priority) -- the proxy sees
every message before the model does, so there's no tool-call budget to
economize and no "model decides whether to look this up" step to gate.
Relevance is decided entirely by embedding similarity at query time.
"""
from __future__ import annotations

import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ContextFile(BaseModel):
    """A single ingested file: the full text plus identifying metadata.

    This is what actually gets injected into the prompt on a match -- in
    full, untouched. Chunking only ever happens on the embedding side; the
    stored content here is never split or summarized.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    source_path: Optional[str] = Field(default=None, description="Original file path, if ingested from disk")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ChunkMatch(BaseModel):
    """A single chunk that cleared the similarity threshold during a query."""

    file_id: str
    chunk_index: int
    similarity: float


class RetrievalHit(BaseModel):
    """A file that matched, with enough detail to debug why."""

    file_id: str
    name: str
    content: str
    best_similarity: float
    matched_chunk_index: int
    tags: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Result of a single query against the store."""

    query_preview: str
    hits: List[RetrievalHit]
    tokens_estimate: int
    truncated: bool
