"""ChromaDB-backed retrieval for linter-lm context injection.

Drop-in replacement for ContextStore: same retrieve() interface,
but queries the shared mcp-hub ChromaDB collections instead of the
local SQLite store. No separate ingestion pipeline needed.

Configure via env vars (set in .env.local):
  CONTEXTSTORE_BACKEND=chroma
  CONTEXTSTORE_CHROMA_DIR=/home/comrade/.local/share/mcp-hub/chroma
  CONTEXTSTORE_CHROMA_COLLECTIONS=homelab,bin,notes
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from . import embeddings
from .models import RetrievalHit, RetrievalResult

CHROMA_DIR = os.environ.get(
    "CONTEXTSTORE_CHROMA_DIR",
    "/home/comrade/.local/share/mcp-hub/chroma",
)
CHROMA_COLLECTIONS = [
    c.strip()
    for c in os.environ.get("CONTEXTSTORE_CHROMA_COLLECTIONS", "homelab,bin,notes").split(",")
    if c.strip()
]
DEFAULT_THRESHOLD = float(os.environ.get("CONTEXTSTORE_SIMILARITY_THRESHOLD", "0.5"))
DEFAULT_TOP_K = int(os.environ.get("CONTEXTSTORE_TOP_K", "3"))
CHARS_PER_TOKEN = 4
MAX_LINE_CHARS = 6_000


def _read_lines(path: str, start_line: int, end_line: int) -> str:
    lines: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if idx < start_line:
                    continue
                if idx > end_line:
                    break
                lines.append(line.rstrip("\n"))
    except OSError:
        return ""
    text = "\n".join(lines)
    return text[:MAX_LINE_CHARS]


class ChromaContextStore:
    """Retrieves context chunks from the mcp-hub ChromaDB collections."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            import chromadb  # deferred so missing package is a soft failure
            self._client = chromadb.PersistentClient(path=CHROMA_DIR)
        return self._client

    async def retrieve(
        self,
        query_text: str,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int = 2048,
    ) -> RetrievalResult:
        preview = query_text[:80]
        empty = RetrievalResult(query_preview=preview, hits=[], tokens_estimate=0, truncated=False)

        if not embeddings.is_enabled():
            return empty

        query_vector = await embeddings.embed(query_text)
        if query_vector is None:
            return empty

        try:
            client = self._get_client()
        except Exception:
            return empty

        # Query all collections, merge by similarity
        candidates: list[tuple[float, str, dict]] = []
        for col_name in CHROMA_COLLECTIONS:
            try:
                col = client.get_collection(col_name)
                raw = col.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    include=["metadatas", "distances"],
                )
                for meta, dist in zip(
                    raw.get("metadatas", [[]])[0],
                    raw.get("distances", [[]])[0],
                ):
                    if not meta:
                        continue
                    # ChromaDB cosine distance: 0=identical → similarity = 1 - dist
                    sim = 1.0 - float(dist)
                    if sim >= threshold:
                        candidates.append((sim, col_name, meta))
            except Exception:
                continue

        candidates.sort(key=lambda x: -x[0])

        hits: List[RetrievalHit] = []
        used_tokens = 0
        truncated = False

        for sim, col_name, meta in candidates[:top_k]:
            path = str(meta.get("path", ""))
            if not path or not Path(path).exists():
                continue
            start = int(meta.get("start_line", 1))
            end = int(meta.get("end_line", start))
            text = _read_lines(path, start, end)
            if not text:
                continue
            cost = max(1, len(text) // CHARS_PER_TOKEN)
            if used_tokens + cost > max_tokens:
                truncated = True
                continue
            hits.append(RetrievalHit(
                file_id=f"{col_name}:{path}:{start}",
                name=f"{Path(path).name}:{start}-{end}",
                content=text,
                best_similarity=round(sim, 4),
                matched_chunk_index=0,
                tags=[col_name],
            ))
            used_tokens += cost

        return RetrievalResult(
            query_preview=preview,
            hits=hits,
            tokens_estimate=used_tokens,
            truncated=truncated,
        )

    # Stub — ChromaDB backend is read-only from linter-lm's perspective.
    def list_files(self):
        return []
