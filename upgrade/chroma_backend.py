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
# Per-file ceiling: cap very large files so one file can't crowd out others.
# 15K chars ≈ 3 750 tokens — comfortably fits any typical homelab doc/script.
MAX_FILE_CHARS = int(os.environ.get("CONTEXTSTORE_MAX_FILE_CHARS", "15000"))


def _read_file(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:MAX_FILE_CHARS]


class ChromaContextStore:
    """Retrieves context from the mcp-hub ChromaDB collections.

    Chunks are used only for matching. When a chunk from file X clears the
    similarity threshold, the *entire file* is injected — same behaviour as
    the original SQLite ContextStore. Multiple chunks from the same file
    collapse to one hit (highest similarity wins).
    """

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
        max_tokens: int = 8192,
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

        # Query all collections; ask for more candidates so deduplication
        # doesn't drop us below top_k unique files.
        n_candidates = top_k * 4
        best_per_file: dict[str, tuple[float, str]] = {}  # path -> (sim, col_name)

        for col_name in CHROMA_COLLECTIONS:
            try:
                col = client.get_collection(col_name)
                raw = col.query(
                    query_embeddings=[query_vector],
                    n_results=n_candidates,
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
                    if sim < threshold:
                        continue
                    path = str(meta.get("path", ""))
                    if not path:
                        continue
                    existing_sim, _ = best_per_file.get(path, (0.0, ""))
                    if sim > existing_sim:
                        best_per_file[path] = (sim, col_name)
            except Exception:
                continue

        # Sort unique files by best chunk similarity, take top_k
        ranked = sorted(best_per_file.items(), key=lambda kv: -kv[1][0])

        hits: List[RetrievalHit] = []
        used_tokens = 0
        truncated = False

        for path, (sim, col_name) in ranked:
            if len(hits) >= top_k:
                break
            if not Path(path).exists():
                continue
            text = _read_file(path)
            if not text:
                continue
            cost = max(1, len(text) // CHARS_PER_TOKEN)
            if used_tokens + cost > max_tokens:
                truncated = True
                continue
            hits.append(RetrievalHit(
                file_id=f"{col_name}:{path}",
                name=Path(path).name,
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
