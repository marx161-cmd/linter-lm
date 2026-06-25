"""ContextStore: ingest arbitrary files, retrieve by embedding similarity.

This is the thing the LinteR-LM proxy imports directly -- no MCP, no
tool-call, no model-visible step at all. The proxy calls `retrieve()` on
the incoming user message before forwarding the request, and splices any
hits into the outbound payload. The model is never told this happened,
same property as the existing sampler-patch mechanism in lintr_core.py.

Deliberately does not implement keyword triggers, cooldown, sticky,
probability, or priority -- those exist in lorebook_mcp to simulate
SillyTavern's World Info behavior for a tool-calling context. There's no
tool call here and no per-turn budget to economize, so relevance is decided
purely by embedding similarity at query time.
"""
from __future__ import annotations

import os
from typing import List, Optional

from . import chunking, embeddings
from .models import ContextFile, RetrievalHit, RetrievalResult
from .store import Store

DEFAULT_DB_PATH = os.environ.get("CONTEXTSTORE_DB_PATH", "./contextstore.db")
DEFAULT_THRESHOLD = float(os.environ.get("CONTEXTSTORE_SIMILARITY_THRESHOLD", "0.5"))
DEFAULT_TOP_K = int(os.environ.get("CONTEXTSTORE_TOP_K", "3"))
CHARS_PER_TOKEN_ESTIMATE = 4  # same crude heuristic as lorebook_mcp's engine.py


class ContextStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, model: Optional[str] = None):
        self.store = Store(db_path)
        # "model" here just namespaces vectors in case you ever swap embedders
        # without wanting old/new vectors compared against each other.
        self.model = model or os.environ.get("CONTEXTSTORE_EMBED_MODEL_NAME", "default")

    # -- ingestion -----------------------------------------------------------
    async def ingest_text(
        self,
        name: str,
        content: str,
        tags: Optional[List[str]] = None,
        source_path: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> ContextFile:
        """Ingest a raw text blob: store the full content untouched, chunk it
        only for embedding, embed every chunk, store all chunk vectors
        pointing back at this file's id. Re-ingesting the same file_id
        replaces its content and chunk vectors cleanly (no stale chunks left
        behind from a previous version).
        """
        existing = self.store.get_file(file_id) if file_id else None
        if existing is not None:
            file = ContextFile(
                id=existing.id,
                name=name,
                content=content,
                tags=tags or [],
                source_path=source_path,
                created_at=existing.created_at,
            )
        else:
            kwargs = dict(name=name, content=content, tags=tags or [], source_path=source_path)
            if file_id:
                kwargs["id"] = file_id
            file = ContextFile(**kwargs)
        self.store.upsert_file(file)

        chunks = chunking.split_to_limit(content)
        if not chunks:
            return file
        vectors_raw = await embeddings.embed_many(chunks)
        vectors = [v for v in vectors_raw if v is not None]
        if vectors:
            self.store.replace_chunk_vectors(file.id, self.model, vectors)
        return file

    async def ingest_file(
        self, path: str, tags: Optional[List[str]] = None, name: Optional[str] = None
    ) -> ContextFile:
        """Convenience wrapper: read a file from disk and ingest it."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return await self.ingest_text(
            name=name or os.path.basename(path),
            content=content,
            tags=tags,
            source_path=path,
        )

    def delete(self, file_id: str) -> bool:
        return self.store.delete_file(file_id)

    def list_files(self) -> List[ContextFile]:
        return self.store.list_files()

    # -- retrieval -----------------------------------------------------------
    async def retrieve(
        self,
        query_text: str,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int = 2048,
    ) -> RetrievalResult:
        """Embed query_text once, compare against every stored chunk vector,
        keep the best-matching chunk per file, return files whose best chunk
        clears `threshold`, highest similarity first, capped at `top_k` and
        trimmed to `max_tokens`.

        Returns an empty result (never raises) if the embedder is down or
        the store is empty -- retrieval failures degrade to "inject
        nothing," they never break the request.
        """
        preview = query_text[:80]
        if not embeddings.is_enabled():
            return RetrievalResult(query_preview=preview, hits=[], tokens_estimate=0, truncated=False)

        query_vector = await embeddings.embed(query_text)
        if query_vector is None:
            return RetrievalResult(query_preview=preview, hits=[], tokens_estimate=0, truncated=False)

        all_vectors = self.store.all_chunk_vectors(self.model)
        if not all_vectors:
            return RetrievalResult(query_preview=preview, hits=[], tokens_estimate=0, truncated=False)

        best_per_file: dict[str, tuple[int, float]] = {}
        for file_id, chunk_index, vector in all_vectors:
            sim = embeddings.cosine_similarity(query_vector, vector)
            current = best_per_file.get(file_id)
            if current is None or sim > current[1]:
                best_per_file[file_id] = (chunk_index, sim)

        ranked = sorted(
            ((fid, idx, sim) for fid, (idx, sim) in best_per_file.items() if sim >= threshold),
            key=lambda t: -t[2],
        )[:top_k]

        hits: List[RetrievalHit] = []
        used_tokens = 0
        truncated = False
        for file_id, chunk_index, sim in ranked:
            file = self.store.get_file(file_id)
            if file is None:
                continue
            cost = max(1, len(file.content) // CHARS_PER_TOKEN_ESTIMATE)
            if used_tokens + cost > max_tokens:
                truncated = True
                continue
            hits.append(
                RetrievalHit(
                    file_id=file.id,
                    name=file.name,
                    content=file.content,
                    best_similarity=round(sim, 4),
                    matched_chunk_index=chunk_index,
                    tags=file.tags,
                )
            )
            used_tokens += cost

        return RetrievalResult(
            query_preview=preview, hits=hits, tokens_estimate=used_tokens, truncated=truncated
        )

    def retrieve_explicit(self, term: str) -> List[ContextFile]:
        """Direct name/tag lookup, bypassing embedding similarity entirely --
        for a deliberate `/term` style pull where the user is naming exactly
        what they want, not describing it. No threshold, no top_k: returns
        everything that matches.
        """
        return self.store.find_files_by_name_or_tag(term)
