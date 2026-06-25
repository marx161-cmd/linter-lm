"""Thin async HTTP client for a resident llama-server instance running an
embedding model (e.g. Qwen3-Embedding-0.6B-GGUF), started separately and kept
loaded in VRAM:

    llama-server -m /path/to/qwen3-embedding-0.6b.gguf --embedding \
        -ngl 999 --device ROCm0 --port 8081 --host 127.0.0.1

This module only ever sends/receives plain text <-> float vectors. It does
not know about files, chunks, or the store -- that separation is deliberate
so the embedder can be swapped (different model, different host, even a
different binary) without touching retrieval logic.
"""
from __future__ import annotations

import os
from typing import List, Optional

import httpx

EMBEDDING_ENDPOINT = os.environ.get(
    "CONTEXTSTORE_EMBEDDING_ENDPOINT", "http://127.0.0.1:18084/v1/embeddings"
)
EMBEDDING_TIMEOUT = float(os.environ.get("CONTEXTSTORE_EMBEDDING_TIMEOUT", "30"))

# llama-server's default --ctx-size for an embedding model bounds how much
# text a single embed() call can usefully cover. This is intentionally
# conservative (chars, not tokens -- no tokenizer dependency here) and is
# what drives chunk sizing in chunking.py.
MAX_CHARS_PER_EMBED_CALL = int(os.environ.get("CONTEXTSTORE_MAX_CHARS_PER_EMBED", "3000"))


def is_enabled() -> bool:
    return bool(EMBEDDING_ENDPOINT)


async def embed(text: str) -> Optional[List[float]]:
    """Embed a single string. Returns None if the embedder is unreachable or
    the request fails -- failures are swallowed deliberately, same rationale
    as the original lorebook_mcp embeddings.py: retrieval is an enhancement
    layered in front of the model, not something that should break the
    request pipeline if the embedding server is briefly down.
    """
    if not EMBEDDING_ENDPOINT or not text.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
            resp = await client.post(EMBEDDING_ENDPOINT, json={"input": text})
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return None


async def embed_many(texts: List[str]) -> List[Optional[List[float]]]:
    """Embed multiple strings. llama-server's /v1/embeddings accepts a list
    input, so a batch ingestion call (many chunks from one file) is one
    request rather than N -- meaningfully faster for ingesting a large file.
    Falls back to per-string calls if the batch request fails, so a single
    malformed chunk can't fail the whole batch.
    """
    if not EMBEDDING_ENDPOINT or not texts:
        return [None] * len(texts)
    try:
        async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
            resp = await client.post(EMBEDDING_ENDPOINT, json={"input": texts})
            resp.raise_for_status()
            data = resp.json()
            rows = data["data"]
            if len(rows) != len(texts):
                raise ValueError("embedding count mismatch")
            return [row["embedding"] for row in rows]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        results = []
        for t in texts:
            results.append(await embed(t))
        return results


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
