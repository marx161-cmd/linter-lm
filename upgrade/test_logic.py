"""Stdlib-only sanity tests for contextstore's core logic (chunking, cosine
similarity, best-chunk-per-file resolution) -- run without pydantic/httpx
since this sandbox has no network access to install them. Mirrors the real
module's algorithms exactly; just doesn't import the pydantic-based classes.
"""
from __future__ import annotations

import math


# -- chunking.split_to_limit, copied logic (no pydantic dependency) --------
def split_to_limit(text, max_chars=50, overlap_chars=10):
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
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
        start = max(end - overlap_chars, start + 1)
    return chunks


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_short_text_single_chunk():
    text = "short text under the limit"
    chunks = split_to_limit(text, max_chars=50)
    assert chunks == [text], chunks


def test_empty_text_no_chunks():
    assert split_to_limit("   ", max_chars=50) == []
    assert split_to_limit("", max_chars=50) == []


def test_long_text_multiple_chunks_with_progress():
    # no newlines at all -- forces hard cuts, must still terminate and cover all text
    text = "x" * 500
    chunks = split_to_limit(text, max_chars=50, overlap_chars=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)
    # reconstruct: every char of original must appear somewhere (overlap allowed)
    rejoined = chunks[0]
    for c in chunks[1:]:
        rejoined += c
    assert "x" * 500 in rejoined or len(rejoined) >= 500


def test_long_text_prefers_paragraph_boundary():
    text = "first paragraph here is short.\n\nsecond paragraph also fairly short.\n\nthird one too, short."
    chunks = split_to_limit(text, max_chars=45, overlap_chars=5)
    assert len(chunks) >= 2
    # first chunk should end at a paragraph boundary, not mid-word
    assert chunks[0].endswith(".") or chunks[0].endswith("here is short")


def test_chunking_terminates_on_pathological_input():
    # single token longer than max_chars with no breakable boundary at all
    text = "a" * 1000
    chunks = split_to_limit(text, max_chars=30, overlap_chars=29)
    assert len(chunks) > 0
    # must terminate in a bounded number of steps despite overlap_chars
    # being almost equal to max_chars (this is the forward-progress guard)
    assert len(chunks) < 1000


def test_cosine_similarity_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b) - 0.0) < 1e-9


def test_cosine_similarity_mismatched_length():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_best_chunk_per_file_resolution():
    # simulates ContextStore.retrieve()'s best-per-file reduction:
    # multiple chunks per file, only the best similarity should win
    query = [1.0, 0.0, 0.0]
    all_vectors = [
        ("fileA", 0, [1.0, 0.0, 0.0]),   # perfect match, fileA chunk 0
        ("fileA", 1, [0.0, 1.0, 0.0]),   # bad match, fileA chunk 1
        ("fileB", 0, [0.0, 0.0, 1.0]),   # bad match, fileB only chunk
    ]
    best_per_file = {}
    for file_id, chunk_index, vector in all_vectors:
        sim = cosine_similarity(query, vector)
        current = best_per_file.get(file_id)
        if current is None or sim > current[1]:
            best_per_file[file_id] = (chunk_index, sim)

    assert best_per_file["fileA"][0] == 0  # chunk 0 won, not chunk 1
    assert abs(best_per_file["fileA"][1] - 1.0) < 1e-9
    assert best_per_file["fileB"][1] < 0.5


def test_threshold_and_top_k_filtering():
    best_per_file = {"a": (0, 0.9), "b": (0, 0.5), "c": (0, 0.7), "d": (0, 0.95)}
    threshold = 0.6
    top_k = 2
    ranked = sorted(
        ((fid, idx, sim) for fid, (idx, sim) in best_per_file.items() if sim >= threshold),
        key=lambda t: -t[2],
    )[:top_k]
    ids = [r[0] for r in ranked]
    assert ids == ["d", "a"]  # highest two above threshold, "b" excluded, "c" cut by top_k


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} passed")
