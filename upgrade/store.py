"""SQLite-backed persistence for context files and their chunk embeddings.

Two tables:
- files: one row per ingested file (id, name, full content, tags)
- chunk_vectors: one row per embedded chunk, each pointing back at its
  parent file via file_id. A file with content under the embed-call size
  limit has exactly one chunk row; longer files have several. Either way,
  retrieval always resolves a matching chunk back to its parent file and
  returns the file's full, untouched content -- chunk rows are never read
  or surfaced on their own outside of debugging.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from .models import ContextFile

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    model TEXT NOT NULL,
    vector TEXT NOT NULL,
    PRIMARY KEY (chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_file_id ON chunk_vectors(file_id);
"""


class Store:
    """Thread-safe wrapper around a single SQLite connection."""

    def __init__(self, db_path: str):
        Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- files -------------------------------------------------------------
    def upsert_file(self, file: ContextFile) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO files (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (file.id, file.model_dump_json()),
            )
            self._conn.commit()

    def get_file(self, file_id: str) -> Optional[ContextFile]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM files WHERE id = ?", (file_id,)
            ).fetchone()
        return ContextFile.model_validate_json(row[0]) if row else None

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self._conn.execute("DELETE FROM chunk_vectors WHERE file_id = ?", (file_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def list_files(self) -> List[ContextFile]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM files").fetchall()
        return [ContextFile.model_validate_json(r[0]) for r in rows]

    def find_files_by_name_or_tag(self, term: str) -> List[ContextFile]:
        """Direct lookup for explicit pulls (e.g. a `/SpectreBoard` style
        deliberate request) -- bypasses embedding similarity entirely.
        Case-insensitive substring match against name, exact match against
        tags.
        """
        term_lower = term.strip().lower()
        if not term_lower:
            return []
        out = []
        for f in self.list_files():
            if term_lower in f.name.lower() or term_lower in [t.lower() for t in f.tags]:
                out.append(f)
        return out

    # -- chunk vectors -------------------------------------------------------
    def replace_chunk_vectors(
        self, file_id: str, model: str, vectors: List[List[float]]
    ) -> None:
        """Delete any existing chunk vectors for this file/model and insert
        the new set. Called on ingest and on re-ingest (content changed),
        so a file is never left with stale chunks from a previous version
        alongside new ones.
        """
        with self._lock:
            self._conn.execute(
                "DELETE FROM chunk_vectors WHERE file_id = ? AND model = ?",
                (file_id, model),
            )
            self._conn.executemany(
                "INSERT INTO chunk_vectors (chunk_id, file_id, chunk_index, model, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (f"{file_id}:{model}:{i}", file_id, i, model, json.dumps(vec))
                    for i, vec in enumerate(vectors)
                ],
            )
            self._conn.commit()

    def all_chunk_vectors(self, model: str) -> List[Tuple[str, int, List[float]]]:
        """Every (file_id, chunk_index, vector) row for a model. Loaded in
        full for a linear similarity scan -- fine at the file counts this is
        designed for (see contextstore README); swap for an ANN index if
        this ever needs to scale past that.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id, chunk_index, vector FROM chunk_vectors WHERE model = ?",
                (model,),
            ).fetchall()
        return [(r[0], r[1], json.loads(r[2])) for r in rows]
