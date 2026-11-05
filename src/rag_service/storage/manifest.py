"""Local ingestion audit and embedding cache; Qdrant owns the authoritative publication alias."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from rag_service.schemas import fingerprint


class Manifest:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                fingerprint TEXT,
                documents INTEGER NOT NULL DEFAULT 0,
                chunks INTEGER NOT NULL DEFAULT 0,
                config_json TEXT NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS active (
                collection TEXT PRIMARY KEY,
                generation_id TEXT NOT NULL REFERENCES generations(id)
            );
            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL
            );
        """)

    def close(self) -> None:
        self._db.close()

    def recover(self) -> None:
        """Call only while holding the exclusive ingestion lock."""
        with self._db:
            self._db.execute(
                "UPDATE generations SET status='failed', error='Interrupted ingestion' "
                "WHERE status='indexing'"
            )

    def begin(self, generation: str, collection: str, config: dict[str, Any]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO generations (id, collection, status, config_json) "
                "VALUES (?, ?, 'indexing', ?)",
                (generation, collection, json.dumps(config, sort_keys=True)),
            )

    def finish(
        self, generation: str, corpus_hash: str, documents: int, chunks: int, *, publish: bool
    ) -> None:
        with self._db:
            self._db.execute(
                "UPDATE generations SET status=?, fingerprint=?, documents=?, chunks=?, "
                "error=NULL WHERE id=?",
                ("ready" if publish else "unchanged", corpus_hash, documents, chunks, generation),
            )
            if publish:
                self._db.execute(
                    "INSERT INTO active SELECT collection, id FROM generations WHERE id=? "
                    "ON CONFLICT(collection) DO UPDATE SET generation_id=excluded.generation_id",
                    (generation,),
                )

    def fail(self, generation: str, error: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE generations SET status='failed', error=? WHERE id=?", (error, generation)
            )

    def active(self, collection: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT g.* FROM generations g JOIN active a ON g.id=a.generation_id "
            "WHERE a.collection=?",
            (collection,),
        ).fetchone()
        return dict(row) if row else None

    def publication_pending(self, generation: str, error: str | None = None) -> None:
        """A remote alias request may succeed even if its acknowledgment is lost."""
        with self._db:
            self._db.execute(
                "UPDATE generations SET status='publication_pending', error=? WHERE id=?",
                (error, generation),
            )

    def pending_publications(self, collection: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM generations WHERE collection=? AND status='publication_pending'",
            (collection,),
        )
        return [dict(row) for row in rows]

    def generations(self, collection: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM generations WHERE collection=? ORDER BY rowid DESC", (collection,)
        )
        return [dict(row) for row in rows]

    def cached_vector(self, namespace: str, text: str) -> list[float] | None:
        row = self._db.execute(
            "SELECT vector_json FROM embedding_cache WHERE key=?", (fingerprint([namespace, text]),)
        ).fetchone()
        if row is None:
            return None
        return [float(value) for value in json.loads(row[0])]

    def cache_vectors(self, namespace: str, vectors: dict[str, list[float]]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?)",
                [
                    (fingerprint([namespace, text]), json.dumps(vector))
                    for text, vector in vectors.items()
                ],
            )
