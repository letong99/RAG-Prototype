"""Durable single-host coordination and publication journal.

All readers and writers must share this directory. Locks serialize admitted readers for now;
this intentionally trades throughput for a small, crash-safe prototype admission boundary.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from filelock import FileLock

from rag_service.schemas import Chunk, Document, IndexMetadata, Record


class DocumentEntry(Record):
    document: Document
    chunk_hashes: dict[str, str]


class Operation(Record):
    operation_id: str
    kind: Literal["full", "incremental", "rollback"]
    index: IndexMetadata
    previous: IndexMetadata | None
    fallback: IndexMetadata | None
    directory: str
    checksums: dict[str, str]


class Ledger:
    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir.resolve()
        self.db = sqlite3.connect(self.data_dir / "lifecycle.sqlite3")
        if self.db.execute("PRAGMA user_version").fetchone()[0] > 1:
            self.db.close()
            raise RuntimeError("Coordinator schema is newer than this application")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS identity (id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS counters (
                corpus TEXT PRIMARY KEY, generation INTEGER NOT NULL, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshots (
                generation TEXT PRIMARY KEY, index_json TEXT NOT NULL, directory TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS publications (
                corpus TEXT PRIMARY KEY, active TEXT NOT NULL, fallback TEXT);
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, corpus TEXT NOT NULL, state TEXT NOT NULL,
                operation_json TEXT NOT NULL, error TEXT);
        """)
        with self.db:
            self.db.execute("PRAGMA user_version=1")
            self.db.execute(
                "INSERT INTO identity SELECT ? WHERE NOT EXISTS (SELECT 1 FROM identity)",
                (uuid4().hex,),
            )
        self.identity = str(self.db.execute("SELECT id FROM identity").fetchone()[0])

    def close(self) -> None:
        self.db.close()

    def allocate(self, corpus: str, *, full: bool) -> tuple[int, int]:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO counters VALUES (?, 0, 0)", (corpus,))
            self.db.execute(
                "UPDATE counters SET generation=generation+?, revision=revision+1 WHERE corpus=?",
                (int(full), corpus),
            )
            row = self.db.execute(
                "SELECT generation, revision FROM counters WHERE corpus=?", (corpus,)
            ).fetchone()
        return int(row[0]), int(row[1])

    def active(self, corpus: str, *, fallback: bool = False) -> IndexMetadata | None:
        column = "fallback" if fallback else "active"
        row = self.db.execute(
            f"SELECT s.index_json FROM publications p JOIN snapshots s ON p.{column}=s.generation "
            "WHERE p.corpus=?",
            (corpus,),
        ).fetchone()
        return IndexMetadata.model_validate_json(row[0]) if row else None

    def entries(self, generation: str) -> dict[str, DocumentEntry]:
        row = self.db.execute(
            "SELECT directory FROM snapshots WHERE generation=?", (generation,)
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "Missing document manifest; restore the coordinator's data directory"
            )
        return read_entries(Path(row[0]))

    def pending(self, corpus: str) -> Operation | None:
        row = self.db.execute(
            "SELECT operation_json FROM operations WHERE corpus=? "
            "AND state NOT IN ('committed', 'abandoned')",
            (corpus,),
        ).fetchone()
        return Operation.model_validate_json(row[0]) if row else None

    def begin(self, operation: Operation, *, abandon_pending: bool = False) -> None:
        if self.pending(operation.index.collection) and not abandon_pending:
            raise RuntimeError("Unresolved ingestion; run recover before starting another update")
        with self.db:
            if abandon_pending:
                self.db.execute(
                    "UPDATE operations SET state='abandoned' WHERE corpus=? AND state!='committed'",
                    (operation.index.collection,),
                )
            self.db.execute(
                "INSERT INTO operations VALUES (?, ?, 'maintenance', ?, NULL)",
                (operation.operation_id, operation.index.collection, operation.model_dump_json()),
            )

    def state(self, operation: Operation, state: str, error: str | None = None) -> None:
        with self.db:
            self.db.execute(
                "UPDATE operations SET state=?, error=? WHERE id=?",
                (state, error, operation.operation_id),
            )

    def commit(self, operation: Operation) -> None:
        index = operation.index
        fallback = operation.previous if operation.kind == "full" else operation.fallback
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)",
                (index.generation_id, index.model_dump_json(), operation.directory),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO publications VALUES (?, ?, ?)",
                (
                    index.collection,
                    index.generation_id,
                    fallback.generation_id if fallback else None,
                ),
            )
            self.db.execute(
                "UPDATE operations SET state='committed', error=NULL WHERE id=?",
                (operation.operation_id,),
            )

    def retired(self, corpus: str) -> list[str]:
        active = self.active(corpus)
        fallback = self.active(corpus, fallback=True)
        retained = {item.generation_id for item in (active, fallback) if item}
        rows = self.db.execute("SELECT generation, index_json FROM snapshots")
        retired = {
            str(row[0])
            for row in rows
            if row[0] not in retained
            and IndexMetadata.model_validate_json(row[1]).collection == corpus
        }
        for row in self.db.execute(
            "SELECT operation_json FROM operations WHERE corpus=? AND state='abandoned'", (corpus,)
        ):
            operation = Operation.model_validate_json(row[0])
            if operation.index.generation_id not in retained:
                retired.add(operation.index.generation_id)
        return sorted(retired)

    @contextmanager
    def admission(self, corpus: str) -> Iterator[IndexMetadata]:
        # This lock covers the entire retrieval request, not just the state check.
        with FileLock(self.data_dir / "retrieval.lock", timeout=0):
            if self.pending(corpus):
                raise RuntimeError("Ingestion maintenance: retrieval unavailable until recovery")
            index = self.active(corpus)
            if index is None:
                raise RuntimeError("No committed index")
            yield index


def read_entries(directory: Path) -> dict[str, DocumentEntry]:
    with (directory / "manifest.jsonl").open() as stream:
        entries = [DocumentEntry.model_validate_json(line) for line in stream]
    return {entry.document.document_id: entry for entry in entries}


def payload_hash(chunk: Chunk) -> str:
    from rag_service.schemas import fingerprint

    return fingerprint(chunk.model_dump())
