"""Prepare replayable mutations without modifying the searchable collection."""

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import TextIO

from rag_service.embeddings.cached import CachedEmbedder
from rag_service.ingestion.lifecycle import DocumentEntry, payload_hash
from rag_service.ports import Chunker, VectorStore
from rag_service.schemas import Chunk, Document, fingerprint

logger = logging.getLogger(__name__)


@dataclass
class StageStats:
    embedded: int = 0
    cached: int = 0
    semantic_computed: int = 0
    semantic_cached: int = 0
    estimated_storage_bytes: int = 128_000_000


class Stager:
    def __init__(
        self, chunker: Chunker, embedder: CachedEmbedder, store: VectorStore, batch_size: int
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size
        self.stats = StageStats()

    def stage(
        self,
        directory: Path,
        changed: Sequence[Document],
        old: Mapping[str, DocumentEntry],
        target: dict[str, DocumentEntry],
        deleted: set[str],
        *,
        full: bool,
        generation: str,
        number: int,
        revision: int,
        processing: str,
    ) -> StageStats:
        with (directory / "mutations.jsonl").open("w") as stream:
            for position, document in enumerate(changed, start=1):
                previous = old.get(document.document_id)
                metadata_only = (
                    not full
                    and previous is not None
                    and previous.document.content_hash == document.content_hash
                )
                if metadata_only:
                    assert previous is not None
                    chunks = self._metadata_chunks(generation, document, previous, revision)
                else:
                    chunks = self._new_chunks(document, number, revision, processing)
                if not chunks:
                    raise ValueError("Document produced no chunks")
                self._write_batches(stream, chunks, metadata_only)
                target[document.document_id] = DocumentEntry(
                    document=document, chunk_hashes={c.chunk_id: payload_hash(c) for c in chunks}
                )
                if not full and previous:
                    obsolete = (
                        previous.chunk_hashes.keys()
                        - target[document.document_id].chunk_hashes.keys()
                    )
                    self._delete_batches(stream, sorted(obsolete))
                if position % 100 == 0 or position == len(changed):
                    stream.flush()
                    logger.info(
                        "Prepared %d/%d changed documents; %d new context embeddings; "
                        "%d new chunk embeddings; estimated additional storage %.1f MB",
                        position,
                        len(changed),
                        self.stats.semantic_computed,
                        self.stats.embedded,
                        self.stats.estimated_storage_bytes / 1_000_000,
                    )
            for key in sorted(deleted & old.keys()):
                self._delete_batches(stream, list(old[key].chunk_hashes))
            stream.flush()
            os.fsync(stream.fileno())
        write_manifest(directory, target)
        return self.stats

    def _metadata_chunks(
        self, generation: str, document: Document, previous: DocumentEntry, revision: int
    ) -> list[Chunk]:
        # Current chunkers depend only on normalized text, not source title/page/citation metadata.
        chunks = list(self.store.iter_chunks(generation, document.document_id))
        if {c.chunk_id: payload_hash(c) for c in chunks} != previous.chunk_hashes:
            raise RuntimeError("Existing document integrity mismatch")
        return [
            chunk.model_copy(
                update={
                    "document_version": document.version,
                    "source_uri": document.source_uri,
                    "title": document.title,
                    "page": document.page,
                    "metadata": document.metadata,
                    "written_revision": revision,
                }
            )
            for chunk in chunks
        ]

    def _new_chunks(
        self, document: Document, number: int, revision: int, processing: str
    ) -> list[Chunk]:
        before, reused = self.embedder.computed_texts, self.embedder.reused_texts
        chunks = [
            chunk.model_copy(
                update={
                    "chunk_id": fingerprint([chunk.chunk_id, processing]),
                    "generation_number": number,
                    "written_revision": revision,
                    "processing_fingerprint": processing,
                }
            )
            for chunk in self.chunker.chunks(document)
        ]
        self.stats.semantic_computed += self.embedder.computed_texts - before
        self.stats.semantic_cached += self.embedder.reused_texts - reused
        return chunks

    def _write_batches(self, stream: TextIO, chunks: Sequence[Chunk], metadata_only: bool) -> None:
        for batch in batched(chunks, self.batch_size):
            vectors = None
            if not metadata_only:
                before = self.embedder.computed_texts
                vectors = self.embedder.embed([c.text for c in batch])
                computed = self.embedder.computed_texts - before
                self.stats.embedded += computed
                self.stats.cached += len(batch) - computed
            payloads = [c.model_dump() for c in batch]
            raw_size = sum(len(c.model_dump_json().encode()) for c in batch)
            raw_size += len(batch) * self.embedder.spec.dimension * 4
            self.stats.estimated_storage_bytes += 3 * raw_size
            stream.write(
                json.dumps(
                    {
                        "kind": "metadata" if metadata_only else "upsert",
                        "chunks": payloads,
                        "vectors": vectors,
                    }
                )
                + "\n"
            )

    def _delete_batches(self, stream: TextIO, ids: Sequence[str]) -> None:
        for batch in batched(ids, self.batch_size):
            stream.write(json.dumps({"kind": "delete", "ids": batch}) + "\n")


def write_manifest(directory: Path, entries: Mapping[str, DocumentEntry]) -> None:
    with (directory / "manifest.jsonl").open("w") as stream:
        for key in sorted(entries):
            stream.write(entries[key].model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def seal(directory: Path) -> dict[str, str]:
    """Hash replay artifacts and fsync directory entries before journaling them."""
    hashes = {}
    for name in ("manifest.jsonl", "mutations.jsonl"):
        with (directory / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    for path in (directory, directory.parent, directory.parent.parent):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return hashes


def check_seal(directory: Path, hashes: Mapping[str, str]) -> None:
    if set(hashes) != {"manifest.jsonl", "mutations.jsonl"}:
        raise RuntimeError("Missing recovery artifact checksums")
    for name, expected in hashes.items():
        with (directory / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError(
                    "Recovery artifact checksum mismatch; restore original artifacts"
                )
