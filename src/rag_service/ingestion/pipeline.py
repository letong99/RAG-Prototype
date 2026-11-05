"""Stage a full corpus snapshot, reuse embeddings, then atomically publish its generation."""

import json
import logging
import os
from collections.abc import Iterable, Iterator
from itertools import batched
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from rag_service.chunking.factory import create_chunker
from rag_service.config import IngestionConfig
from rag_service.embeddings.cached import CachedEmbedder
from rag_service.ingestion.normalization import NORMALIZER_VERSION, normalize
from rag_service.ports import Embedder, VectorStore
from rag_service.schemas import Chunk, IndexMetadata, IngestionResult, SourceDocument, fingerprint
from rag_service.storage.manifest import Manifest

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self, config: IngestionConfig, embedder: Embedder, store: VectorStore, manifest: Manifest
    ) -> None:
        if config.chunk_tokens > embedder.spec.max_tokens:
            raise ValueError("chunk_tokens exceeds the embedding model's usable input budget")
        self.config = config
        self.embedder = embedder
        self.store = store
        self.manifest = manifest
        self.cached_embedder = CachedEmbedder(embedder, manifest, config.batch_size)
        self.chunker = create_chunker(config, self.cached_embedder)

    def ingest(
        self, sources: Iterable[SourceDocument], *, allow_empty: bool = False
    ) -> IngestionResult:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self.config.data_dir / "ingestion.lock", timeout=0):
            self.manifest.recover()
            self._reconcile_publication()
            return self._ingest(sources, allow_empty=allow_empty)

    def _reconcile_publication(self) -> None:
        for pending in self.manifest.pending_publications(self.config.collection):
            active = self.store.active_index(self.config.collection)
            if active is None or active.generation_id != pending["id"]:
                raise RuntimeError(
                    "Previous publication is unresolved. Inspect Qdrant's active alias and the "
                    "publication_pending audit record before starting another ingestion."
                )
            self.manifest.finish(
                active.generation_id,
                active.fingerprint,
                active.documents,
                active.chunks,
                publish=True,
            )

    def _ingest(self, sources: Iterable[SourceDocument], *, allow_empty: bool) -> IngestionResult:
        generation = uuid4().hex
        self.cached_embedder.reset_stats()
        settings = {
            "embedding": self.embedder.spec.model_dump(),
            "chunker": self.chunker.version,
            "chunk_tokens": self.config.chunk_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "chunking_strategy": self.config.chunking_strategy,
            "semantic_percentile": self.config.semantic_percentile,
            "semantic_min_distance": self.config.semantic_min_distance,
            "semantic_window_sentences": self.config.semantic_window_sentences,
            "normalizer": NORMALIZER_VERSION,
        }
        self.manifest.begin(generation, self.config.collection, settings)
        directory = self.config.data_dir / "generations" / generation
        publication_attempted = False
        try:
            directory.mkdir(parents=True)
            identities, chunk_count = self._stage(sources, directory)
            semantic_stats = {
                "semantic_embedded_texts": self.cached_embedder.computed_texts,
                "semantic_cached_texts": self.cached_embedder.reused_texts,
            }
            logger.info("Staged %d documents and %d chunks", len(identities), chunk_count)
            if not identities and not allow_empty:
                raise ValueError(
                    "Refusing to publish an empty corpus; explicitly allow it to clear"
                )
            corpus_hash = fingerprint([settings, sorted(identities.items())])
            active = self.store.active_index(self.config.collection)
            if active and active.fingerprint == corpus_hash:
                self.manifest.finish(
                    generation, corpus_hash, len(identities), chunk_count, publish=False
                )
                return IngestionResult(
                    generation_id=active.generation_id,
                    collection=self.config.collection,
                    status="unchanged",
                    documents=len(identities),
                    chunks=chunk_count,
                    **semantic_stats,
                )
            index = IndexMetadata(
                generation_id=generation,
                collection=self.config.collection,
                fingerprint=corpus_hash,
                embedding=self.embedder.spec,
                documents=len(identities),
                chunks=chunk_count,
            )
            self.store.create(generation, self.embedder.spec.dimension, metadata=index)
            embedded = cached = 0
            for batch in batched(read_chunks(directory / "chunks.jsonl"), self.config.batch_size):
                before = self.cached_embedder.computed_texts
                vectors = self.cached_embedder.embed([chunk.text for chunk in batch])
                new_count = self.cached_embedder.computed_texts - before
                self.store.write(generation, batch, vectors)
                embedded += new_count
                cached += len(batch) - new_count
                if (embedded + cached) % (10 * self.config.batch_size) == 0:
                    logger.info("Indexed %d/%d chunks", embedded + cached, chunk_count)
            if self.store.count(generation) != chunk_count:
                raise RuntimeError("Stored vector count does not match staged chunk count")
            self.manifest.publication_pending(generation)
            publication_attempted = True
            self.store.publish(index)
            self.manifest.finish(
                generation, corpus_hash, len(identities), chunk_count, publish=True
            )
            logger.info(
                "Published %s: %d documents, %d chunks", generation, len(identities), chunk_count
            )
            return IngestionResult(
                generation_id=generation,
                collection=self.config.collection,
                status="ready",
                documents=len(identities),
                chunks=chunk_count,
                embedded_chunks=embedded,
                cached_chunks=cached,
                **semantic_stats,
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if publication_attempted:
                self.manifest.publication_pending(generation, message)
            else:
                self.manifest.fail(generation, message)
            raise

    def _stage(
        self, sources: Iterable[SourceDocument], directory: Path
    ) -> tuple[dict[str, str], int]:
        identities: dict[str, str] = {}
        count = 0
        with (
            (directory / "documents.jsonl").open("w", encoding="utf-8") as documents,
            (directory / "chunks.jsonl").open("w", encoding="utf-8") as chunks,
        ):
            for source in sources:
                document = normalize(source)
                if document.document_id in identities:
                    raise ValueError(f"Duplicate document ID: {document.document_id}")
                identities[document.document_id] = document.version
                documents.write(document.model_dump_json() + "\n")
                for chunk in self.chunker.chunks(document):
                    chunks.write(chunk.model_dump_json() + "\n")
                    count += 1
            for stream in (documents, chunks):
                stream.flush()
                os.fsync(stream.fileno())
        return identities, count


def read_chunks(path: Path) -> Iterator[Chunk]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield Chunk.model_validate(json.loads(line))
