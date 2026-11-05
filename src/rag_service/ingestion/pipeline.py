"""Incremental document ingestion with explicit full rebuilds and durable replay."""

from collections.abc import Iterable, Sequence
from uuid import uuid4

from filelock import FileLock

from rag_service.chunking.factory import create_chunker
from rag_service.config import IngestionConfig
from rag_service.embeddings.cached import CachedEmbedder
from rag_service.ingestion.lifecycle import Ledger, Operation
from rag_service.ingestion.normalization import NORMALIZER_VERSION, normalize
from rag_service.ingestion.recovery import apply_operation
from rag_service.ingestion.staging import Stager, seal
from rag_service.ports import Embedder, VectorStore
from rag_service.schemas import (
    IndexMetadata,
    IngestionResult,
    SourceDocument,
    fingerprint,
)
from rag_service.storage.manifest import Manifest


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

    def settings(self) -> dict[str, object]:
        return {
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

    def ingest(
        self,
        sources: Iterable[SourceDocument],
        *,
        allow_empty: bool = False,
        full_rebuild: bool = False,
        delete_document_ids: Sequence[str] = (),
        available_storage_bytes: int | None = None,
    ) -> IngestionResult:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self.config.data_dir / "ingestion.lock", timeout=0):
            ledger = Ledger(self.config.data_dir)
            try:
                if ledger.pending(self.config.collection):
                    raise RuntimeError("Unresolved ingestion; run recover before another update")
                self.manifest.recover()
                return self._ingest(
                    ledger,
                    sources,
                    allow_empty,
                    full_rebuild,
                    delete_document_ids,
                    available_storage_bytes,
                )
            finally:
                ledger.close()

    def _ingest(
        self,
        ledger: Ledger,
        sources: Iterable[SourceDocument],
        allow_empty: bool,
        full_rebuild: bool,
        delete_ids: Sequence[str],
        available_storage_bytes: int | None,
    ) -> IngestionResult:
        active = self.store.active_index(self.config.collection)
        local = ledger.active(self.config.collection)
        if active != local:
            raise RuntimeError(
                "Remote index and local manifest disagree; restore the coordinator "
                "manifest or migrate the legacy index to a new corpus name"
            )
        if active and active.coordinator_id != ledger.identity:
            raise RuntimeError("This corpus belongs to another ingestion coordinator")
        processing = fingerprint(self.settings())
        full = full_rebuild or active is None
        if active and not full and active.processing_fingerprint != processing:
            raise ValueError("Processing configuration changed; explicitly use --full-rebuild")
        if (
            full
            and self.store.remote
            and (available_storage_bytes is None or available_storage_bytes <= 0)
        ):
            raise ValueError("Remote full builds require measured --available-storage-bytes")
        if full and delete_ids:
            raise ValueError("Full rebuild takes a complete source set, not deletion patches")
        old = ledger.entries(active.generation_id) if active else {}
        target = {} if full else dict(old)
        changed = []
        skipped = 0
        seen: set[str] = set()
        for source in sources:
            document = normalize(source)
            key = document.document_id
            if key in seen:
                raise ValueError(f"Duplicate document ID: {key}")
            seen.add(key)
            previous = old.get(key)
            if not full and previous and previous.document.version == document.version:
                skipped += 1
            else:
                changed.append(document)
        deleted = set(delete_ids)
        if deleted & seen:
            raise ValueError("A document cannot be supplied and deleted in the same operation")
        for key in deleted:
            target.pop(key, None)
        if not target and not changed and not allow_empty:
            raise ValueError("Refusing to publish an empty corpus; explicitly allow it to clear")
        removed = len(deleted & old.keys())
        if active and not full and not changed and not removed:
            return IngestionResult(
                generation_id=active.generation_id,
                status="unchanged",
                collection=active.collection,
                documents=active.documents,
                chunks=active.chunks,
                skipped_documents=skipped,
                ingestion_revision=active.ingestion_revision,
                generation_number=active.generation_number,
            )
        number, revision = ledger.allocate(self.config.collection, full=full)
        if full:
            generation = uuid4().hex
        else:
            assert active is not None
            generation = active.generation_id
            number = active.generation_number
        operation_id = uuid4().hex
        directory = ledger.data_dir / "operations" / operation_id
        directory.mkdir(parents=True)
        self.manifest.begin(operation_id, self.config.collection, self.settings())
        self.cached_embedder.reset_stats()
        try:
            stats = Stager(
                self.chunker, self.cached_embedder, self.store, self.config.batch_size
            ).stage(
                directory,
                changed,
                old,
                target,
                deleted,
                full=full,
                generation=generation,
                number=number,
                revision=revision,
                processing=processing,
            )
            index = IndexMetadata(
                generation_id=generation,
                generation_number=number,
                ingestion_revision=revision,
                coordinator_id=ledger.identity,
                processing_fingerprint=processing,
                collection=self.config.collection,
                embedding=self.embedder.spec,
                documents=len(target),
                chunks=sum(len(e.chunk_hashes) for e in target.values()),
                fingerprint=fingerprint(
                    [processing, sorted((k, e.document.version) for k, e in target.items())]
                ),
            )
            # The SDK does not expose hosting-plan free disk. Require an operator measurement for
            # remote builds and compare it with a conservative estimate, not a claimed exact size.
            if self.store.remote and (full or available_storage_bytes is not None):
                estimated = stats.estimated_storage_bytes
                if available_storage_bytes is None or available_storage_bytes < estimated:
                    raise ValueError(
                        f"Remote ingestion needs sufficient --available-storage-bytes from current "
                        f"server metrics (estimated required headroom: {estimated})"
                    )
            operation = Operation(
                operation_id=operation_id,
                kind="full" if full else "incremental",
                index=index,
                previous=active,
                fallback=ledger.active(self.config.collection, fallback=True),
                directory=str(directory),
                checksums=seal(directory),
            )
            ledger.begin(operation)
            apply_operation(ledger, self.store, self.manifest, operation)
        except Exception as error:
            if ledger.pending(self.config.collection) is None:
                self.manifest.fail(operation_id, type(error).__name__)
            raise
        return IngestionResult(
            generation_id=generation,
            collection=index.collection,
            status="ready",
            documents=index.documents,
            chunks=index.chunks,
            ingestion_revision=revision,
            generation_number=number,
            skipped_documents=skipped,
            updated_documents=len(changed),
            deleted_documents=removed,
            embedded_chunks=stats.embedded,
            cached_chunks=stats.cached,
            semantic_embedded_texts=stats.semantic_computed,
            semantic_cached_texts=stats.semantic_cached,
        )
