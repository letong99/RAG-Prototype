"""Replay durable Qdrant mutations, validate, and commit while retrieval is excluded."""

import json
import logging
from pathlib import Path

from filelock import FileLock

from rag_service.ingestion.lifecycle import Ledger, Operation, payload_hash, read_entries
from rag_service.ingestion.staging import check_seal
from rag_service.ports import VectorStore
from rag_service.schemas import Chunk, IndexMetadata
from rag_service.storage.manifest import Manifest

logger = logging.getLogger(__name__)


def replay_mutations(store: VectorStore, generation: str, path: Path) -> None:
    """Coalesce adjacent upserts, preserving order at every metadata/deletion boundary."""
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    uploaded = 0

    def flush() -> None:
        nonlocal uploaded
        if chunks:
            store.write(generation, chunks, vectors)
            uploaded += len(chunks)
            chunks.clear()
            vectors.clear()
            if uploaded % 4096 == 0:
                logger.info("Uploaded %d chunk vectors", uploaded)

    with path.open() as stream:
        for line in stream:
            mutation = json.loads(line)
            if mutation["kind"] == "upsert":
                for item, vector in zip(mutation["chunks"], mutation["vectors"], strict=True):
                    chunks.append(Chunk.model_validate(item))
                    vectors.append(vector)
                    if len(chunks) == 256:
                        flush()
            else:
                flush()
                if mutation["kind"] == "delete":
                    store.delete_chunks(generation, mutation["ids"])
                elif mutation["kind"] == "metadata":
                    store.update_chunks(
                        generation, [Chunk.model_validate(c) for c in mutation["chunks"]]
                    )
                else:
                    raise ValueError("Unknown journal mutation type")
    flush()
    logger.info("Uploaded %d chunk vectors; mutation replay complete", uploaded)


def verify(store: VectorStore, index: IndexMetadata, directory: Path) -> None:
    entries = read_entries(directory)
    expected = {
        key: value for entry in entries.values() for key, value in entry.chunk_hashes.items()
    }
    if store.count(index.generation_id) != index.chunks:
        raise RuntimeError("Stored vector count does not match planned count")
    seen: set[str] = set()
    for chunk in store.iter_chunks(index.generation_id):
        if chunk.chunk_id in seen or expected.get(chunk.chunk_id) != payload_hash(chunk):
            raise RuntimeError("Chunk identity/version/payload integrity check failed")
        seen.add(chunk.chunk_id)
    if seen != expected.keys():
        raise RuntimeError("Missing or unexpected chunks in stored document set")
    logger.info("Verified all %d stored chunks against the document manifest", len(seen))


def apply_operation(
    ledger: Ledger, store: VectorStore, audit: Manifest, operation: Operation
) -> None:
    index = operation.index
    directory = Path(operation.directory)
    # Acquiring this gate drains the one admitted reader; durable pending state blocks new readers.
    with FileLock(ledger.data_dir / "retrieval.lock", timeout=30):
        try:
            check_seal(directory, operation.checksums)
            current = store.active_index(index.collection)
            if current not in (operation.previous, index):
                raise RuntimeError(
                    "Remote publication changed outside this coordinator; inspect before recovery"
                )
            ledger.state(operation, "applying")
            if not store.exists(index.generation_id):
                if operation.kind != "full":
                    raise RuntimeError("Active collection is missing; restore from fallback")
                store.create(index.generation_id, index.embedding.dimension, metadata=index)
            elif operation.kind == "full":
                store.ensure_indexes(index.generation_id)
            replay_mutations(store, index.generation_id, directory / "mutations.jsonl")
            ledger.state(operation, "verifying")
            verify(store, index, directory)
            ledger.state(operation, "committing")
            store.update_index(index)
            if operation.kind == "full":
                store.publish(index)
            elif operation.kind == "rollback":
                store.restore(index)
            if store.active_index(index.collection) != index:
                raise RuntimeError("Remote metadata publication was not confirmed")
            audit.finish(
                operation.operation_id,
                index.fingerprint,
                index.documents,
                index.chunks,
                publish=True,
            )
            ledger.commit(operation)
        except Exception as error:
            ledger.state(operation, "recovery_required", type(error).__name__)
            raise
    # Cleanup failure must not turn a committed publication back into a pending mutation.
    if operation.kind == "full" and operation.fallback:
        try:
            store.retire(operation.fallback.generation_id)
        except Exception:
            logger.warning("Old fallback retained; run cleanup after inspection")


def recover(data_dir: Path, collection: str, store: VectorStore, audit: Manifest) -> IndexMetadata:
    with FileLock(data_dir / "ingestion.lock", timeout=0):
        ledger = Ledger(data_dir)
        try:
            operation = ledger.pending(collection)
            if operation is None:
                raise ValueError("No pending ingestion to recover")
            apply_operation(ledger, store, audit, operation)
            return operation.index
        finally:
            ledger.close()
