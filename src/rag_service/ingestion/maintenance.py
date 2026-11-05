"""Explicit fallback activation and bounded collection cleanup; no model loading."""

import os
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from rag_service.ingestion.lifecycle import Ledger, Operation
from rag_service.ingestion.recovery import apply_operation
from rag_service.ingestion.staging import seal
from rag_service.ports import VectorStore
from rag_service.schemas import IndexMetadata
from rag_service.storage.manifest import Manifest


def rollback(data_dir: Path, corpus: str, store: VectorStore, audit: Manifest) -> IndexMetadata:
    with FileLock(data_dir / "ingestion.lock", timeout=0):
        ledger = Ledger(data_dir)
        try:
            fallback = ledger.active(corpus, fallback=True)
            if fallback is None:
                raise ValueError("No inactive fallback exists")
            if fallback.coordinator_id != ledger.identity:
                raise RuntimeError("Fallback belongs to another coordinator")
            previous = store.active_index(corpus)
            if previous and previous.coordinator_id != ledger.identity:
                raise RuntimeError("Active collection belongs to another coordinator")
            _, revision = ledger.allocate(corpus, full=False)
            index = fallback.model_copy(update={"ingestion_revision": revision})
            operation_id = uuid4().hex
            directory = ledger.data_dir / "operations" / operation_id
            directory.mkdir(parents=True)
            with (directory / "manifest.jsonl").open("w") as stream:
                for entry in ledger.entries(fallback.generation_id).values():
                    stream.write(entry.model_dump_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            with (directory / "mutations.jsonl").open("w") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            operation = Operation(
                operation_id=operation_id,
                kind="rollback",
                index=index,
                previous=previous,
                fallback=None,
                directory=str(directory),
                checksums=seal(directory),
            )
            audit.begin(operation_id, corpus, {"rollback_from": fallback.ingestion_revision})
            ledger.begin(operation, abandon_pending=True)
            apply_operation(ledger, store, audit, operation)
            return index
        finally:
            ledger.close()


def cleanup(data_dir: Path, corpus: str, store: VectorStore) -> list[str]:
    """Delete only committed, retired collections owned by this coordinator."""
    with FileLock(data_dir / "ingestion.lock", timeout=0):
        ledger = Ledger(data_dir)
        try:
            if ledger.pending(corpus):
                raise RuntimeError("Recover the pending ingestion before cleanup")
            if store.active_index(corpus) != ledger.active(corpus):
                raise RuntimeError("Remote and local publication disagree")
            with FileLock(data_dir / "retrieval.lock", timeout=30):
                retired = ledger.retired(corpus)
                for generation in retired:
                    store.retire(generation)
                return retired
        finally:
            ledger.close()
