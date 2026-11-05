"""Lifecycle invariants, fault injection, and retrieval coordination."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import FakeEmbedder, document
from filelock import FileLock, Timeout

from rag_service.ingestion.lifecycle import Ledger
from rag_service.ingestion.maintenance import cleanup, rollback
from rag_service.ingestion.pipeline import IngestionPipeline
from rag_service.ingestion.recovery import recover
from rag_service.retrieval.service import RetrievalService
from rag_service.schemas import EmbeddingSpec
from rag_service.storage.qdrant import QdrantVectorStore


def ledger_for(pipeline: IngestionPipeline) -> Ledger:
    return Ledger(pipeline.config.data_dir)


def test_unchanged_skips_chunker_embedder_and_all_point_writes(pipeline, monkeypatch):
    first = pipeline.ingest([document("a")])
    fail = Mock(side_effect=AssertionError("unchanged work was rerun"))
    monkeypatch.setattr(pipeline.chunker, "chunks", fail)
    monkeypatch.setattr(pipeline.embedder, "embed", fail)
    monkeypatch.setattr(pipeline.store, "write", fail)
    result = pipeline.ingest([document("a", "  Daily backups are retained for thirty days. \r\n")])
    assert result.status == "unchanged"
    assert result.skipped_documents == 1
    assert result.ingestion_revision == first.ingestion_revision
    fail.assert_not_called()


def test_patch_retains_absent_docs_and_metadata_only_never_embeds(pipeline, monkeypatch):
    first = pipeline.ingest([document("a"), document("b")])
    old_chunks = list(pipeline.store.iter_chunks(first.generation_id, "b"))
    fail = Mock(side_effect=AssertionError("metadata-only work embedded or chunked"))
    monkeypatch.setattr(pipeline.chunker, "chunks", fail)
    monkeypatch.setattr(pipeline.embedder, "embed", fail)
    changed = document("a").model_copy(update={"title": "New title", "source_uri": "test://new"})
    result = pipeline.ingest([changed])
    assert result.documents == 2
    assert result.generation_id == first.generation_id
    assert result.ingestion_revision == 2
    assert list(pipeline.store.iter_chunks(first.generation_id, "b")) == old_chunks
    updated = list(pipeline.store.iter_chunks(first.generation_id, "a"))
    assert all(c.title == "New title" and c.written_revision == 2 for c in updated)
    assert result.embedded_chunks == result.cached_chunks == 0


def test_update_replaces_all_old_chunks_and_explicit_delete(pipeline):
    first = pipeline.ingest([document("a", "word " * 40), document("b")])
    old_ids = {c.chunk_id for c in pipeline.store.iter_chunks(first.generation_id, "a")}
    updated = pipeline.ingest([document("a", "New short text.")], delete_document_ids=["b"])
    chunks = list(pipeline.store.iter_chunks(first.generation_id))
    assert updated.documents == updated.chunks == 1
    assert not old_ids & {c.chunk_id for c in chunks}
    assert chunks[0].written_revision == 2
    assert chunks[0].generation_number == 1
    assert chunks[0].processing_fingerprint


def test_full_rebuild_retains_one_fallback_and_cleans_older(pipeline):
    first = pipeline.ingest([document("a")])
    second = pipeline.ingest([document("b")], full_rebuild=True)
    pipeline.ingest([document("c")])
    assert pipeline.store.count(first.generation_id) == 1
    third = pipeline.ingest([document("d")], full_rebuild=True)
    assert third.generation_number == 3
    assert third.ingestion_revision == 4
    assert not pipeline.store.exists(first.generation_id)
    assert pipeline.store.count(second.generation_id) == 2
    ledger = ledger_for(pipeline)
    try:
        assert ledger.active("default", fallback=True).generation_id == second.generation_id
        assert ledger.active("default").generation_id == third.generation_id
    finally:
        ledger.close()


def test_reader_gate_covers_entire_context_assembly(pipeline):
    pipeline.ingest([document("a")])
    service = RetrievalService(pipeline.config.data_dir, pipeline.store)
    with service.retrieve("default", "backup", pipeline.embedder) as (index, hits):
        assert hits and index.ingestion_revision == 1
        with (
            pytest.raises(Timeout),
            FileLock(pipeline.config.data_dir / "retrieval.lock", timeout=0),
        ):
            pass
    with FileLock(pipeline.config.data_dir / "retrieval.lock", timeout=0):
        pass


@pytest.mark.parametrize("failure_point", ["write", "delete_chunks", "update_index"])
def test_partial_update_blocks_reads_and_replays_without_embeddings(
    pipeline, monkeypatch, failure_point
):
    first = pipeline.ingest([document("a", "old words " * 20)])
    method = getattr(pipeline.store, failure_point)

    def lost_ack(*args, **kwargs):
        method(*args, **kwargs)
        raise TimeoutError("lost response after applied write")

    monkeypatch.setattr(pipeline.store, failure_point, lost_ack)
    with pytest.raises(TimeoutError):
        pipeline.ingest([document("a", "new short text")])
    service = RetrievalService(pipeline.config.data_dir, pipeline.store)
    with (
        pytest.raises(RuntimeError, match="maintenance"),
        service.retrieve("default", "query", pipeline.embedder),
    ):
        pass
    with pytest.raises(RuntimeError, match="Unresolved"):
        pipeline.ingest([document("x")])
    monkeypatch.setattr(pipeline.store, failure_point, method)
    monkeypatch.setattr(pipeline.embedder, "embed", Mock(side_effect=AssertionError("reembedded")))
    restored = recover(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)
    assert restored.generation_id == first.generation_id
    assert restored.ingestion_revision == 2
    assert [c.text for c in pipeline.store.iter_chunks(first.generation_id)] == ["new short text"]
    ledger = ledger_for(pipeline)
    try:
        assert ledger.pending("default") is None
    finally:
        ledger.close()


def test_artifact_corruption_does_not_replay_or_reopen(pipeline, monkeypatch):
    pipeline.ingest([document("a")])
    write = pipeline.store.write
    monkeypatch.setattr(pipeline.store, "write", Mock(side_effect=TimeoutError))
    with pytest.raises(TimeoutError):
        pipeline.ingest([document("a", "changed")])
    ledger = ledger_for(pipeline)
    try:
        operation = ledger.pending("default")
        (Path(operation.directory) / "mutations.jsonl").write_text("{}\n")
    finally:
        ledger.close()
    monkeypatch.setattr(pipeline.store, "write", write)
    with pytest.raises(RuntimeError, match="checksum"):
        recover(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)


def test_rollback_restores_matching_manifest_and_monotonic_revisions(pipeline):
    first = pipeline.ingest([document("a")])
    pipeline.ingest([document("b")], full_rebuild=True)
    pipeline.ingest([document("c")])
    restored = rollback(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)
    assert restored.generation_id == first.generation_id
    assert restored.generation_number == 1
    assert restored.ingestion_revision == 4
    service = RetrievalService(pipeline.config.data_dir, pipeline.store)
    with service.retrieve("default", "query", pipeline.embedder) as (_, hits):
        assert {hit.chunk.document_id for hit in hits} == {"a"}
    updated = pipeline.ingest([document("d")])
    assert updated.generation_number == 1
    assert updated.ingestion_revision == 5
    assert all(c.generation_number == 1 for c in pipeline.store.iter_chunks(first.generation_id))
    rebuilt = pipeline.ingest([document("e")], full_rebuild=True)
    assert rebuilt.generation_number == 3
    assert rebuilt.ingestion_revision == 6


def test_failed_update_can_explicitly_fallback(pipeline, monkeypatch):
    first = pipeline.ingest([document("a")])
    pipeline.ingest([document("b")], full_rebuild=True)
    monkeypatch.setattr(pipeline.store, "write", Mock(side_effect=TimeoutError))
    with pytest.raises(TimeoutError):
        pipeline.ingest([document("c")])
    restored = rollback(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)
    assert restored.generation_id == first.generation_id
    ledger = ledger_for(pipeline)
    try:
        assert ledger.pending("default") is None
        assert ledger.active("default", fallback=True) is None
    finally:
        ledger.close()


def test_uniformly_stale_payload_is_rejected(pipeline):
    first = pipeline.ingest([document("a")])
    chunks = list(pipeline.store.iter_chunks(first.generation_id))
    pipeline.store.update_chunks(
        first.generation_id, [c.model_copy(update={"document_version": "old"}) for c in chunks]
    )
    service = RetrievalService(pipeline.config.data_dir, pipeline.store)
    with (
        pytest.raises(RuntimeError, match="integrity"),
        service.retrieve("default", "query", pipeline.embedder),
    ):
        pass


def test_configuration_change_requires_explicit_rebuild(pipeline):
    pipeline.ingest([document("a")])
    config = pipeline.config.model_copy(update={"chunk_tokens": 10})
    changed = IngestionPipeline(config, pipeline.embedder, pipeline.store, pipeline.manifest)
    with pytest.raises(ValueError, match="full-rebuild"):
        changed.ingest([document("a")])
    assert changed.ingest([document("a")], full_rebuild=True).generation_number == 2


def test_wrong_query_model_and_wrong_coordinator_rejected(pipeline, tmp_path):
    pipeline.ingest([document("a")])
    model = FakeEmbedder()
    model.spec = EmbeddingSpec(model="other", revision="v1", dimension=4, max_tokens=512)
    service = RetrievalService(pipeline.config.data_dir, pipeline.store)
    with pytest.raises(ValueError, match="model"), service.retrieve("default", "query", model):
        pass
    config = pipeline.config.model_copy(update={"data_dir": tmp_path / "other"})
    other = IngestionPipeline(config, pipeline.embedder, pipeline.store, pipeline.manifest)
    with pytest.raises(RuntimeError, match="manifest disagree"):
        other.ingest([document("a")])


def test_missing_and_insufficient_capacity_prevent_remote_mutation(pipeline, monkeypatch):
    monkeypatch.setattr(QdrantVectorStore, "remote", property(lambda self: True))
    create = Mock(side_effect=AssertionError("must not create"))
    monkeypatch.setattr(pipeline.store, "create", create)
    with pytest.raises(ValueError, match="available-storage"):
        pipeline.ingest([document("a")])
    with pytest.raises(ValueError, match="headroom"):
        pipeline.ingest([document("a")], available_storage_bytes=1)
    create.assert_not_called()


def test_no_fallback_at_bootstrap_and_cleanup_preserves_aliases(pipeline):
    first = pipeline.ingest([document("a")])
    with pytest.raises(ValueError, match="No inactive"):
        rollback(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)
    assert cleanup(pipeline.config.data_dir, "default", pipeline.store) == []
    with pytest.raises(RuntimeError, match="alias"):
        pipeline.store.retire(first.generation_id)


def test_commit_ack_loss_is_reconciled(pipeline, monkeypatch):
    pipeline.ingest([document("a")])
    commit = Ledger.commit

    def lost_ack(self, operation):
        commit(self, operation)
        raise TimeoutError("commit acknowledgement lost")

    monkeypatch.setattr(Ledger, "commit", lost_ack)
    with pytest.raises(TimeoutError):
        pipeline.ingest([document("b")])
    monkeypatch.setattr(Ledger, "commit", commit)
    index = recover(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)
    assert index.ingestion_revision == 2
    assert index.documents == 2
    assert pipeline.ingest([document("b")]).status == "unchanged"


def test_creation_ack_loss_recovers_same_generation(pipeline, monkeypatch):
    create = pipeline.store.create

    def lost_ack(*args, **kwargs):
        create(*args, **kwargs)
        raise TimeoutError("collection created but response lost")

    monkeypatch.setattr(pipeline.store, "create", lost_ack)
    with pytest.raises(TimeoutError):
        pipeline.ingest([document("a")])
    monkeypatch.setattr(pipeline.store, "create", create)
    index = recover(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)
    assert index.generation_number == index.ingestion_revision == 1
    assert pipeline.store.count(index.generation_id) == 1


def test_same_count_wrong_chunks_blocks_publication(pipeline, monkeypatch):
    first = pipeline.ingest([document("a")])
    original = pipeline.store.iter_chunks

    def corrupt(generation, document_id=None):
        for chunk in original(generation, document_id):
            yield chunk.model_copy(update={"text": "corrupt text"})

    monkeypatch.setattr(pipeline.store, "iter_chunks", corrupt)
    with pytest.raises(RuntimeError, match="integrity"):
        pipeline.ingest([document("b")])
    assert pipeline.store.active_index("default").ingestion_revision == first.ingestion_revision


def test_metadata_and_delete_ack_loss_replay(pipeline, monkeypatch):
    first = pipeline.ingest([document("a"), document("b")])
    update = pipeline.store.update_chunks

    def lost_ack(*args, **kwargs):
        update(*args, **kwargs)
        raise TimeoutError

    monkeypatch.setattr(pipeline.store, "update_chunks", lost_ack)
    changed = document("a").model_copy(update={"title": "changed"})
    with pytest.raises(TimeoutError):
        pipeline.ingest([changed], delete_document_ids=["b"])
    monkeypatch.setattr(pipeline.store, "update_chunks", update)
    recover(pipeline.config.data_dir, "default", pipeline.store, pipeline.manifest)
    chunks = list(pipeline.store.iter_chunks(first.generation_id))
    assert len(chunks) == 1
    assert chunks[0].title == "changed"


def test_conflicting_delete_does_not_mutate_and_last_delete_requires_opt_in(pipeline):
    first = pipeline.ingest([document("a")])
    with pytest.raises(ValueError, match="supplied and deleted"):
        pipeline.ingest([document("a")], delete_document_ids=["a"])
    with pytest.raises(ValueError, match="empty corpus"):
        pipeline.ingest([], delete_document_ids=["a"])
    assert pipeline.store.count(first.generation_id) == 1
    cleared = pipeline.ingest([], delete_document_ids=["a"], allow_empty=True)
    assert cleared.documents == cleared.chunks == 0
    assert cleared.generation_id == first.generation_id


def test_writer_drains_an_admitted_reader_before_writing(pipeline, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import monotonic, sleep

    pipeline.ingest([document("a")])
    admitted, released = Event(), Event()

    def reader():
        service = RetrievalService(pipeline.config.data_dir, pipeline.store)
        with service.retrieve("default", "query", pipeline.embedder):
            admitted.set()
            ledger = ledger_for(pipeline)
            try:
                deadline = monotonic() + 5
                while ledger.pending("default") is None:
                    if monotonic() > deadline:
                        raise AssertionError("writer did not enter maintenance")
                    sleep(0.01)
            finally:
                ledger.close()
            released.set()

    write = pipeline.store.write

    def guarded_write(*args, **kwargs):
        assert released.is_set()
        write(*args, **kwargs)

    monkeypatch.setattr(pipeline.store, "write", guarded_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(reader)
        assert admitted.wait(5)
        pipeline.ingest([document("b")])
        task.result(timeout=5)


def test_recovery_after_closing_and_reopening_storage(tmp_path, monkeypatch):
    from rag_service.config import IngestionConfig
    from rag_service.storage.manifest import Manifest

    config = IngestionConfig(data_dir=tmp_path, chunking_strategy="token", chunk_tokens=8)
    store = QdrantVectorStore(path=tmp_path / "qdrant")
    audit = Manifest(tmp_path / "manifest.sqlite3")
    pipeline = IngestionPipeline(config, FakeEmbedder(), store, audit)
    first = pipeline.ingest([document("a")])
    write = store.write

    def interrupted(*args, **kwargs):
        write(*args, **kwargs)
        raise TimeoutError("process interrupted after writing")

    monkeypatch.setattr(store, "write", interrupted)
    try:
        with pytest.raises(TimeoutError):
            pipeline.ingest([document("a", "replacement")])
    finally:
        store.close()
        audit.close()
    reopened = QdrantVectorStore(path=tmp_path / "qdrant")
    reopened_audit = Manifest(tmp_path / "manifest.sqlite3")
    try:
        result = recover(tmp_path, "default", reopened, reopened_audit)
        assert result.generation_id == first.generation_id
        assert result.ingestion_revision == 2
        assert [c.text for c in reopened.iter_chunks(result.generation_id)] == ["replacement"]
    finally:
        reopened.close()
        reopened_audit.close()


def test_replay_batches_upserts_without_reordering_mutation_boundaries(pipeline, tmp_path):
    import json
    from unittest.mock import Mock

    from rag_service.ingestion.recovery import replay_mutations

    first = pipeline.ingest([document("a")])
    chunk = next(pipeline.store.iter_chunks(first.generation_id))
    batches = [
        {"kind": "upsert", "chunks": [chunk.model_dump()] * 300, "vectors": [[1.0] * 4] * 300},
        {"kind": "metadata", "chunks": [chunk.model_dump()]},
        {"kind": "upsert", "chunks": [chunk.model_dump()] * 10, "vectors": [[1.0] * 4] * 10},
        {"kind": "delete", "ids": [chunk.chunk_id]},
        {"kind": "upsert", "chunks": [chunk.model_dump()], "vectors": [[1.0] * 4]},
    ]
    path = tmp_path / "replay.jsonl"
    path.write_text("".join(json.dumps(batch) + "\n" for batch in batches))
    calls = []
    store = Mock()
    store.write.side_effect = lambda generation, chunks, vectors: calls.append(
        ("write", len(chunks))
    )
    store.update_chunks.side_effect = lambda generation, chunks: calls.append(
        ("metadata", len(chunks))
    )
    store.delete_chunks.side_effect = lambda generation, ids: calls.append(("delete", len(ids)))
    replay_mutations(store, first.generation_id, path)
    assert calls == [
        ("write", 256),
        ("write", 44),
        ("metadata", 1),
        ("write", 10),
        ("delete", 1),
        ("write", 1),
    ]
