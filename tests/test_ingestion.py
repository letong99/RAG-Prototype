import json
from pathlib import Path

import pytest
from conftest import FakeEmbedder, document
from filelock import FileLock, Timeout

from rag_service.ingestion.pipeline import IngestionPipeline
from rag_service.schemas import EmbeddingSpec
from rag_service.storage.qdrant import QdrantVectorStore


def test_idempotency_and_persistent_vectors(
    pipeline: IngestionPipeline, embedder: FakeEmbedder
) -> None:
    first = pipeline.ingest([document("a")])
    second = pipeline.ingest([document("a")])
    assert first.status == "ready"
    assert second.status == "unchanged"
    assert second.generation_id == first.generation_id
    assert len(embedder.calls) == first.chunks
    assert pipeline.store.count(first.generation_id) == first.chunks


def test_snapshot_update_delete_and_cache(
    pipeline: IngestionPipeline, embedder: FakeEmbedder
) -> None:
    first = pipeline.ingest([document("a"), document("b", "Old policy for weekly backups.")])
    updated = pipeline.ingest([document("a"), document("c", "New policy applies every Friday.")])
    assert updated.generation_id != first.generation_id
    assert updated.cached_chunks == 1
    assert updated.embedded_chunks == 1
    path = pipeline.config.data_dir / "generations" / updated.generation_id / "documents.jsonl"
    ids = {json.loads(line)["document_id"] for line in path.read_text().splitlines()}
    assert ids == {"a", "c"}
    assert pipeline.manifest.active("default")["id"] == updated.generation_id


def test_failure_leaves_previous_generation_active_and_retry_works(
    pipeline: IngestionPipeline, embedder: FakeEmbedder
) -> None:
    first = pipeline.ingest([document("a")])
    embedder.fail = True
    with pytest.raises(RuntimeError, match="inference"):
        pipeline.ingest([document("changed", "Completely different passage.")])
    assert pipeline.manifest.active("default")["id"] == first.generation_id
    assert pipeline.manifest.generations("default")[0]["status"] == "failed"
    embedder.fail = False
    retried = pipeline.ingest([document("changed", "Completely different passage.")])
    assert retried.status == "ready"


def test_invalid_input_does_not_publish(pipeline: IngestionPipeline) -> None:
    with pytest.raises(ValueError, match="Duplicate document"):
        pipeline.ingest([document("a"), document("a")])
    with pytest.raises(ValueError, match="no usable text"):
        pipeline.ingest([document("a", " \n ")])
    with pytest.raises(ValueError, match="empty corpus"):
        pipeline.ingest([])
    assert pipeline.manifest.active("default") is None


def test_clear_requires_explicit_opt_in(pipeline: IngestionPipeline) -> None:
    pipeline.ingest([document("a")])
    cleared = pipeline.ingest([], allow_empty=True)
    assert cleared.documents == cleared.chunks == 0
    assert pipeline.store.count(cleared.generation_id) == 0


def test_recover_interrupted_generation(pipeline: IngestionPipeline) -> None:
    pipeline.manifest.begin("interrupted", "default", {})
    pipeline.ingest([document("a")])
    failed = next(
        row for row in pipeline.manifest.generations("default") if row["id"] == "interrupted"
    )
    assert failed["status"] == "failed"


def test_model_revision_invalidates_cache(
    pipeline: IngestionPipeline, embedder: FakeEmbedder
) -> None:
    pipeline.ingest([document("a")])
    embedder.spec = EmbeddingSpec(model="test-only", revision="v2", dimension=4, max_tokens=512)
    updated = pipeline.ingest([document("a")])
    assert updated.status == "ready"
    assert updated.embedded_chunks == 1


def test_concurrent_ingestion_is_rejected(pipeline: IngestionPipeline) -> None:
    with FileLock(pipeline.config.data_dir / "ingestion.lock"), pytest.raises(Timeout):
        pipeline.ingest([document("a")])


def test_vector_count_mismatch_prevents_publication(
    pipeline: IngestionPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = pipeline.ingest([document("a")])
    monkeypatch.setattr(pipeline.store, "count", lambda generation: 0)
    with pytest.raises(RuntimeError, match="count"):
        pipeline.ingest([document("b", "New source text.")])
    assert pipeline.manifest.active("default")["id"] == previous.generation_id


def test_source_order_does_not_change_snapshot(pipeline: IngestionPipeline) -> None:
    first = pipeline.ingest([document("a"), document("b", "Different evidence.")])
    reordered = pipeline.ingest([document("b", "Different evidence."), document("a")])
    assert reordered.status == "unchanged"
    assert reordered.generation_id == first.generation_id


def test_qdrant_survives_reopening(tmp_path: Path) -> None:
    from conftest import WordTokenizer

    from rag_service.chunking.token_chunker import TokenChunker
    from rag_service.ingestion.normalization import normalize

    chunks = list(TokenChunker(WordTokenizer(), 20, 2).chunks(normalize(document("a"))))
    store = QdrantVectorStore(path=tmp_path / "qdrant")
    store.create("test", 4)
    store.write("test", chunks, [[1.0, 0.0, 0.0, 0.0]])
    store.close()
    reopened = QdrantVectorStore(path=tmp_path / "qdrant")
    try:
        assert reopened.count("test") == 1
    finally:
        reopened.close()


@pytest.mark.parametrize("vector", [[1.0], [float("nan")] * 4, [0.0] * 4])
def test_invalid_vectors_are_rejected(
    pipeline: IngestionPipeline,
    embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    vector: list[float],
) -> None:
    monkeypatch.setattr(embedder, "embed", lambda texts: [vector for _ in texts])
    with pytest.raises(ValueError, match="Embedding"):
        pipeline.ingest([document("a")])
    assert pipeline.manifest.active("default") is None
