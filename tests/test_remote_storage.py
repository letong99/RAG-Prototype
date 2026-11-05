from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import FakeEmbedder, WordTokenizer, document

from rag_service.chunking.token_chunker import TokenChunker
from rag_service.cli import main
from rag_service.config import QdrantSettings
from rag_service.ingestion.normalization import normalize
from rag_service.ingestion.pipeline import IngestionPipeline
from rag_service.schemas import IndexMetadata
from rag_service.storage.manifest import Manifest
from rag_service.storage.qdrant import QdrantVectorStore


def test_remote_is_default_and_missing_url_fails_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "must-not-appear-in-errors")
    assert main(["--data-dir", str(tmp_path), "ingest", "--directory", str(tmp_path)]) == 1
    assert "QDRANT_URL" in caplog.text
    assert "must-not-appear-in-errors" not in caplog.text
    assert not (tmp_path / "cache" / "models").exists()
    assert not (tmp_path / "qdrant").exists()


def test_remote_credentials_and_timeout_are_passed_to_official_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructor = MagicMock()
    monkeypatch.setattr("rag_service.storage.qdrant.QdrantClient", constructor)
    settings = QdrantSettings.model_validate(
        {
            "url": "https://cluster.example:6333",
            "api_key": "private-value",
            "timeout_seconds": 45,
        }
    )
    store = QdrantVectorStore.connect(settings, tmp_path)
    constructor.assert_called_once_with(
        url="https://cluster.example:6333/", api_key="private-value", timeout=45
    )
    assert "private-value" not in settings.model_dump_json()
    assert "private-value" not in repr(settings)
    store.create("new", 4)
    assert constructor.return_value.create_payload_index.call_count == 2
    store.close()


def test_embedded_mode_must_be_explicit_and_cannot_hide_remote_credentials() -> None:
    with pytest.raises(ValueError, match="QDRANT_URL"):
        QdrantSettings()
    with pytest.raises(ValueError, match="credentials"):
        QdrantSettings.model_validate({"mode": "embedded", "api_key": "private"})
    assert QdrantSettings(mode="embedded").mode == "embedded"


def test_official_qdrant_query_ranking_payloads_and_filtering(tmp_path: Path) -> None:
    store = QdrantVectorStore(path=tmp_path / "qdrant")
    chunker = TokenChunker(WordTokenizer(), 20, 0)
    chunks = [
        next(chunker.chunks(normalize(document("a", "DNA repair.")))),
        next(chunker.chunks(normalize(document("b", "Database backups.")))),
    ]
    try:
        store.create("test", 4)
        store.write("test", chunks, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        hits = store.search("test", [1.0, 0.0, 0.0, 0.0], limit=2)
        assert hits[0].chunk.document_id == "a"
        assert hits[0].score > hits[1].score
        assert hits[0].chunk.text == "DNA repair."
        filtered = store.search("test", [1.0, 0.0, 0.0, 0.0], document_ids=["b"])
        assert [hit.chunk.document_id for hit in filtered] == ["b"]
        assert store.search("test", [1.0, 0.0, 0.0, 0.0], document_ids=[]) == []
    finally:
        store.close()


def test_alias_metadata_is_readable_without_local_ingestion_manifest(
    pipeline: IngestionPipeline, tmp_path: Path
) -> None:
    result = pipeline.ingest([document("a")])
    active = pipeline.store.active_index("default")
    assert active is not None
    assert active.generation_id == result.generation_id
    assert active.embedding == pipeline.embedder.spec
    # A worker with an entirely separate local ledger still discovers and reuses the active index.
    other_manifest = Manifest(tmp_path / "another-worker" / "manifest.sqlite3")
    try:
        other = IngestionPipeline(pipeline.config, FakeEmbedder(), pipeline.store, other_manifest)
        assert other.ingest([document("a")]).status == "unchanged"
    finally:
        other_manifest.close()


def test_alias_switch_does_not_change_a_pinned_reader(pipeline: IngestionPipeline) -> None:
    first = pipeline.ingest([document("a")])
    pinned = pipeline.store.active_index("default")
    second = pipeline.ingest([document("b", "Another document.")])
    assert pinned is not None
    assert pinned.generation_id == first.generation_id
    assert pipeline.store.active_index("default").generation_id == second.generation_id
    assert pipeline.store.count(pinned.generation_id) == first.chunks


def test_lost_publication_acknowledgment_reconciles_on_retry(
    pipeline: IngestionPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish = pipeline.store.publish

    def lost_ack(index: IndexMetadata) -> None:
        publish(index)
        raise TimeoutError("Simulated lost publication acknowledgment")

    monkeypatch.setattr(pipeline.store, "publish", lost_ack)
    with pytest.raises(TimeoutError):
        pipeline.ingest([document("a")])
    active = pipeline.store.active_index("default")
    assert active is not None
    assert pipeline.manifest.pending_publications("default")
    monkeypatch.setattr(pipeline.store, "publish", publish)
    retried = pipeline.ingest([document("a")])
    assert retried.status == "unchanged"
    assert retried.generation_id == active.generation_id
    assert pipeline.manifest.pending_publications("default") == []


def test_ambiguous_publication_prevents_blind_republication(
    pipeline: IngestionPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = pipeline.ingest([document("a")])

    def timeout(index: IndexMetadata) -> None:
        raise TimeoutError("Simulated publication timeout")

    monkeypatch.setattr(pipeline.store, "publish", timeout)
    with pytest.raises(TimeoutError):
        pipeline.ingest([document("b", "Changed content.")])
    assert pipeline.store.active_index("default").generation_id == first.generation_id
    with pytest.raises(RuntimeError, match="unresolved"):
        pipeline.ingest([document("b", "Changed content.")])
