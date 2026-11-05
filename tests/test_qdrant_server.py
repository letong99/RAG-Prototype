"""Opt-in SDK contract test against a real Qdrant server; never uses the production URL."""

import os
from uuid import uuid4

import pytest
from conftest import FakeEmbedder, WordTokenizer, document
from qdrant_client import QdrantClient

from rag_service.chunking.token_chunker import TokenChunker
from rag_service.ingestion.normalization import normalize
from rag_service.schemas import IndexMetadata
from rag_service.storage.qdrant import QdrantVectorStore


@pytest.mark.remote
@pytest.mark.skipif(not os.environ.get("RAG_TEST_QDRANT_URL"), reason="no remote test endpoint")
def test_remote_index_publish_query_and_cleanup() -> None:
    url = os.environ["RAG_TEST_QDRANT_URL"]
    api_key = os.environ.get("RAG_TEST_QDRANT_API_KEY")
    generation = "integration_" + uuid4().hex
    store = QdrantVectorStore(url=url, api_key=api_key)
    client = QdrantClient(url=url, api_key=api_key)
    chunk = next(TokenChunker(WordTokenizer(), 20, 0).chunks(normalize(document("fixture"))))
    index = IndexMetadata(
        generation_id=generation,
        collection=generation,
        fingerprint="integration-fixture",
        embedding=FakeEmbedder.spec,
        documents=1,
        chunks=1,
    )
    try:
        store.create(generation, 4, metadata=index)
        store.write(generation, [chunk], [[1.0, 0.0, 0.0, 0.0]])
        store.publish(index)
        assert store.active_index(generation) == index
        hits = store.search(generation, [1.0, 0.0, 0.0, 0.0], document_ids=["fixture"])
        assert hits[0].chunk == chunk
    finally:
        try:
            # Qdrant removes aliases associated with the deleted collection.
            if client.collection_exists(store.collection_name(generation)):
                client.delete_collection(store.collection_name(generation))
        finally:
            client.close()
            store.close()


@pytest.mark.remote
@pytest.mark.skipif(not os.environ.get("RAG_TEST_QDRANT_URL"), reason="no remote test endpoint")
def test_remote_incremental_metadata_delete_and_fallback(tmp_path) -> None:
    from rag_service.config import IngestionConfig
    from rag_service.ingestion.maintenance import rollback
    from rag_service.ingestion.pipeline import IngestionPipeline
    from rag_service.storage.manifest import Manifest

    corpus = "lifecycle_" + uuid4().hex
    store = QdrantVectorStore(
        url=os.environ["RAG_TEST_QDRANT_URL"], api_key=os.environ.get("RAG_TEST_QDRANT_API_KEY")
    )
    client = QdrantClient(
        url=os.environ["RAG_TEST_QDRANT_URL"], api_key=os.environ.get("RAG_TEST_QDRANT_API_KEY")
    )
    audit = Manifest(tmp_path / "manifest.sqlite3")
    config = IngestionConfig(data_dir=tmp_path, collection=corpus, chunking_strategy="token")
    pipeline = IngestionPipeline(config, FakeEmbedder(), store, audit)
    generations = set()
    try:
        # The test endpoint must have at least 256 MB free for these tiny disposable fixtures.
        first = pipeline.ingest([document("a"), document("b")], available_storage_bytes=256_000_000)
        generations.add(first.generation_id)
        changed = document("a").model_copy(update={"title": "Updated citation"})
        second = pipeline.ingest([changed], delete_document_ids=["b"])
        assert second.generation_id == first.generation_id
        assert second.documents == 1
        assert next(store.iter_chunks(first.generation_id)).title == "Updated citation"
        third = pipeline.ingest(
            [document("c")], full_rebuild=True, available_storage_bytes=256_000_000
        )
        generations.add(third.generation_id)
        restored = rollback(tmp_path, corpus, store, audit)
        assert restored.generation_id == first.generation_id
        assert restored.ingestion_revision == 4
    finally:
        # Include a candidate created before an interrupted run returned a generation ID.
        import json

        from rag_service.ingestion.lifecycle import Ledger

        ledger = Ledger(tmp_path)
        try:
            for row in ledger.db.execute("SELECT operation_json FROM operations"):
                generations.add(json.loads(row[0])["index"]["generation_id"])
        finally:
            ledger.close()
        for generation in generations:
            if client.collection_exists(store.collection_name(generation)):
                client.delete_collection(store.collection_name(generation))
        store.close()
        client.close()
        audit.close()
