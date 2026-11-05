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
