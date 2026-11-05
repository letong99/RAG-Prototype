import os
from pathlib import Path

import pytest
from conftest import document

from rag_service.chunking.semantic import SemanticChunker
from rag_service.chunking.token_chunker import TokenChunker
from rag_service.embeddings.cached import CachedEmbedder
from rag_service.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_service.ingestion.normalization import normalize
from rag_service.storage.manifest import Manifest


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("RAG_TEST_MODEL") != "1", reason="opt-in model download")
@pytest.mark.parametrize("strategy", ["token", "semantic"])
def test_real_model_limits_and_embeddings(tmp_path: Path, strategy: str) -> None:
    embedder = SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        Path(".data/cache/models"),
        offline=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    source = normalize(document("a", "β-catenin participates in signaling. " * 300))
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        cached = CachedEmbedder(embedder, manifest, 16)
        chunker = (
            SemanticChunker(cached, 300)
            if strategy == "semantic"
            else TokenChunker(embedder.tokenizer, 300, 40)
        )
        chunks = list(chunker.chunks(source))
        assert len(chunks) > 1
        assert all(chunk.token_count <= 300 for chunk in chunks)
        vectors = cached.embed([chunk.text for chunk in chunks])
        assert len(vectors) == len(chunks)
        assert all(len(vector) == 384 for vector in vectors)
    finally:
        manifest.close()
