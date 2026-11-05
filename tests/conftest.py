import hashlib
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from rag_service.config import IngestionConfig
from rag_service.ingestion.pipeline import IngestionPipeline
from rag_service.schemas import EmbeddingSpec, SourceDocument
from rag_service.storage.manifest import Manifest
from rag_service.storage.qdrant import QdrantVectorStore


class WordTokenizer:
    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in re.finditer(r"\S+", text)]

    def count(self, text: str) -> int:
        return len(self.offsets(text))


class FakeEmbedder:
    """Deterministic test double, never selectable in the production CLI."""

    spec = EmbeddingSpec(model="test-only", revision="v1", dimension=4, max_tokens=512)
    tokenizer = WordTokenizer()

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Simulated inference failure")
        self.calls.extend(texts)
        return [
            [(byte + 1) / 256 for byte in hashlib.sha256(text.encode()).digest()[:4]]
            for text in texts
        ]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def pipeline(tmp_path: Path, embedder: FakeEmbedder) -> Iterator[IngestionPipeline]:
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    store = QdrantVectorStore(path=tmp_path / "qdrant")
    config = IngestionConfig(
        data_dir=tmp_path, chunk_tokens=8, overlap_tokens=2, batch_size=2, chunking_strategy="token"
    )
    try:
        yield IngestionPipeline(config, embedder, store, manifest)
    finally:
        store.close()
        manifest.close()


def document(
    document_id: str, text: str = "Daily backups are retained for thirty days."
) -> SourceDocument:
    return SourceDocument(document_id=document_id, text=text, source_uri=f"test://{document_id}")
