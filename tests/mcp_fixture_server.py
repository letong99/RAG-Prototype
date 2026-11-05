"""Subprocess fixture only: never a selectable production model backend."""

import sys
from pathlib import Path

from conftest import FakeEmbedder, document

from rag_service.config import IngestionConfig
from rag_service.ingestion.pipeline import IngestionPipeline
from rag_service.mcp_server import LocalRuntime, create_server
from rag_service.storage.manifest import Manifest
from rag_service.storage.qdrant import QdrantVectorStore

if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    embedder = FakeEmbedder()
    store = QdrantVectorStore(path=data_dir / "qdrant")
    manifest = Manifest(data_dir / "manifest.sqlite3")
    config = IngestionConfig(data_dir=data_dir, chunking_strategy="token")
    IngestionPipeline(config, embedder, store, manifest).ingest([document("a")])

    def load_embedder():
        print("Simulated noisy model loader")
        return embedder

    try:
        create_server(LocalRuntime(data_dir, "default", store, load_embedder)).run(
            transport="stdio"
        )
    finally:
        store.close()
        manifest.close()
