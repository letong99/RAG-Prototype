"""Inspect segmentation without creating a corpus generation or a vector index."""

import json
from pathlib import Path
from typing import Any

from rag_service.chunking.factory import create_chunker
from rag_service.config import IngestionConfig
from rag_service.embeddings.cached import CachedEmbedder
from rag_service.ingestion.normalization import normalize
from rag_service.schemas import SourceDocument


def preview_file(
    path: Path, config: IngestionConfig, embedder: CachedEmbedder, output: Path | None = None
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Chunk preview accepts a single .txt or .md file")
    document = normalize(
        SourceDocument(
            document_id=path.name,
            text=path.read_text(encoding="utf-8"),
            source_uri=path.as_uri(),
            title=path.stem,
        )
    )
    chunks = list(create_chunker(config, embedder).chunks(document))
    report = {
        "strategy": config.chunking_strategy,
        "model": embedder.spec.model_dump(),
        "max_chunk_tokens": config.chunk_tokens,
        "chunks": [chunk.model_dump() for chunk in chunks],
        "semantic_embedded_texts": embedder.computed_texts,
        "semantic_cached_texts": embedder.reused_texts,
        "indexed": False,
    }
    if output is not None:
        if output.resolve() == path:
            raise ValueError("Preview output must not overwrite its source")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"output": str(output), "chunks": len(chunks), "indexed": False}
    return report
