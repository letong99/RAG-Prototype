"""Choose the default semantic strategy or the retained token baseline for evaluation."""

from rag_service.chunking.semantic import SemanticChunker
from rag_service.chunking.token_chunker import TokenChunker
from rag_service.config import IngestionConfig
from rag_service.ports import Chunker, Embedder


def create_chunker(config: IngestionConfig, embedder: Embedder) -> Chunker:
    if config.chunking_strategy == "token":
        return TokenChunker(embedder.tokenizer, config.chunk_tokens, config.overlap_tokens)
    return SemanticChunker(
        embedder,
        config.chunk_tokens,
        config.overlap_tokens,
        breakpoint_percentile=config.semantic_percentile,
        min_distance=config.semantic_min_distance,
        window_sentences=config.semantic_window_sentences,
    )
