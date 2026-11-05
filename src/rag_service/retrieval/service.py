"""Version-checked dense retrieval for the single-coordinator deployment."""

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from rag_service.ingestion.lifecycle import Ledger, payload_hash
from rag_service.ports import Embedder
from rag_service.retrieval.models import RankedPassage, SearchRequest, SearchResult
from rag_service.retrieval.ranking import Reranker
from rag_service.schemas import IndexMetadata, SearchHit
from rag_service.storage.qdrant import QdrantVectorStore


class RetrievalService:
    def __init__(self, data_dir: Path, store: QdrantVectorStore) -> None:
        self.data_dir = data_dir
        self.store = store

    def search(
        self, request: SearchRequest, embedder: Embedder, *, reranker: Reranker | None = None
    ) -> SearchResult:
        """Retrieve a candidate pool and rerank while holding the ingestion read gate."""
        started = perf_counter()
        with self.retrieve(
            request.collection,
            request.query,
            embedder,
            limit=request.candidate_limit,
            document_ids=request.document_ids,
        ) as (index, hits):
            # Exact chunk identity deduplication retains distinct evidence from a document.
            unique = list({hit.chunk.chunk_id: hit for hit in reversed(hits)}.values())
            unique.reverse()
            if any(not math.isfinite(hit.score) for hit in unique):
                raise RuntimeError("Retriever returned non-finite scores")
            scores = (
                reranker.score(request.query, [hit.chunk.text for hit in unique])
                if (reranker and unique)
                else []
            )
            if reranker and (
                len(scores) != len(unique) or any(not math.isfinite(score) for score in scores)
            ):
                raise RuntimeError("Reranker returned invalid scores")
            positions = list(range(len(unique)))
            if reranker:
                positions.sort(key=lambda position: (-scores[position], position))
            passages = [
                RankedPassage(
                    rank=rank,
                    dense_rank=position + 1,
                    dense_score=unique[position].score,
                    rerank_score=scores[position] if reranker else None,
                    chunk=unique[position].chunk,
                )
                for rank, position in enumerate(positions[: request.top_n], start=1)
            ]
            return SearchResult(
                request=request,
                index=index,
                ranker=reranker.spec if reranker else None,
                candidate_count=len(unique),
                passages=passages,
                elapsed_seconds=perf_counter() - started,
            )

    @contextmanager
    def retrieve(
        self,
        collection: str,
        query: str,
        embedder: Embedder,
        *,
        limit: int = 6,
        document_ids: Sequence[str] | None = None,
    ) -> Iterator[tuple[IndexMetadata, list[SearchHit]]]:
        """Keep this context open through reranking and context assembly to drain safely."""
        if not query.strip() or len(query) > 16000:
            raise ValueError("Query must contain text and be at most 16000 characters")
        if not 1 <= limit <= 100:
            raise ValueError("Candidate limit must be between 1 and 100")
        if embedder.tokenizer.count(query) > embedder.spec.max_tokens:
            raise ValueError("Query exceeds embedding token budget")
        ledger = Ledger(self.data_dir)
        try:
            with ledger.admission(collection) as index:
                if (
                    index.coordinator_id != ledger.identity
                    or self.store.active_index(collection) != index
                ):
                    raise RuntimeError("Remote publication and coordinator manifest disagree")
                if embedder.spec != index.embedding:
                    raise ValueError("Query embedder must match the committed model specification")
                entries = ledger.entries(index.generation_id)
                hits = self.store.search(
                    index.generation_id,
                    embedder.embed([query])[0],
                    limit=limit,
                    document_ids=document_ids,
                )
                for hit in hits:
                    chunk = hit.chunk
                    entry = entries.get(chunk.document_id)
                    if (
                        entry is None
                        or chunk.document_version != entry.document.version
                        or chunk.processing_fingerprint != index.processing_fingerprint
                        or chunk.generation_number != index.generation_number
                        or entry.chunk_hashes.get(chunk.chunk_id) != payload_hash(chunk)
                    ):
                        raise RuntimeError(
                            "Retrieval integrity mismatch; context cannot be returned"
                        )
                yield index, hits
        finally:
            ledger.close()
